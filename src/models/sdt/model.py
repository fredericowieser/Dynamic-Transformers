from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import (Qwen2DecoderLayer,
                                                      Qwen2RMSNorm)

from ..base.block import DynamicBlock
from ..base.causal_lm import BaseForCausalLM
from ..base.priors import BasePriorNetwork
from ..base.routers import BaseSurpriseRouter, CausalRouter
from ..configs import SDTConfig


class SDTPriorNetwork(BasePriorNetwork):
    def __init__(self, config, model_params: Dict):
        super().__init__(config, model_cfg=model_params)
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(x))


class SDTRouter(BaseSurpriseRouter):
    def __init__(self, config, model_params: Dict):
        super().__init__(config, capacity_attr="capacity")

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        actual_residual, predicted_residual = args[0], args[1]
        beta_ce, beta_cu = kwargs["beta_ce"], kwargs["beta_cu"]

        d = float(actual_residual.shape[-1])
        D_st = torch.sum(actual_residual.pow(2), dim=-1) / d
        D_ch = torch.sum((actual_residual - predicted_residual).pow(2), dim=-1) / d

        g_cont, stats = self._get_vpr_signals(D_st, D_ch, beta_ce, beta_cu)

        # Also return binary targets for training the causal router
        B, T = g_cont.shape
        k = max(1, int(T * self.capacity))
        _, topk_idx = g_cont.topk(k, dim=-1)
        binary_targets = torch.zeros_like(g_cont)
        binary_targets.scatter_(1, topk_idx, 1.0)

        return g_cont, binary_targets, stats


class SDTDecisionLayer(nn.Module):
    def __init__(self, config, model_params: Dict, layer_idx: int):
        super().__init__()
        self.block = DynamicBlock(Qwen2DecoderLayer(config, layer_idx))
        self.prior = SDTPriorNetwork(config, model_params=model_params)

    def forward(self, hidden_states, **kwargs):
        original_hidden = hidden_states
        out = self.block(hidden_states, **kwargs)
        processed_hidden = out[0] if isinstance(out, tuple) else out

        prior_output = self.prior(original_hidden)
        prior_hidden = original_hidden + prior_output

        actual_residual = processed_hidden - original_hidden
        predicted_residual = prior_hidden - original_hidden

        prior_loss = F.mse_loss(prior_hidden, processed_hidden.detach())

        return processed_hidden, actual_residual, predicted_residual, prior_loss


class SDTPair(nn.Module):
    def __init__(self, hf_layer: Qwen2DecoderLayer, config, model_params: Dict, layer_idx: int):
        super().__init__()
        self.decision = SDTDecisionLayer(config, model_params, layer_idx)
        self.router = SDTRouter(config, model_params=model_params)
        self.dynamic = DynamicBlock(hf_layer)

        self.train_causal_router = model_params.get("train_causal_router", True)
        if self.train_causal_router:
            self.causal_router = CausalRouter(
                config, layer_idx=layer_idx, capacity_attr="capacity", model_cfg=model_params
            )
        else:
            self.causal_router = None
        self.model_params = model_params

    def forward(self, hidden_states, **kwargs):
        processed_hidden, actual_res, predicted_res, prior_loss = self.decision(
            hidden_states, **kwargs
        )

        layer_losses = {"sdt_prior_loss": prior_loss}
        router_stats = {}

        if self.training:
            g_cont, binary_targets, surprise_stats = self.router(
                actual_res, predicted_res, **kwargs
            )
            router_stats.update(surprise_stats)

            if self.causal_router is not None:
                causal_logits, _, _ = self.causal_router(hidden_states.detach())
                causal_loss = F.binary_cross_entropy_with_logits(
                    causal_logits, binary_targets.detach()
                )
                layer_losses["sdt_causal_router_loss"] = causal_loss

                # Calculate causal router accuracy
                causal_preds = (torch.sigmoid(causal_logits) > 0.5).float()
                causal_accuracy = (causal_preds == binary_targets.detach()).float().mean().item()
                router_stats["causal_router_accuracy"] = causal_accuracy

            router_stats["g_cont"] = g_cont.mean().item()

            B, T, D = hidden_states.shape
            k = max(1, int(T * self.router.capacity))
            gating_scores, topk_idx = g_cont.topk(k, dim=-1)
            batch_idx = torch.arange(B, device=g_cont.device).unsqueeze(1).expand(-1, k)

            router_stats["selected_tokens_proportion"] = topk_idx.numel() / (B * T)

            gating_scores_for_selected = gating_scores.reshape(-1)  # Reshape to 1D

            final_hidden_states, _, _ = self.dynamic.process_selected(
                processed_hidden,
                batch_indices=batch_idx.reshape(-1),
                token_indices=topk_idx.reshape(-1),
                gating_scores=gating_scores_for_selected,
                use_soft_gating=True,
                **kwargs,
            )
        else:  # Inference
            use_causal = self.model_params.get("use_causal_router_in_validation", True)

            if use_causal and self.causal_router is not None:
                causal_logits, _, _ = self.causal_router(hidden_states)

                B, T, D = hidden_states.shape
                k = max(1, int(T * self.causal_router.capacity))
                gating_scores, topk_idx = causal_logits.topk(k, dim=-1)
                batch_idx = torch.arange(B, device=causal_logits.device).unsqueeze(1).expand(-1, k)

                gating_scores_for_selected = gating_scores.reshape(-1)  # Reshape to 1D

                final_hidden_states, _, _ = self.dynamic.process_selected(
                    processed_hidden,
                    batch_indices=batch_idx.reshape(-1),
                    token_indices=topk_idx.reshape(-1),
                    gating_scores=gating_scores_for_selected,
                    use_soft_gating=False,
                    **kwargs,
                )
                router_stats["inferred_selected_tokens_proportion"] = topk_idx.numel() / (B * T)
            else:  # Use training-time router for validation
                # Run the main router to get selection decisions
                g_cont, binary_targets, surprise_stats = self.router(
                    actual_res, predicted_res, **kwargs
                )
                router_stats.update(surprise_stats)
                router_stats["g_cont"] = g_cont.mean().item()

                B, T, D = hidden_states.shape
                k = max(1, int(T * self.router.capacity))
                gating_scores, topk_idx = g_cont.topk(k, dim=-1)
                batch_idx = torch.arange(B, device=g_cont.device).unsqueeze(1).expand(-1, k)

                gating_scores_for_selected = gating_scores.reshape(-1)  # Reshape to 1D

                final_hidden_states, _, _ = self.dynamic.process_selected(
                    processed_hidden,
                    batch_indices=batch_idx.reshape(-1),
                    token_indices=topk_idx.reshape(-1),
                    gating_scores=gating_scores_for_selected,
                    use_soft_gating=True,  # Use soft gating as in training
                    **kwargs,
                )
                router_stats["selected_tokens_proportion"] = topk_idx.numel() / (B * T)

        return final_hidden_states, layer_losses, router_stats


class SDTForCausalLM(BaseForCausalLM):
    config_class = SDTConfig
    _supports_flash_attn_2 = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        for i in range(0, self.config.num_hidden_layers - 1, 2):
            self.model.layers[i] = SDTPair(self.model.layers[i + 1], config, self.model_params, i)
            self.model.layers[i + 1] = nn.Identity()

    def _run_layers(
        self,
        hidden_states,
        mask_mapping,
        position_ids,
        past_key_values,
        use_cache,
        cache_position,
        position_embeddings,
        output_attentions,
        **kwargs,
    ):
        all_losses = []
        all_router_stats: Dict[str, Any] = {}
        all_g_cont_values = []  # Initialize for collecting g_cont
        cfg = self.config

        beta_ce, beta_cu = 1.0, 1.0  # Default if no schedule is found
        if hasattr(self.config, "beta_schedule"):
            sched_cfg = self.config.beta_schedule
            global_step = kwargs.get("global_step", 0)
            max_steps = kwargs.get("max_steps", 100000)  # max_steps from training loop
            warmup = sched_cfg["warmup_steps"]

            # Calculate betas based on schedule, regardless of training mode
            if global_step > warmup:
                progress = (global_step - warmup) / (max_steps - warmup)
                progress = min(progress, 1.0)
                beta_ce = sched_cfg["beta_ce_start"] + progress * (
                    sched_cfg["beta_ce_end"] - sched_cfg["beta_ce_start"]
                )
                beta_cu = sched_cfg["beta_cu_start"] + progress * (
                    sched_cfg["beta_cu_end"] - sched_cfg["beta_cu_start"]
                )
            else:
                beta_ce = sched_cfg["beta_ce_start"]
                beta_cu = sched_cfg["beta_cu_start"]

        layer_args = {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
            "output_attentions": output_attentions,
            "beta_ce": beta_ce,
            "beta_cu": beta_cu,
            **kwargs,
        }

        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, SDTPair):
                layer_args["attention_mask"] = mask_mapping[layer.dynamic.layer.attention_type]
                hidden_states, losses, stats = layer(
                    hidden_states,
                    **layer_args,
                )
                if self.training:
                    all_losses.append(losses)
                    for key, value in stats.items():
                        all_router_stats[f"sdt/layer_{i}/{key}"] = value

                    # Collect g_cont for logging
                    if "g_cont" in stats:
                        all_g_cont_values.append(stats["g_cont"])

            elif isinstance(layer, nn.Identity):
                continue
            else:  # Standard Qwen2DecoderLayer
                layer_args["attention_mask"] = mask_mapping[layer.attention_type]
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    **layer_args,
                )
                hidden_states = (
                    layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
                )

        aux = {}
        if self.training:
            # FIX: Use a robust loop for aggregating losses instead of a brittle comprehension.
            agg_losses = {}
            if all_losses:
                all_keys = set(k for l in all_losses for k in l)
                for k in all_keys:
                    key_losses = [l[k] for l in all_losses if k in l and l.get(k) is not None]
                    if key_losses:
                        agg_losses[k] = torch.mean(torch.stack(key_losses))
            aux["unscaled_losses"] = agg_losses
            aux["router_stats"] = all_router_stats
            aux["beta_ce"] = beta_ce
            aux["beta_cu"] = beta_cu

            # Add g_cont mean across layers to router_stats
            if all_g_cont_values:
                aux["router_stats"]["sdt_g_cont_mean_across_layers"] = sum(all_g_cont_values) / len(
                    all_g_cont_values
                )

        return hidden_states, aux

    def get_trainable_parameters(self):
        params_map = {"sdt_prior": [], "sdt_router": [], "sdt_causal_router": [], "base_model": []}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "prior" in n:
                params_map["sdt_prior"].append(p)
            elif "causal_router" in n and self.model_params.get("train_causal_router", True):
                params_map["sdt_causal_router"].append(p)
            elif "router" in n:
                params_map["sdt_router"].append(p)
            else:
                params_map["base_model"].append(p)
        return [{"name": k, "params": v} for k, v in params_map.items() if v]
