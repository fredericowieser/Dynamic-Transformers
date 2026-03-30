from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
)

from ..base.block import DynamicBlock
from ..base.causal_lm import BaseForCausalLM
from ..base.priors import BasePriorNetwork
from ..base.routers import BaseSurpriseRouter, UnifiedCausalRouter
from ..configs import SDTConfig


class SDTPriorNetwork(BasePriorNetwork):
    def __init__(self, config, model_params: Dict):
        super().__init__(config)
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(self.norm(x))


class SDTRouter(BaseSurpriseRouter):
    def __init__(self, config, model_params: Dict):
        super().__init__(config, capacity_attr="capacity")

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        actual_residual, mu_q, log_var_q = args[0], args[1], args[2]
        beta_ce, beta_cu = kwargs["beta_ce"], kwargs["beta_cu"]
        c = getattr(self.config, "posterior_variance_c", 1.0)

        d = float(actual_residual.shape[-1])
        D_st = torch.sum(actual_residual.pow(2), dim=-1) / d
        
        # D_ch is now the KL Divergence
        D_ch = self.compute_kl_divergence(actual_residual, mu_q, log_var_q, c)

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

        # Get mean and log variance from prior
        mu_q, log_var_q = self.prior(original_hidden)

        actual_residual = processed_hidden - original_hidden

        return processed_hidden, actual_residual, mu_q, log_var_q


class SDTPair(nn.Module):
    def __init__(self, hf_layer: Qwen2DecoderLayer, config, model_params: Dict, layer_idx: int):
        super().__init__()
        self.decision = SDTDecisionLayer(config, model_params, layer_idx)
        self.router = SDTRouter(config, model_params=model_params)
        self.dynamic = DynamicBlock(hf_layer)
        self.causal_router = UnifiedCausalRouter(config)
        self.model_params = model_params
        self.config = config
        self.causal_threshold = getattr(config, "causal_threshold", 0.5)

    def forward(self, hidden_states, use_causal_router: bool = False, **kwargs):
        # Decision Phase
        processed_hidden, actual_res, mu_q, log_var_q = self.decision(
            hidden_states, **kwargs
        )

        layer_losses = {}
        router_stats = {}
        if self.training:
            c = getattr(self.config, "posterior_variance_c", 1.0)
            kl_div_per_token = BaseSurpriseRouter.compute_kl_divergence(actual_res.detach(), mu_q, log_var_q, c)
            layer_losses["sdt_prior_loss"] = kl_div_per_token.mean()

        B, T, D = hidden_states.shape

        if use_causal_router:
            causal_logits, _, _ = self.causal_router(hidden_states)
            gating_probs = torch.sigmoid(causal_logits)
            
            selected_mask = gating_probs >= self.causal_threshold
            batch_idx, token_idx = selected_mask.nonzero(as_tuple=True)
            gating_scores_for_selected = gating_probs[selected_mask]
            
            if B * T > 0:
                router_stats["selected_tokens_proportion"] = (selected_mask.sum() / (B * T)).item()
            router_stats["g_cont"] = gating_probs.mean().item()
        else:
            # Routing Phase
            g_cont, binary_targets, surprise_stats = self.router(
                actual_res, mu_q, log_var_q, **kwargs
            )
            
            causal_logits, causal_loss, causal_acc = self.causal_router(hidden_states.detach(), targets=binary_targets)

            router_stats.update(surprise_stats)
            router_stats["g_cont"] = g_cont.mean().item()
            
            layer_losses["causal_router_loss"] = causal_loss
            if causal_acc is not None:
                router_stats["causal_router_acc"] = causal_acc # Tensor for gathering

            # Dynamic Execution Phase
            k = max(1, int(T * self.router.capacity))
            gating_scores, topk_idx = g_cont.topk(k, dim=-1)

            # Sort indices to ensure causal processing (past cannot see future)
            topk_idx, sort_indices = topk_idx.sort(dim=-1)
            gating_scores = gating_scores.gather(dim=-1, index=sort_indices)

            batch_idx = torch.arange(B, device=g_cont.device).unsqueeze(1).expand(-1, k).reshape(-1)
            token_idx = topk_idx.reshape(-1)
            gating_scores_for_selected = gating_scores.reshape(-1)

            if B * T > 0:
                router_stats["selected_tokens_proportion"] = topk_idx.numel() / (B * T)

        final_hidden_states, _, _ = self.dynamic.process_selected(
            processed_hidden,
            batch_indices=batch_idx,
            token_indices=token_idx,
            gating_scores=gating_scores_for_selected,
            use_soft_gating=True,
            **kwargs,
        )

        return final_hidden_states, layer_losses, router_stats


class SDTForCausalLM(BaseForCausalLM):
    config_class = SDTConfig
    _supports_flash_attn_2 = False

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
        if self.training and self.gradient_checkpointing:
            if not hidden_states.requires_grad:
                hidden_states = hidden_states + (self.gradient_checkpointing_trigger * 0.0)

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

        use_causal_router = kwargs.get("use_causal_router", getattr(self.config, "use_causal_router_in_validation", False))
        if self.training:
            use_causal_router = False

        layer_args = {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
            "output_attentions": output_attentions,
            "beta_ce": beta_ce,
            "beta_cu": beta_cu,
            "use_causal_router": use_causal_router,
            **kwargs,
        }

        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, SDTPair):
                layer_attn_mask = mask_mapping[layer.dynamic.layer.attention_type]
                
                if self.gradient_checkpointing and self.training:
                    # Clean up checkpoint call to ensure hidden_states is the primary positional arg
                    hidden_states, losses, stats = torch.utils.checkpoint.checkpoint(
                        layer.__call__,
                        hidden_states,
                        use_reentrant=False,
                        attention_mask=layer_attn_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        output_attentions=output_attentions,
                        beta_ce=beta_ce,
                        beta_cu=beta_cu,
                        use_causal_router=use_causal_router,
                        **kwargs,
                    )
                else:
                    hidden_states, losses, stats = layer(
                        hidden_states,
                        attention_mask=layer_attn_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        output_attentions=output_attentions,
                        beta_ce=beta_ce,
                        beta_cu=beta_cu,
                        use_causal_router=use_causal_router,
                        **kwargs,
                    )
                all_losses.append(losses)
                for key, value in stats.items():
                    all_router_stats[f"sdt/layer_{i}/{key}"] = value

                # Collect g_cont for logging
                if "g_cont" in stats:
                    all_g_cont_values.append(stats["g_cont"])

            elif isinstance(layer, nn.Identity):
                continue
            else:  # Standard Qwen2DecoderLayer
                attn_mask = mask_mapping[layer.attention_type]
                
                if self.gradient_checkpointing and self.training:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        layer.__call__,
                        hidden_states,
                        use_reentrant=False,
                        attention_mask=attn_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states=hidden_states,
                        attention_mask=attn_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                    hidden_states = (
                        layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
                    )

        aux = {}
        # FIX: Aggregating losses robustly
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
