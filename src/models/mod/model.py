from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from ..base.block import DynamicBlock
from ..base.causal_lm import BaseForCausalLM
from ..base.routers import BaseRouter
from ..configs import MoDConfig


class MoDRouter(BaseRouter):
    def __init__(self, config, layer_idx: int, model_params: Dict):
        super().__init__(config, capacity_attr="capacity")
        self.router = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states, **kwargs):
        logits = self.router(hidden_states).squeeze(-1)

        B, T = logits.shape
        k = max(1, int(T * self.capacity))
        if k > T:
            k = T

        gating_scores, topk_idx = logits.topk(k, dim=-1)

        binary_targets = torch.zeros_like(logits)
        binary_targets.scatter_(1, topk_idx, 1.0)

        # Return the raw, unscaled loss
        router_bce_loss = F.binary_cross_entropy_with_logits(logits, binary_targets)

        # Calculate Z-loss (L2 penalty on logits) to prevent drift
        router_z_loss = torch.mean(logits**2)

        return logits, router_bce_loss, router_z_loss, binary_targets, gating_scores, topk_idx


class MoDCausalRouter(nn.Module):
    """
    A small MLP to predict the top-k selection of the main router for causal inference,
    as described in the MoD paper (Section 3.5, Method 2).
    It takes the same input as the main router.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        # A simple 2-layer MLP
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_size // 4, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.fc1(hidden_states)
        x = self.act(x)
        x = self.fc2(x)
        return x.squeeze(-1)


class MoDLayer(nn.Module):
    def __init__(self, hf_layer: Qwen2DecoderLayer, config, model_params: Dict):
        super().__init__()
        self.block = DynamicBlock(hf_layer)
        self.router = MoDRouter(config, layer_idx=0, model_params=model_params)

        self.train_causal_router = model_params.get("train_causal_router", True)
        if self.train_causal_router:
            self.causal_router = MoDCausalRouter(config)
        else:
            self.causal_router = None
        self.model_params = model_params

    def forward(self, hidden_states, training: bool, **kwargs):
        layer_losses = {}
        layer_metrics = {}

        if training:
            (
                scores,
                router_bce_loss,
                router_z_loss,
                binary_targets,
                gating_scores,
                topk_idx,
            ) = self.router(hidden_states)
            layer_losses["mod_aux_loss"] = router_bce_loss
            layer_losses["mod_z_loss"] = router_z_loss * 1e-4  # Apply coefficient here

            # FIX: Apply sigmoid to ensure gating scores are probabilities (0-1) instead of raw logits.
            # This prevents exploding residuals where the scalar multiplier grows unbounded.
            gating_probs = torch.sigmoid(gating_scores)

            if self.causal_router is not None:
                predictor_logits = self.causal_router(hidden_states.detach())
                predictor_loss = F.binary_cross_entropy_with_logits(
                    predictor_logits, binary_targets.detach()
                )
                layer_losses["mod_predictor_loss"] = predictor_loss

            B, T, D = hidden_states.shape
            k = topk_idx.shape[1]
            batch_idx = torch.arange(B, device=hidden_states.device).unsqueeze(1).expand(-1, k)

            gating_scores_for_selected = gating_probs.reshape(-1)  # Reshape to 1D

            new_states, _, _ = self.block.process_selected(
                hidden_states,
                batch_indices=batch_idx.reshape(-1),
                token_indices=topk_idx.reshape(-1),
                gating_scores=gating_scores_for_selected,
                use_soft_gating=True,
                **kwargs,
            )

            router_probs = torch.sigmoid(scores)
            num_selected = int(self.router.capacity * scores.shape[1])
            layer_metrics = {
                "router_logits_mean": scores.mean().item(),
                "router_logits_std": scores.std().item(),
                "router_probs_mean": router_probs.mean().item(),
                "router_probs_std": router_probs.std().item(),
                "selected_tokens_proportion": num_selected / scores.shape[1],
                "g_cont": router_probs.mean().item(),
            }

        else:  # Inference
            use_causal = self.model_params.get("use_causal_router_in_validation", True)

            if use_causal and self.causal_router is not None:
                predictor_logits = self.causal_router(hidden_states)
                is_selected = predictor_logits > 0

                batch_idx, token_idx = is_selected.nonzero(as_tuple=True)

                new_states, _, _ = self.block.process_selected(
                    hidden_states,
                    batch_indices=batch_idx,
                    token_indices=token_idx,
                    gating_scores=None,
                    use_soft_gating=False,
                    **kwargs,
                )

                layer_metrics = {
                    "inferred_selected_tokens_proportion": (
                        is_selected.sum() / is_selected.numel()
                    ).item()
                }
            else:  # Use training-time router for validation
                scores, router_bce_loss, binary_targets, gating_scores, topk_idx = self.router(
                    hidden_states
                )

                # FIX: Apply sigmoid to ensure gating scores are probabilities (0-1) instead of raw logits.
                gating_probs = torch.sigmoid(gating_scores)

                B, T, D = hidden_states.shape
                k = topk_idx.shape[1]  # k is already determined by topk_idx
                batch_idx = torch.arange(B, device=hidden_states.device).unsqueeze(1).expand(-1, k)

                gating_scores_for_selected = gating_probs.reshape(-1)  # Reshape to 1D

                new_states, _, _ = self.block.process_selected(
                    hidden_states,
                    batch_indices=batch_idx.reshape(-1),
                    token_indices=topk_idx.reshape(-1),
                    gating_scores=gating_scores_for_selected,
                    use_soft_gating=True,  # Use soft gating as in training
                    **kwargs,
                )

                router_probs = torch.sigmoid(scores)
                num_selected = int(self.router.capacity * scores.shape[1])
                layer_metrics = {
                    "router_logits_mean": scores.mean().item(),
                    "router_logits_std": scores.std().item(),
                    "router_probs_mean": router_probs.mean().item(),
                    "router_probs_std": router_probs.std().item(),
                    "selected_tokens_proportion": num_selected / scores.shape[1],
                    "g_cont": router_probs.mean().item(),
                }

        return new_states, layer_losses, layer_metrics


class MoDForCausalLM(BaseForCausalLM):
    config_class = MoDConfig
    _supports_flash_attn_2 = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # Replace standard layers with MoD layers in-place
        for i in range(self.config.num_hidden_layers):
            if i % 2 == 1:
                self.model.layers[i] = MoDLayer(self.model.layers[i], config, self.model_params)

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
        all_mod_metrics = []
        all_selected_tokens_proportions = []  # To collect for logging
        all_g_cont_values = []  # Initialize for collecting g_cont

        layer_args = {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
            "output_attentions": output_attentions,
            **kwargs,
        }

        for layer in self.model.layers:
            if isinstance(layer, MoDLayer):
                layer_args["attention_mask"] = mask_mapping[layer.block.layer.attention_type]
                hidden_states, losses, mod_metrics = layer(
                    hidden_states,
                    training=self.training,
                    **layer_args,
                )
                all_losses.append(losses)
                all_mod_metrics.append(mod_metrics)
                if "selected_tokens_proportion" in mod_metrics:
                    all_selected_tokens_proportions.append(
                        mod_metrics["selected_tokens_proportion"]
                    )

                # Collect g_cont for logging
                if "g_cont" in mod_metrics:
                    all_g_cont_values.append(mod_metrics["g_cont"])

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
        # Aggregate losses
        agg_losses = {}
        if all_losses:
            all_keys = set(k for l in all_losses for k in l)
            for k in all_keys:
                key_losses = [l[k] for l in all_losses if k in l and l.get(k) is not None]
                if key_losses:
                    agg_losses[k] = torch.mean(torch.stack(key_losses))
        aux["unscaled_losses"] = agg_losses

        # Aggregate metrics
        agg_metrics = {}
        if all_mod_metrics:
            metric_keys = all_mod_metrics[0].keys()
            for key in metric_keys:
                agg_metrics[f"mod/{key}"] = (
                    torch.tensor([m[key] for m in all_mod_metrics]).mean().item()
                )
        aux["router_stats"] = agg_metrics

        if all_selected_tokens_proportions:
            aux["router_stats"]["mod_selected_tokens_proportion_mean"] = sum(
                all_selected_tokens_proportions
            ) / len(all_selected_tokens_proportions)

        # Add g_cont mean across layers to router_stats
        if all_g_cont_values:
            aux["router_stats"]["mod_g_cont_mean_across_layers"] = sum(all_g_cont_values) / len(
                all_g_cont_values
            )

        return hidden_states, aux

    def get_trainable_parameters(self):
        params_map = {"mod_router": [], "mod_causal_router": [], "base_model": []}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "causal_router" in n and self.model_params.get("train_causal_router", True):
                params_map["mod_causal_router"].append(p)
            elif "router" in n:
                params_map["mod_router"].append(p)
            else:
                params_map["base_model"].append(p)
        return [{"name": k, "params": v} for k, v in params_map.items() if v]
