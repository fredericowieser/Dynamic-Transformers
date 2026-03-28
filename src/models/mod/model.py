from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from ..base.block import DynamicBlock
from ..base.causal_lm import BaseForCausalLM
from ..base.routers import BaseRouter, UnifiedCausalRouter
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


class MoDLayer(nn.Module):
    def __init__(self, hf_layer: Qwen2DecoderLayer, config, model_params: Dict):
        super().__init__()
        self.block = DynamicBlock(hf_layer)
        self.router = MoDRouter(config, layer_idx=0, model_params=model_params)
        self.causal_router = UnifiedCausalRouter(config)
        self.model_params = model_params
        self.causal_threshold = getattr(config, "causal_threshold", 0.5)

    def forward(self, hidden_states, training: bool, use_causal_router: bool = False, **kwargs):
        layer_losses = {}
        layer_metrics = {}
        B, T, D = hidden_states.shape

        if use_causal_router:
            # INFERENCE MODE (Autoregressive / Causal)
            causal_logits, _, _ = self.causal_router(hidden_states)
            gating_probs = torch.sigmoid(causal_logits)
            
            selected_mask = gating_probs >= self.causal_threshold
            batch_idx, token_idx = selected_mask.nonzero(as_tuple=True)
            gating_scores_for_selected = gating_probs[selected_mask]
            
            if B * T > 0:
                layer_metrics["selected_tokens_proportion"] = (selected_mask.sum() / (B * T)).item()
            layer_metrics["g_cont"] = gating_probs.mean().item()
        else:
            # TRAINING MODE (Non-Causal Top-K)
            (
                scores,
                router_bce_loss,
                router_z_loss,
                binary_targets,
                gating_scores,
                topk_idx,
            ) = self.router(hidden_states)
            
            causal_logits, causal_loss, causal_acc = self.causal_router(hidden_states.detach(), targets=binary_targets)
            
            topk_idx, sort_indices = topk_idx.sort(dim=-1)
            gating_scores = gating_scores.gather(dim=-1, index=sort_indices)
            
            if training:
                layer_losses["mod_aux_loss"] = router_bce_loss
                layer_losses["mod_z_loss"] = router_z_loss * 1e-4
                layer_losses["mod_causal_router_loss"] = causal_loss
            
            if causal_acc is not None:
                layer_metrics["causal_router_acc"] = causal_acc.item()
                
            gating_probs = torch.sigmoid(gating_scores)
            
            k = topk_idx.shape[1]
            batch_idx = torch.arange(B, device=hidden_states.device).unsqueeze(1).expand(-1, k).reshape(-1)
            token_idx = topk_idx.reshape(-1)
            gating_scores_for_selected = gating_probs.reshape(-1)
            
            router_probs = torch.sigmoid(scores)
            num_selected = int(self.router.capacity * scores.shape[1])
            layer_metrics.update({
                "router_logits_mean": scores.mean().item(),
                "router_logits_std": scores.std().item(),
                "router_probs_mean": router_probs.mean().item(),
                "router_probs_std": router_probs.std().item(),
                "selected_tokens_proportion": num_selected / scores.shape[1] if scores.shape[1] > 0 else 0,
                "g_cont": router_probs.mean().item(),
            })

        new_states, _, _ = self.block.process_selected(
            hidden_states,
            batch_indices=batch_idx,
            token_indices=token_idx,
            gating_scores=gating_scores_for_selected,
            use_soft_gating=True,
            **kwargs,
        )

        return new_states, layer_losses, layer_metrics


class MoDForCausalLM(BaseForCausalLM):
    config_class = MoDConfig
    _supports_flash_attn_2 = False

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
            "use_causal_router": use_causal_router,
            **kwargs,
        }

        for layer in self.model.layers:
            if isinstance(layer, MoDLayer):
                layer_attn_mask = mask_mapping[layer.block.layer.attention_type]
                
                if getattr(self, "gradient_checkpointing", False) and self.training:
                    # Clean up checkpoint call to ensure hidden_states is the primary positional arg
                    hidden_states, losses, mod_metrics = torch.utils.checkpoint.checkpoint(
                        layer.__call__,
                        hidden_states,
                        use_reentrant=False,
                        training=self.training,
                        use_causal_router=False,
                        attention_mask=layer_attn_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        output_attentions=output_attentions,
                        **kwargs,
                    )
                else:
                    hidden_states, losses, mod_metrics = layer(
                        hidden_states,
                        training=self.training,
                        use_causal_router=False,
                        attention_mask=layer_attn_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        output_attentions=output_attentions,
                        **kwargs,
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
                attn_mask = mask_mapping[layer.attention_type]
                
                if getattr(self, "gradient_checkpointing", False) and self.training:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        layer.__call__,
                        hidden_states,
                        attn_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                        use_reentrant=False,
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
