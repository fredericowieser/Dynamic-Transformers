import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from ..base.causal_lm import BaseForCausalLM
from ..base.block import DynamicBlock
from ..base.routers import BaseRouter
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

class MoDRouter(BaseRouter):
    def __init__(self, config, layer_idx: int, model_params: Dict):
        super().__init__(config, capacity_attr='capacity', model_cfg=model_params)
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
        
        return logits, router_bce_loss, binary_targets, gating_scores, topk_idx

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
        self.causal_router = MoDCausalRouter(config)

    def forward(self, hidden_states, training: bool, **kwargs):
        layer_losses = {}
        layer_metrics = {}

        if training:
            scores, router_bce_loss, binary_targets, gating_scores, topk_idx = self.router(hidden_states)
            layer_losses['mod_router_bce_loss'] = router_bce_loss
            
            predictor_logits = self.causal_router(hidden_states.detach())
            predictor_loss = F.binary_cross_entropy_with_logits(predictor_logits, binary_targets.detach())
            layer_losses['mod_causal_predictor_loss'] = predictor_loss

            B, T, D = hidden_states.shape
            k = topk_idx.shape[1]
            batch_idx = torch.arange(B, device=hidden_states.device).unsqueeze(1).expand(-1, k)

            new_states, _, _ = self.block.process_selected(
                hidden_states,
                batch_indices=batch_idx.reshape(-1),
                token_indices=topk_idx.reshape(-1),
                gating_scores=gating_scores.reshape(-1),
                use_soft_gating=True,
                **kwargs
            )
            
            router_probs = torch.sigmoid(scores)
            num_selected = int(self.router.capacity * scores.shape[1])
            layer_metrics = {
                'router_logits_mean': scores.mean().item(),
                'router_logits_std': scores.std().item(),
                'router_probs_mean': router_probs.mean().item(),
                'router_probs_std': router_probs.std().item(),
                'selected_tokens_proportion': num_selected / scores.shape[1],
            }

        else: # Inference
            predictor_logits = self.causal_router(hidden_states)
            is_selected = predictor_logits > 0
            
            batch_idx, token_idx = is_selected.nonzero(as_tuple=True)

            new_states, _, _ = self.block.process_selected(
                hidden_states,
                batch_indices=batch_idx,
                token_indices=token_idx,
                gating_scores=None,
                use_soft_gating=False,
                **kwargs
            )
            
            layer_metrics = {
                'inferred_selected_tokens_proportion': (is_selected.sum() / is_selected.numel()).item()
            }

        return new_states, layer_losses, layer_metrics


class MoDForCausalLM(BaseForCausalLM):
    _supports_flash_attn_2 = True

    def __init__(self, config, model_type: str, **kwargs):
        super().__init__(config, model_type=model_type, **kwargs)
        # Replace standard layers with MoD layers in-place
        for i in range(self.config.num_hidden_layers):
            if i % 2 == 1:
                self.model.layers[i] = MoDLayer(self.model.layers[i], config, self.model_params)

    def _run_layers(self, hidden_states, mask_mapping, position_ids, past_key_values, use_cache, cache_position, position_embeddings, output_attentions, **kwargs):
        all_losses = []
        all_mod_metrics = []
        
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
                if self.training:
                    all_losses.append(losses)
                    all_mod_metrics.append(mod_metrics)
            else: # Standard Qwen2DecoderLayer
                layer_args["attention_mask"] = mask_mapping[layer.attention_type]
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    **layer_args,
                )
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
                
        aux = {}
        if self.training:
            # Aggregate losses
            agg_losses = {k: torch.mean(torch.stack([l[k] for l in all_losses])) for k in all_losses[0] if all_losses}
            aux['unscaled_losses'] = agg_losses

            # Aggregate metrics
            agg_metrics = {}
            if all_mod_metrics:
                metric_keys = all_mod_metrics[0].keys()
                for key in metric_keys:
                    agg_metrics[f"mod/{key}"] = torch.tensor([m[key] for m in all_mod_metrics]).mean().item()
            aux['router_stats'] = agg_metrics
            
        return hidden_states, aux

    def get_trainable_parameters(self):
        params_map = {'mod_router': [], 'mod_causal_router': [], 'base_model': []}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'causal_router' in n:
                params_map['mod_causal_router'].append(p)
            elif 'router' in n:
                params_map['mod_router'].append(p)
            else:
                params_map['base_model'].append(p)
        return [{'name': k, 'params': v} for k, v in params_map.items() if v]
