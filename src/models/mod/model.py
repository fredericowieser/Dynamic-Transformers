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
        self.aux_loss_weight = model_params.get('mod', {}).get('aux_loss_weight', 0.01)

    def forward(self, hidden_states, **kwargs):
        logits = self.router(hidden_states).squeeze(-1)
        
        B, T = logits.shape
        k = max(1, int(T * self.capacity))
        if k > T:
            k = T
            
        _, topk_idx = logits.topk(k, dim=-1)
        
        binary_targets = torch.zeros_like(logits)
        binary_targets.scatter_(1, topk_idx, 1.0)

        # Auxiliary loss to train the router, as described in MoD paper, Section 3.5.
        # This loss encourages the router to produce high scores for tokens that are
        # selected (top-k) and low scores for those that are not. This makes the
        # router's behavior more predictable for causal inference.
        # The equation is: L_aux = BCE(logits, topk_mask) * weight
        aux_loss = F.binary_cross_entropy_with_logits(logits, binary_targets) * self.aux_loss_weight
        
        return logits, aux_loss, binary_targets

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
        self.predictor_loss_weight = model_params.get('mod', {}).get('predictor_loss_weight', 0.01)


    def forward(self, hidden_states, training: bool, **kwargs):
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device)
        layer_metrics = {}

        if training:
            # Main router pass (non-causal top-k)
            scores, main_aux_loss, binary_targets = self.router(hidden_states)
            total_aux_loss += main_aux_loss
            
            # Causal router (predictor) pass to train it for inference
            # Use .detach() on inputs/targets to not affect the main model's gradients
            predictor_logits = self.causal_router(hidden_states.detach())
            predictor_loss = F.binary_cross_entropy_with_logits(predictor_logits, binary_targets.detach())
            total_aux_loss += predictor_loss * self.predictor_loss_weight

            # Gather tokens based on main router for the forward pass
            _, batch_idx, token_idx, gating_scores = self.router.select_tokens(scores, hidden_states)
            
            new_states, _, _ = self.block.process_selected(
                hidden_states, batch_idx, token_idx, gating_scores, use_soft_gating=True, **kwargs
            )
            
            # Logging metrics from the training router
            router_probs = torch.sigmoid(scores)
            num_selected = int(self.router.capacity * scores.shape[1])
            layer_metrics = {
                'router_logits_mean': scores.mean().item(),
                'router_logits_std': scores.std().item(),
                'router_probs_mean': router_probs.mean().item(),
                'router_probs_std': router_probs.std().item(),
                'selected_tokens_proportion': num_selected / scores.shape[1],
                'predictor_loss': predictor_loss.item(),
            }

        else: # Inference
            # Use the trained causal router for causal routing
            predictor_logits = self.causal_router(hidden_states)
            is_selected = predictor_logits > 0
            
            batch_idx, token_idx = is_selected.nonzero(as_tuple=True)

            new_states, _, _ = self.block.process_selected(
                hidden_states, batch_idx, token_idx, gating_scores=None, use_soft_gating=False, **kwargs
            )
            
            # Logging metrics from the inference predictor
            layer_metrics = {
                'inferred_selected_tokens_proportion': (is_selected.sum() / is_selected.numel()).item()
            }

        return new_states, total_aux_loss, layer_metrics


class MoDForCausalLM(BaseForCausalLM):
    _supports_flash_attn_2 = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._mod_wrappers = nn.ModuleDict({
            str(i): MoDLayer(self.model.layers[i], config, self.model_params) for i in range(self.config.num_hidden_layers) if i % 2 == 1
        })

    def _run_layers(self, hidden_states, mask_mapping, position_ids, past_key_values, use_cache, cache_position, position_embeddings, output_attentions, **kwargs):
        total_aux = torch.tensor(0.0, device=hidden_states.device)
        all_mod_metrics = []
        
        for i, layer in enumerate(self.model.layers):
            attn_mask = mask_mapping[layer.attention_type]
            if str(i) in self._mod_wrappers:
                hidden_states, aux, mod_metrics = self._mod_wrappers[str(i)](
                    hidden_states, training=self.training,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=use_cache,
                )
                total_aux += aux
                if self.training: # Only collect metrics during training
                    all_mod_metrics.append(mod_metrics)
            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                
        aux = {'aux_loss': total_aux} if self.training else {}
        
        if all_mod_metrics:
            agg_metrics = {}
            # Get keys from the first layer's metrics
            metric_keys = all_mod_metrics[0].keys()
            for key in metric_keys:
                # Average the metric across all MoD layers
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