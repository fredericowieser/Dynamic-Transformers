import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from ..base.causal_lm import BaseForCausalLM
from ..base.block import DynamicBlock
from ..base.routers import CausalRouter, BaseRouter
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

        aux_loss = F.binary_cross_entropy_with_logits(logits, binary_targets) * self.aux_loss_weight
        
        return logits, aux_loss, {}

class MoDLayer(nn.Module):
    def __init__(self, hf_layer: Qwen2DecoderLayer, config, model_params: Dict):
        super().__init__()
        self.block = DynamicBlock(hf_layer)
        self.router = MoDRouter(config, layer_idx=0, model_params=model_params)
        self.causal_router = CausalRouter(config, layer_idx=0, capacity_attr='capacity', model_cfg=model_params)

    def forward(self, hidden_states, training: bool, **kwargs):
        scores, aux_loss, _ = (self.router(hidden_states) if training else self.causal_router(hidden_states))
        _, batch_idx, token_idx, gating_scores = self.router.select_tokens(scores, hidden_states)
        new_states, _, _ = self.block.process_selected(
            hidden_states, batch_idx, token_idx, gating_scores, use_soft_gating=training, **kwargs
        )
        return new_states, aux_loss if aux_loss is not None else torch.tensor(0.0, device=hidden_states.device)

class MoDForCausalLM(BaseForCausalLM):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._mod_wrappers = nn.ModuleDict({
            str(i): MoDLayer(self.model.layers[i], config, self.model_params) for i in range(self.config.num_hidden_layers) if i % 2 == 1
        })

    def _run_layers(self, hidden_states, mask_mapping, position_ids, past_key_values, use_cache, cache_position, position_embeddings, output_attentions, **kwargs):
        total_aux = torch.tensor(0.0, device=hidden_states.device)
        for i, layer in enumerate(self.model.layers):
            attn_mask = mask_mapping[layer.attention_type]
            if i in self._mod_wrappers:
                hidden_states, aux = self._mod_wrappers[i](
                    hidden_states, training=self.training,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=use_cache,
                )
                total_aux += aux
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
        return hidden_states, aux

    def get_trainable_parameters(self):
        params_map = {'router': [], 'causal_router': [], 'base_model': []}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'router' in n and 'causal_router' not in n:
                params_map['router'].append(p)
            elif 'causal_router' in n:
                params_map['causal_router'].append(p)
            else:
                params_map['base_model'].append(p)
        return [{'name': k, 'params': v} for k, v in params_map.items() if v]