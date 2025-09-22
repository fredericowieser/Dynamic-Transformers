import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from ..base.priors import BasePriorNetwork
from ..base.routers import BaseSurpriseRouter, CausalRouter
from ..base.block import DynamicBlock
from omegaconf import DictConfig
from ..base.causal_lm import BaseForCausalLM


class STTTransitionNetwork(BasePriorNetwork):
    """Implements the STT transition network with pre-normalization: MLP(RMSNorm(x))."""
    def __init__(self, config):
        super().__init__(config)
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(x))

class STTPredictiveRouter(BaseSurpriseRouter):
    """Implements the STT surprise calculation and returns binary targets."""
    def __init__(self, config, layer_idx: int, model_cfg: Dict = None):
        super().__init__(config, capacity_attr='stt_capacity')

    def forward(self, actual_residual: torch.Tensor, predicted_residual: torch.Tensor, beta_ce: float, beta_cu: float, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        d = float(actual_residual.shape[-1])
        D_st = torch.sum(actual_residual.pow(2), dim=-1) / d
        D_ch = torch.sum((actual_residual - predicted_residual).pow(2), dim=-1) / d
        
        g_cont, stats = self._get_vpr_signals(D_st, D_ch, beta_ce, beta_cu)
        
        B, T = g_cont.shape
        k = max(1, int(T * self.capacity))
        _, topk_idx = g_cont.topk(k, dim=-1)
        binary_targets = torch.zeros_like(g_cont)
        binary_targets.scatter_(1, topk_idx, 1.0)

        return g_cont, binary_targets, stats

class STTLayer(nn.Module):
    """A self-contained STT Layer implementing the full teacher-student paradigm."""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.block = DynamicBlock(config, layer_idx)
        self.transition_network = STTTransitionNetwork(config)
        self.predictive_router = STTPredictiveRouter(config, layer_idx)
        self.causal_router = CausalRouter(config, layer_idx, 'stt_capacity')
        self.config = config

    def forward(self, hidden_states, **kwargs):
        original_hidden = hidden_states
        processed_hidden = self.block(hidden_states, **kwargs)[0]
        
        aux_loss = None
        router_stats = {} # Initialize router_stats
        if self.training:
            actual_residual = processed_hidden - original_hidden
            
            prev_final_states = torch.cat([
                torch.zeros_like(processed_hidden[:, :1, :]),
                processed_hidden[:, :-1, :]
            ], dim=1)
            predicted_residual = self.transition_network(prev_final_states)
            
            tpn_loss = F.mse_loss(predicted_residual, actual_residual.detach())
            
            g_cont, binary_targets, pred_router_stats = self.predictive_router(
                actual_residual, predicted_residual, **kwargs
            )
            router_stats.update(pred_router_stats)
            
            causal_logits, _, causal_router_stats = self.causal_router(original_hidden)
            router_stats.update(causal_router_stats)
            causal_loss = F.binary_cross_entropy_with_logits(causal_logits, binary_targets.detach())
            
            aux_loss = (self.config.tpn_loss_weight * tpn_loss) + \
                       (self.config.causal_loss_weight * causal_loss)
            
            # During training, no tokens are skipped to ensure stable gradient flow
            final_hidden_states = processed_hidden
        
        else: # Inference logic
            causal_logits, _, causal_router_stats = self.causal_router(original_hidden)
            router_stats.update(causal_router_stats)
            k = max(1, int(hidden_states.shape[1] * self.causal_router.capacity))
            _, topk_indices = causal_logits.topk(k, dim=-1)
            
            mask = torch.zeros_like(causal_logits, dtype=torch.bool).scatter_(1, topk_indices, True)
            
            # Mix original and processed states based on the routing decision
            final_hidden_states = torch.where(mask.unsqueeze(-1), processed_hidden, original_hidden)

        return final_hidden_states, aux_loss, router_stats


class STTForCausalLM(BaseForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self._setup_layers()
    
    def _setup_layers(self):
        for i in range(self.config.num_hidden_layers):
                        if i % 2 == 1: # Dynamic STT layer
                            self.layers.append(STTLayer(self.config, i))
                        else: # Standard layer
                            self.layers.append(DynamicBlock(self.config, i))
    def _forward_layers(self, hidden_states, **kwargs):
        total_aux_loss = 0
        all_router_stats = {} # Initialize all_router_stats here
        for layer in self.layers:
            if isinstance(layer, STTLayer):
                hidden_states, aux_loss, router_stats = layer(hidden_states, **kwargs)
                if aux_loss is not None:
                    total_aux_loss += aux_loss
                all_router_stats.update(router_stats)
            else: # Standard DynamicBlock
                hidden_states = layer(hidden_states, **kwargs)[0]
            
        return {"hidden_states": hidden_states, "aux_loss": total_aux_loss, "router_stats": all_router_stats}

    def get_trainable_parameters(self):
        return self._create_param_groups({
            'transition_network': 'transition_network',
            'predictive_router': 'predictive_router',
            'causal_router': 'causal_router'
        })