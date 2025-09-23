import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import logging

from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
from ..base.block import DynamicBlock
from ..base.routers import BaseSurpriseRouter, STTCausalRouter
from ..base.priors import BasePriorNetwork
from ..base.causal_lm import BaseForCausalLM

log = logging.getLogger(__name__)

class STTTransitionNetwork(BasePriorNetwork):
    """Implements the STT transition network with pre-normalization: MLP(RMSNorm(x))."""
    def __init__(self, config, model_params: Dict):
        super().__init__(config, model_cfg=model_params)
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(x))

class STTPredictiveRouter(BaseSurpriseRouter):
    def __init__(self, config, layer_idx: int, model_params: Dict):
        super().__init__(config, capacity_attr='capacity', model_cfg=model_params)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        actual_residual = args[0]
        predicted_residual = args[1]
        beta_ce = kwargs['beta_ce']
        beta_cu = kwargs['beta_cu']

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
    """Wraps an HF layer and adds STT routing."""
    def __init__(self, hf_layer: Qwen2DecoderLayer, config, model_params: Dict):
        super().__init__()
        self.block = DynamicBlock(hf_layer)
        self.transition_network = STTTransitionNetwork(config, model_params=model_params)
        self.predictive_router = STTPredictiveRouter(config, layer_idx=0, model_params=model_params)
        self.causal_router = STTCausalRouter(config, layer_idx=0, capacity_attr='capacity', model_cfg=model_params)
        self.config = config
        self.model_params = model_params

    def forward(self, hidden_states, **kwargs):
        original_hidden = hidden_states
        router_stats = {}
        layer_losses = {}

        if self.training:
            out = self.block(hidden_states, **kwargs)
            processed_hidden = out[0] if isinstance(out, tuple) else out
            
            actual_residual = processed_hidden - original_hidden

            prev_final_states = torch.cat(
                [torch.zeros_like(processed_hidden[:, :1, :]), processed_hidden[:, :-1, :]], dim=1
            )
            predicted_residual = self.transition_network(prev_final_states)
            layer_losses['stt_tpn_loss'] = F.mse_loss(predicted_residual, actual_residual.detach())

            g_cont, binary_targets, pred_stats = self.predictive_router(
                actual_residual, predicted_residual, **kwargs
            )
            router_stats.update(pred_stats)

            causal_logits, _, causal_stats = self.causal_router(original_hidden)
            router_stats.update(causal_stats)
            layer_losses['stt_causal_router_loss'] = F.binary_cross_entropy_with_logits(causal_logits, binary_targets.detach())
            
            final_hidden_states = processed_hidden
        
        else: # Inference
            causal_logits, _, causal_stats = self.causal_router(original_hidden)
            router_stats.update(causal_stats)
            
            B, T, D = hidden_states.shape
            k = max(1, int(T * self.causal_router.capacity))
            gating_scores, topk_idx = causal_logits.topk(k, dim=-1)
            
            batch_idx = torch.arange(B, device=causal_logits.device).unsqueeze(1).expand(-1, k)

            final_hidden_states, _, _ = self.block.process_selected(
                original_hidden,
                batch_indices=batch_idx.reshape(-1),
                token_indices=topk_idx.reshape(-1),
                gating_scores=gating_scores.reshape(-1),
                use_soft_gating=False,
                **kwargs
            )

        return final_hidden_states, layer_losses, router_stats

class STTForCausalLM(BaseForCausalLM):
    _supports_flash_attn_2 = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        for i in range(self.config.num_hidden_layers):
            if i % 2 == 1:
                self.model.layers[i] = STTLayer(self.model.layers[i], config, self.model_params)

    def _run_layers(
        self,
        hidden_states: torch.Tensor,
        mask_mapping: Dict[str, torch.Tensor],
        position_ids: torch.LongTensor,
        past_key_values: Optional[Any],
        use_cache: bool,
        cache_position: Optional[torch.LongTensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        output_attentions: bool,
        **kwargs,
    ):
        all_losses = []
        all_router_stats: Dict[str, Any] = {}

        beta_ce, beta_cu = 1.0, 1.0 # Default if no schedule is found
        if hasattr(self.config, 'beta_schedule'):
            sched_cfg = self.config.beta_schedule
            global_step = kwargs.get('global_step', 0)
            max_steps = kwargs.get('max_steps', 100000) # max_steps from training loop
            warmup = sched_cfg['warmup_steps']
            
            # Calculate betas based on schedule, regardless of training mode
            if global_step > warmup:
                progress = (global_step - warmup) / (max_steps - warmup)
                progress = min(progress, 1.0)
                beta_ce = sched_cfg['beta_ce_start'] + progress * (sched_cfg['beta_ce_end'] - sched_cfg['beta_ce_start'])
                beta_cu = sched_cfg['beta_cu_start'] + progress * (sched_cfg['beta_cu_end'] - sched_cfg['beta_cu_start'])
            else:
                beta_ce = sched_cfg['beta_ce_start']
                beta_cu = sched_cfg['beta_cu_start']

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
            if isinstance(layer, STTLayer):
                layer_args["attention_mask"] = mask_mapping[layer.block.layer.attention_type]
                hidden_states, losses, rstats = layer(
                    hidden_states,
                    **layer_args,
                )
                if self.training:
                    all_losses.append(losses)
                    for key, value in rstats.items():
                        all_router_stats[f"stt/layer_{i}/{key}"] = value
            else: # Standard Qwen2DecoderLayer
                layer_args["attention_mask"] = mask_mapping[layer.attention_type]
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    **layer_args,
                )
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        aux = {}
        if self.training:
            agg_losses = {k: torch.mean(torch.stack([l[k] for l in all_losses])) for k in all_losses[0] if all_losses}
            aux['unscaled_losses'] = agg_losses
            aux['router_stats'] = all_router_stats
            aux['beta_ce'] = beta_ce
            aux['beta_cu'] = beta_cu

        return hidden_states, aux

    def get_trainable_parameters(self):
        # Groups for optimizer
        params_map = {
            'stt_transition_network': [],
            'stt_predictive_router': [],
            'stt_causal_router': [],
            'base_model': [],
        }
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'transition_network' in n:
                params_map['stt_transition_network'].append(p)
            elif 'predictive_router' in n:
                params_map['stt_predictive_router'].append(p)
            elif 'causal_router' in n:
                params_map['stt_causal_router'].append(p)
            else:
                params_map['base_model'].append(p)
        return [{'name': k, 'params': v} for k, v in params_map.items() if v]