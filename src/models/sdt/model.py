import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
)
from ..base.block import DynamicBlock
from ..base.priors import BasePriorNetwork
from ..base.routers import BaseSurpriseRouter, CausalRouter
from ..base.causal_lm import BaseForCausalLM

class SDTPriorNetwork(BasePriorNetwork):
    def __init__(self, config, model_params: Dict):
        super().__init__(config, model_cfg=model_params)
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(x))

class SDTRouter(BaseSurpriseRouter):
    def __init__(self, config, model_params: Dict):
        super().__init__(config, capacity_attr='capacity', model_cfg=model_params)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        actual_residual, predicted_residual = args[0], args[1]
        beta_ce, beta_cu = kwargs.get('beta_ce', 1.0), kwargs.get('beta_cu', 1.0)

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
        self.causal_router = CausalRouter(config, layer_idx=layer_idx, capacity_attr='capacity', model_cfg=model_params)
        self.model_params = model_params

    def forward(self, hidden_states, **kwargs):
        processed_hidden, actual_res, predicted_res, prior_loss = self.decision(hidden_states, **kwargs)
        
        layer_losses = {"sdt_prior_loss": prior_loss}
        router_stats = {}

        if self.training:
            g_cont, binary_targets, surprise_stats = self.router(actual_res, predicted_res, **kwargs)
            router_stats.update(surprise_stats)

            causal_logits, _, _ = self.causal_router(hidden_states.detach())
            causal_loss = F.binary_cross_entropy_with_logits(causal_logits, binary_targets.detach())
            layer_losses['sdt_causal_router_loss'] = causal_loss
            
            B, T, D = hidden_states.shape
            k = max(1, int(T * self.router.capacity))
            gating_scores, topk_idx = g_cont.topk(k, dim=-1)
            batch_idx = torch.arange(B, device=g_cont.device).unsqueeze(1).expand(-1, k)

            final_hidden_states, _, _ = self.dynamic.process_selected(
                processed_hidden, 
                batch_indices=batch_idx.reshape(-1),
                token_indices=topk_idx.reshape(-1),
                gating_scores=gating_scores.reshape(-1),
                use_soft_gating=True,
                **kwargs
            )
        else: # Inference
            causal_logits, _, _ = self.causal_router(hidden_states)
            
            B, T, D = hidden_states.shape
            k = max(1, int(T * self.causal_router.capacity))
            gating_scores, topk_idx = causal_logits.topk(k, dim=-1)
            batch_idx = torch.arange(B, device=causal_logits.device).unsqueeze(1).expand(-1, k)

            final_hidden_states, _, _ = self.dynamic.process_selected(
                processed_hidden,
                batch_indices=batch_idx.reshape(-1),
                token_indices=topk_idx.reshape(-1),
                gating_scores=gating_scores.reshape(-1),
                use_soft_gating=False,
                **kwargs
            )

        return final_hidden_states, layer_losses, router_stats

class SDTForCausalLM(BaseForCausalLM):
    _supports_flash_attn_2 = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        for i in range(0, self.config.num_hidden_layers - 1, 2):
            self.model.layers[i] = SDTPair(self.model.layers[i+1], config, self.model_params, i)
            self.model.layers[i+1] = nn.Identity()

    def _run_layers(self, hidden_states, mask_mapping, position_ids, past_key_values, use_cache, cache_position, position_embeddings, output_attentions, **kwargs):
        all_losses = []
        all_router_stats = {}
        cfg = self.config

        beta_ce, beta_cu = 1.0, 1.0
        if self.training and hasattr(cfg, 'beta_schedule'):
            sched_cfg = cfg.beta_schedule
            global_step = kwargs.get('global_step', 0)
            max_steps = kwargs.get('max_steps', 100000)
            warmup = sched_cfg.warmup_steps
            
            if global_step > warmup:
                progress = (global_step - warmup) / (max_steps - warmup)
                progress = min(progress, 1.0)
                beta_ce = sched_cfg.beta_ce_start + progress * (sched_cfg.beta_ce_end - sched_cfg.beta_ce_start)
                beta_cu = sched_cfg.beta_cu_start + progress * (sched_cfg.beta_cu_end - sched_cfg.beta_cu_start)
            else:
                beta_ce = sched_cfg.beta_ce_start
                beta_cu = sched_cfg.beta_cu_start

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

            elif isinstance(layer, nn.Identity):
                continue
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
        params_map = {'prior': [], 'router': [], 'causal_router': [], 'base_model': []}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'prior' in n:
                params_map['prior'].append(p)
            elif 'causal_router' in n:
                params_map['causal_router'].append(p)
            elif 'router' in n:
                params_map['router'].append(p)
            else:
                params_map['base_model'].append(p)
        return [{'name': k, 'params': v} for k, v in params_map.items() if v]
