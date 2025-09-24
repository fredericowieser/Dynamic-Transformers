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
    def __init__(self, config):
        super().__init__(config)
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
        return g_cont, stats

class STTLayer(nn.Module):
    """Wraps an HF layer and adds STT routing."""
    def __init__(self, hf_layer: Qwen2DecoderLayer, config):
        super().__init__()
        self.block = DynamicBlock(hf_layer)
        self.transition_network = STTTransitionNetwork(config)
        self.predictive_router = STTPredictiveRouter(config, layer_idx=0, capacity_attr='capacity')
        
        self.train_causal_router = getattr(config, 'train_causal_router', True)
        if self.train_causal_router:
            self.causal_router = STTCausalRouter(config, layer_idx=0, capacity_attr='capacity')
        else:
            self.causal_router = None
        self.config = config

    def forward(self, hidden_states, **kwargs):
        original_hidden = hidden_states
        router_stats = {}
        layer_losses = {}
        
        use_g_threshold = self.model_params.get('stt', {}).get('use_g_threshold_selection', False)
        g_threshold = self.model_params.get('stt', {}).get('g_threshold', 0.5)

        if self.training:
            # First pass through the block to get processed_hidden and residuals
            out = self.block(hidden_states, **kwargs) # This is the first TF block
            processed_hidden = out[0] if isinstance(out, tuple) else out
            
            actual_residual = processed_hidden - original_hidden

            prev_final_states = torch.cat(
                [torch.zeros_like(processed_hidden[:, :1, :]), processed_hidden[:, :-1, :]], dim=1
            )
            predicted_residual = self.transition_network(prev_final_states)
            layer_losses['stt_tpn_loss'] = F.mse_loss(predicted_residual, actual_residual.detach())

            g_cont, pred_stats = self.predictive_router(
                actual_residual, predicted_residual, **kwargs
            )
            router_stats.update(pred_stats)

            # FIX: Generate binary targets for the causal router based on the configured
            # routing mode (threshold vs. top-k), ensuring the student learns the correct objective.
            B, T = g_cont.shape
            if use_g_threshold:
                binary_targets = (g_cont >= g_threshold).float()
            else: # Top-K
                k = max(1, int(T * self.predictive_router.capacity))
                _, topk_idx = g_cont.topk(k, dim=-1)
                binary_targets = torch.zeros_like(g_cont)
                binary_targets.scatter_(1, topk_idx, 1.0)

            if self.causal_router is not None:
                causal_logits, _, causal_stats = self.causal_router(original_hidden)
                router_stats.update(causal_stats)
                layer_losses['stt_causal_router_loss'] = F.binary_cross_entropy_with_logits(causal_logits, binary_targets.detach())
                
                causal_preds = (torch.sigmoid(causal_logits) > 0.5).float()
                causal_accuracy = (causal_preds == binary_targets.detach()).float().mean().item()
                router_stats['causal_router_accuracy'] = causal_accuracy

            # Determine selected tokens for the second pass based on the teacher's g_cont
            if use_g_threshold:
                selected_mask = (g_cont >= g_threshold)
                batch_indices, token_indices = selected_mask.nonzero(as_tuple=True)
                gating_scores_for_selected = g_cont[selected_mask]
                router_stats['selected_tokens_proportion'] = (selected_mask.sum() / (B * T)).item()
            else:
                k = max(1, int(T * self.predictive_router.capacity))
                gating_scores_values, topk_idx = g_cont.topk(k, dim=-1)
                gating_scores_for_selected = gating_scores_values.reshape(-1) # Reshape to 1D
                batch_indices = torch.arange(B, device=g_cont.device).unsqueeze(1).expand(-1, k).reshape(-1)
                token_indices = topk_idx.reshape(-1)
                router_stats['selected_tokens_proportion'] = (topk_idx.numel() / (B * T))

            # Apply the second TF block only to selected tokens
            final_hidden_states, _, _ = self.block.process_selected(
                original_hidden, # STT uses original_hidden for process_selected
                batch_indices=batch_indices,
                token_indices=token_indices,
                gating_scores=gating_scores_for_selected,
                use_soft_gating=True,
                **kwargs
            )
            return final_hidden_states, layer_losses, router_stats, g_cont # Return g_cont tensor
        
        else: # Inference
            if not self.training:
                log.debug("--- STTLayer VALIDATION TRACE ---")
                log.debug(f"original_hidden: shape={original_hidden.shape}, dtype={original_hidden.dtype}, mean={original_hidden.mean():.4f}, min={original_hidden.min():.4f}, max={original_hidden.max():.4f}")

            use_causal = self.model_params.get('use_causal_router_in_validation', True)

            if use_causal and self.causal_router is not None:
                causal_logits, _, causal_stats = self.causal_router(original_hidden)
                router_stats.update(causal_stats)
                
                if not self.training:
                    log.debug(f"causal_logits: shape={causal_logits.shape}, dtype={causal_logits.dtype}, mean={causal_logits.mean():.4f}, min={causal_logits.min():.4f}, max={causal_logits.max():.4f}")

                B, T, D = hidden_states.shape

                if use_g_threshold:
                    selected_mask = (torch.sigmoid(causal_logits) >= g_threshold)
                    if not self.training:
                        log.debug(f"selected_mask (threshold={g_threshold}): shape={selected_mask.shape}, dtype={selected_mask.dtype}, sum={selected_mask.sum()}")
                    batch_indices, token_indices = selected_mask.nonzero(as_tuple=True)
                    router_stats['inferred_selected_tokens_proportion'] = (selected_mask.sum() / (B * T)).item()
                else:
                    k = max(1, int(T * self.causal_router.capacity))
                    gating_scores, topk_idx = causal_logits.topk(k, dim=-1)
                    if not self.training:
                        log.debug(f"topk_idx (k={k}): shape={topk_idx.shape}, dtype={topk_idx.dtype}")
                    batch_indices = torch.arange(B, device=causal_logits.device).unsqueeze(1).expand(-1, k).reshape(-1)
                    token_indices = topk_idx.reshape(-1)
                    router_stats['inferred_selected_tokens_proportion'] = (topk_idx.numel() / (B * T))

                if not self.training:
                    log.debug(f"batch_indices: shape={batch_indices.shape}, numel={batch_indices.numel()}")
                    log.debug(f"token_indices: shape={token_indices.shape}, numel={token_indices.numel()}")

                final_hidden_states, _, _ = self.block.process_selected(
                    original_hidden,
                    batch_indices=batch_indices,
                    token_indices=token_indices,
                    gating_scores=None, # No gating scores for hard gating
                    use_soft_gating=False,
                    **kwargs
                )
                return final_hidden_states, layer_losses, router_stats, None # No g_cont tensor in inference
            else: # Use training-time predictive router for validation
                # Compute residuals needed for predictive router
                out = self.block(hidden_states, **kwargs)
                processed_hidden = out[0] if isinstance(out, tuple) else out
                actual_residual = processed_hidden - original_hidden

                prev_final_states = torch.cat(
                    [torch.zeros_like(processed_hidden[:, :1, :]), processed_hidden[:, :-1, :]], dim=1
                )
                predicted_residual = self.transition_network(prev_final_states)

                # Run predictive router
                g_cont, pred_stats = self.predictive_router(
                    actual_residual, predicted_residual, **kwargs
                )
                router_stats.update(pred_stats)

                B, T, D = hidden_states.shape

                if use_g_threshold:
                    selected_mask = (g_cont >= g_threshold)
                    batch_indices, token_indices = selected_mask.nonzero(as_tuple=True)
                    router_stats['selected_tokens_proportion'] = (selected_mask.sum() / (B * T)).item()
                else:
                    k = max(1, int(T * self.predictive_router.capacity))
                    _, topk_idx = g_cont.topk(k, dim=-1)
                    batch_indices = torch.arange(B, device=g_cont.device).unsqueeze(1).expand(-1, k).reshape(-1)
                    token_indices = topk_idx.reshape(-1)
                    router_stats['selected_tokens_proportion'] = (topk_idx.numel() / (B * T))

                final_hidden_states, _, _ = self.block.process_selected(
                    original_hidden,
                    batch_indices=batch_indices,
                    token_indices=token_indices,
                    gating_scores=None, # No gating scores for hard gating
                    # FIX: Use hard gating (use_soft_gating=False) during validation to get a realistic
                    # measure of the routing performance, which is crucial for benchmarking the POC.
                    use_soft_gating=False,
                    **kwargs
                )
                return final_hidden_states, layer_losses, router_stats, g_cont # Return g_cont tensor

class STTForCausalLM(BaseForCausalLM):
    _supports_flash_attn_2 = True

    def __init__(self, config, model_type: str = None, **kwargs):
        super().__init__(config, model_type=model_type, **kwargs)
        for i in range(self.config.num_hidden_layers):
            if i % 2 == 1:
                self.model.layers[i] = STTLayer(self.model.layers[i], config)

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
        all_g_cont_values = [] # To collect g_cont for regularization

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
                hidden_states, losses, rstats, g_cont_tensor = layer(
                    hidden_states,
                    **layer_args,
                )
                all_losses.append(losses)
                for key, value in rstats.items():
                    all_router_stats[f"stt/layer_{i}/{key}"] = value
                
                # Collect g_cont for regularization if available
                if g_cont_tensor is not None:
                    all_g_cont_values.append(g_cont_tensor)

            else: # Standard Qwen2DecoderLayer
                layer_args["attention_mask"] = mask_mapping[layer.attention_type]
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    **layer_args,
                )
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else hidden_states

        aux = {}
        # FIX: Use a robust loop for aggregating losses instead of a brittle comprehension.
        # This ensures that all auxiliary losses are correctly collected and added to the total loss.
        agg_losses = {}
        if all_losses:
            all_keys = set(k for l in all_losses for k in l)
            for k in all_keys:
                key_losses = [l[k] for l in all_losses if k in l and l.get(k) is not None]
                if key_losses:
                    agg_losses[k] = torch.mean(torch.stack(key_losses))
        aux['unscaled_losses'] = agg_losses
        
        aux['router_stats'] = all_router_stats
        aux['beta_ce'] = beta_ce
        aux['beta_cu'] = beta_cu

        # Add g_cont regularization loss if applicable
        if self.training and all_g_cont_values:
            avg_g_cont_across_layers = torch.mean(torch.stack(all_g_cont_values))
            aux['unscaled_losses']['stt_g_reg_loss'] = avg_g_cont_across_layers
            aux['router_stats']['stt_g_cont_mean_across_layers'] = avg_g_cont_across_layers.item()

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
            elif 'causal_router' in n and self.model_params.get('train_causal_router', True):
                params_map['stt_causal_router'].append(p)
            else:
                params_map['base_model'].append(p)
        return [{'name': k, 'params': v} for k, v in params_map.items() if v]