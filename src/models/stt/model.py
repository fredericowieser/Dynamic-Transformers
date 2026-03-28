import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import (Qwen2DecoderLayer,
                                                      Qwen2RMSNorm)

from ..base.block import DynamicBlock
from ..base.causal_lm import BaseForCausalLM
from ..base.priors import BasePriorNetwork
from ..base.routers import BaseSurpriseRouter, UnifiedCausalRouter
from ..configs import STTConfig

log = logging.getLogger(__name__)


class STTTransitionNetwork(BasePriorNetwork):
    """Implements the STT transition network with pre-normalization: MLP(RMSNorm(x))."""

    def __init__(self, config):
        super().__init__(config)
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(x))


class STTPredictiveRouter(BaseSurpriseRouter):
    def __init__(self, config, layer_idx: int, capacity_attr: str):
        super().__init__(config, capacity_attr=capacity_attr)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        actual_residual = args[0]
        predicted_residual = args[1]
        beta_ce = kwargs["beta_ce"]
        beta_cu = kwargs["beta_cu"]

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
        self.predictive_router = STTPredictiveRouter(config, layer_idx=0, capacity_attr="capacity")
        self.causal_router = UnifiedCausalRouter(config)
        self.config = config
        self.causal_threshold = getattr(config, "causal_threshold", 0.5)

    def forward(self, hidden_states, use_causal_router: bool = False, **kwargs):
        original_hidden = hidden_states
        router_stats = {}
        layer_losses = {}
        B, T, D = hidden_states.shape

        use_g_threshold = getattr(self.config, "use_g_threshold_selection", False)
        g_threshold = getattr(self.config, "g_threshold", 0.5)
        
        g_cont_for_loss = None

        if use_causal_router:
            causal_logits, _, _ = self.causal_router(hidden_states)
            gating_probs = torch.sigmoid(causal_logits)
            
            selected_mask = gating_probs >= self.causal_threshold
            batch_indices, token_indices = selected_mask.nonzero(as_tuple=True)
            gating_scores_for_selected = gating_probs[selected_mask]
            
            if B * T > 0:
                router_stats["selected_tokens_proportion"] = (selected_mask.sum() / (B * T)).item()
            router_stats["g_cont"] = gating_probs.mean().item()
            
            final_hidden_states, _, _ = self.block.process_selected(
                original_hidden,
                batch_indices=batch_indices,
                token_indices=token_indices,
                gating_scores=gating_scores_for_selected,
                use_soft_gating=True,
                **kwargs,
            )
            return final_hidden_states, layer_losses, router_stats, g_cont_for_loss

        # First pass through the block to get processed_hidden and residuals
        out = self.block(hidden_states, **kwargs)
        processed_hidden = out[0] if isinstance(out, tuple) else out

        actual_residual = processed_hidden - original_hidden

        prev_final_states = torch.cat(
            [torch.zeros_like(processed_hidden[:, :1, :]), processed_hidden[:, :-1, :]], dim=1
        )
        predicted_residual = self.transition_network(prev_final_states)
        if self.training:
            layer_losses["stt_tpn_loss"] = F.mse_loss(predicted_residual, actual_residual.detach())

        g_cont, pred_stats = self.predictive_router(
            actual_residual, predicted_residual, **kwargs
        )
        router_stats.update(pred_stats)

        # Determine selected tokens for the second pass based on g_cont
        if use_g_threshold:
            selected_mask = g_cont >= g_threshold
            binary_targets = selected_mask.float()
            
            batch_indices, token_indices = selected_mask.nonzero(as_tuple=True)
            gating_scores_for_selected = g_cont[selected_mask]
            if B * T > 0:
                router_stats["selected_tokens_proportion"] = (selected_mask.sum() / (B * T)).item()
        else:
            k = max(1, int(T * self.predictive_router.capacity))
            gating_scores_values, topk_idx = g_cont.topk(k, dim=-1)

            binary_targets = torch.zeros_like(g_cont)
            binary_targets.scatter_(1, topk_idx, 1.0)

            # Sort indices to ensure causal processing (past cannot see future)
            topk_idx, sort_indices = topk_idx.sort(dim=-1)
            gating_scores_values = gating_scores_values.gather(dim=-1, index=sort_indices)

            gating_scores_for_selected = gating_scores_values.reshape(-1)
            batch_indices = (
                torch.arange(B, device=g_cont.device).unsqueeze(1).expand(-1, k).reshape(-1)
            )
            token_indices = topk_idx.reshape(-1)
            if B * T > 0:
                router_stats["selected_tokens_proportion"] = topk_idx.numel() / (B * T)

        causal_logits, causal_loss, causal_acc = self.causal_router(hidden_states.detach(), targets=binary_targets)
        
        if self.training:
            layer_losses["stt_causal_router_loss"] = causal_loss
        if causal_acc is not None:
            router_stats["causal_router_acc"] = causal_acc.item()

        # Apply the second TF block only to selected tokens, always using soft gating
        final_hidden_states, _, _ = self.block.process_selected(
            original_hidden,  # STT uses original_hidden for process_selected
            batch_indices=batch_indices,
            token_indices=token_indices,
            gating_scores=gating_scores_for_selected,
            use_soft_gating=True,  # Unification point
            **kwargs,
        )

        # Only return g_cont for regularization loss during training
        g_cont_for_loss = g_cont if self.training else None

        return final_hidden_states, layer_losses, router_stats, g_cont_for_loss


class STTForCausalLM(BaseForCausalLM):
    config_class = STTConfig
    _supports_flash_attn_2 = False

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
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
        all_g_cont_values = []  # To collect g_cont for regularization

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
            if isinstance(layer, STTLayer):
                layer_attn_mask = mask_mapping[layer.block.layer.attention_type]
                
                if getattr(self, "gradient_checkpointing", False) and self.training:
                    # Clean up checkpoint call to ensure hidden_states is the primary positional arg
                    hidden_states, losses, rstats, g_cont_tensor = torch.utils.checkpoint.checkpoint(
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
                    hidden_states, losses, rstats, g_cont_tensor = layer(
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
                for key, value in rstats.items():
                    all_router_stats[f"stt/layer_{i}/{key}"] = value

                # Collect g_cont for regularization if available
                if g_cont_tensor is not None:
                    all_g_cont_values.append(g_cont_tensor)

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
                        layer_outputs[0] if isinstance(layer_outputs, tuple) else hidden_states
                    )

        aux = {}
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

        # Add g_cont regularization loss if applicable
        if self.training and all_g_cont_values:
            avg_g_cont_across_layers = torch.mean(torch.stack(all_g_cont_values))
            aux["unscaled_losses"]["stt_g_reg_loss"] = avg_g_cont_across_layers
            aux["router_stats"]["stt_g_cont_mean_across_layers"] = avg_g_cont_across_layers.item()

        return hidden_states, aux
