import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from .priors import TDTFTransitionNetwork
from .routers import TDTFPredictiveRouter, TDTFCausalRouter

log = logging.getLogger(__name__)


class TDTFLayer(nn.Module):
    """Single TDTF layer implementing teacher-student paradigm."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Ensure required config keys exist (fail fast)
        for k in ["tpn_loss_weight", "causal_loss_weight", "prior_ffn_intermediate_size_factor"]:
            if not hasattr(config, k):
                raise ValueError(f"Missing config.{k}")

        # Standard transformer block
        self.transformer_block = Qwen2DecoderLayer(config, layer_idx)

        # Teacher components
        self.transition_network = TDTFTransitionNetwork(config)
        self.predictive_router = TDTFPredictiveRouter(config, layer_idx)

        # Student component
        self.causal_router = TDTFCausalRouter(config, layer_idx)

        # Loss weights
        self.tpn_loss_weight = float(config.tpn_loss_weight)
        self.causal_loss_weight = float(config.causal_loss_weight)

    def forward_training(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple] = None,
        *,
        beta_ce: float,
        beta_cu: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass during training (teacher mode)."""
        B, T, D = hidden_states.shape
        x_original = hidden_states

        # Dense block execution (teacher)
        layer_outputs = self.transformer_block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
        )

        if isinstance(layer_outputs, tuple):
            x_posterior = layer_outputs[0]
            cache = layer_outputs[1] if len(layer_outputs) > 1 else None
            attn_weights = layer_outputs[2] if len(layer_outputs) > 2 else None
        else:
            x_posterior = layer_outputs
            cache, attn_weights = None, None

        actual_residual = x_posterior - x_original

        # TPN prediction using previous token's FINAL state (x_{t-1}^(l))
        prev_final_states = torch.cat(
            [
                torch.zeros(B, 1, D, device=hidden_states.device, dtype=hidden_states.dtype),
                x_posterior[:, :-1, :],
            ],
            dim=1,
        )
        predicted_residual = self.transition_network(prev_final_states)

        # Debugging: Log magnitudes of residuals
        if self.training:
            log.debug(f"Layer {self.layer_idx} - actual_residual shape: {actual_residual.shape}, mean: {actual_residual.mean().item():.6f}, std: {actual_residual.std().item():.6f}")
            log.debug(f"Layer {self.layer_idx} - predicted_residual shape: {predicted_residual.shape}, mean: {predicted_residual.mean().item():.6f}, std: {predicted_residual.std().item():.6f}")

        # TPN loss (1/d-scaled via MSE on vectors)
        tpn_loss = F.mse_loss(predicted_residual, actual_residual.detach())

        # Predictive router with scheduled β
        g_cont, binary_targets, vpr_stats = self.predictive_router(
            actual_residual,
            predicted_residual,
            beta_ce=beta_ce,
            beta_cu=beta_cu,
        )

        # Student (causal) router learns to match TopK targets
        causal_scores, _, causal_stats = self.causal_router.compute_routing_scores(x_original)
        causal_loss = F.binary_cross_entropy_with_logits(causal_scores, binary_targets.detach())

        # Combine stats
        router_stats = {**vpr_stats, **causal_stats}

        return {
            "hidden_states": x_posterior,
            "tpn_loss": tpn_loss,
            "causal_loss": causal_loss,
            "g_continuous": g_cont,
            "binary_targets": binary_targets,
            "past_key_value": cache,
            "attention_weights": attn_weights,
            "router_stats": router_stats,
        }

    def forward_inference(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass during inference (student mode)."""
        B, T, D = hidden_states.shape

        # For autoregressive generation (T == 1): capacity-enforced selection or threshold as configured.
        if T == 1:
            causal_scores, _, causal_stats = self.causal_router.compute_routing_scores(hidden_states)
            probs = torch.sigmoid(causal_scores)

            mode = self.causal_router.routing_mode
            if mode == "topk":
                k = max(1, int(self.causal_router.capacity * B))
                _, topk_idx = probs.squeeze(-1).topk(k, dim=0)
                selected_mask = torch.zeros_like(probs, dtype=torch.bool)
                selected_mask[topk_idx, 0] = True
            elif mode == "threshold":
                selected_mask = probs > 0.5
            else:
                raise ValueError(f"Unknown student_routing_mode: {mode}")

            if not selected_mask.any():
                return {
                    "hidden_states": hidden_states,
                    "routing_decisions": selected_mask.float(),
                    "past_key_value": past_key_value,
                    "attention_weights": None,
                    "router_stats": {
                        **causal_stats,
                        "processed_tokens": 0,
                        "total_tokens": B * T,
                        "processing_ratio": 0.0,
                    },
                }

            sel_idx = selected_mask.squeeze(-1).nonzero(as_tuple=True)[0]
            x_sel = hidden_states[sel_idx, :, :]  # [k,1,D]
            attn_mask_sel = attention_mask[sel_idx, :] if attention_mask is not None else None
            pos_ids_sel = position_ids[sel_idx, :] if position_ids is not None else None
            pos_emb_sel = None
            if position_embeddings is not None:
                cos, sin = position_embeddings
                pos_emb_sel = (cos[sel_idx, :, :], sin[sel_idx, :, :])

            layer_outputs = self.transformer_block(
                x_sel,
                attention_mask=attn_mask_sel,
                position_ids=pos_ids_sel,
                past_key_value=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                position_embeddings=pos_emb_sel,
            )

            if isinstance(layer_outputs, tuple):
                x_sel_out = layer_outputs[0]
                cache = layer_outputs[1] if len(layer_outputs) > 1 else None
                attn_weights = layer_outputs[2] if len(layer_outputs) > 2 else None
            else:
                x_sel_out = layer_outputs
                cache, attn_weights = None, None

            output_states = hidden_states.clone()
            output_states[sel_idx, :, :] = x_sel_out

            return {
                "hidden_states": output_states,
                "routing_decisions": selected_mask.float(),
                "past_key_value": cache,
                "attention_weights": attn_weights,
                "router_stats": {
                    **causal_stats,
                    "processed_tokens": sel_idx.numel(),
                    "total_tokens": B * T,
                    "processing_ratio": float(sel_idx.numel()) / float(B * T),
                },
            }

        # For batched evaluation (T > 1), run dense for correctness.
        layer_outputs = self.transformer_block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
        )
        if isinstance(layer_outputs, tuple):
            x_out = layer_outputs[0]
            cache = layer_outputs[1] if len(layer_outputs) > 1 else None
            attn_weights = layer_outputs[2] if len(layer_outputs) > 2 else None
        else:
            x_out, cache, attn_weights = layer_outputs, None, None

        return {
            "hidden_states": x_out,
            "routing_decisions": None,
            "past_key_value": cache,
            "attention_weights": attn_weights,
            "router_stats": {
                "processed_tokens": B * T,
                "total_tokens": B * T,
                "processing_ratio": 1.0,
            },
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if self.training:
            # β values must be provided by caller during training
            if "beta_ce" not in kwargs or "beta_cu" not in kwargs:
                raise ValueError("Missing beta_ce/beta_cu in TDTFLayer.forward (training mode)")
            return self.forward_training(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                use_cache,
                output_attentions,
                position_embeddings,
                beta_ce=kwargs["beta_ce"],
                beta_cu=kwargs["beta_cu"],
            )
        else:
            return self.forward_inference(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                use_cache,
                output_attentions,
                position_embeddings,
            )