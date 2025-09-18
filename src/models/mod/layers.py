import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from .routers import MoDRouter, CausalMoDRouter


class MoDLayer(nn.Module):
    """MoD layer that routes tokens to computation or residual connection."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Ensure config has attention implementation
        if not hasattr(config, '_attn_implementation'):
            config._attn_implementation = 'eager'

        # Standard Qwen2 decoder block
        self.block = Qwen2DecoderLayer(config, layer_idx)

        # MoD router
        self.router = MoDRouter(config, layer_idx)
        self.causal_router = CausalMoDRouter(config, layer_idx)
        self.causal_loss_weight = getattr(config, 'causal_loss_weight', 0.01)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict, Optional[Tuple], Optional[torch.Tensor]]:
        """Forward pass with conditional computation.

        Selected tokens go through the transformer block,
        others bypass via residual connection.
        """
        B, T, D = hidden_states.shape

        if self.training:
            # Get routing decision
            scores, aux_loss, stats = self.router.compute_routing_scores(hidden_states)

            # Select top-k tokens
            k = max(1, int(T * self.router.capacity))
            _, topk_indices = scores.topk(k, dim=-1)

            # Causal router training
            causal_scores, _, _ = self.causal_router.compute_routing_scores(hidden_states)
            
            # Create binary targets for causal router
            binary_targets = torch.zeros_like(scores)
            binary_targets.scatter_(1, topk_indices, 1)

            causal_loss = F.binary_cross_entropy_with_logits(causal_scores, binary_targets.detach())
            
            if aux_loss is not None:
                aux_loss += self.causal_loss_weight * causal_loss
            else:
                aux_loss = self.causal_loss_weight * causal_loss
            
            selected_hidden, batch_idx, token_idx, selected_scores = self.router.select_tokens(
                scores, hidden_states
            )

        else:
            # Inference with causal router
            scores, aux_loss, stats = self.causal_router.compute_routing_scores(hidden_states)
            selected_hidden, batch_idx, token_idx, selected_scores = self.causal_router.select_tokens(
                scores, hidden_states
            )

        # Track statistics
        stats['selected_tokens'] = batch_idx.numel()
        stats['total_tokens'] = B * T

        # If no tokens selected, skip computation
        if batch_idx.numel() == 0:
            return (
                hidden_states,
                aux_loss,
                stats,
                past_key_values if use_cache else None,
                None
            )

        # Reshape for processing (add batch dimension)
        num_selected = selected_hidden.shape[0]
        selected_hidden = selected_hidden.unsqueeze(0)  # [1, num_selected, D]

        # Create attention mask for selected tokens if needed
        selected_attn_mask = None
        if attention_mask is not None and num_selected > 0:
            selected_attn_mask = _prepare_4d_causal_attention_mask(
                None, (1, num_selected), selected_hidden, 0
            )

        # Gather position information for selected tokens
        selected_pos_ids = None
        if position_ids is not None and num_selected > 0:
            pos_2d = position_ids.reshape(-1)
            flat_idx = batch_idx * T + token_idx
            selected_pos_ids = pos_2d[flat_idx].unsqueeze(0)  # [1, num_selected]

        # Gather position embeddings for selected tokens
        selected_pos_emb = None
        if position_embeddings is not None and num_selected > 0:
            cos, sin = position_embeddings
            gathered_cos = cos[batch_idx, token_idx].unsqueeze(0)  # [1, num_selected, head_dim]
            gathered_sin = sin[batch_idx, token_idx].unsqueeze(0)
            selected_pos_emb = (gathered_cos, gathered_sin)

        # Process selected tokens through transformer block
        layer_outputs = self.block(
            selected_hidden,
            attention_mask=selected_attn_mask,
            position_ids=selected_pos_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            position_embeddings=selected_pos_emb,
        )

        # Handle different return formats from Qwen2DecoderLayer
        if isinstance(layer_outputs, tuple):
            processed = layer_outputs[0].squeeze(0)  # [num_selected, D]
            cache = layer_outputs[1] if len(layer_outputs) > 1 else None
            attn_weights = layer_outputs[2] if len(layer_outputs) > 2 else None
        else:
            processed = layer_outputs.squeeze(0)  # [num_selected, D]
            cache = None
            attn_weights = None

        # Apply gating based on router scores (soft selection)
        gate_values = torch.sigmoid(selected_scores).unsqueeze(-1)  # [num_selected, 1]

        # Gated update: mix processed and original based on gate
        selected_hidden_flat = selected_hidden.squeeze(0)  # [num_selected, D]
        gated_processed = gate_values * processed + (1 - gate_values) * selected_hidden_flat

        # Scatter processed tokens back
        output = self.router.scatter_tokens(
            gated_processed,
            hidden_states,
            batch_idx,
            token_idx
        )

        return output, aux_loss, stats, cache, attn_weights
