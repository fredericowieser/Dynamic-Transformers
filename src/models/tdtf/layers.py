import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from .priors import TDTFTransitionNetwork
from .routers import TDTFPredictiveRouter, TDTFCausalRouter


class TDTFLayer(nn.Module):
    """Single TDTF layer implementing teacher-student training paradigm.

    During training: Uses both TPN + Predictive Router (teacher) and Causal Router (student)
    During inference: Uses only Causal Router for token gating decisions
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.training_mode = True  # Track if we're in training vs inference mode

        # Ensure config has attention implementation
        if not hasattr(config, '_attn_implementation'):
            config._attn_implementation = 'eager'

        # Standard transformer block
        self.transformer_block = Qwen2DecoderLayer(config, layer_idx)

        # Training components (teacher model)
        self.transition_network = TDTFTransitionNetwork(config)
        self.predictive_router = TDTFPredictiveRouter(config, layer_idx)

        # Inference component (student model)
        self.causal_router = TDTFCausalRouter(config, layer_idx)

        # Loss weights
        self.tpn_loss_weight = getattr(config, 'tpn_loss_weight')
        self.causal_loss_weight = getattr(config, 'causal_loss_weight')

    def forward_training(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass during training (teacher mode)."""
        B, T, D = hidden_states.shape

        # Store input state (original)
        x_original = hidden_states

        # Compute TF block output (posterior/ground truth)
        layer_outputs = self.transformer_block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
        )

        # Handle different return formats
        if isinstance(layer_outputs, tuple):
            x_posterior = layer_outputs[0]
            cache = layer_outputs[1] if len(layer_outputs) > 1 else None
            attn_weights = layer_outputs[2] if len(layer_outputs) > 2 else None
        else:
            x_posterior = layer_outputs
            cache = None
            attn_weights = None

        # Compute actual residual
        actual_residual = x_posterior - x_original

        # TPN prediction using previous token's final state
        # For t=1, use zero vector; for t>1, use x_{t-1}^(l)
        prev_final_states = torch.cat([
            torch.zeros(B, 1, D, device=hidden_states.device, dtype=hidden_states.dtype),
            x_posterior[:, :-1, :]  # Use final states from previous positions
        ], dim=1)  # [B, T, D]

        predicted_residual = self.transition_network(prev_final_states)

        # Compute TPN loss
        tpn_loss = F.mse_loss(predicted_residual, actual_residual.detach())

        # Predictive router (teacher)
        g_continuous, binary_targets = self.predictive_router(actual_residual, predicted_residual)

        # Causal router (student) training
        causal_scores, _, causal_stats = self.causal_router.compute_routing_scores(x_original)
        causal_probs = torch.sigmoid(causal_scores)

        # Causal router loss (BCE with binary targets from teacher)
        causal_loss = F.binary_cross_entropy_with_logits(causal_scores, binary_targets.detach())

        return {
            'hidden_states': x_posterior,  # Use posterior as output
            'tpn_loss': tpn_loss,
            'causal_loss': causal_loss,
            'g_continuous': g_continuous,
            'binary_targets': binary_targets,
            'causal_probs': causal_probs,
            'past_key_value': cache,
            'attention_weights': attn_weights,
            'router_stats': causal_stats,
        }

    def forward_inference(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass during inference (student mode)."""
        B, T, D = hidden_states.shape

        # Get causal routing decisions
        causal_scores, _, causal_stats = self.causal_router.compute_routing_scores(hidden_states)

        # Binarize decisions (simple thresholding)
        # TODO: Could also implement TopK of generated prefix for better control
        routing_decisions = (torch.sigmoid(causal_scores) > 0.5).float()  # [B, T]

        # Initialize output
        output_states = hidden_states.clone()
        cache = None
        attn_weights = None
        processed_tokens = 0

        # Process each token based on routing decision
        # FIX: Vectorized token processing for efficiency
        selected_batch_indices, selected_token_indices = routing_decisions.nonzero(as_tuple=True)
        num_selected_tokens = selected_batch_indices.numel()

        if num_selected_tokens > 0:
            selected_hidden_states = hidden_states[selected_batch_indices, selected_token_indices]
            selected_hidden_states = selected_hidden_states.unsqueeze(0) # [1, num_selected_tokens, D]

            selected_attention_mask = _prepare_4d_causal_attention_mask(
                None, (1, num_selected_tokens), selected_hidden_states, 0
            )

            selected_position_ids = None
            if position_ids is not None:
                selected_position_ids = position_ids[selected_batch_indices, selected_token_indices].unsqueeze(0)

            selected_position_embeddings = None
            if position_embeddings is not None:
                cos, sin = position_embeddings
                selected_cos = cos[selected_batch_indices, selected_token_indices].unsqueeze(0)
                selected_sin = sin[selected_batch_indices, selected_token_indices].unsqueeze(0)
                selected_position_embeddings = (selected_cos, selected_sin)

            layer_outputs = self.transformer_block(
                selected_hidden_states,
                attention_mask=selected_attention_mask,
                position_ids=selected_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                position_embeddings=selected_position_embeddings,
            )

            if isinstance(layer_outputs, tuple):
                processed_tokens = layer_outputs[0].squeeze(0)
                cache = layer_outputs[1] if len(layer_outputs) > 1 else None
                attn_weights = layer_outputs[2] if len(layer_outputs) > 2 else None
            else:
                processed_tokens = layer_outputs.squeeze(0)
                cache = None
                attn_weights = None

            output_states[selected_batch_indices, selected_token_indices] = processed_tokens

        processed_tokens = num_selected_tokens # Update processed_tokens count


        causal_stats['processed_tokens'] = processed_tokens
        causal_stats['total_tokens'] = B * T
        causal_stats['processing_ratio'] = processed_tokens / (B * T) if B * T > 0 else 0.0

        return {
            'hidden_states': output_states,
            'routing_decisions': routing_decisions,
            'past_key_value': cache,
            'attention_weights': attn_weights,
            'router_stats': causal_stats,
        }

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
    ) -> Dict[str, Any]:
        """Forward pass - delegates to training or inference mode."""
        if self.training:
            return self.forward_training(
                hidden_states, attention_mask, position_ids, past_key_values,
                use_cache, output_attentions, position_embeddings, **kwargs
            )
        else:
            return self.forward_inference(
                hidden_states, attention_mask, position_ids, past_key_values,
                use_cache, output_attentions, position_embeddings, **kwargs
            )
