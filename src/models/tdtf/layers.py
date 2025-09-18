import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

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
        causal_loss = F.binary_cross_entropy(causal_probs, binary_targets.detach())

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
        for t in range(T):
            for b in range(B):
                if routing_decisions[b, t] == 1.0:
                    # Process this token through TF block
                    token_input = hidden_states[b:b+1, t:t+1, :]  # [1, 1, D]

                    # Prepare attention mask for single token
                    token_mask = None
                    if attention_mask is not None:
                        token_mask = attention_mask[b:b+1, :, t:t+1, :t+1]  # Causal mask up to position t

                    # Prepare position info
                    token_pos_ids = None
                    if position_ids is not None:
                        token_pos_ids = position_ids[b:b+1, t:t+1]

                    token_pos_emb = None
                    if position_embeddings is not None:
                        cos, sin = position_embeddings
                        token_pos_emb = (cos[b:b+1, t:t+1], sin[b:b+1, t:t+1])

                    # Process through TF block
                    layer_outputs = self.transformer_block(
                        token_input,
                        attention_mask=token_mask,
                        position_ids=token_pos_ids,
                        past_key_value=past_key_values,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        position_embeddings=token_pos_emb,
                    )

                    if isinstance(layer_outputs, tuple):
                        processed_token = layer_outputs[0]  # [1, 1, D]
                        if cache is None and layer_outputs[1] is not None:
                            cache = layer_outputs[1]
                        if attn_weights is None and len(layer_outputs) > 2:
                            attn_weights = layer_outputs[2]
                    else:
                        processed_token = layer_outputs

                    # Update output
                    output_states[b, t, :] = processed_token[0, 0, :]
                    processed_tokens += 1
                # else: token passes through unchanged (residual connection)

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
