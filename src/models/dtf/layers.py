import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from .priors import DTFPriorNetwork
from .routers import DTFRouter, CausalDTFRouter


class DTFDecisionLayer(nn.Module):
    """Decision layer that computes the three states needed for routing.

    Processes input through both a standard transformer block (posterior)
    and a lightweight prior network, providing all states for routing decisions.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Ensure config has attention implementation
        if not hasattr(config, '_attn_implementation'):
            config._attn_implementation = 'eager'

        # Standard transformer block for posterior
        self.block = Qwen2DecoderLayer(config, layer_idx)
        # Lightweight prior network for change prediction
        self.prior_network = DTFPriorNetwork(config)

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
        """Compute original, posterior, and prior states.

        Returns dictionary containing all three states plus auxiliary loss.
        """
        # Store original input
        original = hidden_states

        # Process through standard transformer layer to get posterior
        layer_outputs = self.block(
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
            posterior = layer_outputs[0]
            cache = layer_outputs[1] if len(layer_outputs) > 1 else None
            attn_weights = layer_outputs[2] if len(layer_outputs) > 2 else None
        else:
            posterior = layer_outputs
            cache = None
            attn_weights = None

        # Compute prior prediction (lightweight)
        prior = self.prior_network(hidden_states)

        # Compute prior loss for training (teach prior to predict posterior)
        prior_loss = None
        if self.training:
            prior_loss = F.mse_loss(prior, posterior.detach())

        return {
            'original': original,
            'posterior': posterior,
            'prior': prior,
            'prior_loss': prior_loss,
            'past_key_value': cache,
            'attention_weights': attn_weights,
        }


class DTFDynamicLayer(nn.Module):
    """Dynamic layer that processes selected tokens based on surprise.

    Uses routing decisions from decision layer to conditionally process
    tokens through a second transformer block.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Ensure config has attention implementation
        if not hasattr(config, '_attn_implementation'):
            config._attn_implementation = 'eager'

        # Standard transformer block for processing selected tokens
        self.block = Qwen2DecoderLayer(config, layer_idx)
        # Router for token selection
        self.router = DTFRouter(config, layer_idx)
        self.causal_router = CausalDTFRouter(config, layer_idx)
        self.causal_loss_weight = getattr(config, 'causal_loss_weight', 0.01)

    def forward(
        self,
        hidden_states: torch.Tensor,
        decision_output: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any], Optional[Tuple], Optional[torch.Tensor]]:
        """Process selected tokens through transformer block.

        Args:
            hidden_states: Current hidden states (from decision layer posterior)
            decision_output: Dictionary with original, posterior, prior states
            Other args: Standard transformer layer arguments

        Returns:
            Updated hidden states, routing stats, cache, attention weights
        """
        B, T, D = hidden_states.shape

        # Extract states from decision layer
        original = decision_output['original']
        posterior = decision_output['posterior']
        prior = decision_output['prior']

        # Ensure all have batch dimension
        if original.dim() == 2:
            original = original.unsqueeze(0)
        if posterior.dim() == 2:
            posterior = posterior.unsqueeze(0)
        if prior.dim() == 2:
            prior = prior.unsqueeze(0)

        if self.training:
            # Compute routing scores based on surprise
            scores, _, stats = self.router.compute_routing_scores(
                hidden_states, original, posterior, prior
            )

            # Select top-k tokens
            k = max(1, int(T * self.router.capacity))
            _, topk_indices = scores.topk(k, dim=-1)

            # Causal router training
            causal_scores, _, _ = self.causal_router.compute_routing_scores(hidden_states)
            
            # Create binary targets for causal router
            binary_targets = torch.zeros_like(scores)
            binary_targets.scatter_(1, topk_indices, 1)

            causal_loss = F.binary_cross_entropy_with_logits(causal_scores, binary_targets.detach())
            
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

        # If no tokens selected, return input unchanged
        if batch_idx.numel() == 0:
            return hidden_states, aux_loss, stats, None, None

        # Reshape for processing
        num_selected = selected_hidden.shape[0]
        selected_hidden = selected_hidden.unsqueeze(0)  # [1, num_selected, D]

        # Create attention mask for selected tokens
        selected_attn_mask = None
        if attention_mask is not None and num_selected > 0:
            selected_attn_mask = _prepare_4d_causal_attention_mask(
                None, (1, num_selected), selected_hidden, 0
            )

        # Gather position information
        selected_pos_ids = None
        if position_ids is not None and num_selected > 0:
            pos_2d = position_ids.reshape(-1)
            flat_idx = batch_idx * T + token_idx
            selected_pos_ids = pos_2d[flat_idx].unsqueeze(0)

        # Gather position embeddings
        selected_pos_emb = None
        if position_embeddings is not None and num_selected > 0:
            cos, sin = position_embeddings
            gathered_cos = cos[batch_idx, token_idx].unsqueeze(0)
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

        # Handle return format
        if isinstance(layer_outputs, tuple):
            processed = layer_outputs[0].squeeze(0)  # [num_selected, D]
            cache = layer_outputs[1] if len(layer_outputs) > 1 else None
            attn_weights = layer_outputs[2] if len(layer_outputs) > 2 else None
        else:
            processed = layer_outputs.squeeze(0)
            cache = None
            attn_weights = None

        # Re-merge selected tokens back into the original sequence
        final_hidden_states = hidden_states.clone()

        # FIX: Corrected undefined 'g_bin' variable and applied G_cont weighting as per spec.
        # Use batch_idx and token_idx from self.router.select_tokens for indexing.
        if self.training:
            # Apply G_cont weighting to the TF-Block output for selected tokens
            # The spec formula is H_i^(l+1) = H_{post,i}^{(l)} + G_{cont,i} * TF-Block(H_{post,i}^{(l)}) if i in S
            # `processed` contains the output of the second TF-Block for selected tokens.
            # To get TF-Block(H_{post,i}^{(l)}), we subtract the input to that block (which is hidden_states[batch_idx, token_idx]).
            tf_block_output_for_selected = processed - hidden_states[batch_idx, token_idx]
            weighted_tf_block_output = (selected_scores.unsqueeze(-1) * tf_block_output_for_selected).to(final_hidden_states.dtype)
            final_hidden_states[batch_idx, token_idx] = hidden_states[batch_idx, token_idx] + weighted_tf_block_output
        else:
            # During inference, it's a hard gate: either processed or original H_post
            final_hidden_states[batch_idx, token_idx] = processed

        return final_hidden_states, aux_loss, stats, cache, attn_weights
