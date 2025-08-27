import torch
import torch.nn as nn
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from ..blocks.mod_router import MoDTokenRouter
from ..blocks.qwen_block import Qwen2Block


class MoDLayer(nn.Module):
    """
    Implements a Mixture-of-Depths (MoD) layer using a fully-batched,
    numerically stable gather-process-scatter approach. This version is
    significantly more performant than a batch-iterative approach on parallel
    hardware.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.router = MoDTokenRouter(config.hidden_size)
        self.block = Qwen2Block(config, layer_idx)
        self.capacity_gamma = config.capacity_gamma

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """
        Forward pass for the MoD layer.

        Note: This implementation does not support `use_cache=True`. Managing a
        sparse KV cache in a fully batched context is a complex problem that
        often requires specialized kernels. For training and evaluation, disabling
        the cache is a standard and safe approach.
        """
        if use_cache:
            raise NotImplementedError(
                "The fully-batched MoDLayer does not support use_cache=True."
            )

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 1. --- ROUTING ---
        # Get a scalar importance score for each token in the entire batch.
        # Shape: (batch_size, seq_len)
        router_weights = self.router(hidden_states)

        # Determine the number of tokens to process per sequence (k)
        k = max(1, int(self.capacity_gamma * seq_len))

        # Expert-choice routing: select the top-k tokens for each sequence
        top_k_weights, _ = torch.topk(router_weights, k, dim=1, sorted=False)
        
        # The threshold is the k-th largest weight for each sequence
        threshold = top_k_weights[:, -1].unsqueeze(1)
        is_selected = (router_weights >= threshold).to(torch.bool)

        # 2. --- GATHER ---
        # Get the batch and token indices of all selected tokens across the batch
        batch_indices, token_indices = is_selected.nonzero(as_tuple=True)

        # If no tokens are selected across the entire batch, return the original input
        if batch_indices.numel() == 0:
            return (hidden_states, None, None)

        # Gather the selected tokens into a single, flat tensor.
        # This tensor contains all tokens that need processing from the entire batch.
        # Shape: (num_selected_tokens, hidden_dim)
        selected_tokens = hidden_states[batch_indices, token_indices]

        # 3. --- PROCESS ---
        # For processing, we treat the gathered tokens as a single "mega-sequence"
        # in a batch of size 1. This is highly efficient.
        num_selected_tokens = selected_tokens.shape[0]
        selected_tokens_batched = selected_tokens.unsqueeze(0) # Shape: (1, num_selected_tokens, hidden_dim)

        # Create a new causal attention mask for this temporary mega-sequence.
        processing_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=None,
            input_shape=(1, num_selected_tokens),
            inputs_embeds=selected_tokens_batched,
            past_key_values_length=0,
        )

        # Gather the corresponding position_ids for the selected tokens.
        selected_pos_ids = position_ids[batch_indices, token_indices].unsqueeze(0) if position_ids is not None else None

        # Process the gathered tokens in a single forward pass.
        block_outputs = self.block(
            hidden_states=selected_tokens_batched,
            attention_mask=processing_attention_mask,
            position_ids=selected_pos_ids,
            use_cache=False,  # Caching is disabled
            **kwargs,
        )
        processed_tokens = block_outputs[0].squeeze(0) # Shape: (num_selected_tokens, hidden_dim)

        # 4. --- SCATTER ---
        # Create a clean tensor to scatter the results into.
        final_hidden_states = hidden_states.clone()

        # --- START OF FAITHFUL IMPLEMENTATION & NaN FIX ---
        # The paper states to scale the "output of the function f", which is the
        # change (delta) computed by the block, not the entire output.
        # This is also more numerically stable.
        delta = processed_tokens - selected_tokens
        
        # Gather the router weights for only the selected tokens
        selected_router_weights = router_weights[batch_indices, token_indices]
        
        # Scale the delta by the original, un-normalized router weights.
        scaled_delta = delta * selected_router_weights.unsqueeze(-1).to(delta.dtype)
        
        # Add the scaled delta back to the original selected tokens
        updated_selected_tokens = selected_tokens + scaled_delta
        # --- END OF FAITHFUL IMPLEMENTATION & NaN FIX ---

        # Scatter the updated tokens back to their original positions.
        final_hidden_states[batch_indices, token_indices] = updated_selected_tokens

        # Return in the standard format (hidden_states, present_key_value, attention_weights)
        return (final_hidden_states, None, None)
