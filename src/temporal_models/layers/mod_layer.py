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

        # Get router weights for token importance scoring
        router_weights = self.router(hidden_states)

        # Calculate capacity per sequence
        k = max(1, int(self.capacity_gamma * seq_len))

        # Select top-k tokens per sequence
        top_k_weights, _ = torch.topk(router_weights, k, dim=1, sorted=False)
        
        threshold = top_k_weights[:, -1].unsqueeze(1)
        is_selected = (router_weights >= threshold).to(torch.bool)

        # Gather selected token indices
        batch_indices, token_indices = is_selected.nonzero(as_tuple=True)

        if batch_indices.numel() == 0:
            return (hidden_states, None, None)

        # Gather selected tokens for processing
        selected_tokens = hidden_states[batch_indices, token_indices]

        # Process tokens as single batch
        num_selected_tokens = selected_tokens.shape[0]
        selected_tokens_batched = selected_tokens.unsqueeze(0)

        # Create attention mask for processing
        processing_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=None,
            input_shape=(1, num_selected_tokens),
            inputs_embeds=selected_tokens_batched,
            past_key_values_length=0,
        )

        # Gather position IDs for selected tokens
        selected_pos_ids = position_ids[batch_indices, token_indices].unsqueeze(0) if position_ids is not None else None

        block_outputs = self.block(
            hidden_states=selected_tokens_batched,
            attention_mask=processing_attention_mask,
            position_ids=selected_pos_ids,
            use_cache=False,
            **kwargs,
        )
        processed_tokens = block_outputs[0].squeeze(0)

        # Scatter processed tokens back
        final_hidden_states = hidden_states.clone()

        # Scale delta by router weights for numerical stability
        delta = processed_tokens - selected_tokens
        
        selected_router_weights = router_weights[batch_indices, token_indices]
        
        scaled_delta = delta * selected_router_weights.unsqueeze(-1).to(delta.dtype)
        
        updated_selected_tokens = selected_tokens + scaled_delta

        final_hidden_states[batch_indices, token_indices] = updated_selected_tokens

        return (final_hidden_states, None, None)
