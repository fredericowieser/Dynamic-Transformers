import torch
import torch.nn as nn

from ..blocks.mod_router import MoDTokenRouter
from ..blocks.qwen_block import Qwen2Block


class MoDLayer(nn.Module):
    """
    Implements a Mixture-of-Depths (MoD) layer using a numerically stable,
    batch-iterative approach to process only selected tokens.
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
        batch_size, seq_len, _ = hidden_states.shape
        router_weights = self.router(hidden_states)

        # Determine the number of tokens to process (k)
        k = max(1, int(self.capacity_gamma * seq_len))
        
        # Expert-choice routing: select the top-k tokens for each sequence in the batch
        top_k_weights, _ = torch.topk(router_weights, k, dim=1, sorted=True)
        threshold = top_k_weights[:, -1].unsqueeze(1)
        is_selected = router_weights >= threshold

        # Initialize the output as a copy of the input for the residual path
        final_hidden_states = hidden_states.clone()
        
        # This will hold the KV cache from the last processed batch item
        # Note: In a batched context, handling KV cache properly for sparse layers
        # is complex. For simplicity, we'll just return the cache of the last item.
        # For real-world use, more sophisticated cache management would be needed.
        present_key_value = None
        
        # Process each sequence in the batch individually to ensure stability
        for i in range(batch_size):
            selected_indices = is_selected[i].nonzero().squeeze(-1)

            # Skip if no tokens are selected for this sequence
            if selected_indices.numel() == 0:
                continue

            # Gather the selected tokens into a new *dense* tensor
            selected_tokens = hidden_states[i, selected_indices]
            
            # Gather corresponding positional IDs
            selected_pos_ids = position_ids[i, selected_indices].unsqueeze(0) if position_ids is not None else None
            
            # Create the attention mask for the selected tokens
            current_attention_mask = None
            if attention_mask is not None:
                # Handle 4D attention mask by indexing
                if attention_mask.dim() == 4:
                    num_heads = attention_mask.shape[1]
                    selected_attn_mask = attention_mask[i, :, selected_indices][:, :, selected_indices]
                    current_attention_mask = selected_attn_mask.unsqueeze(0)
                else: # Handle 2D mask (like from Flash Attention)
                    current_attention_mask = attention_mask[i, selected_indices].unsqueeze(0)
            
            # Process the small, dense tensor of selected tokens. Unsqueeze to create a batch of 1.
            block_outputs = self.block(
                hidden_states=selected_tokens.unsqueeze(0),
                attention_mask=current_attention_mask,
                position_ids=selected_pos_ids,
                use_cache=use_cache,
                **kwargs,
            )
            processed_tokens = block_outputs[0].squeeze(0)

            # Scatter the processed results back into the final output tensor.
            # Scale by their router weights as described in the MoD paper.
            scaled_processed_tokens = processed_tokens * router_weights[i, selected_indices].unsqueeze(-1).to(processed_tokens.dtype)
            final_hidden_states[i, selected_indices] = scaled_processed_tokens
            
            if use_cache and len(block_outputs) > 1:
                present_key_value = block_outputs[1]

        # Return in the same format as a standard block
        return (final_hidden_states, present_key_value, None)