import torch
import torch.nn as nn

from ..blocks.mod_router import MoDTokenRouter
from ..blocks.qwen_block import Qwen2Block


class MoDLayer(nn.Module):
    """
    Implements a Mixture-of-Depths (MoD) layer.

    This layer uses a router to select a fixed capacity of tokens (top-k)
    to be processed by a standard transformer block. All other tokens
    bypass the block via a simple residual connection, saving computation.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.router = MoDTokenRouter(config.hidden_size)
        self.block = Qwen2Block(config, layer_idx)
        self.capacity_gamma = config.capacity_gamma

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """
        Forward pass for the MoD layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch, seq_len, hidden_size).
            **kwargs: Additional arguments to be passed to the transformer block (e.g., attention_mask).

        Returns:
            A tuple containing the output hidden states and other optional outputs from the block.
        """
        batch_size, seq_len, dim = hidden_states.shape
        router_weights = self.router(hidden_states)

        # Determine the number of tokens to process (k) based on capacity_gamma
        k = max(1, int(self.capacity_gamma * seq_len))

        # Expert-choice routing: select the top-k tokens to process
        top_k_weights, _ = torch.topk(router_weights, k, dim=1, sorted=True)
        threshold = top_k_weights[:, -1].unsqueeze(1)
        is_selected = (router_weights >= threshold).unsqueeze(-1)

        # To maintain a static graph, we create a tensor of selected tokens and zero out the rest
        selected_hidden_states = torch.where(
            is_selected, hidden_states, torch.zeros_like(hidden_states)
        )

        # The block processes the full sequence, but non-selected tokens are zero vectors
        block_output, *rest = self.block(selected_hidden_states, **kwargs)

        # For selected tokens, the output is the result of the block's computation.
        # For non-selected tokens, the output is the original hidden_state (residual connection).
        # We also scale the output of the processed tokens by their router weights as in the paper.
        final_hidden_states = torch.where(
            is_selected,
            block_output * router_weights.unsqueeze(-1).to(block_output.dtype),
            hidden_states,
        )

        return (final_hidden_states,) + tuple(rest)