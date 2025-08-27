import torch
import torch.nn as nn

class MoDTokenRouter(nn.Module):
    """
    A simple router that assigns a scalar weight to each token based on a
    linear projection. Used for Mixture-of-Depths routing.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        # The router is a simple linear projection from the hidden dimension to a single scalar.
        self.gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): The input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            torch.Tensor: A tensor of router weights for each token. Shape: (batch, seq_len).
        """
        return self.gate(hidden_states).squeeze(-1)