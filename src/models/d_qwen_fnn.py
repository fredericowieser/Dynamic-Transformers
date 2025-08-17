# src/models/prior_ffn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PriorFeedForward(nn.Module):
    """
    A small feed-forward network for prior predictions in DynamicQwen layers.
    It takes an `intermediate_size_factor` to scale the hidden dimension
    for its internal intermediate size, allowing for more flexible bottlenecking.
    """
    def __init__(self, config, intermediate_size_factor: float = 2.0):
        super().__init__()
        hidden_size = config.hidden_size
        # Calculate intermediate_size based on factor, with a minimum of 1
        intermediate_size = max(1, int(hidden_size * intermediate_size_factor))

        # two projection layers and one gating projection (SwiGLU-like)
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

        # activation and dropout
        self.act = nn.SiLU()  # approximate SwiGLU gating
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout", 0.0))

        # init weights
        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights using a normal distribution and biases to zeros.
        """
        for p in self.parameters():
            if p.ndim > 1:
                # Standard deviation of 0.02 is a common heuristic for transformers
                nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PriorFeedForward network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, seq_len, hidden_size).

        Returns:
            torch.Tensor: Output tensor with shape (batch, seq_len, hidden_size).
        """
        # SwiGLU-like gating mechanism
        gate = self.act(self.w1(x)) * self.w3(x)
        out = self.w2(gate)
        return self.dropout(out)