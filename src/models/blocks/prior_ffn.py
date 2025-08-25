import torch
import torch.nn as nn
import math  # <-- ADD THIS IMPORT
import logging # <-- ADD THIS IMPORT

log = logging.getLogger(__name__) # <-- ADD THIS LINE

class PriorFeedForward(nn.Module):
    """
    A small feed-forward network for prior predictions in the VPR architecture.
    It takes an `intermediate_size_factor` to scale the hidden dimension
    for its internal intermediate size, allowing for more flexible bottlenecking.
    """

    def __init__(self, config, intermediate_size_factor: float = 2.0):
        super().__init__()
        hidden_size = config.hidden_size
        
        # Calculate the raw size
        raw_intermediate_size = hidden_size * intermediate_size_factor

        # Round up to the nearest integer
        rounded_up_size = math.ceil(raw_intermediate_size)
        
        # Ensure the size is an even number by adding 1 if it's odd
        even_size = rounded_up_size + (rounded_up_size % 2)

        # Enforce a minimum size of 2 to ensure it's a valid, non-zero even number
        intermediate_size = max(2, even_size)

        log.info(
            f"Initialized PriorFeedForward with intermediate_size={intermediate_size} "
            f"(factor={intermediate_size_factor}, raw_size={raw_intermediate_size:.2f})"
        )

        # two projection layers and one gating projection (SwiGLU-like)
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

        # activation and dropout
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout", 0.0))

        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights using a normal distribution and biases to zeros.
        """
        for p in self.parameters():
            if p.ndim > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PriorFeedForward network.
        """
        # SwiGLU-like gating mechanism
        gate = self.act(self.w1(x)) * self.w3(x)
        out = self.w2(gate)
        return self.dropout(out)