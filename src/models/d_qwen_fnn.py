import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    A small feed-forward network for prior predictions in DynamicQwen layers.
    Matches the hidden_size → intermediate_size → hidden_size structure
    (with SwiGLU/SiLU-like gating), loosely following the Qwen MLP design.
    """
    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        # Qwen intermediate size is usually 4 * hidden_size
        intermediate = getattr(config, "intermediate_size", 4 * hidden)

        # two projection layers and one gating projection
        self.w1 = nn.Linear(hidden, intermediate, bias=False)
        self.w3 = nn.Linear(hidden, intermediate, bias=False)
        self.w2 = nn.Linear(intermediate, hidden, bias=False)

        # activation and dropout
        self.act = nn.SiLU()  # approximate SwiGLU gating
        # --- START OF CHANGE ---
        # Qwen2Config uses 'hidden_dropout' for general dropout.
        # Use getattr for robustness, falling back to 0.0 if not found.
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout", 0.0))
        # --- END OF CHANGE ---

        # init weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.ndim > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_size)
        returns: (batch, seq_len, hidden_size)
        """
        # SwiGLU-like: gate = w1(x) * w3(x), then project
        gate = self.act(self.w1(x)) * self.w3(x)
        out = self.w2(gate)
        return self.dropout(out)