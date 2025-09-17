import torch
import torch.nn as nn


class MoDRouter(nn.Module):
    """MoD importance-based token routing."""

    def __init__(self, config):
        super().__init__()
        self.capacity = config.capacity_gamma
        self.gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        """Compute importance scores and selection mask."""
        scores = self.gate(hidden_states).squeeze(-1)
        batch_size, seq_len = scores.shape
        k = int(self.capacity * seq_len)

        if k >= seq_len:
            mask = torch.ones_like(scores)
        else:
            _, top_indices = torch.topk(scores, k, dim=-1)
            mask = torch.zeros_like(scores)
            mask.scatter_(1, top_indices, 1)

        return mask, scores