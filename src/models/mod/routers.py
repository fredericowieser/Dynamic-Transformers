import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..base.router import BaseRouter


class MoDRouter(BaseRouter):
    """Router for MoD token selection based on learned importance scores."""

    def __init__(self, config, layer_idx: int):
        capacity = getattr(config, 'mod_capacity')
        super().__init__(capacity)

        self.layer_idx = layer_idx

        # Simple linear router as per MoD paper: r_i = w^T * x_i
        self.router = nn.Linear(config.hidden_size, 1, bias=False)

        # Auxiliary loss weight for load balancing
        self.aux_loss_weight = getattr(config, 'mod_aux_loss_weight')

    def compute_routing_scores(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """Compute importance scores for each token.

        Args:
            hidden_states: [B, T, D]

        Returns:
            scores: Router scores [B, T]
            aux_loss: Load balancing loss
            stats: Routing statistics
        """
        B, T, D = hidden_states.shape

        # Compute routing scores: r_i = w^T * x_i
        router_logits = self.router(hidden_states).squeeze(-1)  # [B, T]

        # Compute auxiliary loss for load balancing during training
        aux_loss = None
        if self.training:
            # Encourage uniform routing across tokens
            k = max(1, int(T * self.capacity))
            target_load = k / T  # Expected fraction of tokens selected

            # Actual load (using sigmoid as soft selection)
            actual_load = torch.sigmoid(router_logits).mean()

            # MSE loss between actual and target load
            aux_loss = self.aux_loss_weight * ((actual_load - target_load) ** 2)

        stats = {
            'layer_idx': self.layer_idx,
            'capacity': self.capacity,
            'mean_score': router_logits.mean().item(),
            'std_score': router_logits.std().item(),
        }

        return router_logits, aux_loss, stats


class CausalMoDRouter(BaseRouter):
    """Causal router for MoD token selection.
    This router is used during inference to make decisions based on past tokens.
    """

    def __init__(self, config, layer_idx: int):
        capacity = getattr(config, 'mod_capacity')
        super().__init__(capacity)
        self.layer_idx = layer_idx
        self.router = nn.Linear(2 * config.hidden_size, 1, bias=False)

    def compute_routing_scores(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """Computes causal routing scores for MoD.

        Args:
            hidden_states: Current token states [B, T, D]

        Returns:
            scores: Routing logits [B, T]
            aux_loss: None
            stats: Routing statistics
        """
        B, T, D = hidden_states.shape

        # Prepare causal input: [x_t^(l-1) || x_{t-1}^(l-1)]
        # For t=0, x_{-1}^(l-1) should be zero vector
        prev_states = torch.cat([
            torch.zeros(B, 1, D, device=hidden_states.device, dtype=hidden_states.dtype),
            hidden_states[:, :-1, :]  # Shift right by 1
        ], dim=1)  # [B, T, D]

        # Concatenate current and previous states
        causal_input = torch.cat([hidden_states, prev_states], dim=-1)  # [B, T, 2*D]

        router_logits = self.router(causal_input).squeeze(-1)

        stats = {
            'layer_idx': self.layer_idx,
            'capacity': self.capacity,
            'mean_score': router_logits.mean().item(),
            'std_score': router_logits.std().item(),
        }

        return router_logits, None, stats