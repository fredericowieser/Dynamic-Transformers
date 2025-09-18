import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from ..base.router import BaseRouter


class DTFRouter(BaseRouter):
    """Predictive router for DTF token selection based on surprise metrics.

    Implements soft VPR criteria using expected (CE) and unexpected (CU) change
    to determine which tokens need computational updates.
    """

    def __init__(self, config, layer_idx: int):
        # DTF uses 12.5% capacity by default
        capacity = getattr(config, 'dtf_capacity')
        super().__init__(capacity)

        self.layer_idx = layer_idx

        # Learnable parameters for routing criteria
        self.beta_ce = nn.Parameter(torch.tensor(getattr(config, 'beta_ce_init')))
        self.beta_cu = nn.Parameter(torch.tensor(getattr(config, 'beta_cu_init')))
        self.cu_detection_multiplier = nn.Parameter(
            torch.tensor(getattr(config, 'cu_detection_multiplier_init'))
        )
        self.ce_criterion_offset = nn.Parameter(
            torch.tensor(getattr(config, 'ce_criterion_offset_init'))
        )

    def _get_capacity(self, config) -> float:
        """DTF uses γ (gamma) for capacity, typically 50%."""
        return getattr(config, 'capacity_gamma')

    def compute_routing_scores(
        self,
        hidden_states: torch.Tensor,
        original: torch.Tensor,
        posterior: torch.Tensor,
        prior: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Compute surprise-based routing scores.

        Args:
            hidden_states: Current hidden states (not used, for API consistency)
            original: Original input state (static hypothesis)
            posterior: Output from full transformer block (ground truth)
            prior: Prediction from prior network (change hypothesis)

        Returns:
            scores: Combined CE + CU routing scores
            aux_loss: None (prior loss computed separately)
            stats: Routing statistics
        """
        B, T, D = original.shape

        # Compute surprise metrics (MSE as proxy for KL divergence)
        # Static surprise: error of assuming no change
        cu = (original - posterior).norm(dim=-1)  # [B, T]
        # Change surprise: error of prior prediction
        ce = (posterior - prior).norm(dim=-1)  # [B, T]

        # Compute routing criteria with learnable parameters
        # CE criterion: Is prior prediction better than static?
        cu_criterion = self.beta_cu * cu
        ce_criterion = self.beta_ce * (ce + self.ce_criterion_offset)

        # Combined routing score
        scores = cu_criterion + ce_criterion  # [B, T]

        # Compute gating signal for soft selection
        gate_signal = torch.sigmoid(scores)

        stats = {
            'layer_idx': self.layer_idx,
            'capacity': self.capacity,
            'avg_cu': cu.mean().item(),
            'avg_ce': ce.mean().item(),
            'avg_gate': gate_signal.mean().item(),
        }

            # Compute prior loss
            prior_loss = self.prior_loss_weight * prior_dist.entropy().mean()

            # Compute causal loss
            # TODO: Implement actual causal loss for DTF based on specification.
            # For now, using a placeholder.
            causal_loss = self.causal_loss_weight * torch.tensor(0.0, device=hidden_states.device)
        return scores, None, stats


class CausalDTFRouter(BaseRouter):
    """Causal router for DTF token selection.
    This router is used during inference to make decisions based on past tokens.
    """

    def __init__(self, config, layer_idx: int):
        capacity = getattr(config, 'dtf_capacity')
        super().__init__(capacity)
        self.layer_idx = layer_idx
        self.router = nn.Linear(2 * config.hidden_size, 1, bias=False)

    def compute_routing_scores(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """Computes causal routing scores for DTF.

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