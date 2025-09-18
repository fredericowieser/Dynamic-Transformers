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
        # D_st,i = (1/d) ||H_post,i - H_orig,i||_2^2
        # D_ch,i = (1/d) ||H_post,i - H_prior,i||_2^2
        # Note: The spec uses 'd' for hidden dimension, which is config.hidden_size
        d = float(original.shape[-1])
        D_st = torch.sum((posterior - original).pow(2), dim=-1) / d  # [B, T]
        D_ch = torch.sum((posterior - prior).pow(2), dim=-1) / d  # [B, T]

        # Compute Moving Average of D_st for CU criterion
        # The spec implies a temporal MA. For a single forward pass, we approximate
        # this by taking the MA over the current sequence.
        # This is a simplification and might need refinement for true temporal MA across batches.
        ma_D_st = torch.mean(D_st, dim=-1, keepdim=True) # [B, 1]

        # Ensure beta parameters are positive using Softplus
        beta_ce_positive = torch.nn.functional.softplus(self.beta_ce)
        beta_cu_positive = torch.nn.functional.softplus(self.beta_cu)

        # Compute routing criteria with learnable parameters
        # CE_i = D_st,i - (D_ch,i - log(o_ce + epsilon))
        # CU_i = D_st,i - (m_cu * MA(D_st,i))
        epsilon = 1e-10 # As per DTF-Report.md
        CE_i = D_st - (D_ch - torch.log(self.ce_criterion_offset + epsilon)) # [B, T]
        CU_i = D_st - (self.cu_detection_multiplier * ma_D_st) # [B, T]

        # Convert raw criteria into probabilities using scaled sigmoid functions
        S_CE = torch.sigmoid(beta_ce_positive * CE_i) # [B, T]
        S_CU = torch.sigmoid(beta_cu_positive * CU_i) # [B, T]

        # Combine probabilities using a probabilistic OR to form a final continuous gating signal G_cont
        # G_cont = S_CE + S_CU - (S_CE * S_CU)
        G_cont = S_CE + S_CU - (S_CE * S_CU) # [B, T]

        stats = {
            'layer_idx': self.layer_idx,
            'capacity': self.capacity,
            'D_st_mean': D_st.mean().item(),
            'D_ch_mean': D_ch.mean().item(),
            'CE_i_mean': CE_i.mean().item(),
            'CU_i_mean': CU_i.mean().item(),
            'S_CE_mean': S_CE.mean().item(),
            'S_CU_mean': S_CU.mean().item(),
            'G_cont_mean': G_cont.mean().item(),
            'beta_ce': self.beta_ce.item(),
            'beta_cu': self.beta_cu.item(),
            'o_ce': self.ce_criterion_offset.item(),
            'm_cu': self.cu_detection_multiplier.item(),
        }

        # The scores returned here are G_cont, which will be used for TopK selection in DTFDynamicLayer
        return G_cont, None, stats


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