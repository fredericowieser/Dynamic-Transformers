import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from ..base.router import BaseRouter


class TDTFPredictiveRouter(nn.Module):
    """Non-causal Predictive Router for training (teacher model).

    Uses actual residual and predicted residual to calculate continuous gate values
    based on static and change surprise metrics with VPR event criteria.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Learnable parameters for VPR criteria (initialized as per spec)
        self.o_ce = nn.Parameter(torch.tensor(getattr(config, 'o_ce_init')))
        self.m_cu = nn.Parameter(torch.tensor(getattr(config, 'm_cu_init')))
        self.beta_ce = nn.Parameter(torch.tensor(getattr(config, 'beta_ce_init')))
        self.beta_cu = nn.Parameter(torch.tensor(getattr(config, 'beta_cu_init')))

        # Capacity for TopK selection
        self.capacity = getattr(config, 'tdtf_capacity')  # γ parameter

        # Moving average window for CU detection
        self.ma_window = getattr(config, 'ma_window')
        self.register_buffer('static_surprise_history', torch.zeros(self.ma_window))
        self.register_buffer('history_pointer', torch.tensor(0))

    def compute_surprise_metrics(self, actual_residual: torch.Tensor, predicted_residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute static and change surprise metrics.

        Args:
            actual_residual: Actual residual update [B, T, D]
            predicted_residual: TPN predicted residual [B, T, D]

        Returns:
            D_st: Static surprise (magnitude of actual update)
            D_ch: Change surprise (TPN prediction error)
        """
        B, T, D = actual_residual.shape

        # Static surprise: magnitude of the actual update
        D_st = (actual_residual.norm(dim=-1) ** 2) / D  # [B, T]

        # Change surprise: TPN's prediction error
        D_ch = ((actual_residual - predicted_residual).norm(dim=-1) ** 2) / D  # [B, T]

        return D_st, D_ch

    def update_moving_average(self, D_st: torch.Tensor):
        """Update moving average of static surprise for CU detection."""
        if not self.training:
            return

        # Flatten across batch and time
        D_st_flat = D_st.flatten()

        for val in D_st_flat:
            idx = self.history_pointer % self.ma_window
            self.static_surprise_history[idx] = val.item()
            self.history_pointer += 1

    def get_moving_average(self, D_st: torch.Tensor) -> torch.Tensor:
        """Get moving average for CU detection."""
        if self.history_pointer < self.ma_window:
            # Not enough history, use current mean
            return D_st.mean()
        else:
            return self.static_surprise_history.mean()

    def compute_vpr_criteria(self, D_st: torch.Tensor, D_ch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute VPR event criteria.

        Args:
            D_st: Static surprise [B, T]
            D_ch: Change surprise [B, T]

        Returns:
            CE: Expected change criteria [B, T]
            CU: Unexpected change criteria [B, T]
        """
        # Expected Event (CE): D_st - (D_ch - log(o_ce + eps))
        CE = D_st - (D_ch - torch.log(self.o_ce + 1e-10))

        # Unexpected Event (CU): D_st - (m_cu * MA(D_st))
        ma_d_st = self.get_moving_average(D_st)
        CU = D_st - (self.m_cu * ma_d_st)

        return CE, CU

    def compute_continuous_gate(self, CE: torch.Tensor, CU: torch.Tensor) -> torch.Tensor:
        """Convert criteria to continuous gate values using sigmoid and probabilistic OR.

        Args:
            CE: Expected change criteria [B, T]
            CU: Unexpected change criteria [B, T]

        Returns:
            Continuous gate values g_t^(l) ∈ [0,1]
        """
        # Convert to probabilities using learnable inverse temperatures
        beta_ce_pos = F.softplus(self.beta_ce)
        beta_cu_pos = F.softplus(self.beta_cu)

        S_CE = torch.sigmoid(beta_ce_pos * CE)
        S_CU = torch.sigmoid(beta_cu_pos * CU)

        # Probabilistic OR: P(A or B) = P(A) + P(B) - P(A)P(B)
        g_continuous = S_CE + S_CU - (S_CE * S_CU)

        return g_continuous

    def forward(self, actual_residual: torch.Tensor, predicted_residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute continuous gate values and binary targets.

        Args:
            actual_residual: Actual residual from TF block [B, T, D]
            predicted_residual: TPN predicted residual [B, T, D]

        Returns:
            g_continuous: Continuous gate values [B, T]
            binary_targets: TopK binary mask for causal router training [B, T]
        """
        B, T, D = actual_residual.shape

        # Compute surprise metrics
        D_st, D_ch = self.compute_surprise_metrics(actual_residual, predicted_residual)

        # Update moving average (training only)
        if self.training:
            self.update_moving_average(D_st)

        # Compute VPR criteria
        CE, CU = self.compute_vpr_criteria(D_st, D_ch)

        # Compute continuous gate values
        g_continuous = self.compute_continuous_gate(CE, CU)

        # Generate binary targets by TopK selection
        k = max(1, int(T * self.capacity))
        _, topk_idx = g_continuous.topk(k, dim=-1)  # [B, k]

        binary_targets = torch.zeros_like(g_continuous)  # [B, T]
        batch_idx = torch.arange(B, device=g_continuous.device).unsqueeze(1)  # [B, 1]
        binary_targets[batch_idx, topk_idx] = 1.0

        return g_continuous, binary_targets


class TDTFCausalRouter(BaseRouter):
    """Causal Router for inference (student model).

    Simple linear layer that makes causal routing decisions using only
    pre-computation states from current and previous tokens.
    """

    def __init__(self, config, layer_idx: int):
        capacity = getattr(config, 'tdtf_capacity')
        super().__init__(capacity)

        self.layer_idx = layer_idx
        hidden_size = config.hidden_size

        # Linear layer for causal prediction: input is [x_t^(l-1) || x_{t-1}^(l-1)]
        self.router_linear = nn.Linear(2 * hidden_size, 1, bias=True)

        # Initialize to reasonable values
        nn.init.normal_(self.router_linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.router_linear.bias)

    def compute_routing_scores(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Compute causal routing scores.

        Args:
            hidden_states: Current token states [B, T, D]

        Returns:
            scores: Routing logits [B, T]
            aux_loss: None
            stats: Routing statistics
        """
        B, T, D = hidden_states.shape

        # Prepare causal input: [x_t^(l-1) || x_{t-1}^(l-1)]
        # For t=1, x_0^(l-1) should be zero vector
        prev_states = torch.cat([
            torch.zeros(B, 1, D, device=hidden_states.device, dtype=hidden_states.dtype),
            hidden_states[:, :-1, :]  # Shift right by 1
        ], dim=1)  # [B, T, D]

        # Concatenate current and previous states
        causal_input = torch.cat([hidden_states, prev_states], dim=-1)  # [B, T, 2*D]

        # Compute routing logits
        logits = self.router_linear(causal_input).squeeze(-1)  # [B, T]

        # Convert to probabilities for statistics
        probs = torch.sigmoid(logits)

        stats = {
            'layer_idx': self.layer_idx,
            'capacity': self.capacity,
            'avg_prob': probs.mean().item(),
            'max_prob': probs.max().item(),
            'min_prob': probs.min().item(),
        }

        return logits, None, stats
