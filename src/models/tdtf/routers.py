import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

from ..base.router import BaseRouter


class TDTFPredictiveRouter(nn.Module):
    """Non-causal Predictive Router for training (teacher model).

    - o_ce and m_cu are learnable (as in spec).
    - β_ce and β_cu are NOT learnable; they must be provided at forward time (scheduled in the trainer).
    - Uses per-sequence moving average for CU.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Required config values (fail fast if missing)
        if not hasattr(config, "o_ce_init"):
            raise ValueError("Missing config.o_ce_init")
        if not hasattr(config, "m_cu_init"):
            raise ValueError("Missing config.m_cu_init")
        if not hasattr(config, "tdtf_capacity"):
            raise ValueError("Missing config.tdtf_capacity")
        if not hasattr(config, "ma_window"):
            raise ValueError("Missing config.ma_window")

        # Learnable CE/CU bias terms
        self.raw_o_ce = nn.Parameter(torch.tensor(float(config.o_ce_init)))
        self.raw_m_cu = nn.Parameter(torch.tensor(float(config.m_cu_init)))

        # Capacity (γ), MA window (W)
        self.capacity = float(config.tdtf_capacity)
        self.ma_window = int(config.ma_window)

    def compute_surprise_metrics(
        self,
        actual_residual: torch.Tensor,
        predicted_residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """D_st: [B, T], D_ch: [B, T]"""
        # D_st: (1/d) * ||Δx||^2
        D_st = (actual_residual.pow(2).sum(dim=-1)) / actual_residual.shape[-1]
        # D_ch: (1/d) * ||Δx - Δx_hat||^2
        D_ch = ((actual_residual - predicted_residual).pow(2).sum(dim=-1)) / actual_residual.shape[-1]
        return D_st, D_ch

    def moving_average(self, D_st: torch.Tensor) -> torch.Tensor:
        """Per-sequence causal moving average MA(D_st) with window W.
        Returns [B, T]."""
        B, T = D_st.shape
        W = min(self.ma_window, T)
        x = D_st.unsqueeze(1)  # [B,1,T]

        # Manual causal padding: pad only on the left
        # F.avg_pool1d expects padding as (padding_left, padding_right) for the last dimension.
        # We need W-1 padding on the left.
        padded_x = F.pad(x, (W - 1, 0), 'constant', 0)

        # Apply avg_pool1d with no internal padding, as we've already padded
        ma = F.avg_pool1d(padded_x, kernel_size=W, stride=1, count_include_pad=False)
        return ma.squeeze(1)  # [B, T]

    def compute_vpr_criteria(
        self,
        D_st: torch.Tensor,
        D_ch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CE and CU (pre-temperature)."""
        # Enforce positivity on o_ce and m_cu
        o_ce_pos = F.softplus(self.raw_o_ce)
        m_cu_pos = F.softplus(self.raw_m_cu)

        # CE = D_st - (D_ch - log(o_ce))
        CE = D_st - (D_ch - torch.log(o_ce_pos + 1e-10))

        # CU = D_st - (m_cu * MA(D_st))
        MA = self.moving_average(D_st)
        CU = D_st - (m_cu_pos * MA)

        return CE, CU

    def compute_continuous_gate(
        self,
        CE: torch.Tensor,
        CU: torch.Tensor,
        beta_ce: float,
        beta_cu: float,
    ) -> Dict[str, torch.Tensor]:
        """Apply inverse temperatures (scheduled scalars) and probabilistic OR."""
        if beta_ce is None or beta_cu is None:
            raise ValueError("beta_ce and beta_cu must be provided (scheduled scalars)")

        # Apply sigmoids with provided temperatures
        S_CE = torch.sigmoid(torch.tensor(beta_ce, device=CE.device, dtype=CE.dtype) * CE)
        S_CU = torch.sigmoid(torch.tensor(beta_cu, device=CU.device, dtype=CU.dtype) * CU)

        g_cont = S_CE + S_CU - (S_CE * S_CU)

        return {"g_cont": g_cont, "S_CE": S_CE, "S_CU": S_CU}

    def forward(
        self,
        actual_residual: torch.Tensor,
        predicted_residual: torch.Tensor,
        *,
        beta_ce: float,
        beta_cu: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Returns g_continuous [B, T], binary_targets [B, T], and stats."""
        B, T, _ = actual_residual.shape

        D_st, D_ch = self.compute_surprise_metrics(actual_residual, predicted_residual)
        CE, CU = self.compute_vpr_criteria(D_st, D_ch)
        out = self.compute_continuous_gate(CE, CU, beta_ce=beta_ce, beta_cu=beta_cu)

        g_cont = out["g_cont"]

        # TopK per sequence for capacity
        k = max(1, int(T * self.capacity))
        _, topk_idx = g_cont.topk(k, dim=-1)  # [B, k]
        binary_targets = torch.zeros_like(g_cont)
        batch_idx = torch.arange(B, device=g_cont.device).unsqueeze(1)
        binary_targets[batch_idx, topk_idx] = 1.0

        stats = {
            "layer_idx": self.layer_idx,
            "S_CE_mean": out["S_CE"].mean().item(),
            "S_CU_mean": out["S_CU"].mean().item(),
            "G_cont_mean": g_cont.mean().item(),
        }

        return g_cont, binary_targets, stats


class TDTFCausalRouter(BaseRouter):
    """Causal Router for inference (student).

    student_routing_mode must be provided in config: 'topk' or 'threshold'.
    """

    def __init__(self, config, layer_idx: int):
        if not hasattr(config, "tdtf_capacity"):
            raise ValueError("Missing config.tdtf_capacity")
        if not hasattr(config, "student_routing_mode"):
            raise ValueError("Missing config.student_routing_mode")
        super().__init__(float(config.tdtf_capacity))

        self.layer_idx = layer_idx
        self.routing_mode = config.student_routing_mode  # 'topk' or 'threshold'

        hidden_size = config.hidden_size
        self.router_linear = nn.Linear(2 * hidden_size, 1, bias=True)
        nn.init.normal_(self.router_linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.router_linear.bias)

    def compute_routing_scores(self, hidden_states: torch.Tensor, **kwargs):
        B, T, D = hidden_states.shape
        prev_states = torch.cat(
            [
                torch.zeros(B, 1, D, device=hidden_states.device, dtype=hidden_states.dtype),
                hidden_states[:, :-1, :],
            ],
            dim=1,
        )
        x = torch.cat([hidden_states, prev_states], dim=-1)  # [B,T,2D]
        logits = self.router_linear(x).squeeze(-1)  # [B,T]
        probs = torch.sigmoid(logits)

        stats = {
            "layer_idx": self.layer_idx,
            "capacity": self.capacity,
            "avg_prob": probs.mean().item(),
            "max_prob": probs.max().item(),
            "min_prob": probs.min().item(),
        }
        return logits, None, stats