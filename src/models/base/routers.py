import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


class BaseRouter(nn.Module, ABC):
    """Abstract base class for all routing modules."""

    def __init__(self, config, capacity_attr: str):
        super().__init__()
        self.config = config
        self.capacity = getattr(config, capacity_attr, 0.5)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        raise NotImplementedError

    def select_tokens(self, scores: torch.Tensor, hidden_states: torch.Tensor):
        B, T, D = hidden_states.shape
        k = max(1, int(T * self.capacity))

        if k > T:
            k = T

        topk_vals, topk_idx = scores.topk(k, dim=-1)
        batch_idx = torch.arange(B, device=scores.device).unsqueeze(1).expand(-1, k)

        selected_hidden = hidden_states[batch_idx, topk_idx]

        return (
            selected_hidden.reshape(-1, D),
            batch_idx.reshape(-1),
            topk_idx.reshape(-1),
            topk_vals.reshape(-1),
        )


class UnifiedCausalRouter(nn.Module):
    """
    Auxiliary causal predictor (x_t only) based on the original MoD architecture.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Architecture strictly matching MoD auxiliary predictor references
        self.router = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_size // 2, 1, bias=False)
        )

    def forward(self, hidden_states: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: [B, T, D]. MUST BE DETACHED during training to prevent 
                           gradients from flowing into the main LM.
            targets: [B, T] binary mask (1.0 for route, 0.0 for skip).
        """
        # Strictly x_t only
        logits = self.router(hidden_states).squeeze(-1)
        
        loss = None
        accuracy = None
        
        if targets is not None:
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            with torch.no_grad():
                preds = (logits > 0.0).float()
                accuracy = (preds == targets).float().mean()
                
        return logits, loss, accuracy


class BaseSurpriseRouter(BaseRouter):
    """Abstracts the common surprise-based routing logic for SDT and STT."""

    def __init__(self, config, capacity_attr: str):
        super().__init__(config, capacity_attr)

        o_ce_init_val = torch.tensor(float(getattr(config, "o_ce_init", 1.0)))
        if getattr(config, "learn_o_ce", False):
            self.raw_o_ce = nn.Parameter(o_ce_init_val)
        else:
            self.register_buffer("raw_o_ce", o_ce_init_val)

        m_cu_init_val = torch.tensor(float(getattr(config, "m_cu_init", 1.1)))
        if getattr(config, "learn_m_cu", False):
            self.raw_m_cu = nn.Parameter(m_cu_init_val)
        else:
            self.register_buffer("raw_m_cu", m_cu_init_val)

        self.ma_window = int(getattr(config, "ma_window", 100))

    @staticmethod
    def compute_kl_divergence(mu_p: torch.Tensor, mu_q: torch.Tensor, log_var_q: torch.Tensor, c: float) -> torch.Tensor:
        """
        Computes the normalized KL Divergence D_KL(p||q) / d.
        Uses the log-variance trick for numerical stability (exp(-log_var) replaces division by variance).
        """
        # Precision-weighted squared error + posterior variance
        precision_weighted_term = ((mu_p - mu_q).pow(2) + c) * torch.exp(-log_var_q)
        
        # Sum over dimension d, but we use mean to divide by d implicitly, 
        # keeping the scale of D_ch identical to the old MSE formulation.
        # D_KL/d = 0.5 * mean(log_var_q + precision_weighted_term)
        kl_div_normalized = 0.5 * torch.mean(log_var_q + precision_weighted_term, dim=-1)
        
        return kl_div_normalized

    def _moving_average(self, d_st: torch.Tensor) -> torch.Tensor:
        B, T = d_st.shape
        W = min(self.ma_window, T)
        if W <= 1:
            return d_st
        padded = F.pad(d_st.unsqueeze(1), (W - 1, 0), "replicate")
        return F.avg_pool1d(padded, kernel_size=W, stride=1).squeeze(1)

    def _get_vpr_signals(self, D_st, D_ch, beta_ce, beta_cu):
        o_ce = self.raw_o_ce
        m_cu = self.raw_m_cu

        CE = D_st - (D_ch - torch.log(o_ce + 1e-10))
        CU = D_st - (m_cu * self._moving_average(D_st.detach()))

        S_CE = torch.sigmoid(beta_ce * CE)
        S_CU = torch.sigmoid(beta_cu * CU)

        # g_cont = S_CE + S_CU - (S_CE * S_CU)
        g_cont = S_CE
        return g_cont, {
            "S_CE_mean": S_CE.mean().item(),
            "S_CU_mean": S_CU.mean().item(),
            "g_cont_mean": g_cont.mean().item(),
            "o_ce": o_ce.item(),
            "m_cu": m_cu.item(),
        }
