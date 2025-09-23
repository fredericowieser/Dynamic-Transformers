import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import copy
from typing import Dict
from omegaconf import DictConfig, OmegaConf
import logging

log = logging.getLogger(__name__)

# ... (rest of the imports)

class BaseRouter(nn.Module, ABC):
    """Abstract base class for all routing modules."""
    def __init__(self, config, capacity_attr: str, model_cfg: Dict = None):
        super().__init__()
        self.config = config
        self.capacity = model_cfg.get(capacity_attr, 0.5)

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

        return selected_hidden.reshape(-1, D), \
               batch_idx.reshape(-1), topk_idx.reshape(-1), topk_vals.reshape(-1)

class CausalRouter(BaseRouter):
    """Unified CausalRouter for MoD, SDT, and STT inference."""
    def __init__(self, config, layer_idx: int, capacity_attr: str, model_cfg: Dict = None):
        super().__init__(config, capacity_attr, model_cfg=model_cfg)
        self.router = nn.Linear(2 * config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        B, T, D = hidden_states.shape
        prev = torch.cat([torch.zeros(B, 1, D, device=hidden_states.device, dtype=hidden_states.dtype), hidden_states[:, :-1, :]], dim=1)
        logits = self.router(torch.cat([hidden_states, prev], dim=-1)).squeeze(-1)
        return logits, None, {}

class BaseSurpriseRouter(BaseRouter):
    """Abstracts the common surprise-based routing logic for SDT and STT."""
    def __init__(self, config, capacity_attr: str, model_cfg: Dict = None):
        super().__init__(config, capacity_attr, model_cfg=model_cfg)
        
        o_ce_init_val = torch.tensor(float(model_cfg.get('o_ce_init', 1.0)))
        if model_cfg.get('learn_o_ce', False):
            self.raw_o_ce = nn.Parameter(o_ce_init_val)
        else:
            self.register_buffer('raw_o_ce', o_ce_init_val)

        m_cu_init_val = torch.tensor(float(model_cfg.get('m_cu_init', 1.1)))
        if model_cfg.get('learn_m_cu', False):
            self.raw_m_cu = nn.Parameter(m_cu_init_val)
        else:
            self.register_buffer('raw_m_cu', m_cu_init_val)

        self.ma_window = int(model_cfg.get('ma_window', 100))
    
    def _moving_average(self, d_st: torch.Tensor) -> torch.Tensor:
        B, T = d_st.shape
        W = min(self.ma_window, T)
        if W <= 1: return d_st
        padded = F.pad(d_st.unsqueeze(1), (W - 1, 0), 'replicate')
        return F.avg_pool1d(padded, kernel_size=W, stride=1).squeeze(1)

    def _get_vpr_signals(self, D_st, D_ch, beta_ce, beta_cu):
        o_ce_pos = F.softplus(self.raw_o_ce)
        m_cu_pos = F.softplus(self.raw_m_cu)
        
        CE = D_st - (D_ch - torch.log(o_ce_pos + 1e-10))
        CU = D_st - (m_cu_pos * self._moving_average(D_st.detach()))
        
        S_CE = torch.sigmoid(torch.tensor(beta_ce, device=CE.device) * CE)
        S_CU = torch.sigmoid(torch.tensor(beta_cu, device=CU.device) * CU)
        
        g_cont = S_CE + S_CU - (S_CE * S_CU)
        return g_cont, {"S_CE_mean": S_CE.mean().item(), "S_CU_mean": S_CU.mean().item(), "o_ce_pos": o_ce_pos.item(), "m_cu_pos": m_cu_pos.item()}