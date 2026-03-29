import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP


class BasePriorNetwork(nn.Module):
    """Abstracts the creation of a lightweight feed-forward network to predict mean and log-variance."""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        mlp_config = copy.deepcopy(config)
        factor = getattr(config, "prior_ffn_intermediate_size_factor", 0.25)
        raw_size = config.hidden_size * factor
        rounded_size = int(raw_size + 0.999)
        intermediate_size = max(2, rounded_size + (rounded_size % 2))
        mlp_config.intermediate_size = intermediate_size
        
        # Shared feature extractor
        self.mlp = Qwen2MLP(mlp_config)
        # Projection to 2 * D (Mean and Log-Variance)
        self.mu_logvar_proj = nn.Linear(config.hidden_size, 2 * config.hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.mlp(x)
        mu_logvar = self.mu_logvar_proj(features)
        # Split the last dimension into mu_q and log_var_q
        mu_q, log_var_q = mu_logvar.chunk(2, dim=-1)
        return mu_q, log_var_q
