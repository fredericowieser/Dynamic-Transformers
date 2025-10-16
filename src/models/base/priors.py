import copy
from typing import Dict

import torch.nn as nn
from omegaconf import DictConfig
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP


class BasePriorNetwork(nn.Module):
    """Abstracts the creation of a lightweight feed-forward network."""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        mlp_config = copy.deepcopy(config)
        factor = getattr(config, "prior_ffn_intermediate_size_factor", 0.25)
        raw_size = config.hidden_size * factor
        rounded_size = int(raw_size + 0.999)
        intermediate_size = max(2, rounded_size + (rounded_size % 2))
        mlp_config.intermediate_size = intermediate_size
        self.mlp = Qwen2MLP(mlp_config)

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward pass.") or (
            "Subclasses must implement the forward pass."
        )
