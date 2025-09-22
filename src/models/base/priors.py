import copy
import torch.nn as nn
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from typing import Dict
from omegaconf import DictConfig

class BasePriorNetwork(nn.Module):
    """Abstracts the creation of a lightweight feed-forward network."""
    def __init__(self, config: Qwen2Config, model_cfg: Dict = None):
        super().__init__()
        self.config = config
        mlp_config = copy.deepcopy(config)
        factor = config.prior_ffn_intermediate_size_factor
        raw_size = config.hidden_size * factor
        rounded_size = int(raw_size + 0.999)
        intermediate_size = max(2, rounded_size + (rounded_size % 2))
        mlp_config.intermediate_size = intermediate_size
        self.mlp = Qwen2MLP(mlp_config)

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward pass.")or("Subclasses must implement the forward pass.")