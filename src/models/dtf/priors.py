from omegaconf import OmegaConf
import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm
from transformers import Qwen2Config


class DTFPriorNetwork(nn.Module):
    """Lightweight network for computing prior predictions.

    Implements the change hypothesis by predicting the posterior state
    using minimal computation (SwiGLU with reduced intermediate dim).
    """

    def __init__(self, config):
        super().__init__()
        # Create a new config for the MLP with a reduced intermediate size
        mlp_config = Qwen2Config.from_dict(OmegaConf.to_container(config, resolve=True))
        mlp_config.intermediate_size = int(config.hidden_size * getattr(config, 'prior_ffn_intermediate_size_factor'))

        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen2MLP(mlp_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply lightweight transformation to predict posterior state."""
        # Residual connection with normalized FFN
        return x + self.mlp(self.norm(x))
