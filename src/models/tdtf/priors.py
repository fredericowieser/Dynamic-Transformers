import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP


class TDTFTransitionNetwork(nn.Module):
    """Transition Network (TPN) for predicting residual updates.

    Implements the change hypothesis by predicting the residual update
    of the current token using the final output state of the previous token.
    """

    def __init__(self, config):
        super().__init__()
        # Create a new config for the MLP with a reduced intermediate size
        mlp_config = config
        mlp_config.intermediate_size = int(config.hidden_size * getattr(config, 'prior_ffn_intermediate_size_factor'))

        self.mlp = Qwen2MLP(mlp_config)

    def forward(self, x: nn.Module) -> nn.Module:
        """Predict residual update from previous token's output state."""
        return self.mlp(x)
