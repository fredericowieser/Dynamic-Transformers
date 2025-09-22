import torch.nn as nn
from transformers import Qwen2ForCausalLM, Qwen2Config


class StandardTransformerForCausalLM(Qwen2ForCausalLM):
    """Standard transformer model using Qwen2 architecture.

    This is essentially a wrapper around Qwen2ForCausalLM that ensures
    we use the exact same architecture (RMSNorm, SwiGLU MLP, RoPE, etc.)
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self._init_weights_if_needed(config)

    def _init_weights_if_needed(self, config):
        """Initialize weights using Qwen's approach if training from scratch."""
        if hasattr(config, 'init_from_scratch') and config.init_from_scratch:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights matching Qwen's initialization strategy."""
        std = getattr(self.config, 'initializer_range')

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif hasattr(module, 'weight') and hasattr(module.weight, 'data'):
            # For RMSNorm layers
            module.weight.data.fill_(1.0)

    @classmethod
    def from_pretrained_or_random(cls, model_name_or_config, from_scratch=False, **kwargs):
        """Load pretrained or initialize from scratch."""
        if from_scratch:
            if isinstance(model_name_or_config, str):
                # Load config from pretrained but initialize weights randomly
                config = Qwen2Config.from_pretrained(model_name_or_config)
            else:
                config = model_name_or_config

            config.init_from_scratch = True

            # Override any size parameters if provided
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            model = cls(config)
            return model
        else:
            # Load pretrained weights
            return cls.from_pretrained(model_name_or_config, **kwargs)

    def get_trainable_parameters(self):
        """Get parameter groups for training."""
        # All parameters are trainable in standard transformer
        return [
            {
                'params': self.parameters(),
                'lr_scale': 1.0,
                'name': 'all'
            }
        ]
