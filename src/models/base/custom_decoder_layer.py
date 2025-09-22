import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.cache_utils import Cache
from transformers.integrations import use_kernel_forward_from_hub
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm # Import original MLP and RMSNorm

from .custom_attention import CustomQwen2Attention # Import our custom attention

class CustomQwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = CustomQwen2Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Assuming config.layer_types is available and relevant for attention_type
        self.attention_type = config.layer_types[layer_idx] if hasattr(config, 'layer_types') and len(config.layer_types) > layer_idx else "full_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None, # Passed from BaseForCausalLM
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, present_key_value, attentions = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings, # Pass position_embeddings
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        # Return hidden_states, present_key_value, attentions
        return hidden_states, present_key_value, attentions
