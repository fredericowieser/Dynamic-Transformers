import torch
import torch.nn as nn
import logging
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)

log = logging.getLogger(__name__)

class Qwen2Block(nn.Module):
    """
    A standalone, reusable Qwen2 transformer block.
    This module encapsulates the standard self-attention and MLP layers,
    including residual connections and layer normalization.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = Qwen2MLP(config)
        
        if not hasattr(self.self_attn, "rotary_emb") or self.self_attn.rotary_emb is None:
            log.warning(
                f"Layer {layer_idx}: Qwen2Attention unexpectedly missing rotary_emb. "
                "Initializing it manually as a fallback."
            )
            # --- START OF MODIFICATION ---
            # Corrected the constructor call to pass the config object directly.
            self.self_attn.rotary_emb = Qwen2RotaryEmbedding(config=self.config)
            # --- END OF MODIFICATION ---

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        
        residual = hidden_states
        hidden_states_norm = self.input_layernorm(hidden_states)

        kv_seq_len = hidden_states_norm.shape[1]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        # The rotary_emb forward pass in the modular implementation expects `position_ids`
        cos, sin = self.self_attn.rotary_emb(hidden_states_norm, position_ids=position_ids)
        position_embeddings = (cos, sin)
        
        attn_outputs = self.self_attn(
            hidden_states_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        attn_output = attn_outputs[0]
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states_norm = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states_norm)
        hidden_states = residual + mlp_output

        outputs = (hidden_states,) + attn_outputs[1:]
        return outputs