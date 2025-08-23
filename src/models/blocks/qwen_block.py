import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2RMSNorm,
)


class Qwen2Block(nn.Module):
    """
    A standalone, reusable Qwen2 transformer block.
    This module encapsulates the standard self-attention and MLP layers,
    including residual connections and layer normalization.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = Qwen2MLP(config)
        self.config = config # Store config for easy access

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        Forward pass for a standard Qwen2 transformer block.
        """
        residual = hidden_states
        hidden_states_norm = self.input_layernorm(hidden_states)

        # --- START OF MODIFICATION: Create Rotary Embeddings ---
        # This logic is critical for Qwen2Attention and was missing.
        if not hasattr(self.self_attn, "rotary_emb"):
             raise AttributeError("The Qwen2Attention layer is missing the `rotary_emb` attribute.")

        # Determine the sequence length to use for rotary embeddings
        # This handles cases where we process a subset of tokens (like in MoD/Dynamic layers)
        # but need the full context length for position calculations.
        kv_seq_len = hidden_states_norm.shape[1]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        # The rotary embedding module might need to be re-initialized if the sequence length exceeds its cache
        cos, sin = self.self_attn.rotary_emb(hidden_states_norm, seq_len=kv_seq_len)
        position_embeddings = (cos, sin)
        # --- END OF MODIFICATION ---

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings, # Pass the created embeddings
        )
        attn_output = attn_outputs[0]
        hidden_states = residual + attn_output

        # Fully Connected
        residual = hidden_states
        hidden_states_norm = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states_norm)
        hidden_states = residual + mlp_output

        outputs = (hidden_states,) + attn_outputs[1:]
        return outputs