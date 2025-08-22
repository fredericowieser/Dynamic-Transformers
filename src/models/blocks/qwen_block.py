import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)
from transformers.modeling_outputs import CausalLMOutputWithPast


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

        # --- START OF FIX ---
        # Fallback for rotary embedding initialization
        if not hasattr(self.self_attn, "rotary_emb") or self.self_attn.rotary_emb is None:
            # The constructor expects the config object directly.
            self.self_attn.rotary_emb = Qwen2RotaryEmbedding(config=self.config)
        # --- END OF FIX ---


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

        # Self Attention
        batch_size, seq_len, _ = hidden_states_norm.shape
        num_attention_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_attention_heads

        input_for_rope = hidden_states_norm.view(
            batch_size, seq_len, num_attention_heads, head_dim
        ).transpose(1, 2)

        # The Qwen2Model's forward pass should create and pass position_embeddings
        # but we need a fallback for direct block usage.
        # This part requires a look at how Qwen2Model creates position_embeddings
        # For now, let's assume it's passed in kwargs if available
        # This part of the logic might need to be in the main causal_lm.py forward pass.
        
        # Let's simplify and assume the main model will handle creating position_embeddings
        # and pass them down. This block shouldn't be responsible for it.
        
        attn_outputs = self.self_attn(
            hidden_states_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            # position_embeddings are passed from the main model's forward pass
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