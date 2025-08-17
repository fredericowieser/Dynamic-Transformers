# src/models/decision_qwen_decoder_layer.py

import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)
from src.models.d_qwen_fnn import PriorFeedForward

import logging
log = logging.getLogger(__name__)

class DecisionQwenDecoderLayer(nn.Module):
    """
    Implements the 'Decision Sub-Layer' for the Dynamic Qwen architecture.
    This layer wraps the original Qwen2 attention and MLP, and adds a PriorFeedForward
    network for predictive purposes.

    It computes the standard transformer block output and a 'prior' prediction,
    passing both (along with the original input) to the subsequent Dynamic layer.
    """
    def __init__(self, config, layer_idx: int, load_from_pretrained: bool = False, original_layer_state_dict: dict = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Original Qwen2DecoderLayer components
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config, layer_idx) # Layer norm is part of Qwen2Attention already.
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen2MLP(config) # Qwen2MLP already contains its SiLU and Linear layers

        # New Prior FFN components
        # Using prior_ffn_intermediate_size_factor from config
        prior_ffn_factor = getattr(config, "prior_ffn_intermediate_size_factor", 2.0)
        self.prior_ffn = PriorFeedForward(config, intermediate_size_factor=prior_ffn_factor)
        self.prior_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Ensure rotary_emb is initialized in attention module if missing
        if not hasattr(self.self_attn, "rotary_emb") or self.self_attn.rotary_emb is None:
            log.warning(
                f"Layer {self.layer_idx}: Qwen2Attention unexpectedly missing or having None rotary_emb. Initializing it manually as a fallback."
            )
            self.self_attn.rotary_emb = Qwen2RotaryEmbedding(self.config, base=config.rope_theta)

        if load_from_pretrained and original_layer_state_dict is not None:
            # Filter and load state_dict for original components
            # This requires careful mapping if parameter names differ significantly
            # within the wrapped Qwen2Attention/Qwen2MLP.
            # Assuming direct loading for now, `strict=False` will ignore new `prior_ffn` params.
            self.load_state_dict(original_layer_state_dict, strict=False)
            log.info(f"Loaded pre-trained weights for DecisionQwenDecoderLayer {self.layer_idx}.")


    def forward(
        self,
        hidden_states: torch.Tensor, # Input from previous Dynamic Layer (or initial embeddings)
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs, # Accept extra kwargs for compatibility
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...] | None, torch.Tensor | None]:
        """
        Forward pass for the DecisionQwenDecoderLayer.

        Args:
            hidden_states (torch.Tensor): Input sequence embeddings. (B, S, D)
            attention_mask (torch.Tensor | None): Attention mask. (B, 1, S, S)
            position_ids (torch.LongTensor | None): Position IDs. (B, S)
            past_key_values (tuple[torch.Tensor] | None): Cached key/value states.
            output_attentions (bool): Whether to return attentions.
            use_cache (bool): Whether to return cached key/value states.

        Returns:
            tuple:
                - original_input_to_block (torch.Tensor): The original `hidden_states` passed into this layer. (B, S, D)
                - posterior_full_path_output (torch.Tensor): Output after Attention + MLP from this layer. (B, S, D)
                - prior_hidden_states (torch.Tensor): Output of the Prior FFN. (B, S, D)
                - present_key_value (tuple[torch.Tensor] | None): Updated key/value cache for attention.
                - attn_weights (torch.Tensor | None): Attention weights if `output_attentions` is True.
        """
        original_input_to_block = hidden_states # Z^{n-1}

        # --- Standard Qwen2 Decoder Layer Logic (Attention + MLP) ---
        # 1. Input LayerNorm + Self-Attention
        hidden_states_pre_attn_ln = self.input_layernorm(hidden_states)

        # Prepare input for rotary embedding calculation
        batch_size, seq_len, hidden_size_ = hidden_states_pre_attn_ln.shape
        num_attention_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_attention_heads

        input_for_rope = hidden_states_pre_attn_ln.view(
            batch_size, seq_len, num_attention_heads, head_dim
        ).transpose(1, 2) # (B, N_H, T, H_D)

        cos, sin = self.self_attn.rotary_emb(input_for_rope, position_ids)
        position_embeddings = (cos, sin)

        attn_outputs = self.self_attn(
            hidden_states_pre_attn_ln,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            layer_idx=self.layer_idx, # Critical for Qwen2's KV caching in Attention
            position_embeddings=position_embeddings, # Explicitly pass pre-calculated embeddings
        )
        attention_output = attn_outputs[0] # (B, S, D)
        present_key_value = attn_outputs[2] if use_cache else None
        attn_weights = attn_outputs[1] if output_attentions else None

        hidden_states_after_attn = original_input_to_block + attention_output # Residual connection

        # 2. Post-Attention LayerNorm + MLP
        residual_mlp = hidden_states_after_attn
        hidden_states_pre_mlp_ln = self.post_attention_layernorm(hidden_states_after_attn)
        mlp_output = self.mlp(hidden_states_pre_mlp_ln) # (B, S, D)
        posterior_full_path_output = residual_mlp + mlp_output # (B, S, D)

        # --- Prior FFN Logic ---
        # H^{D_n}_{\text{prior}} = LN(\text{PriorFFN}(Z^{n-1})) + Z^{n-1}
        prior_input_ln = self.prior_layernorm(original_input_to_block)
        prior_ffn_output = self.prior_ffn(prior_input_ln) # (B, S, D)
        prior_hidden_states = original_input_to_block + prior_ffn_output # Residual connection

        return (
            original_input_to_block, # Z^{n-1} (input)
            posterior_full_path_output, # H^{D_n}_{trans} (standard transformer output)
            prior_hidden_states, # H^{D_n}_{prior} (prior prediction)
            present_key_value,
            attn_weights,
        )