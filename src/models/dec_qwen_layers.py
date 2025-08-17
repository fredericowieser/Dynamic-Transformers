import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)
from src.models.d_qwen_fnn import PriorFeedForward # Renamed from d_qwen_fnn

import logging
log = logging.getLogger(__name__)

class DecisionQwenDecoderLayer(nn.Module):
    """
    Implements the 'Decision Sub-Layer' for the Dynamic Qwen architecture.
    This layer wraps the original Qwen2 attention and MLP, and adds a PriorFeedForward
    network for predictive purposes.

    It computes the standard transformer block output and a 'prior' prediction.
    It returns its *actual output* (to be fed to the next layer) AND the signals
    needed by the VPR router in the *subsequent Dynamic Layer*.
    """
    def __init__(self, config, layer_idx: int, load_from_pretrained: bool = False, original_layer_state_dict: dict = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Original Qwen2DecoderLayer components
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen2MLP(config)

        # New Prior FFN components
        prior_ffn_factor = getattr(config, "prior_ffn_intermediate_size_factor", 2.0)
        self.prior_ffn = PriorFeedForward(config, intermediate_size_factor=prior_ffn_factor)
        self.prior_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Ensure rotary_emb is initialized in attention module if missing
        if not hasattr(self.self_attn, "rotary_emb") or self.self_attn.rotary_emb is None:
            log.warning(
                f"Layer {self.layer_idx}: Qwen2Attention unexpectedly missing or having None rotary_emb. Initializing it manually as a fallback."
            )
            self.self_attn.rotary_emb = Qwen2RotaryEmbedding(self.config)

        if load_from_pretrained and original_layer_state_dict is not None:
            self.load_state_dict(original_layer_state_dict, strict=False)
            log.info(f"Loaded pre-trained weights for DecisionQwenDecoderLayer {self.layer_idx}.")


    def forward(
        self,
        hidden_states: torch.Tensor, # This is the input to *this* Decision layer (e.g., Z^{n-1} or output from previous Dynamic)
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs, # Accept extra kwargs for compatibility
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...] | None, torch.Tensor | None]:
        """
        Forward pass for the DecisionQwenDecoderLayer.

        Args:
            hidden_states (torch.Tensor): Input sequence embeddings to THIS layer. (B, S, D)
            attention_mask (torch.Tensor | None): Attention mask. (B, 1, S, S)
            position_ids (torch.LongTensor | None): Position IDs. (B, S)
            past_key_values (tuple[torch.Tensor] | None): Cached key/value states for THIS layer's attention.
            output_attentions (bool): Whether to return attentions for THIS layer.
            use_cache (bool): Whether to return cached key/value states for THIS layer.

        Returns:
            tuple:
                - output_hidden_states (torch.Tensor): The main hidden states output of THIS layer. (B, S, D).
                                                       This will be the input to the next layer in the sequence.
                - vpr_signal_original_input (torch.Tensor): The original `hidden_states` passed into THIS layer. (B, S, D).
                                                            Used as `original_input_to_block` for next Dynamic Layer's VPRRouter.
                - vpr_signal_posterior_output (torch.Tensor): Output after Attention + MLP from THIS layer. (B, S, D).
                                                              Used as `posterior_full_path_output` for next Dynamic Layer's VPRRouter.
                - vpr_signal_prior_hidden_states (torch.Tensor): Output of the Prior FFN from THIS layer. (B, S, D).
                                                                 Used as `prior_hidden_states` for next Dynamic Layer's VPRRouter.
                - present_key_value (tuple[torch.Tensor] | None): Updated key/value cache from THIS layer's attention.
                - attn_weights (torch.Tensor | None): Attention weights from THIS layer's attention.
        """
        vpr_signal_original_input = hidden_states # This is Z^{n-1} or output from previous Dynamic

        # --- Standard Qwen2 Decoder Layer Logic (Attention + MLP) ---
        hidden_states_pre_attn_ln = self.input_layernorm(hidden_states)

        batch_size, seq_len, _ = hidden_states_pre_attn_ln.shape
        num_attention_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_attention_heads

        input_for_rope = hidden_states_pre_attn_ln.view(
            batch_size, seq_len, num_attention_heads, head_dim
        ).transpose(1, 2)

        cos, sin = self.self_attn.rotary_emb(input_for_rope, position_ids)
        position_embeddings = (cos, sin)

        attn_outputs = self.self_attn(
            hidden_states_pre_attn_ln,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            layer_idx=self.layer_idx,
            position_embeddings=position_embeddings,
        )
        attention_output = attn_outputs[0]
        present_key_value = attn_outputs[2] if use_cache else None
        attn_weights = attn_outputs[1] if output_attentions else None

        hidden_states_after_attn = vpr_signal_original_input + attention_output # Residual connection

        vpr_signal_posterior_output_residual = hidden_states_after_attn
        hidden_states_pre_mlp_ln = self.post_attention_layernorm(hidden_states_after_attn)
        mlp_output = self.mlp(hidden_states_pre_mlp_ln)
        vpr_signal_posterior_output = vpr_signal_posterior_output_residual + mlp_output # Result of *this* layer's full compute

        # This `vpr_signal_posterior_output` is also the hidden states that will be
        # passed as the `output_hidden_states` of this Decision Layer.
        output_hidden_states = vpr_signal_posterior_output


        # --- Prior FFN Logic ---
        # H^{D_n}_{\text{prior}} = LN(\text{PriorFFN}(Z^{n-1})) + Z^{n-1}
        # Use vpr_signal_original_input as the input for the prior FFN.
        prior_input_ln = self.prior_layernorm(vpr_signal_original_input)
        prior_ffn_output = self.prior_ffn(prior_input_ln)
        vpr_signal_prior_hidden_states = vpr_signal_original_input + prior_ffn_output

        return (
            output_hidden_states,
            vpr_signal_original_input,
            vpr_signal_posterior_output,
            vpr_signal_prior_hidden_states,
            present_key_value,
            attn_weights,
        )