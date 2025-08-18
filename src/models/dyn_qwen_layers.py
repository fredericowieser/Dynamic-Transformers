import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)
from src.models.vpr_router import VPRRouter # Import the new VPRRouter

import logging
log = logging.getLogger(__name__)

class DynamicQwenDecoderLayer(nn.Module):
    """
    Implements the 'Dynamic Sub-Layer' for the Dynamic Qwen architecture.
    This layer contains its OWN Qwen2 Attention and MLP blocks.
    It uses a VPRRouter, informed by outputs from the *preceding Decision Layer*,
    to decide which tokens undergo full computation (Attention + MLP) within *this* layer,
    and which tokens bypass for computational savings (identity connection).
    The routing decision is differentiable via a Straight-Through Estimator (STE)
    and applies the continuous gating signal to scale the change introduced by the computation.
    """
    def __init__(self, config, layer_idx: int, load_from_pretrained: bool = False, original_layer_state_dict: dict = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Core Transformer components for this layer (these are OWN to Dynamic layer)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen2MLP(config)

        # VPR Router specific to this layer
        self.vpr_router = VPRRouter(config, layer_idx)

        # Ensure rotary_emb is initialized in attention module if missing
        if not hasattr(self.self_attn, "rotary_emb") or self.self_attn.rotary_emb is None:
            log.warning(
                f"Layer {self.layer_idx}: Qwen2Attention unexpectedly missing or having None rotary_emb. Initializing it manually as a fallback."
            )
            self.self_attn.rotary_emb = Qwen2RotaryEmbedding(self.config)

        # Load pre-trained weights for this layer's components
        if load_from_pretrained and original_layer_state_dict is not None:
            # Load the state_dict for self_attn, mlp, and layernorms of this layer
            # Using strict=False because prior_ffn and vpr_router won't be in original_layer_state_dict
            # for Decision/Dynamic layers, so it's correct here.
            self.load_state_dict(original_layer_state_dict, strict=False)
            log.info(f"Loaded pre-trained weights for DynamicQwenDecoderLayer {self.layer_idx} (main blocks).")

    def forward(
        self,
        hidden_states: torch.Tensor, # Input to *this* Dynamic layer (output from preceding Decision Layer)
        # Signals from the *preceding* Decision Layer (layer_idx - 1)
        prev_decision_original_input: torch.Tensor, # Z^{n-1} from previous Decision Layer (used by router)
        prev_decision_posterior_output: torch.Tensor, # H^{D_n}_{trans} from previous Decision Layer (used by router)
        prev_decision_prior_output: torch.Tensor, # H^{D_n}_{prior} from previous Decision Layer (used by router)
        prior_loss_from_decision: torch.Tensor, # ADDED: Prior loss from the *preceding Decision Layer*

        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None, # KV cache for *this* layer's attention
        output_attentions: bool = False, # Flag if attentions should be returned
        use_cache: bool = False, # Flag if KV cache should be returned
        current_iter: int = 0, # Global training step for router's internal state
    ) -> tuple[torch.Tensor, ...]:
        """
        Forward pass of the DynamicQwenDecoderLayer, performing VPR-based gating
        on its OWN internal computation.
        """
        # Call the VPR Router using signals from the *preceding Decision Layer*
        (
            gate_vec_binary, # (B, T) or (B,) binary routing decision (for stats and forward hard decision)
            avg_ce_proportion,
            avg_cu_proportion,
            _, _, # d_st_tok, d_ch_tok (not directly returned from router)
            combined_gating_signal_continuous, # (B, T) or (B,) continuous signal (for backward pass)
            router_beta_ce, # VPRRouter learnable param
            router_beta_cu, # VPRRouter learnable param
            router_cu_detection_multiplier, # VPRRouter learnable param
            router_ce_criterion_offset, # VPRRouter learnable param
        ) = self.vpr_router(
            original_input_to_block=prev_decision_original_input,
            posterior_full_path_output=prev_decision_posterior_output,
            prior_hidden_states=prev_decision_prior_output,
            capacity_gamma=self.config.capacity_gamma,
            is_training=self.training,
        )

        # Prepare for conditional computation with Straight-Through Estimator (STE)
        # gate_forward is the binary decision for the forward pass, ensuring static compute budget.
        # gate_backward is the continuous signal for the backward pass, enabling differentiability.
        if self.vpr_router.token_wise_gating:
            # (B, T) -> (B, T, 1) for broadcasting
            gate_forward = gate_vec_binary.unsqueeze(-1)
            gate_backward = combined_gating_signal_continuous.unsqueeze(-1)
        else:
            # (B,) -> (B, 1, 1) for broadcasting across sequence dimension
            gate_forward = gate_vec_binary.view(-1, 1, 1)
            gate_backward = combined_gating_signal_continuous.view(-1, 1, 1)

        # STE: In forward, use gate_forward. In backward, gradients flow through gate_backward.
        gate_ste = gate_forward + (gate_backward - gate_forward).detach()

        # Store the original input to *this* Dynamic layer for the identity path
        original_input_to_dynamic_block = hidden_states # This is the output from the Decision Layer before this Dynamic Layer

        # --- Perform this layer's (Dynamic) Transformer computation ---
        # 1. Input LayerNorm + Self-Attention
        hidden_states_pre_attn_ln = self.input_layernorm(original_input_to_dynamic_block)

        batch_size, seq_len, hidden_size_ = hidden_states_pre_attn_ln.shape
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
            layer_idx=self.layer_idx, # Critical for KV caching
            position_embeddings=position_embeddings,
        )
        attention_output = attn_outputs[0] # Output of attention module

        present_key_value = attn_outputs[2] if use_cache else None
        attn_weights = attn_outputs[1] if output_attentions else None

        # Apply residual connection for attention
        hidden_states_after_attn = original_input_to_dynamic_block + attention_output

        # 2. Post-Attention LayerNorm + MLP
        hidden_states_pre_mlp_ln = self.post_attention_layernorm(hidden_states_after_attn)
        mlp_output = self.mlp(hidden_states_pre_mlp_ln) # Output of MLP module

        # --- APPLY NEW GATING LOGIC ---
        # Original input: X_{in,i} = original_input_to_dynamic_block_i
        # Full computation without residual: f_{S_n}(X_{\text{out},i}) -> here this is attention_output + mlp_output
        # This is the 'delta_output' or 'change' that the full computation would add
        delta_output = attention_output + mlp_output

        # According to the formula: g_i * delta_output + original_input_to_dynamic_block
        # BUT only for selected tokens. For non-selected, it's just original_input_to_dynamic_block.
        # This is achieved by using gate_ste (binary in forward) to select where to apply this scaled delta.
        scaled_delta_output = gate_ste * (combined_gating_signal_continuous.unsqueeze(-1) * delta_output) # (B, T, 1) * (B, T, C)
        # Note: combined_gating_signal_continuous must be expanded to (B, T, 1) for broadcasting
        # The .unsqueeze(-1) is applied above when gate_forward/backward are prepared.

        hidden_states_final = original_input_to_dynamic_block + scaled_delta_output
        # For tokens where gate_ste is 0, scaled_delta_output becomes 0, so hidden_states_final = original_input_to_dynamic_block.
        # For tokens where gate_ste is 1, scaled_delta_output = combined_gating_signal_continuous * delta_output,
        # so hidden_states_final = original_input_to_dynamic_block + combined_gating_signal_continuous * delta_output.


        # Prepare outputs to match expected Hugging Face Transformer layer output signature,
        # followed by our custom metrics.
        outputs = (hidden_states_final,)

        if use_cache:
            outputs += (present_key_value,)

        if output_attentions:
            outputs += (attn_weights,)

        # Append dynamic gating metrics for trainer logging.
        # Pass `gate_vec_binary` for statistics because it represents the actual hard decision count.
        # Also pass the router's learnable parameters and the prior_loss from the Decision Layer.
        return outputs + (
            avg_ce_proportion,
            avg_cu_proportion,
            gate_vec_binary,
            prior_loss_from_decision, # ADDED: Prior loss from Decision Layer
            router_beta_ce, # VPRRouter learnable param
            router_beta_cu, # VPRRouter learnable param
            router_cu_detection_multiplier, # VPRRouter learnable param
            router_ce_criterion_offset, # VPRRouter learnable param
        )