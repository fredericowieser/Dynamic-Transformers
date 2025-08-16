# src/models/d_qwen_layers.py

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)
from src.models.d_qwen_fnn import FeedForward

import logging
log = logging.getLogger(__name__)

class DynamicQwenDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx: int, load_from_pretrained: bool = False, original_layer_state_dict: dict = None):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx

        self.prior_ffn = FeedForward(config)
        self.prior_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if isinstance(self.self_attn, nn.Module) and hasattr(self.self_attn, 'base_model') and isinstance(self.self_attn.base_model, Qwen2Attention):
            base_attn_module = self.self_attn.base_model
        elif isinstance(self.self_attn, Qwen2Attention):
            base_attn_module = self.self_attn
        else:
            base_attn_module = self.self_attn

        if not hasattr(base_attn_module, "rotary_emb") or base_attn_module.rotary_emb is None:
            log.warning(
                f"Layer {self.layer_idx}: Qwen2Attention unexpectedly missing or having None rotary_emb. Initializing it manually as a fallback."
            )
            head_dim = config.hidden_size // config.num_attention_heads

            # --- START OF CHANGE ---
            base_attn_module.rotary_emb = Qwen2RotaryEmbedding(
                self.config,
            )
            # --- END OF CHANGE ---

        if load_from_pretrained and original_layer_state_dict is not None:
            self.load_state_dict(original_layer_state_dict, strict=False)
            log.info(f"Loaded pre-trained weights for DynamicQwenDecoderLayer {self.layer_idx}.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        current_iter: int = 0,
        dynamic_k: float = 0.5,
        ce_bias: float = 0.0,
        gate_warmup_iters: int = 1000,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """
        Forward pass of the Dynamic Qwen Decoder Layer.
        This method implements the core dynamic gating mechanism where each layer
        can choose between using its full computation (posterior) or bypassing it
        based on complexity measures.
        """

        # Store original input for gating decision
        original_input_to_block = hidden_states  # (B, T, C)

        # Qwen2DecoderLayer components are now directly accessible
        # First RMSNorm
        hidden_states_pre_attn_ln = self.input_layernorm(hidden_states)

        # Self Attention
        # The Qwen2Attention module is typically called like this internally by Qwen2DecoderLayer.
        # We need to ensure position_ids are correctly passed.
        # If Qwen2Attention expects position_embeddings to be precomputed (which the error suggests it might be in some cases),
        # we'd do that here. However, `Qwen2Attention`'s default `forward` *should* compute it from `position_ids`.
        # The error implies a problem with `position_embeddings` itself when it should be `None` and processed.
        # Passing `layer_idx` is crucial for Qwen2Attention for KV caching.

        # --- START OF CHANGE ---
        # Get the base attention module if PEFT is applied
        if isinstance(self.self_attn, nn.Module) and hasattr(self.self_attn, 'base_model'):
            base_attn = self.self_attn.base_model
        else:
            base_attn = self.self_attn

        # Prepare input for rotary embedding calculation
        # Qwen2RotaryEmbedding expects input shape (batch_size, num_heads, seq_len, head_dim)
        # Replicate Llama's approach using the hidden_states directly for reshaping
        batch_size, seq_len, hidden_size_ = hidden_states_pre_attn_ln.shape
        num_attention_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_attention_heads

        # This dummy input assumes `hidden_states_pre_attn_ln` is suitable as `x` for rotary_emb
        input_for_rope = hidden_states_pre_attn_ln.view(
            batch_size, seq_len, num_attention_heads, head_dim
        ).transpose(1, 2) # (B, N_H, T, H_D)

        cos, sin = base_attn.rotary_emb(input_for_rope, position_ids)
        position_embeddings = (cos, sin)

        attn_outputs = self.self_attn(
            hidden_states_pre_attn_ln,
            attention_mask=attention_mask,
            # position_ids=position_ids, # No longer needed here, passed via position_embeddings
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            layer_idx=self.layer_idx, # Critical for Qwen2's KV caching in Attention
            position_embeddings=position_embeddings, # Explicitly pass pre-calculated embeddings
        )
        # --- END OF CHANGE ---

        attention_output = attn_outputs[0]  # (B, T, C)
        hidden_states_after_attn = original_input_to_block + attention_output # Add residual connection

        # MLP "Posterior"
        residual_mlp = hidden_states_after_attn
        hidden_states_pre_mlp_ln = self.post_attention_layernorm(hidden_states_after_attn)
        posterior_mlp_output = self.mlp(hidden_states_pre_mlp_ln)  # (B, T, C)
        posterior_full_path_output = residual_mlp + posterior_mlp_output  # (B, T, C)

        # Dynamic "Prior" FFN
        # Similar to Llama, use previous hidden state (or initial block input) for prior FFN.
        # Let's use `original_input_to_block` for `prior_ffn` as `prior_input` for now.
        # Llama's `prev_attention_output` suggests using the output of the *previous* layer's attention for current layer's prior.
        # To align perfectly, we'd need to thread `attention_output` from previous layer or redefine `prior_input`.
        # For simplicity, let's use the input to the current layer as `prior_input` after its layernorm.
        prior_input = self.prior_layernorm(original_input_to_block) # Apply layernorm to input for prior
        prior_prediction = self.prior_ffn(prior_input)  # (B, T, C)


        # Dynamic Gating Logic
        d_st_tok = F.mse_loss(posterior_full_path_output, original_input_to_block, reduction="none").mean(-1)  # (B, T)
        d_ch_tok = F.mse_loss(posterior_full_path_output, prior_prediction, reduction="none").mean(-1)  # (B, T)

        # Determine if token_wise_gating is enabled from config
        token_wise_gating = getattr(self.config, "token_wise", True) # Default True if not in config

        # Apply gating strategy
        if token_wise_gating:
            D_st, D_ch = d_st_tok, d_ch_tok  # Shape: (B, T)
        else:
            D_st = d_st_tok.mean(dim=1)  # Shape: (B,)
            D_ch = d_ch_tok.mean(dim=1)  # Shape: (B,)

        # Apply warmup bias if configured
        if gate_warmup_iters > 0:
            bias_scale = max(
                0.0,
                1.0 - current_iter / gate_warmup_iters,
            )
            beta = D_ch.detach().mean() * bias_scale
            D_ch = D_ch - beta

        # Compute gate conditions
        CE = D_st > D_ch - ce_bias
        CU = D_st > dynamic_k * D_st.detach().mean() # Note: D_st.detach().mean() computes mean across B*T for token-wise, or B for block-wise

        # Combine conditions
        gate_vec = (CE | CU).float()

        # Apply gate
        if token_wise_gating:
            # gate_vec: (B, T) -> (B, T, 1)
            gate = gate_vec.unsqueeze(-1)
        else:
            # gate_vec: (B,) -> (B, 1, 1)
            gate = gate_vec.view(-1, 1, 1)

        hidden_states_final = gate * posterior_full_path_output + (1.0 - gate) * original_input_to_block

        # Prior Loss
        prior_loss = F.mse_loss(prior_prediction, posterior_full_path_output.detach())

        # Prepare outputs (mirroring Qwen2DecoderLayer output format + custom metrics)
        outputs = (hidden_states_final,) # Primary output

        if output_attentions:
            outputs += (attn_outputs[1],) # Attention weights are at index 1 of attn_outputs
        if use_cache:
            outputs += (attn_outputs[2],) # Present_key_value is at index 2 of attn_outputs

        # Add dynamic gating metrics for trainer logging
        avg_ce_proportion = CE.float().mean()
        avg_cu_proportion = CU.float().mean()
        # gate_vec_for_stats should reflect actual gates for logging
        gate_vec_for_stats = gate_vec # Keep original shape (B,T) or (B,) before unsqueezing/viewing for mix

        # The order of these custom metrics MUST match what `DynamicQwenForCausalLM.forward` expects
        # (avg_ce_proportion, avg_cu_proportion, prior_loss, gate_vec_for_stats)
        return outputs + (avg_ce_proportion, avg_cu_proportion, prior_loss, gate_vec_for_stats)


def patch_qwen_layers(model):
    """
    Replaces each Qwen2DecoderLayer in model.model.layers with DynamicQwenDecoderLayer.
    It transfers the state_dict from the original layers.
    """
    log.info("Patching Qwen model layers to DynamicQwenDecoderLayer.")
    new_layers = nn.ModuleList()
    for i, layer in enumerate(model.model.layers):
        original_layer_state_dict = layer.state_dict()
        # Ensure state_dict is on CPU before passing, constructor handles device move
        original_layer_state_dict = {k: v.cpu() for k, v in original_layer_state_dict.items()}

        custom_layer = DynamicQwenDecoderLayer(
            model.config,
            layer_idx=i,
            load_from_pretrained=True,
            original_layer_state_dict=original_layer_state_dict,
        )
        # Move to device as done in Llama
        device = next(model.parameters()).device # Get current device of the model
        try:
            # Check if custom_layer is on 'meta' device or needs to be moved
            layer_device = next(custom_layer.parameters()).device
            if str(layer_device) == "meta":
                custom_layer = custom_layer.to_empty(device=device)
            else:
                custom_layer = custom_layer.to(device)
        except StopIteration:
            # If layer has no parameters (e.g., still meta and empty), move it to device
            custom_layer = custom_layer.to_empty(device=device)

        new_layers.append(custom_layer)
        log.info(f"Successfully re-instantiated layer {i} as DynamicQwenDecoderLayer for loading.")
    model.model.layers = new_layers
    return model