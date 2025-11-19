import logging
from typing import Optional, Tuple

import torch
import torch._dynamo
import torch.nn as nn
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask
from transformers.models.qwen2.modeling_qwen2 import (Qwen2DecoderLayer,
                                                      apply_rotary_pos_emb,
                                                      repeat_kv)

# Try to import the custom kernel
try:
    from src.kernels.sparse_causal_flash_attn import sparse_causal_attention
    _SPARSE_ATTN_KERNEL_AVAILABLE = True
except ImportError:
    _SPARSE_ATTN_KERNEL_AVAILABLE = False

log = logging.getLogger(__name__)


class DynamicBlock(nn.Module):
    """
    Wraps an existing HF Qwen2DecoderLayer and provides a method for dynamically
    processing a subset of tokens in a sequence.
    """

    def __init__(self, layer: Qwen2DecoderLayer):
        super().__init__()
        self.layer = layer  # reference to HF layer (weights are shared)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        return self.layer(hidden_states, **kwargs)

    @torch._dynamo.disable
    def process_selected(
        self,
        hidden_states: torch.Tensor,
        batch_indices: torch.Tensor,
        token_indices: torch.Tensor,
        gating_scores: Optional[torch.Tensor] = None,
        use_soft_gating: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[Tuple]]:
        """
        Processes a packed tensor of selected tokens. This method is used when the number
        of selected tokens is dynamic per sequence. Flash attention is disabled to prevent errors.
        """
        if batch_indices.numel() == 0:
            return hidden_states, None, None

        B, T, D = hidden_states.shape
        num_selected_total = token_indices.numel()

        # Check if we can use the optimized kernel (fixed k per batch item)
        is_fixed_k = (num_selected_total > 0) and (num_selected_total % B == 0)
        
        # Use custom kernel only if available, enabled, and for fixed-k routing
        if _SPARSE_ATTN_KERNEL_AVAILABLE and self.layer.self_attn._attn_implementation == "flash_attention_2" and is_fixed_k:
            
            k = num_selected_total // B
            topk_idx = token_indices.view(B, k)
            batch_idx_gather = torch.arange(B, device=hidden_states.device).unsqueeze(1)

            # --- Manual Qwen2DecoderLayer forward pass for selected tokens ---
            attn_layer = self.layer.self_attn
            mlp_layer = self.layer.mlp
            input_ln = self.layer.input_layernorm
            post_attn_ln = self.layer.post_attention_layernorm

            residual = hidden_states
            normed_hidden_states = input_ln(hidden_states)

            # 1. Attention
            q_full = attn_layer.q_proj(normed_hidden_states).view(B, T, attn_layer.num_heads, attn_layer.head_dim)
            k_full = attn_layer.k_proj(normed_hidden_states).view(B, T, attn_layer.num_key_value_heads, attn_layer.head_dim)
            v_full = attn_layer.v_proj(normed_hidden_states).view(B, T, attn_layer.num_key_value_heads, attn_layer.head_dim)

            cos, sin = kwargs["position_embeddings"]
            position_ids = kwargs["position_ids"]
            
            # Apply rotary embeddings to the full Q/K tensors
            q_full_rotary, k_full_rotary = apply_rotary_pos_emb(q_full, k_full, cos, sin, position_ids)
            
            # Gather the Q for selected tokens
            q_selected_rotary = q_full_rotary[batch_idx_gather, topk_idx]

            # Repeat K/V for Grouped-Query Attention
            k_full_repeated = repeat_kv(k_full_rotary, attn_layer.num_key_value_groups)
            v_full_repeated = repeat_kv(v_full, attn_layer.num_key_value_groups)

            # Call the custom sparse attention kernel
            attn_output_selected = sparse_causal_attention(
                q_selected_rotary,
                k_full_repeated,
                v_full_repeated,
                topk_idx,
                attn_layer.softmax_scale
            ) # Shape: [B, k, H, D_h]

            attn_output_selected = attn_output_selected.reshape(B, k, -1)
            attn_output_selected = attn_layer.o_proj(attn_output_selected) # Shape: [B, k, D]

            # 2. First Residual
            selected_residual = residual[batch_idx_gather, topk_idx]
            hidden_states_after_attn_selected = selected_residual + attn_output_selected

            # 3. MLP
            mlp_residual_selected = hidden_states_after_attn_selected
            normed_mlp_input_selected = post_attn_ln(hidden_states_after_attn_selected)
            mlp_output_selected = mlp_layer(normed_mlp_input_selected)

            # 4. Second Residual
            final_processed_tokens = mlp_residual_selected + mlp_output_selected

            # 5. Scatter back
            final_hidden_states = hidden_states.clone()
            original_selected_tokens = hidden_states[batch_idx_gather, topk_idx]
            delta = final_processed_tokens - original_selected_tokens

            if use_soft_gating:
                if gating_scores is None:
                    raise ValueError("gating_scores must be provided for soft gating")
                # gating_scores is flattened, needs to be [B, k, 1]
                gating_scores_reshaped = gating_scores.view(B, k, 1)
                delta = delta * gating_scores_reshaped.to(delta.dtype)
            
            updated_tokens = original_selected_tokens + delta
            final_hidden_states.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, D), updated_tokens)

            # The custom kernel doesn't return these, so we return None
            present_key_value = None
            attention_weights = None

            return final_hidden_states, present_key_value, attention_weights

        # --- Fallback to original eager implementation for variable-k or if kernel is unavailable ---
        else:
            selected_tokens = hidden_states[batch_indices, token_indices]
            num_selected = selected_tokens.shape[0]
            selected_tokens_batched = selected_tokens.unsqueeze(0)

            position_ids = kwargs.get("position_ids")
            position_embeddings = kwargs.get("position_embeddings")

            selected_attn_mask = _prepare_4d_causal_attention_mask(
                None, (1, num_selected), selected_tokens_batched, 0
            )

            selected_pos_ids = None
            if position_ids is not None:
                # Ensure position_ids has at least 2 dimensions before trying to access them
                if position_ids.dim() >= 2:
                    selected_pos_ids = position_ids[batch_indices, token_indices].unsqueeze(0)
                else:
                    # Fallback for 1D position_ids, though less likely with batches
                    selected_pos_ids = position_ids[token_indices].unsqueeze(0)

            selected_pos_emb = None
            if position_embeddings is not None:
                cos, sin = position_embeddings
                selected_pos_emb = (
                    cos[batch_indices, token_indices].unsqueeze(0),
                    sin[batch_indices, token_indices].unsqueeze(0),
                )

            # TODO: Investigate how to make packed sequences compatible with flash attention.
            # The current implementation in Hugging Face transformers (4.43.2) throws a
            # `cu_seqlens` error when processing a packed tensor of tokens gathered from
            # across a batch. As a workaround, we temporarily disable flash attention
            # for this specific operation.
            original_attn_impl = None
            if hasattr(self.layer.self_attn, "_attn_implementation"):
                original_attn_impl = self.layer.self_attn._attn_implementation
                if original_attn_impl == "flash_attention_2":
                    self.layer.self_attn._attn_implementation = "eager"

            try:
                out = self.layer(
                    hidden_states=selected_tokens_batched,
                    attention_mask=selected_attn_mask,
                    position_ids=selected_pos_ids,
                    position_embeddings=selected_pos_emb,
                    use_cache=kwargs.get("use_cache", False),
                )
            finally:
                if original_attn_impl is not None:
                    self.layer.self_attn._attn_implementation = original_attn_impl

            processed_tokens = out[0].squeeze(0) if isinstance(out, tuple) else out.squeeze(0)
            present_key_value = (
                out[1]
                if kwargs.get("use_cache", False) and isinstance(out, tuple) and len(out) > 1
                else None
            )
            attention_weights = out[2] if isinstance(out, tuple) and len(out) > 2 else None

            final_hidden_states = hidden_states.clone()

            if use_soft_gating:
                if gating_scores is None:
                    raise ValueError("gating_scores must be provided for soft gating")
                delta = processed_tokens - selected_tokens
                scaled_delta = delta * gating_scores.unsqueeze(-1).to(delta.dtype)
                updated_tokens = selected_tokens + scaled_delta
                final_hidden_states[batch_indices, token_indices] = updated_tokens
            else:
                final_hidden_states[batch_indices, token_indices] = processed_tokens

            return final_hidden_states, present_key_value, attention_weights
