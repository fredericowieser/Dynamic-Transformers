import torch
import torch.nn as nn
import torch._dynamo
from typing import Tuple, Optional
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

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

        selected_tokens = hidden_states[batch_indices, token_indices]
        num_selected = selected_tokens.shape[0]
        selected_tokens_batched = selected_tokens.unsqueeze(0)

        position_ids = kwargs.get('position_ids')
        position_embeddings = kwargs.get('position_embeddings')
        
        selected_attn_mask = _prepare_4d_causal_attention_mask(None, (1, num_selected), selected_tokens_batched, 0)

        # DIAGNOSTIC: Force attention_mask to None to test hypothesis
        selected_attn_mask = None
        
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
            selected_pos_emb = (cos[batch_indices, token_indices].unsqueeze(0), sin[batch_indices, token_indices].unsqueeze(0))

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
                use_cache=kwargs.get('use_cache', False),
            )
        finally:
            if original_attn_impl is not None:
                self.layer.self_attn._attn_implementation = original_attn_impl

        processed_tokens = out[0].squeeze(0) if isinstance(out, tuple) else out.squeeze(0)
        present_key_value = out[1] if kwargs.get('use_cache', False) and isinstance(out, tuple) and len(out) > 1 else None
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
