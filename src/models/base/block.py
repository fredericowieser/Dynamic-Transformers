import torch
import torch.nn as nn
from typing import Tuple
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

class DynamicBlock(nn.Module):
    """
    Wraps an existing HF Qwen2DecoderLayer and provides a gather-process-scatter path.
    """
    def __init__(self, layer: Qwen2DecoderLayer):
        super().__init__()
        self.layer = layer  # reference to HF layer (weights are shared)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        return self.layer(hidden_states, **kwargs)

    def process_selected(
        self,
        hidden_states: torch.Tensor,
        batch_indices: torch.Tensor,
        token_indices: torch.Tensor,
        gating_scores: torch.Tensor,
        use_soft_gating: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Tuple, Tuple]:
        if batch_indices.numel() == 0:
            return hidden_states, None, None

        # 1. Gather
        selected_tokens = hidden_states[batch_indices, token_indices]
        num_selected = selected_tokens.shape[0]
        selected_tokens_batched = selected_tokens.unsqueeze(0)

        # 2. Prepare inputs for the block
        position_ids = kwargs.get('position_ids')
        position_embeddings = kwargs.get('position_embeddings')
        
        selected_attn_mask = _prepare_4d_causal_attention_mask(None, (1, num_selected), selected_tokens_batched, 0)
        selected_pos_ids = position_ids[batch_indices, token_indices].unsqueeze(0) if position_ids is not None else None
        
        selected_pos_emb = None
        if position_embeddings is not None:
            cos, sin = position_embeddings
            selected_pos_emb = (cos[batch_indices, token_indices].unsqueeze(0), sin[batch_indices, token_indices].unsqueeze(0))

        # 3. Process
        out = self.layer(
            hidden_states=selected_tokens_batched,
            attention_mask=selected_attn_mask,
            position_ids=selected_pos_ids,
            position_embeddings=selected_pos_emb,
            use_cache=kwargs.get('use_cache', False),
        )
        # HF Qwen2DecoderLayer returns a Tensor (B, L, H). Old custom code returned a tuple.
        if isinstance(out, tuple):
            processed_tokens = out[0].squeeze(0)
            present_key_value = out[1] if kwargs.get('use_cache', False) and len(out) > 1 else None
            attention_weights = out[2] if len(out) > 2 else None
        else:
            processed_tokens = out.squeeze(0)  # out is (1, L, H)
            present_key_value = None
            attention_weights = None

        # 4. Scatter
        final_hidden_states = hidden_states.clone()
        
        if use_soft_gating:
            # Aligns with the OLD code's logic for MoD and SDT
            delta = processed_tokens - selected_tokens
            scaled_delta = delta * gating_scores.unsqueeze(-1).to(delta.dtype)
            updated_tokens = selected_tokens + scaled_delta
            final_hidden_states[batch_indices, token_indices] = updated_tokens
        else:
            # Hard update for inference
            final_hidden_states[batch_indices, token_indices] = processed_tokens

        return final_hidden_states, present_key_value, attention_weights