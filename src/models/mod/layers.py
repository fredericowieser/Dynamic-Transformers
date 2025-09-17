import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from .router import MoDRouter


class MoDLayer(nn.Module):
    """MoD layer with top-k token selection."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.block = Qwen2DecoderLayer(config, layer_idx)
        self.router = MoDRouter(config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        # Get routing decision
        mask, scores = self.router(hidden_states)
        batch_idx, token_idx = mask.nonzero(as_tuple=True)

        if batch_idx.numel() == 0:
            return hidden_states, None

        # Gather selected tokens
        selected = hidden_states[batch_idx, token_idx].unsqueeze(0)

        # Process selected tokens
        attn_mask = _prepare_4d_causal_attention_mask(
            None, (1, selected.shape[1]), selected, 0
        )
        pos_ids = position_ids[batch_idx, token_idx].unsqueeze(0) if position_ids is not None else None

        layer_outputs = self.block(selected, attn_mask, pos_ids, **kwargs)
        processed = layer_outputs[0].squeeze(0)

        # Scatter back
        output = hidden_states.clone()
        output[batch_idx, token_idx] = processed

        stats = {"avg_score": scores.mean().item(), "selected_ratio": mask.mean().item()}
        return output, stats