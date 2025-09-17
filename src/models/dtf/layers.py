import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from .router import DTFRouter


class PriorFFN(nn.Module):
    """Lightweight FFN for prior prediction."""

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = int(hidden_size * config.prior_ffn_intermediate_size_factor)

        self.norm = nn.RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.up = nn.Linear(hidden_size, intermediate_size)
        self.down = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.down(self.act(self.up(self.norm(x))))


class DTFDecisionLayer(nn.Module):
    """Decision layer that computes original, posterior, and prior states."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.block = Qwen2DecoderLayer(config, layer_idx)
        self.prior_ffn = PriorFFN(config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        original = hidden_states
        posterior = self.block(hidden_states, attention_mask, position_ids, **kwargs)[0]
        prior = self.prior_ffn(hidden_states)

        prior_loss = F.mse_loss(prior, posterior.detach()) if self.training else None

        return {
            "original": original,
            "posterior": posterior,
            "prior": prior,
            "prior_loss": prior_loss
        }


class DTFDynamicLayer(nn.Module):
    """Dynamic layer that processes selected tokens."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.block = Qwen2DecoderLayer(config, layer_idx)
        self.router = DTFRouter(config)

    def forward(self, hidden_states, decision_output, position_ids=None, **kwargs):
        # Get routing decision
        mask, signal, stats = self.router(
            decision_output["original"],
            decision_output["posterior"],
            decision_output["prior"]
        )

        batch_idx, token_idx = mask.nonzero(as_tuple=True)

        if batch_idx.numel() == 0:
            return hidden_states, stats

        # Gather selected tokens
        selected = hidden_states[batch_idx, token_idx].unsqueeze(0)
        gate_vals = signal[batch_idx, token_idx]

        # Process selected tokens
        attn_mask = _prepare_4d_causal_attention_mask(
            None, (1, selected.shape[1]), selected, 0
        )
        pos_ids = position_ids[batch_idx, token_idx].unsqueeze(0) if position_ids is not None else None

        processed = self.block(selected, attn_mask, pos_ids, **kwargs)[0].squeeze(0)

        # Apply gating and scatter back
        delta = (processed - selected.squeeze(0)) * gate_vals.unsqueeze(-1)
        output = hidden_states.clone()
        output[batch_idx, token_idx] += delta

        return output, stats