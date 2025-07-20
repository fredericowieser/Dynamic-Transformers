import logging
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaMLP,
    LlamaRotaryEmbedding,
)
from typing import Tuple, Optional

log = logging.getLogger(__name__)


class FeedForward(nn.Module):
    """A standard Feed-Forward Network, as used in Llama."""

    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.0)
    def forward(self, x):
        x = self.w2(self.act_fn(self.w1(x)) * self.w3(x))
        return self.dropout(x)


class DynamicLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.prior_ffn = FeedForward(config)
        self.prior_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ):
        original_input = hidden_states

        # Standard Llama path
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )[0]
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + mlp_out

        # Prior FFN
        prev_attn = F.pad(attn_out[:, :-1, :], (0, 0, 1, 0))
        prior_input = self.prior_layernorm(prev_attn)
        prior_pred = self.prior_ffn(prior_input)

        # Dynamic gating
        d_st = F.mse_loss(hidden_states, original_input, reduction="none").mean(-1)
        d_ch = F.mse_loss(hidden_states, prior_pred, reduction="none").mean(-1)

        gate = (d_st > d_ch).float().view(-1, 1, 1)
        hidden_states = gate * hidden_states + (1 - gate) * original_input

        prior_loss = F.mse_loss(prior_pred, hidden_states.detach())

        return hidden_states, prior_loss, gate.mean()