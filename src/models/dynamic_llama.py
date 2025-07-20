import logging
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

log = logging.getLogger(__name__)


class FeedForward(nn.Module):
    """A standard Feed-Forward Network, as used in Llama."""

    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.w2(self.act_fn(self.w1(x)) * self.w3(x))
        return self.dropout(x)


class DynamicLlamaDecoderLayer(LlamaDecoderLayer):
    """
    A custom version of the LlamaDecoderLayer that inserts our Prior FFN.
    We inherit from the original to reuse as much of the existing logic as possible.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        # Add the new components: the prior FFN and its own LayerNorm
        self.prior_ffn = FeedForward(config)
        self.prior_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Initialize the new components' weights.
        log.info(f"Initializing new prior_ffn for layer {layer_idx}")
        for module in [self.prior_ffn, self.prior_layernorm]:
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.normal_(param, mean=0.0, std=0.02)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_value: tuple[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # Standard Llama Decoder Path
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            **kwargs,
        )
        attention_output = attn_outputs[0]
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        posterior_mlp_output = self.mlp(hidden_states)
        hidden_states = residual + posterior_mlp_output

        # The prior predicts the current state based on the *previous* attention output.
        prev_attention_output = F.pad(attention_output[:, :-1, :], (0, 0, 1, 0))
        prior_input = self.prior_layernorm(prev_attention_output)
        prior_prediction = self.prior_ffn(prior_input)

        # Calculate the loss for the prior FFN.
        prior_loss = F.mse_loss(prior_prediction, posterior_mlp_output.detach())

        # Standard LlamaDecoderLayer returns a tuple. We append our loss.
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[2],)
        outputs += (prior_loss,)

        return outputs