import logging
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaMLP,
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
    """
    A custom version of the LlamaDecoderLayer that inserts our Prior FFN.
    We inherit from the original to reuse as much of the existing logic as possible.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.self_attn = LlamaAttention(config, layer_idx) # Explicitly create LlamaAttention
        self.mlp = LlamaMLP(config)

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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,  # Input as position_ids
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        current_iter: int = 0,
        gate_warmup_iters: int = 1000,
        dynamic_k: float = 0.5,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        
        original_input_to_block = hidden_states

        # Standard Llama Decoder Path
        residual_attn = hidden_states
        hidden_states_pre_attn_ln = self.input_layernorm(hidden_states)
        
        # Compute position_embeddings from position_ids if available
        if position_ids is not None:
            # Assuming access to RotaryEmbedding; you might need to pass it from the model level
            # In standard Transformers, RotaryEmbedding is part of the model config
            rotary_emb = LlamaRotaryEmbedding(self.config)  # Instantiate if not already available
            position_embeddings = rotary_emb(position_ids, seq_len=hidden_states.shape[1])  # Compute RoPE embeddings
        else:
            position_embeddings = None  # Fallback if no position_ids

        # Explicitly prepare arguments for self.self_attn
        attn_args = {
            "hidden_states": hidden_states_pre_attn_ln,
            "output_attentions": output_attentions,
            "use_cache": use_cache,
        }
        if attention_mask is not None:
            attn_args["attention_mask"] = attention_mask
        if position_embeddings is not None:
            attn_args["position_embeddings"] = position_embeddings  # Pass the computed embeddings
        if past_key_value is not None:
            attn_args["past_key_value"] = past_key_value
        
        # Call self_attn with explicitly constructed arguments
        attn_outputs = self.self_attn(**attn_args)
        
        attention_output = attn_outputs[0]  # (B, T, C)
        hidden_states_after_attn = residual_attn + attention_output

        # Main MLP part (Posterior)
        residual_mlp = hidden_states_after_attn
        hidden_states_pre_mlp_ln = self.post_attention_layernorm(hidden_states_after_attn)
        posterior_mlp_output = self.mlp(hidden_states_pre_mlp_ln)  # (B, T, C)
        posterior_full_path_output = residual_mlp + posterior_mlp_output

        # Prior FFN prediction
        prev_attention_output = F.pad(attention_output[:, :-1, :], (0, 0, 1, 0))
        prior_input = self.prior_layernorm(prev_attention_output)
        prior_prediction = self.prior_ffn(prior_input)  # (B, T, C)

        # --- Dynamic Gating Logic ---
        d_st_tok = F.mse_loss(posterior_full_path_output, original_input_to_block, reduction="none").mean(-1)
        d_ch_tok = F.mse_loss(posterior_full_path_output, prior_prediction, reduction="none").mean(-1)

        D_st = d_st_tok.mean(dim=1)  # (B,)
        D_ch = d_ch_tok.mean(dim=1)  # (B,)

        bias_scale = max(0.0, 1.0 - current_iter / gate_warmup_iters)
        beta = D_ch.detach().mean() * bias_scale
        D_ch_biased = D_ch + beta

        CE = D_st > D_ch_biased
        CU = D_st > dynamic_k * D_st.detach().mean()

        gate_vec = (CE | CU).float()  # (B,)

        # Mix the block output based on the gate
        gate = gate_vec.view(-1, 1, 1)  # Reshape for broadcasting
        hidden_states_final = gate * posterior_full_path_output + (1.0 - gate) * original_input_to_block

        # Prediction-loss for the prior FFN
        prior_loss = F.mse_loss(prior_prediction, posterior_full_path_output.detach())

        # Prepare outputs
        outputs = (hidden_states_final,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[2],)
        outputs += (prior_loss,)
        outputs += (gate_vec,)

        return outputs