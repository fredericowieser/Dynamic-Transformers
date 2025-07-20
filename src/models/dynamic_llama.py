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
        attention_mask: Optional[torch.Tensor] = None, # Made Optional for clarity, consistent with HF
        position_ids: Optional[torch.LongTensor] = None, # Made Optional
        past_key_value: Optional[Tuple[torch.Tensor]] = None, # Made Optional
        output_attentions: bool = False,
        use_cache: bool = False,
        current_iter: int = 0, # Custom argument
        gate_warmup_iters: int = 1000, # Custom argument
        dynamic_k: float = 0.5, # Custom argument
        **kwargs, # Capture any other arguments passed to this layer's forward, don't pass them on
    ) -> Tuple[torch.Tensor, ...]:
        
        original_input_to_block = hidden_states # (B, T, C)

        # Standard Llama Decoder Path
        # Self-attention part
        residual_attn = hidden_states
        hidden_states_pre_attn_ln = self.input_layernorm(hidden_states)
        
        # Explicitly prepare arguments for self.self_attn
        attn_args = {
            "hidden_states": hidden_states_pre_attn_ln,
            "output_attentions": output_attentions,
            "use_cache": use_cache,
        }
        if attention_mask is not None:
            attn_args["attention_mask"] = attention_mask
        if position_ids is not None:
            attn_args["position_embeddings"] = position_ids
        if past_key_value is not None:
            attn_args["past_key_value"] = past_key_value
        
        # Call self_attn with explicitly constructed arguments
        attn_outputs = self.self_attn(**attn_args)
        
        attention_output = attn_outputs[0] # (B, T, C)
        hidden_states_after_attn = residual_attn + attention_output # This is 'mha_out'

        # Main MLP part (Posterior)
        residual_mlp = hidden_states_after_attn
        hidden_states_pre_mlp_ln = self.post_attention_layernorm(hidden_states_after_attn)
        posterior_mlp_output = self.mlp(hidden_states_pre_mlp_ln) # (B, T, C)
        posterior_full_path_output = residual_mlp + posterior_mlp_output # (B, T, C)

        # Prior FFN prediction
        prev_attention_output = F.pad(attention_output[:, :-1, :], (0, 0, 1, 0))
        prior_input = self.prior_layernorm(prev_attention_output)
        prior_prediction = self.prior_ffn(prior_input) # (B, T, C)

        # --- Dynamic Gating Logic (inspired by nanoGPT) ---
        d_st_tok = F.mse_loss(posterior_full_path_output, original_input_to_block, reduction="none").mean(-1) # (B, T)
        d_ch_tok = F.mse_loss(posterior_full_path_output, prior_prediction, reduction="none").mean(-1) # (B, T)

        D_st = d_st_tok.mean(dim=1) # (B,)
        D_ch = d_ch_tok.mean(dim=1) # (B,)

        bias_scale = max(0.0, 1.0 - current_iter / gate_warmup_iters)
        beta = D_ch.detach().mean() * bias_scale # Scalar, mean over batch
        D_ch_biased = D_ch + beta # (B,)

        CE = D_st > D_ch_biased # (B,) bool
        CU = D_st > dynamic_k * D_st.detach().mean() # (B,) bool

        gate_vec = (CE | CU).float() # (B,) - 1.0 means activate posterior, 0.0 means activate original input
        
        # Mix the block output based on the gate
        gate = gate_vec.view(-1, 1, 1) # Reshape for broadcasting (B,1,1)
        hidden_states_final = gate * posterior_full_path_output + (1.0 - gate) * original_input_to_block # (B, T, C)

        # Prediction-loss for the prior FFN
        prior_loss = F.mse_loss(prior_prediction, posterior_full_path_output.detach()) # Scalar

        # Prepare outputs (standard Llama output + prior_loss + gate_vec)
        outputs = (hidden_states_final,) # New primary output
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[2],)
        outputs += (prior_loss,) # The specific loss for the prior FFN
        outputs += (gate_vec,) # The gate decision vector for logging or auxiliary uses

        return outputs