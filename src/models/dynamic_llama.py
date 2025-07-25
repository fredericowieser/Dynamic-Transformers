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


class DynamicLlamaBlockWiseDecoderLayer(LlamaDecoderLayer):
    """
    A custom version of the LlamaDecoderLayer that inserts our Prior FFN.
    We inherit from the original to reuse as much of the existing logic as possible.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        # Explicitly create LlamaAttention.
        # This will initialize its internal self.rotary_emb.
        self.self_attn = LlamaAttention(config, layer_idx)
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
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        current_iter: int = 0,
        gate_warmup_iters: int = 1000,
        dynamic_k: float = 0.5,
        **kwargs,
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
        if past_key_value is not None:
            attn_args["past_key_value"] = past_key_value

        # FIX: Handle 'position_embeddings' if required by a non-standard LlamaAttention
        # The LlamaAttention usually computes cos/sin from position_ids internally.
        # If your LlamaAttention.forward truly requires 'position_embeddings'
        # as a separate argument, it expects the (cos, sin) tuple from rotary_emb.
        # We need to manually compute it here using the LlamaAttention's own rotary_emb module.
        position_embeddings = None
        if position_ids is not None:
            # We need a tensor with the correct shape for rotary_emb, specifically its head_dim.
            # LlamaRotaryEmbedding's forward signature is `forward(value_states, position_ids, seq_len=None)`.
            # value_states typically has shape (batch_size, num_heads, seq_len, head_dim).
            # We can create a dummy tensor from hidden_states to match the expected shape and dtype.
            
            # hidden_states_pre_attn_ln is (B, T, C)
            batch_size, seq_len, hidden_size = hidden_states_pre_attn_ln.shape
            num_heads = self.self_attn.num_heads
            head_dim = self.self_attn.head_dim # LlamaAttention sets this attribute

            # Create a dummy tensor of (B, num_heads, T, head_dim) for rotary_emb's first arg.
            # The actual values don't matter, only shape and device/dtype.
            dummy_value_states = hidden_states_pre_attn_ln.reshape(
                batch_size, seq_len, num_heads, head_dim
            ).transpose(1, 2) # -> (B, num_heads, T, head_dim)

            # Compute cos and sin using the attention layer's own rotary embedding module
            cos, sin = self.self_attn.rotary_emb(dummy_value_states, position_ids)
            position_embeddings = (cos, sin)
        
        # If position_embeddings is computed, add it to attn_args.
        # This is passed as a *keyword argument* here, but the error suggests it might be a *positional* one.
        # If it's a positional argument, the order matters. But keyword args are usually more robust.
        # Let's add it as a keyword, as that's the most common way.
        if position_embeddings is not None:
            attn_args["position_embeddings"] = position_embeddings
        # The `position_ids` argument itself should ideally NOT be passed if `position_embeddings` is.
        # The original LlamaAttention uses `position_ids` to *compute* `cos, sin`.
        # If `position_embeddings` is expected, it means the `cos, sin` are directly provided.
        # So we explicitly *do not* pass `position_ids` here to avoid a conflict.
        # If position_ids is expected to also be present in this non-standard LlamaAttention,
        # then the initial problem was even deeper.


        # Call self_attn with explicitly constructed arguments
        # If the LlamaAttention is truly modified to require 'position_embeddings' as
        # a *positional* argument, this might still fail if it's the Nth positional arg.
        # But this is the most logical way to pass pre-computed embeddings.
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

        # Dynamic Gating Logic (inspired by nanoGPT)
        d_st_tok = F.mse_loss(posterior_full_path_output, original_input_to_block, reduction="none").mean(-1) # (B, T)
        d_ch_tok = F.mse_loss(posterior_full_path_output, prior_prediction, reduction="none").mean(-1) # (B, T)

        D_st = d_st_tok.mean(dim=1) # (B,)
        D_ch = d_ch_tok.mean(dim=1) # (B,)

        # Training Time Biasing
        if gate_warmup_iters > 0:
            bias_scale = max(0.0, 1.0 - current_iter / gate_warmup_iters)
            beta = D_ch.detach().mean() * bias_scale # Scalar, mean over batch
            D_ch = D_ch - beta # (B,)

        CE = D_st > D_ch # (B,) bool
        CU = D_st > dynamic_k * D_st.detach().mean() # (B,) bool

        gate_vec = (CE | CU).float() # (B,) - 1.0 means activate posterior, 0.0 means activate original input

        # Calculate average proportions of CE and CU
        avg_ce_proportion = CE.float().mean()
        avg_cu_proportion = CU.float().mean()
        
        # Mix the block output based on the gate
        gate = gate_vec.view(-1, 1, 1) # Reshape for broadcasting (B,1,1)
        hidden_states_final = gate * posterior_full_path_output + (1.0 - gate) * original_input_to_block # (B, T, C)

        # Prediction-loss for the prior FFN
        prior_loss = F.mse_loss(prior_prediction, posterior_full_path_output.detach()) # Scalar

        # Prepare outputs
        outputs = (hidden_states_final,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[2],)
        
        # --- NEW: Add metrics to output tuple ---
        outputs += (avg_ce_proportion, avg_cu_proportion, prior_loss, gate_vec)

        return outputs


class DynamicLlamaTokenWiseDecoderLayer(LlamaDecoderLayer):
    """
    A custom version of the LlamaDecoderLayer that inserts our Prior FFN.
    We inherit from the original to reuse as much of the existing logic as possible.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = LlamaAttention(config, layer_idx)
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
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        current_iter: int = 0,
        gate_warmup_iters: int = 1000,
        dynamic_k: float = 0.5,
        **kwargs,
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
        if past_key_value is not None:
            attn_args["past_key_value"] = past_key_value
        position_embeddings = None
        if position_ids is not None:
            batch_size, seq_len, hidden_size = hidden_states_pre_attn_ln.shape
            num_heads = self.self_attn.num_heads
            head_dim = self.self_attn.head_dim
            dummy_value_states = hidden_states_pre_attn_ln.reshape(
                batch_size, seq_len, num_heads, head_dim
            ).transpose(1, 2) # -> (B, num_heads, T, head_dim)
            cos, sin = self.self_attn.rotary_emb(dummy_value_states, position_ids)
            position_embeddings = (cos, sin)
        if position_embeddings is not None: attn_args["position_embeddings"] = position_embeddings
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

        # Dynamic Gating Logic (inspired by nanoGPT)
        d_st_tok = F.mse_loss(posterior_full_path_output, original_input_to_block, reduction="none").mean(-1) # (B, T)
        d_ch_tok = F.mse_loss(posterior_full_path_output, prior_prediction, reduction="none").mean(-1) # (B, T)

        D_st = d_st_tok # (B, T)
        D_ch = d_ch_tok # (B, T)

        # Training Time Biasing
        if gate_warmup_iters > 0:
            bias_scale = max(0.0, 1.0 - current_iter / gate_warmup_iters)
            beta = D_ch.detach().mean() * bias_scale # Scalar, mean over batch
            D_ch = D_ch - beta # (B,)

        CE = D_st > D_ch # (B, T) bool
        CU = D_st > dynamic_k * D_st.detach().mean() # (B, T) bool

        gate_vec = (CE | CU).float() # (B, T) - 1.0 means activate posterior, 0.0 means activate original input

        # Calculate average proportions of CE and CU
        avg_ce_proportion = CE.float().mean()
        avg_cu_proportion = CU.float().mean()

        # Mix the block output based on the gate
        gate = gate_vec.unsqueeze(-1) # Reshape for broadcasting (B, T)
        hidden_states_final = gate * posterior_full_path_output + (1.0 - gate) * original_input_to_block # (B, T, C)

        # Prediction-loss for the prior FFN
        prior_loss = F.mse_loss(prior_prediction, posterior_full_path_output.detach()) # Scalar

        # Prepare outputs
        outputs = (hidden_states_final,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[2],)
        
        # --- NEW: Add metrics to output tuple ---
        outputs += (avg_ce_proportion, avg_cu_proportion, prior_loss, gate_vec.mean(dim=1))
        
        return outputs