import logging

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from ..blocks.prior_ffn import PriorFeedForward
from ..blocks.qwen_block import Qwen2Block
from ..qwen.modeling_outputs import DecisionLayerOutput

log = logging.getLogger(__name__)


class DecisionLayer(nn.Module):
    """
    Implements the 'Decision Sub-Layer' for the VPR architecture.

    This layer wraps a standard Qwen2 transformer block and adds a
    PriorFeedForward network to generate predictive signals for the
    VPR router in the subsequent Dynamic Layer.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.block = Qwen2Block(config, layer_idx=layer_idx)

        # New Prior FFN components
        prior_ffn_factor = getattr(config, "prior_ffn_intermediate_size_factor", 2.0)
        self.prior_ffn = PriorFeedForward(
            config, intermediate_size_factor=prior_ffn_factor
        )
        self.prior_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> DecisionLayerOutput:
        
        vpr_signal_original_input = hidden_states

        # --- Standard Qwen2 Decoder Layer Logic ---
        # The complex logic is now encapsulated in the reusable block
        block_outputs = self.block(hidden_states, **kwargs)
        posterior_full_path_output = block_outputs[0]
        present_key_value = block_outputs[1]
        attn_weights = block_outputs[2] if len(block_outputs) > 2 else None

        # --- Prior FFN Logic ---
        prior_input_ln = self.prior_layernorm(vpr_signal_original_input)
        prior_ffn_output = self.prior_ffn(prior_input_ln)
        vpr_signal_prior_hidden_states = vpr_signal_original_input + prior_ffn_output

        # Calculate prior loss for monitoring
        prior_loss = F.mse_loss(
            vpr_signal_prior_hidden_states, posterior_full_path_output.detach()
        )

        return DecisionLayerOutput(
            hidden_states=posterior_full_path_output,
            vpr_signal_original_input=vpr_signal_original_input,
            vpr_signal_posterior_output=posterior_full_path_output,
            vpr_signal_prior_hidden_states=vpr_signal_prior_hidden_states,
            present_key_value=present_key_value,
            attention_weights=attn_weights,
            prior_loss=prior_loss,
        )