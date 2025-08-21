import torch
import torch.nn as nn

from ..blocks.qwen_block import Qwen2Block
from ..blocks.vpr_router import VPRRouter
from ..qwen.modeling_outputs import DecisionLayerOutput, DynamicLayerOutput


class DynamicLayer(nn.Module):
    """
    Implements the 'Dynamic Sub-Layer' for the VPR architecture.

    This layer uses a VPRRouter, informed by outputs from the preceding
    DecisionLayer, to decide which tokens undergo full computation.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.block = Qwen2Block(config, layer_idx=layer_idx)
        self.vpr_router = VPRRouter(config, layer_idx)
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        decision_output: DecisionLayerOutput,
        **kwargs,
    ) -> DynamicLayerOutput:
        
        (
            gate_vec_binary,
            avg_ce_proportion,
            avg_cu_proportion,
            _, _,
            combined_gating_signal,
            router_beta_ce,
            router_beta_cu,
            router_cu_multiplier,
            router_ce_offset,
        ) = self.vpr_router(
            original_input_to_block=decision_output.vpr_signal_original_input,
            posterior_full_path_output=decision_output.vpr_signal_posterior_output,
            prior_hidden_states=decision_output.vpr_signal_prior_hidden_states,
            capacity_gamma=self.config.capacity_gamma,
            is_training=self.training,
        )

        # Straight-Through Estimator (STE) for differentiable routing
        gate_forward = gate_vec_binary.unsqueeze(-1)
        gate_backward = combined_gating_signal.unsqueeze(-1)
        gate_ste = gate_forward + (gate_backward - gate_forward).detach()

        # Perform the block's computation on all tokens
        # The gating is applied to the *output* of the block
        block_output, present_key_value, attn_weights = self.block(hidden_states, **kwargs)
        
        # Calculate the change (delta) introduced by the block
        delta_output = block_output - hidden_states

        # Apply the gate to the delta and add it back to the original input
        # For tokens where gate_ste is 0, this adds nothing.
        # For tokens where gate_ste is 1, it adds the scaled change.
        final_hidden_states = hidden_states + (gate_ste * delta_output)

        return DynamicLayerOutput(
            hidden_states=final_hidden_states,
            present_key_value=present_key_value,
            attention_weights=attn_weights,
            avg_ce_proportion=avg_ce_proportion,
            avg_cu_proportion=avg_cu_proportion,
            combined_gating_signal=combined_gating_signal,
            gate_vector=gate_vec_binary,
            prior_loss=decision_output.prior_loss,
            router_beta_ce=router_beta_ce,
            router_beta_cu=router_beta_cu,
            router_cu_detection_multiplier=router_cu_multiplier,
            router_ce_criterion_offset=router_ce_offset,
        )