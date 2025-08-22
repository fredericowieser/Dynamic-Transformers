# Filename: dynamic_layer.py

import torch
import torch.nn as nn

from ..blocks.qwen_block import Qwen2Block
from ..blocks.vpr_router import VPRRouter
from ..qwen.modeling_outputs import DecisionLayerOutput, DynamicLayerOutput


class DynamicLayer(nn.Module):
    """
    Implements the 'Dynamic Layer' for the VPR architecture, mimicking the
    Mixture-of-Depths (MoD) structural logic.

    This layer uses a VPRRouter to select a top-k capacity of tokens.
    Only these selected tokens are processed by the transformer block; all
    others are passed through via a residual connection. The continuous
    gating signal from the router is used to scale the block's update,
    analogous to the router weights in a standard MoD layer.
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

        # --- REFACTORED MoD-style FORWARD PASS ---

        # 1. Create a boolean mask for selected tokens.
        is_selected = gate_vec_binary.unsqueeze(-1).bool()

        # 2. To maintain a static graph, create a tensor of selected tokens
        #    and zero out the rest before passing to the block.
        selected_hidden_states = torch.where(
            is_selected, hidden_states, torch.zeros_like(hidden_states)
        )

        # 3. The block processes the full sequence, but non-selected tokens are zero vectors.
        block_outputs = self.block(selected_hidden_states, **kwargs)
        block_output = block_outputs[0]
        present_key_value = block_outputs[1]
        attn_weights = block_outputs[2] if len(block_outputs) > 2 else None
        
        # 4. Calculate the change (delta) introduced by the block relative to the original input.
        delta_output = block_output - hidden_states

        # 5. Scale the delta by the continuous VPR gating signal. This is analogous
        #    to scaling by router_weights in the MoDLayer.
        scaled_delta = delta_output * combined_gating_signal.unsqueeze(-1)

        # 6. Apply the scaled update only to the tokens selected by the binary gate.
        final_hidden_states = hidden_states + torch.where(
            is_selected,
            scaled_delta,
            torch.zeros_like(scaled_delta),
        )

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