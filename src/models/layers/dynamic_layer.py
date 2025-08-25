import torch
import torch.nn as nn

from ..blocks.qwen_block import Qwen2Block
from ..blocks.vpr_router import VPRRouter
from ..qwen.modeling_outputs import DecisionLayerOutput, DynamicLayerOutput


class DynamicLayer(nn.Module):
    """
    Implements the 'Dynamic Sub-Layer' for the VPR architecture using a
    numerically stable, batch-iterative approach.
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
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> DynamicLayerOutput:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        (
            gate_vec_binary,
            s_ce_stats,
            s_cu_stats,
            g_cont_stats,
            _, _,
            combined_gating_signal,
            beta_ce, beta_cu,
            cu_multiplier, ce_offset,
        ) = self.vpr_router(
            original_input_to_block=decision_output.vpr_signal_original_input,
            posterior_full_path_output=decision_output.vpr_signal_posterior_output,
            prior_hidden_states=decision_output.vpr_signal_prior_hidden_states,
            capacity_gamma=self.config.capacity_gamma,
            is_training=self.training,
        )
        
        # Initialize the output as a copy for the residual path
        final_hidden_states = hidden_states.clone()
        present_key_value = None

        # Process each sequence in the batch individually for stability
        for i in range(batch_size):
            selected_indices = gate_vec_binary[i].nonzero().squeeze(-1)

            if selected_indices.numel() == 0:
                continue

            # Gather inputs for the selected tokens
            selected_tokens = hidden_states[i, selected_indices]
            selected_pos_ids = position_ids[i, selected_indices].unsqueeze(0) if position_ids is not None else None
            
            current_attention_mask = None
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    selected_attn_mask = attention_mask[i, :, selected_indices][:, :, selected_indices]
                    current_attention_mask = selected_attn_mask.unsqueeze(0)
                else:
                    current_attention_mask = attention_mask[i, selected_indices].unsqueeze(0)

            # Compute the block output only for the selected tokens
            block_outputs = self.block(
                hidden_states=selected_tokens.unsqueeze(0),
                attention_mask=current_attention_mask,
                position_ids=selected_pos_ids,
                use_cache=use_cache,
                **kwargs,
            )
            processed_tokens = block_outputs[0].squeeze(0)

            # The "delta" is the change introduced by the block
            delta_output = processed_tokens - selected_tokens
            
            # Straight-Through-Estimator logic applied only to the selected tokens
            # For gradients, use the continuous signal; for forward pass, use binary gate (implicitly 1 here)
            continuous_signal_selected = combined_gating_signal[i, selected_indices].unsqueeze(-1)
            
            # Re-scale delta by the continuous signal for backpropagation
            scaled_delta = delta_output * continuous_signal_selected
            
            # Apply the change back to the original input positions
            final_hidden_states[i, selected_indices] = selected_tokens + scaled_delta

            if use_cache and len(block_outputs) > 1:
                present_key_value = block_outputs[1]

        return DynamicLayerOutput(
            hidden_states=final_hidden_states,
            present_key_value=present_key_value,
            attention_weights=None,  # Not currently captured in this implementation
            s_ce_stats=s_ce_stats,
            s_cu_stats=s_cu_stats,
            g_cont_stats=g_cont_stats,
            combined_gating_signal=combined_gating_signal,
            gate_vector=gate_vec_binary,
            prior_loss=decision_output.prior_loss,
            router_beta_ce=beta_ce,
            router_beta_cu=beta_cu,
            router_cu_detection_multiplier=cu_multiplier,
            router_ce_criterion_offset=ce_offset,
        )