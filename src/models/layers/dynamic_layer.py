import torch
import torch.nn as nn
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from ..blocks.qwen_block import Qwen2Block
from ..blocks.vpr_router import VPRRouter
from ..qwen.modeling_outputs import DecisionLayerOutput, DynamicLayerOutput


class DynamicLayer(nn.Module):
    """
    Implements the 'Dynamic Sub-Layer' for the VPR architecture using a
    numerically stable, fully batched approach.
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
        attention_mask: torch.Tensor | None = None, # Expects 4D causal mask
        position_ids: torch.LongTensor | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> DynamicLayerOutput:
        
        # 1. --- ROUTING ---
        (
            gate_vec_binary, s_ce_stats, s_cu_stats, g_cont_stats,
            _, _, combined_gating_signal, beta_ce, beta_cu,
            cu_multiplier, ce_offset,
        ) = self.vpr_router(
            original_input_to_block=decision_output.vpr_signal_original_input,
            posterior_full_path_output=decision_output.vpr_signal_posterior_output,
            prior_hidden_states=decision_output.vpr_signal_prior_hidden_states,
            capacity_gamma=self.config.capacity_gamma,
            is_training=self.training,
        )
        
        # 2. --- GATHER ---
        batch_indices, token_indices = gate_vec_binary.nonzero(as_tuple=True)

        if batch_indices.numel() == 0:
            return DynamicLayerOutput(
                hidden_states=hidden_states, present_key_value=None, attention_weights=None,
                s_ce_stats=s_ce_stats, s_cu_stats=s_cu_stats, g_cont_stats=g_cont_stats,
                combined_gating_signal=combined_gating_signal, gate_vector=gate_vec_binary,
                prior_loss=decision_output.prior_loss, router_beta_ce=beta_ce, router_beta_cu=beta_cu,
                router_cu_detection_multiplier=cu_multiplier, router_ce_criterion_offset=ce_offset,
            )

        selected_tokens = hidden_states[batch_indices, token_indices]
        continuous_signal_selected = combined_gating_signal[batch_indices, token_indices]

        # 3. --- PROCESS ---
        num_selected_tokens = selected_tokens.shape[0]
        selected_tokens_batched = selected_tokens.unsqueeze(0)
        
        processing_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=None, input_shape=(1, num_selected_tokens),
            inputs_embeds=selected_tokens_batched, past_key_values_length=0
        )
        
        selected_pos_ids = position_ids[batch_indices, token_indices].unsqueeze(0) if position_ids is not None else None

        block_outputs = self.block(
            hidden_states=selected_tokens_batched, attention_mask=processing_attention_mask,
            position_ids=selected_pos_ids, use_cache=use_cache, **kwargs,
        )
        processed_tokens = block_outputs[0].squeeze(0)

        # 4. --- SCATTER & APPLY STE ---
        final_hidden_states = hidden_states.clone()
        delta_output = processed_tokens - selected_tokens
        scaled_delta = delta_output * continuous_signal_selected.unsqueeze(-1)
        updated_selected_states = selected_tokens + scaled_delta

        # --- START OF FIX ---
        # Explicitly cast the source tensor to match the destination's dtype before scattering.
        final_hidden_states[batch_indices, token_indices] = updated_selected_states.to(final_hidden_states.dtype)
        # --- END OF FIX ---

        return DynamicLayerOutput(
            hidden_states=final_hidden_states,
            present_key_value=block_outputs[1] if use_cache and len(block_outputs) > 1 else None,
            attention_weights=None, 
            s_ce_stats=s_ce_stats, s_cu_stats=s_cu_stats, g_cont_stats=g_cont_stats,
            combined_gating_signal=combined_gating_signal, gate_vector=gate_vec_binary,
            prior_loss=decision_output.prior_loss, router_beta_ce=beta_ce, router_beta_cu=beta_cu,
            router_cu_detection_multiplier=cu_multiplier, router_ce_criterion_offset=ce_offset,
        )