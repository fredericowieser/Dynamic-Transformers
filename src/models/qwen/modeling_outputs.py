from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers.modeling_outputs import ModelOutput


@dataclass
class DecisionLayerOutput:
    """
    Structured output for a VPR Decision Layer.
    """
    hidden_states: torch.Tensor
    vpr_signal_original_input: torch.Tensor
    vpr_signal_posterior_output: torch.Tensor
    vpr_signal_prior_hidden_states: torch.Tensor
    present_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]
    attention_weights: Optional[torch.Tensor]
    prior_loss: torch.Tensor


@dataclass
class DynamicLayerOutput:
    """
    Structured output for a VPR Dynamic Layer.
    """
    hidden_states: torch.Tensor
    present_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]
    attention_weights: Optional[torch.Tensor]
    avg_ce_proportion: torch.Tensor
    avg_cu_proportion: torch.Tensor
    combined_gating_signal: torch.Tensor
    gate_vector: torch.Tensor
    prior_loss: torch.Tensor
    router_beta_ce: float
    router_beta_cu: float
    router_cu_detection_multiplier: float
    router_ce_criterion_offset: float


@dataclass
class DynamicCausalLMOutput(ModelOutput):
    """
    Structured output for the DynamicQwenForCausalLM model,
    unifying outputs for both VPR and MoD architectures.
    """
    logits: torch.Tensor
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None

    # --- VPR Specific Metrics ---
    prior_loss: Optional[torch.Tensor] = None
    avg_ce_proportion: Optional[torch.Tensor] = None
    avg_cu_proportion: Optional[torch.Tensor] = None
    combined_gating_signal_mean: Optional[torch.Tensor] = None
    ce_proportions_per_layer: Optional[List[torch.Tensor]] = None
    cu_proportions_per_layer: Optional[List[torch.Tensor]] = None
    avg_beta_ce: Optional[torch.Tensor] = None
    avg_beta_cu: Optional[torch.Tensor] = None
    avg_cu_detection_multiplier: Optional[torch.Tensor] = None
    avg_ce_criterion_offset: Optional[torch.Tensor] = None

    # --- Shared Metrics ---
    gate_vectors_per_layer: Optional[List[torch.Tensor]] = None