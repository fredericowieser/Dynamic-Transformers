from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast


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
    s_ce_stats: dict
    s_cu_stats: dict
    g_cont_stats: dict
    combined_gating_signal: torch.Tensor
    gate_vector: torch.Tensor
    prior_loss: torch.Tensor
    router_beta_ce: float
    router_beta_cu: float
    router_cu_detection_multiplier: float
    router_ce_criterion_offset: float


@dataclass
class VPRCausalLMOutput(CausalLMOutputWithPast):
    """
    Custom output for the VPR architecture that inherits from the standard
    CausalLMOutputWithPast to be compatible with PEFT wrappers.
    All custom VPR metrics are bundled into a single dictionary.
    """
    vpr_metrics: Optional[Dict[str, torch.Tensor]] = None

@dataclass
class DynamicCausalLMOutput(ModelOutput):
    """
    A general output class, now primarily for the MoD architecture.
    """
    logits: torch.Tensor
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None