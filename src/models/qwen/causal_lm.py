import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import DynamicQwenConfig
from .modeling_outputs import DynamicCausalLMOutput, DecisionLayerOutput
from ..layers.decision_layer import DecisionLayer
from ..layers.dynamic_layer import DynamicLayer
from ..layers.mod_layer import MoDLayer
from ..blocks.qwen_block import Qwen2Block
from ..utils.patching import patch_and_populate_layers

logger = logging.getLogger(__name__)


class DynamicQwenForCausalLM(Qwen2ForCausalLM):
    config_class = DynamicQwenConfig

    def __init__(self, config: DynamicQwenConfig):
        super().__init__(config)
        self._freeze_main_transformer_blocks = getattr(
            config, "freeze_main_transformer_blocks", False
        )

    def _apply_main_block_freezing(self):
        """Applies or removes gradient requirements for main transformer blocks."""
        # This logic needs to be adapted to the new layer structure
        for layer in self.model.layers:
            block_to_freeze = None
            if hasattr(layer, 'block'): # For DecisionLayer, DynamicLayer, MoDLayer
                block_to_freeze = layer.block
            elif isinstance(layer, Qwen2Block): # For standard layers in MoD
                block_to_freeze = layer

            if block_to_freeze:
                for p in block_to_freeze.parameters():
                    p.requires_grad = not self._freeze_main_transformer_blocks


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        current_iter: int = 0,
    ) -> Union[Tuple, DynamicCausalLMOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        
        # Create position_ids if they are not provided
        if position_ids is None:
            past_key_values_length = 0
            if past_key_values is not None:
                past_key_values_length = past_key_values[0][0].shape[2]
            
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, hidden_states.shape[1] + past_key_values_length, dtype=torch.long, device=device
            ).unsqueeze(0)

        # Layer loop
        all_dynamic_layer_outputs = []
        next_past_key_values = [] if use_cache else None

        # VPR architecture alternates between Decision and Dynamic layers
        if self.config.dynamic_architecture == "vpr":
            for i in range(0, len(self.model.layers), 2):
                decision_layer = self.model.layers[i]
                dynamic_layer = self.model.layers[i+1]
                
                layer_args = {
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "use_cache": use_cache,
                    "output_attentions": output_attentions
                }

                decision_output = decision_layer(hidden_states, **layer_args)
                dynamic_output = dynamic_layer(decision_output.hidden_states, decision_output=decision_output, **layer_args)

                hidden_states = dynamic_output.hidden_states
                all_dynamic_layer_outputs.append(dynamic_output)
                if use_cache:
                    next_past_key_values.extend([decision_output.present_key_value, dynamic_output.present_key_value])

        # MoD architecture alternates between standard and MoD layers
        elif self.config.dynamic_architecture == "mod":
            for i, layer in enumerate(self.model.layers):
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=use_cache,
                    output_attentions=output_attentions
                )
                hidden_states = layer_outputs[0]
                if use_cache:
                    next_past_key_values.append(layer_outputs[1])
        
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if not return_dict:
            return (logits,)

        # Aggregate metrics for the trainer
        # This part should be adapted to handle both VPR and MoD outputs gracefully
        # For now, focusing on fixing the VPR path
        if self.config.dynamic_architecture == "vpr":
             return DynamicCausalLMOutput(
                logits=logits,
                past_key_values=tuple(next_past_key_values) if use_cache else None,
                prior_loss=torch.stack([o.prior_loss for o in all_dynamic_layer_outputs]).mean(),
                gate_vectors_per_layer=[o.gate_vector for o in all_dynamic_layer_outputs],
                avg_ce_proportion=torch.stack([o.avg_ce_proportion for o in all_dynamic_layer_outputs]).mean(),
                avg_cu_proportion=torch.stack([o.avg_cu_proportion for o in all_dynamic_layer_outputs]).mean(),
                combined_gating_signal_mean=torch.stack([o.combined_gating_signal.mean() for o in all_dynamic_layer_outputs]).mean(),
                avg_beta_ce=torch.tensor([o.router_beta_ce for o in all_dynamic_layer_outputs]).mean(),
                avg_beta_cu=torch.tensor([o.router_beta_cu for o in all_dynamic_layer_outputs]).mean(),
                avg_cu_detection_multiplier=torch.tensor([o.router_cu_detection_multiplier for o in all_dynamic_layer_outputs]).mean(),
                avg_ce_criterion_offset=torch.tensor([o.router_ce_criterion_offset for o in all_dynamic_layer_outputs]).mean(),
            )
        else:
            return DynamicCausalLMOutput(
                logits=logits,
                past_key_values=tuple(next_past_key_values) if use_cache else None,
            )


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Pop custom config from kwargs
        model_cfg = kwargs.pop("model_cfg", {})
        
        # Create the correct config object
        config = DynamicQwenConfig.from_pretrained(pretrained_model_name_or_path, **model_cfg)

        # Load the base HF model with the potentially modified config
        base_hf_model = Qwen2ForCausalLM.from_pretrained(
            pretrained_model_name_or_path, config=config, *model_args, **kwargs
        )

        # Create an instance of our custom model
        custom_model = cls(config)

        # Transfer core components
        custom_model.model.embed_tokens = base_hf_model.model.embed_tokens
        custom_model.model.norm = base_hf_model.model.norm
        custom_model.lm_head = base_hf_model.lm_head

        # Let the utility function handle layer patching
        patch_and_populate_layers(custom_model, config, base_hf_model.model.layers)

        # Apply freezing configuration
        custom_model._apply_main_block_freezing()

        del base_hf_model
        return custom_modeljjjjjjjjjjj