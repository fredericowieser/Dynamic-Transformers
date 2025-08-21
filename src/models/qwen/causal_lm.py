import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import DynamicQwenConfig
from .modeling_outputs import DynamicCausalLMOutput
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
        for layer in self.model.layers:
            # Main block is always under the 'block' attribute now
            if hasattr(layer, 'block'):
                for n, p in layer.block.named_parameters():
                    p.requires_grad = not self._freeze_main_transformer_blocks
            elif isinstance(layer, Qwen2Block): # For MoD layers that are just the block
                 for n, p in layer.named_parameters():
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
        **kwargs,
    ) -> Union[Tuple, DynamicCausalLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,  # Force return_dict to handle outputs consistently
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        if not return_dict:
            # This part is for compatibility if you ever need to return tuples
            # It will need to be adapted based on which metrics are available
            return (logits,) + outputs[1:]

        # Create a unified output object
        # Populate with metrics if they exist in the model's output
        return DynamicCausalLMOutput(
            logits=logits,
            past_key_values=outputs.past_key_values,
            attentions=outputs.attentions,
            prior_loss=getattr(outputs, "prior_loss", None),
            gate_vectors_per_layer=getattr(outputs, "gate_vectors", None),
            # ... populate all other metrics from the structured output ...
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Pop custom config from kwargs
        config_from_kwargs = kwargs.pop("config", None)
        model_cfg = kwargs.pop("model_cfg", {})

        # Create the correct config object
        if config_from_kwargs:
            config = DynamicQwenConfig.from_dict({**config_from_kwargs.to_dict(), **model_cfg})
        else:
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
        return custom_model