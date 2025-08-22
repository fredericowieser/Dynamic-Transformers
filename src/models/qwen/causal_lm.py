import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from .config import DynamicQwenConfig
from .modeling_outputs import DynamicCausalLMOutput, DecisionLayerOutput
from ..layers.decision_layer import DecisionLayer
from ..layers.dynamic_layer import DynamicLayer
from ..layers.mod_layer import MoDLayer
from ..blocks.qwen_block import Qwen2Block
from ..utils.patching import patch_and_populate_layers

logger = logging.getLogger(__name__)


class DynamicQwenForCausalLM(Qwen2ForCausalLM):
    """
    A Qwen2-based model extended with dynamic computation capabilities.

    This model can be configured to use one of two dynamic architectures:
    1.  **VPR (Variational Predictive Routing):** An architecture with alternating
        Decision and Dynamic layers that use a predictive mechanism for routing.
    2.  **MoD (Mixture-of-Depths):** An architecture that interleaves standard
        transformer blocks with MoD layers that process a fixed capacity of
        tokens, allowing others to bypass computation.

    The architecture is controlled by the `dynamic_architecture` attribute in the
    `DynamicQwenConfig`.
    """
    config_class = DynamicQwenConfig

    def __init__(self, config: DynamicQwenConfig):
        super().__init__(config)
        self._freeze_main_transformer_blocks = getattr(
            config, "freeze_main_transformer_blocks", False
        )
        # Instantiate the rotary embedding layer at the top level
        self.rotary_emb = Qwen2RotaryEmbedding(config)


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
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> DynamicCausalLMOutput:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        
        batch_size, seq_length = (input_ids.shape if input_ids is not None else inputs_embeds.shape[:2])

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[-2]
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            ).unsqueeze(0)

        # Prepare 4D attention mask
        _attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        
        # Create positional embeddings before the layer loop
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        all_dynamic_layer_outputs = []
        decision_output = None

        # Manually iterate through the custom layers
        for idx, decoder_layer in enumerate(self.model.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            layer_args = {
                "attention_mask": _attention_mask,
                "position_ids": position_ids,
                "past_key_value": past_key_value,
                "output_attentions": output_attentions,
                "use_cache": use_cache,
                "position_embeddings": position_embeddings,
            }

            if self.config.dynamic_architecture == "vpr":
                if isinstance(decoder_layer, DecisionLayer):
                    decision_output = decoder_layer(hidden_states, **layer_args)
                    hidden_states = decision_output.hidden_states
                    layer_outputs = (hidden_states, decision_output.present_key_value, decision_output.attention_weights)
                elif isinstance(decoder_layer, DynamicLayer):
                    if decision_output is None:
                        raise ValueError("DynamicLayer in VPR must be preceded by a DecisionLayer.")
                    dynamic_output = decoder_layer(hidden_states, decision_output=decision_output, **layer_args)
                    hidden_states = dynamic_output.hidden_states
                    layer_outputs = (hidden_states, dynamic_output.present_key_value, dynamic_output.attention_weights)
                    all_dynamic_layer_outputs.append(dynamic_output)
                else:
                    raise TypeError(f"Unexpected layer type {type(decoder_layer)} for VPR architecture.")
            else: # MoD or standard block logic
                layer_outputs = decoder_layer(hidden_states, **layer_args)
                hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
            if output_attentions:
                all_self_attns += (layer_outputs[2 if use_cache else 1],)

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # Aggregate metrics for VPR architecture
        prior_loss = avg_ce = avg_cu = cs_mean = beta_ce = beta_cu = cu_multi = ce_offset = None
        ce_per_layer = cu_per_layer = gate_vectors = None
        if self.config.dynamic_architecture == "vpr" and all_dynamic_layer_outputs:
            prior_loss = torch.stack([o.prior_loss for o in all_dynamic_layer_outputs]).mean()
            gate_vectors = [o.gate_vector for o in all_dynamic_layer_outputs]
            avg_ce = torch.stack([o.avg_ce_proportion for o in all_dynamic_layer_outputs]).mean()
            avg_cu = torch.stack([o.avg_cu_proportion for o in all_dynamic_layer_outputs]).mean()
            cs_mean = torch.stack([o.combined_gating_signal.mean() for o in all_dynamic_layer_outputs]).mean()
            ce_per_layer = [o.avg_ce_proportion for o in all_dynamic_layer_outputs]
            cu_per_layer = [o.avg_cu_proportion for o in all_dynamic_layer_outputs]
            beta_ce = torch.tensor([o.router_beta_ce for o in all_dynamic_layer_outputs]).mean()
            beta_cu = torch.tensor([o.router_beta_cu for o in all_dynamic_layer_outputs]).mean()
            cu_multi = torch.tensor([o.router_cu_detection_multiplier for o in all_dynamic_layer_outputs]).mean()
            ce_offset = torch.tensor([o.router_ce_criterion_offset for o in all_dynamic_layer_outputs]).mean()
        
        return DynamicCausalLMOutput(
            logits=logits,
            past_key_values=next_decoder_cache,
            attentions=all_self_attns,
            prior_loss=prior_loss,
            gate_vectors_per_layer=gate_vectors,
            avg_ce_proportion=avg_ce,
            avg_cu_proportion=avg_cu,
            combined_gating_signal_mean=cs_mean,
            ce_proportions_per_layer=ce_per_layer,
            cu_proportions_per_layer=cu_per_layer,
            avg_beta_ce=beta_ce,
            avg_beta_cu=beta_cu,
            avg_cu_detection_multiplier=cu_multi,
            avg_ce_criterion_offset=ce_offset
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model_cfg = kwargs.pop("model_cfg", {})
        config = DynamicQwenConfig.from_pretrained(pretrained_model_name_or_path, **model_cfg)

        base_hf_model = Qwen2ForCausalLM.from_pretrained(
            pretrained_model_name_or_path, config=config, *model_args, **kwargs
        )
        custom_model = cls(config)

        custom_model.model.embed_tokens = base_hf_model.model.embed_tokens
        custom_model.model.norm = base_hf_model.model.norm
        custom_model.lm_head = base_hf_model.lm_head

        patch_and_populate_layers(custom_model, config, base_hf_model.model.layers)
        custom_model._apply_main_block_freezing()

        del base_hf_model
        return custom_model