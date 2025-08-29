import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM, Qwen2Model
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from .config import DynamicQwenConfig
from .modeling_outputs import DynamicCausalLMOutput, VPRCausalLMOutput
from ..layers.decision_layer import DecisionLayer
from ..layers.dynamic_layer import DynamicLayer
from ..layers.mod_layer import MoDLayer
from ..blocks.qwen_block import Qwen2Block
from ..utils.patching import populate_weights_from_source_layers

logger = logging.getLogger(__name__)


class DynamicQwenForCausalLM(Qwen2ForCausalLM):
    config_class = DynamicQwenConfig

    def __init__(self, config: DynamicQwenConfig):
        # We override __init__ completely to build the correct layer structure
        super(Qwen2ForCausalLM, self).__init__(config)

        # Build the model with dynamic layers from the start
        self.model = Qwen2Model(config) # This creates embed_tokens and norm
        
        # Replace the standard layers with our dynamic layer structure
        dynamic_layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.dynamic_architecture == "vpr":
                if i % 2 == 0:
                    dynamic_layers.append(DecisionLayer(config, layer_idx=i))
                else:
                    dynamic_layers.append(DynamicLayer(config, layer_idx=i))
            elif config.dynamic_architecture == "mod":
                if (i + 1) % 2 == 0:
                    dynamic_layers.append(MoDLayer(config, layer_idx=i))
                else:
                    dynamic_layers.append(Qwen2Block(config, layer_idx=i))
            else:
                raise ValueError(f"Unknown dynamic_architecture: '{config.dynamic_architecture}'")
        self.model.layers = dynamic_layers

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._freeze_main_transformer_blocks = getattr(config, "freeze_main_transformer_blocks", False)

        # Initialize weights and apply final processing
        self.post_init()
        # This needs to be called after post_init, which initializes all weights
        self._apply_main_block_freezing()

    def _apply_main_block_freezing(self):
        for layer in self.model.layers:
            block_to_freeze = None
            if hasattr(layer, 'block'):
                block_to_freeze = layer.block
            elif isinstance(layer, Qwen2Block):
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
        **kwargs,
    ) -> Union[Tuple, DynamicCausalLMOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # --- START OF FIX: More robustly disable KV Caching for VPR during generation ---
        # The gather-scatter logic in VPR is incompatible with the KV cache.
        # We detect generation by simply checking if a cache is passed.
        if self.config.dynamic_architecture == "vpr" and past_key_values is not None:
            use_cache = False
        # --- END OF FIX ---

        if self.config.dynamic_architecture == "mod":
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        batch_size, seq_length, _ = hidden_states.shape
        
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            ).unsqueeze(0)
        
        # --- START OF MODIFICATION ---
        # Ensure position_ids are expanded to match the batch size
        if position_ids.shape[0] != batch_size:
            position_ids = position_ids.expand(batch_size, -1)
        # --- END OF MODIFICATION ---

        causal_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        all_dynamic_layer_outputs = []
        next_past_key_values = [] if use_cache else None

        if self.config.dynamic_architecture == "vpr":
            for i in range(0, len(self.model.layers), 2):
                decision_layer = self.model.layers[i]
                dynamic_layer = self.model.layers[i+1]
                
                # --- START OF FIX: Handle past_key_values for each layer ---
                past_kv_decision = past_key_values[i] if past_key_values is not None else None
                past_kv_dynamic = past_key_values[i+1] if past_key_values is not None else None

                common_args = {
                    "attention_mask": causal_mask,
                    "position_ids": position_ids,
                    "use_cache": use_cache,
                    "output_attentions": output_attentions,
                    **kwargs,
                }

                decision_output = decision_layer(
                    hidden_states, past_key_value=past_kv_decision, **common_args
                )
                dynamic_output = dynamic_layer(
                    decision_output.hidden_states,
                    decision_output=decision_output,
                    past_key_value=past_kv_dynamic,
                    **common_args,
                )
                # --- END OF FIX ---

                hidden_states = dynamic_output.hidden_states
                all_dynamic_layer_outputs.append(dynamic_output)
                if use_cache:
                    next_past_key_values.extend([decision_output.present_key_value, dynamic_output.present_key_value])

        elif self.config.dynamic_architecture == "mod":
            for i, layer in enumerate(self.model.layers):
                # --- START OF FIX: Handle past_key_values for each layer ---
                past_kv = past_key_values[i] if past_key_values is not None else None
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs,
                )
                # --- END OF FIX ---
                hidden_states = layer_outputs[0]
                if use_cache:
                    next_past_key_values.append(layer_outputs[1])
        
        
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if self.config.dynamic_architecture == "vpr" and all_dynamic_layer_outputs:
            def aggregate_stats(outputs_list, key_name):
                # Gathers stats from each layer and computes the mean of each stat across layers
                stats = [o.__getattribute__(key_name) for o in outputs_list]
                return {
                    'mean': torch.stack([s['mean'] for s in stats]).mean(),
                    'std': torch.stack([s['std'] for s in stats]).mean(),
                    'min': torch.stack([s['min'] for s in stats]).mean(),
                    'max': torch.stack([s['max'] for s in stats]).mean(),
                }
            s_ce_stats_agg = aggregate_stats(all_dynamic_layer_outputs, 's_ce_stats')
            s_cu_stats_agg = aggregate_stats(all_dynamic_layer_outputs, 's_cu_stats')
            g_cont_stats_agg = aggregate_stats(all_dynamic_layer_outputs, 'g_cont_stats')
            def aggregate_router_param_stats(outputs_list, param_name):
                # Gathers a scalar parameter from each layer and computes its mean and std
                values = torch.tensor([o.__getattribute__(param_name) for o in outputs_list], device=outputs_list[0].hidden_states.device)
                return {
                    'mean': values.mean(),
                    'std': values.std()
                }
            beta_ce_stats_agg = aggregate_router_param_stats(all_dynamic_layer_outputs, 'router_beta_ce')
            beta_cu_stats_agg = aggregate_router_param_stats(all_dynamic_layer_outputs, 'router_beta_cu')
            cu_multiplier_stats_agg = aggregate_router_param_stats(all_dynamic_layer_outputs, 'router_cu_detection_multiplier')
            ce_offset_stats_agg = aggregate_router_param_stats(all_dynamic_layer_outputs, 'router_ce_criterion_offset')

        if not return_dict:
            return (logits,)

        if self.config.dynamic_architecture == "vpr":
            vpr_metrics_dict = {
                "prior_loss": torch.stack([o.prior_loss for o in all_dynamic_layer_outputs]).mean(),
                "gate_vectors_per_layer": [o.gate_vector for o in all_dynamic_layer_outputs],
                "s_ce_stats": s_ce_stats_agg,
                "s_cu_stats": s_cu_stats_agg,
                "g_cont_stats": g_cont_stats_agg,
                "router_beta_ce_stats": beta_ce_stats_agg,
                "router_beta_cu_stats": beta_cu_stats_agg,
                "router_cu_multiplier_stats": cu_multiplier_stats_agg,
                "router_ce_offset_stats": ce_offset_stats_agg,
            }

            return VPRCausalLMOutput(
                logits=logits,
                past_key_values=tuple(next_past_key_values) if use_cache else None,
                loss=None,  # Loss is calculated in the training script
                vpr_metrics=vpr_metrics_dict,
            )
        else:
            return DynamicCausalLMOutput(
                logits=logits,
                past_key_values=tuple(next_past_key_values) if use_cache else None,
            )


    @classmethod
    def from_vanilla_checkpoint(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Factory method to CONVERT a vanilla HF checkpoint into a dynamic one.
        This should only be called once at the beginning of training.
        """
        logger.info(f"Converting vanilla checkpoint '{pretrained_model_name_or_path}' to dynamic architecture.")
        model_cfg = kwargs.pop("model_cfg", {})
        config = DynamicQwenConfig.from_pretrained(pretrained_model_name_or_path, **model_cfg)
        
        # 1. Create our custom model. Its __init__ method builds the correct layer structure.
        custom_model = cls(config)

        # 2. Load the vanilla pre-trained model to get its weights.
        kwargs.pop('config', None)
        vanilla_model = Qwen2ForCausalLM.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        
        # 3. Copy weights from the vanilla model to our custom model.
        custom_model.model.embed_tokens.load_state_dict(vanilla_model.model.embed_tokens.state_dict())
        custom_model.model.norm.load_state_dict(vanilla_model.model.norm.state_dict())
        custom_model.lm_head.load_state_dict(vanilla_model.lm_head.state_dict())
        
        # Use the utility to populate weights for the transformer layers.
        populate_weights_from_source_layers(custom_model, vanilla_model.model.layers)
        
        del vanilla_model
        return custom_model