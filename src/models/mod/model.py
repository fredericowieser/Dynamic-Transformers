import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..base.causal_lm import BaseDynamicCausalLM
from .layers import MoDLayer


class MoDForCausalLM(BaseDynamicCausalLM):
    """MoD (Mixture-of-Depths) model for causal language modeling."""

    def setup_dynamic_layers(self, config):
        """Setup MoD-specific layers."""
        layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layers.append(MoDLayer(config, i))
        self.model.layers = layers

    def copy_base_weights(self, base_model):
        """Copy weights from base model to MoD layers."""
        for i, layer in enumerate(self.model.layers):
            if i < len(base_model.model.layers):
                layer.block.load_state_dict(base_model.model.layers[i].state_dict())

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Embed tokens
        hidden_states = self.model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds

        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)

        # Process through layers
        all_hidden_states = () if output_hidden_states else None
        router_stats = {}

        for i, layer in enumerate(self.model.layers):
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states,)

            hidden_states, stats = layer(hidden_states, attention_mask, position_ids, **kwargs)

            if stats:
                router_stats[f"layer_{i}"] = stats

        # Final norm
        hidden_states = self.model.norm(hidden_states)

        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        # Compute logits
        logits = self.lm_head(hidden_states)

        # Compute loss
        loss = self.compute_loss(logits, labels)

        if not return_dict:
            output = (logits,) + (all_hidden_states,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )