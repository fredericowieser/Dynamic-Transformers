import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..base.causal_lm import BaseDynamicCausalLM
from .layers import DTFDecisionLayer, DTFDynamicLayer


class DTFForCausalLM(BaseDynamicCausalLM):
    """DTF (Dynamic Transformer) model for causal language modeling."""

    def setup_dynamic_layers(self, config):
        """Setup DTF-specific layers."""
        layers = nn.ModuleList()
        for i in range(0, config.num_hidden_layers, 2):
            layers.append(DTFDecisionLayer(config, i))
            if i + 1 < config.num_hidden_layers:
                layers.append(DTFDynamicLayer(config, i + 1))
        self.model.layers = layers
        self.prior_weight = getattr(config, 'prior_loss_weight', 0.05)

    def copy_base_weights(self, base_model):
        """Copy weights from base model to DTF layers."""
        base_idx = 0
        for layer in self.model.layers:
            if isinstance(layer, DTFDecisionLayer) and base_idx < len(base_model.model.layers):
                layer.block.load_state_dict(base_model.model.layers[base_idx].state_dict())
                base_idx += 1
            elif isinstance(layer, DTFDynamicLayer) and base_idx < len(base_model.model.layers):
                layer.block.load_state_dict(base_model.model.layers[base_idx].state_dict())
                base_idx += 1

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
        total_prior_loss = 0
        router_stats = {}

        for i, layer in enumerate(self.model.layers):
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states,)

            if isinstance(layer, DTFDecisionLayer):
                decision_output = layer(hidden_states, attention_mask, position_ids)
                hidden_states = decision_output["posterior"]

                if decision_output["prior_loss"] is not None:
                    total_prior_loss += decision_output["prior_loss"]

                # Process next dynamic layer if it exists
                if i + 1 < len(self.model.layers) and isinstance(self.model.layers[i + 1], DTFDynamicLayer):
                    hidden_states, stats = self.model.layers[i + 1](
                        hidden_states, decision_output, position_ids
                    )
                    router_stats[f"layer_{i+1}"] = stats

            elif isinstance(layer, DTFDynamicLayer):
                # Already processed with decision layer
                continue

        # Final norm
        hidden_states = self.model.norm(hidden_states)

        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        # Compute logits
        logits = self.lm_head(hidden_states)

        # Compute loss
        loss = self.compute_loss(logits, labels)
        if loss is not None and total_prior_loss > 0:
            loss += self.prior_weight * total_prior_loss

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