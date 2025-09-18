import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Dict, Any

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from ..base.dynamic_model import BaseDynamicModel
from .layers import DTFDecisionLayer, DTFDynamicLayer


class DTFForCausalLM(BaseDynamicModel):
    """DTF (Dynamic Transformer) model for causal language modeling.

    Alternates between decision layers and dynamic layers to implement
    surprise-based conditional computation.
    """

    def __init__(self, config):
        super().__init__(config)
        self.prior_loss_weight = getattr(config, 'prior_loss_weight')
        self.causal_loss_weight = getattr(config, 'causal_loss_weight')
        self._setup_layers()

    def _setup_layers(self):
        """Setup alternating decision and dynamic layers."""
        self.layers = nn.ModuleList()

        for i in range(0, self.config.num_hidden_layers, 2):
            # Add decision layer
            self.layers.append(DTFDecisionLayer(self.config, i))
            # Add dynamic layer if not the last
            if i + 1 < self.config.num_hidden_layers:
                self.layers.append(DTFDynamicLayer(self.config, i + 1))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass with surprise-based routing."""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        B, T, D = hidden_states.shape

        # Setup position ids
        if position_ids is None:
            position_ids = torch.arange(T, device=hidden_states.device).unsqueeze(0).expand(B, -1)

        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (B, T), hidden_states, 0
            )

        # Get rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        # Process through layers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        total_prior_loss = torch.tensor(0.0, device=hidden_states.device)
        total_causal_loss = torch.tensor(0.0, device=hidden_states.device)
        total_router_stats = {}
        decision_output = None

        for i, layer in enumerate(self.layers):
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            if isinstance(layer, DTFDecisionLayer):
                # Process decision layer
                decision_output = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    position_embeddings=position_embeddings,
                )

                # Update hidden states to posterior
                hidden_states = decision_output['posterior']

                # Accumulate prior loss
                if decision_output['prior_loss'] is not None:
                    total_prior_loss += decision_output['prior_loss']

                if use_cache:
                    next_decoder_cache += (decision_output['past_key_value'],)

                if output_attentions and decision_output['attention_weights'] is not None:
                    all_attentions += (decision_output['attention_weights'],)

            elif isinstance(layer, DTFDynamicLayer) and decision_output is not None:
                # Process dynamic layer using decision outputs
                hidden_states, aux_loss, stats, cache, attn_weights = layer(
                    hidden_states,
                    decision_output,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    position_embeddings=position_embeddings,
                )

                # Accumulate causal loss
                if aux_loss is not None:
                    total_causal_loss += aux_loss

                # Aggregate stats
                for k, v in stats.items():
                    if k not in total_router_stats:
                        total_router_stats[k] = 0
                    total_router_stats[k] += v

                if use_cache and cache is not None:
                    next_decoder_cache += (cache,)

                if output_attentions and attn_weights is not None:
                    all_attentions += (attn_weights,)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Get logits
        logits = self.lm_head(hidden_states)

        # Compute loss
        loss = self.compute_loss(logits, labels)

        # Add prior loss
        if loss is not None and total_prior_loss > 0:
            loss = loss + self.prior_loss_weight * total_prior_loss

        # Add causal loss
        if loss is not None and total_causal_loss > 0:
            loss = loss + self.causal_loss_weight * total_causal_loss

        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "prior_loss": total_prior_loss,
            "causal_loss": total_causal_loss,
            "router_stats": total_router_stats,
        }