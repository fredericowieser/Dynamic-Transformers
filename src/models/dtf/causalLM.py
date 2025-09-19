import logging
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
        self.causal_loss_weight = getattr(config, 'causal_loss_weight')
        self._setup_layers()

        # FIX: Freeze main transformer blocks if configured
        if getattr(config, 'freeze_base_model', False):
            self.freeze_main_transformer_blocks()

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

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        B, T, D = hidden_states.shape

        # Always prepare a 4D causal attention mask
        causal_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (B, T), hidden_states, past_key_values_length
        )

        # Setup position ids
        if position_ids is None:
            position_ids = torch.arange(T, device=hidden_states.device).unsqueeze(0).expand(B, -1)

        # Prepare attention mask
        if attention_mask is not None:
            current_attention_mask = attention_mask
            if current_attention_mask is not None:
                if self.config.attn_implementation != "flash_attention_2":
                    current_attention_mask = _prepare_4d_causal_attention_mask(
                        current_attention_mask, (B, T), hidden_states, 0
                    )
                else:
                    # Temporarily set attention_mask to None for Flash Attention 2 debugging
                    current_attention_mask = None

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

        # Add prior loss (handled in training loop based on schedule)
        # if loss is not None and total_prior_loss > 0:
        #     loss = loss + self.prior_loss_weight * total_prior_loss

        # Calculate averaged router stats
        averaged_router_stats = {}
        num_dynamic_layers = len(self.layers) // 2 # Assuming alternating Decision and Dynamic layers
        if num_dynamic_layers > 0:
            for k, v in total_router_stats.items():
                averaged_router_stats[k] = v / num_dynamic_layers

        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "prior_loss": total_prior_loss,
            "router_stats": averaged_router_stats,
        }

    def copy_weights_from_pretrained(self, pretrained_model):
        """Copy weights from a pretrained Qwen2 model to the DTF model.

        This method copies weights for shared components (embeddings, norms, LM head)
        and for the Qwen2Attention and Qwen2MLP parts within Decision/Dynamic layers.
        PriorFFN and DTFRouter are left with their random initialization.
        """
        super().copy_weights_from_pretrained(pretrained_model)
        logging.info("Starting weight copying from pretrained model to DTF model.")

        # Copy weights for each layer
        for i, layer in enumerate(self.layers):
            # Determine the corresponding pretrained layer index
            # Both Decision and Dynamic layers map to the same original Qwen2 layer for weight copying
            pretrained_layer_idx = i
            pretrained_layer = pretrained_model.model.layers[pretrained_layer_idx]

            logging.info(
                f"  DTF Layer {i} ({type(layer).__name__}) will copy weights from "
                f"Pretrained Layer {pretrained_layer_idx} ({type(pretrained_layer).__name__})."
            )

            if isinstance(layer, DTFDecisionLayer):
                # Decision layer contains a standard Qwen2DecoderLayer's components
                # and a PriorFFN (which should be randomly initialized).
                layer.block.self_attn.load_state_dict(pretrained_layer.self_attn.state_dict())
                logging.info(f"    Copied self_attn weights for DTF Layer {i}.")
                layer.block.mlp.load_state_dict(pretrained_layer.mlp.state_dict())
                logging.info(f"    Copied mlp weights for DTF Layer {i}.")
                layer.block.input_layernorm.load_state_dict(pretrained_layer.input_layernorm.state_dict())
                logging.info(f"    Copied input_layernorm weights for DTF Layer {i}.")
                layer.block.post_attention_layernorm.load_state_dict(pretrained_layer.post_attention_layernorm.state_dict())
                logging.info(f"    Copied post_attention_layernorm weights for DTF Layer {i}.")

            elif isinstance(layer, DTFDynamicLayer):
                # Dynamic layer contains a second standard Qwen2DecoderLayer's components
                # and a DTFRouter (which should be randomly initialized).
                layer.block.self_attn.load_state_dict(pretrained_layer.self_attn.state_dict())
                logging.info(f"    Copied self_attn weights for DTF Layer {i}.")
                layer.block.mlp.load_state_dict(pretrained_layer.mlp.state_dict())
                logging.info(f"    Copied mlp weights for DTF Layer {i}.")
                layer.block.input_layernorm.load_state_dict(pretrained_layer.input_layernorm.state_dict())
                logging.info(f"    Copied input_layernorm weights for DTF Layer {i}.")
                layer.block.post_attention_layernorm.load_state_dict(pretrained_layer.post_attention_layernorm.state_dict())
                logging.info(f"    Copied post_attention_layernorm weights for DTF Layer {i}.")
        logging.info("Finished copying weights to DTF model.")

    def get_trainable_parameters(self) -> List[Dict[str, Any]]:
        """Returns parameter groups for differential learning rates.

        Groups parameters into: base model, PriorFFN, and Predictive Router.
        Only includes parameters where requires_grad is True.
        """
        base_model_params = []
        prior_ffn_params = []
        router_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Check for PriorFFN parameters
            if "prior_network" in name:
                prior_ffn_params.append(param)
            # Check for Router parameters
            elif "router" in name:
                router_params.append(param)
            # All other trainable parameters go to base_model_params
            else:
                base_model_params.append(param)

        # Define learning rate scales as per Feature-Spec.md
        # These scales will be multiplied by the base_lr from the config.
        # The actual base_lr for each group is defined in the config.
        # Here we just provide the relative scales.
        param_groups = []
        if base_model_params:
            param_groups.append({
                'params': base_model_params,
                'lr_scale': getattr(self.config, 'base_model_lr_scale', 1.0),
                'name': 'base_model'
            })
        if prior_ffn_params:
            param_groups.append({
                'params': prior_ffn_params,
                'lr_scale': getattr(self.config, 'prior_ffn_lr_scale', 1.0),
                'name': 'prior_ffn'
            })
        if router_params:
            param_groups.append({
                'params': router_params,
                'lr_scale': getattr(self.config, 'router_lr_scale', 1.0),
                'name': 'predictive_router'
            })

        return param_groups