import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Dict, Any

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from ..base.dynamic_model import BaseDynamicModel
from .layers import TDTFLayer


class TDTFForCausalLM(BaseDynamicModel):
    """TDTF (Temporal Dynamic Transformer) model for causal language modeling.

    Implements student-teacher framework with training-time predictive router
    and inference-time causal router for conditional computation.
    """

    def __init__(self, config):
        super().__init__(config)

        # Loss weights
        self.tpn_loss_weight = getattr(config, 'tpn_loss_weight')
        self.causal_loss_weight = getattr(config, 'causal_loss_weight')

        self._setup_layers()

        # FIX: Freeze main transformer blocks if configured
        if getattr(config, 'freeze_base_model', False):
            self.freeze_main_transformer_blocks()

    def _setup_layers(self):
        """Setup TDTF layers."""
        self.layers = nn.ModuleList()

        for i in range(self.config.num_hidden_layers):
            self.layers.append(TDTFLayer(self.config, i))

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
        """Forward pass with temporal dynamic routing."""

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

        total_tpn_loss = 0.0
        total_causal_loss = 0.0
        total_router_stats = {}

        for i, layer in enumerate(self.layers):
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            # Forward through TDTF layer
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
            )

            # Update hidden states
            hidden_states = layer_output['hidden_states']

            # Accumulate losses (training only)
            if self.training:
                if 'tpn_loss' in layer_output and layer_output['tpn_loss'] is not None:
                    total_tpn_loss += layer_output['tpn_loss']
                if 'causal_loss' in layer_output and layer_output['causal_loss'] is not None:
                    total_causal_loss += layer_output['causal_loss']

            # Accumulate router stats
            if 'router_stats' in layer_output:
                for k, v in layer_output['router_stats'].items():
                    if k not in total_router_stats:
                        total_router_stats[k] = []
                    total_router_stats[k].append(v)

            # Handle caching and attention outputs
            if use_cache and 'past_key_value' in layer_output:
                next_decoder_cache += (layer_output['past_key_value'],)

            if output_attentions and 'attention_weights' in layer_output:
                if layer_output['attention_weights'] is not None:
                    all_attentions += (layer_output['attention_weights'],)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Get logits
        logits = self.lm_head(hidden_states)

        # Compute loss
        loss = self.compute_loss(logits, labels)

        # Add auxiliary losses (training only)
        if loss is not None and self.training:
            if total_tpn_loss > 0:
                loss = loss + self.tpn_loss_weight * total_tpn_loss
            if total_causal_loss > 0:
                loss = loss + self.causal_loss_weight * total_causal_loss

        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "tpn_loss": total_tpn_loss,
            "causal_loss": total_causal_loss,
            "router_stats": total_router_stats,
        }

    def copy_weights_from_pretrained(self, pretrained_model):
        """Copy weights from a pretrained Qwen2 model to the TDTF model.

        This method copies weights for shared components (embeddings, norms, LM head)
        and for the Qwen2DecoderLayer parts within TDTFLayer.
        TDTFTransitionNetwork, TDTFPredictiveRouter, and TDTFCausalRouter are left
        with their random initialization.
        """
        super().copy_weights_from_pretrained(pretrained_model)

        log.info("Starting weight copying from pretrained model to TDTF model.")
        # Copy weights for each layer
        for i, layer in enumerate(self.layers):
            pretrained_layer = pretrained_model.model.layers[i] # Corresponding layer in pretrained model

            log.info(f"  TDTF Layer {i} ({type(layer).__name__}) will copy weights from Pretrained Layer {i} ({type(pretrained_layer).__name__}).")

            if isinstance(layer, TDTFLayer):
                # TDTFLayer contains a standard Qwen2DecoderLayer as its 'transformer_block'
                layer.transformer_block.load_state_dict(pretrained_layer.state_dict())
        log.info("Finished copying weights to TDTF model.")

    def get_trainable_parameters(self) -> List[Dict[str, Any]]:
        """Returns parameter groups for differential learning rates.

        Groups parameters into: base model, TPN, and Predictive Router.
        Only includes parameters where requires_grad is True.
        """
        base_model_params = []
        tpn_params = []
        predictive_router_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Check for TPN parameters
            if "transition_network" in name:
                tpn_params.append(param)
            # Check for Predictive Router parameters
            elif "predictive_router" in name:
                predictive_router_params.append(param)
            # All other trainable parameters go to base_model_params
            else:
                base_model_params.append(param)

        # Define learning rate scales
        param_groups = []
        if base_model_params:
            param_groups.append({
                'params': base_model_params,
                'lr_scale': getattr(self.config, 'base_model_lr_scale', 1.0),
                'name': 'base_model'
            })
        if tpn_params:
            param_groups.append({
                'params': tpn_params,
                'lr_scale': getattr(self.config, 'tpn_lr_scale', 1.0),
                'name': 'tpn'
            })
        if predictive_router_params:
            param_groups.append({
                'params': predictive_router_params,
                'lr_scale': getattr(self.config, 'predictive_router_lr_scale', 1.0),
                'name': 'predictive_router'
            })

        return param_groups
