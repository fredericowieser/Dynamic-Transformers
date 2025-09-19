import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Dict, Any

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from ..base.dynamic_model import BaseDynamicModel
from .layers import MoDLayer


class MoDForCausalLM(BaseDynamicModel):
    """MoD (Mixture-of-Depths) model for causal language modeling."""

    def __init__(self, config):
        super().__init__(config)
        self.total_aux_loss_weight = getattr(config, 'mod_aux_loss_weight')
        self.causal_loss_weight = getattr(config, 'causal_loss_weight')
        self._setup_layers()

        # FIX: Freeze main transformer blocks if configured
        if getattr(config, 'freeze_base_model', False):
            self.freeze_main_transformer_blocks()

    def _setup_layers(self):
        """Setup MoD layers - apply to every other layer as per paper."""
        self.layers = nn.ModuleList()

        for i in range(self.config.num_hidden_layers):
            # Apply MoD to every other layer (or based on config)
            if i % 2 == 1:  # Apply to odd layers
                self.layers.append(MoDLayer(self.config, i))
            else:
                # Standard Qwen2 layer for even layers
                if not hasattr(self.config, '_attn_implementation'):
                    self.config._attn_implementation = 'eager'
                self.layers.append(Qwen2DecoderLayer(self.config, i))

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
        """Forward pass with mixed depth computation."""

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

        total_aux_loss = torch.tensor(0.0, device=hidden_states.device)
        total_router_stats = []

        for i, layer in enumerate(self.layers):
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            # Process through layer (MoD or standard)
            if isinstance(layer, MoDLayer):
                # MoD layer with routing
                hidden_states, aux_loss, stats, cache, attn_weights = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    position_embeddings=position_embeddings,
                )

                if aux_loss is not None:
                    total_aux_loss += aux_loss

                total_router_stats.append(stats)

            else:
                # Standard Qwen2 layer
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    position_embeddings=position_embeddings,
                )

                # Handle different return formats
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                    cache = layer_outputs[1] if use_cache else None
                    attn_weights = layer_outputs[2] if output_attentions else None
                else:
                    hidden_states = layer_outputs
                    cache = None
                    attn_weights = None

            if use_cache:
                next_decoder_cache += (cache,)

            if output_attentions and attn_weights is not None:
                all_attentions += (attn_weights,)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Get logits
        logits = self.lm_head(hidden_states)

        # Compute loss
        loss = self.compute_loss(logits, labels)

        # Add auxiliary loss
        if loss is not None and total_aux_loss > 0:
            loss = loss + self.total_aux_loss_weight * total_aux_loss

        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "aux_loss": total_aux_loss,
            "router_stats": total_router_stats,
        }

    def copy_weights_from_pretrained(self, pretrained_model):
        """Copy weights from a pretrained Qwen2 model to the MoD model.

        This method copies weights for shared components (embeddings, norms, LM head)
        and for the Qwen2DecoderLayer parts within MoDLayer and standard Qwen2DecoderLayers.
        MoDRouter and CausalMoDRouter are left with their random initialization.
        """
        super().copy_weights_from_pretrained(pretrained_model)

        # Copy weights for each layer
        for i, layer in enumerate(self.layers):
            pretrained_layer = pretrained_model.model.layers[i] # Corresponding layer in pretrained model

            if isinstance(layer, MoDLayer):
                # MoDLayer contains a standard Qwen2DecoderLayer as its 'block'
                layer.block.load_state_dict(pretrained_layer.state_dict())
            elif isinstance(layer, Qwen2DecoderLayer):
                # Standard Qwen2DecoderLayer
                layer.load_state_dict(pretrained_layer.state_dict())