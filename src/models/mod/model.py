"""Mixture of Depths (MoD) model implementation using Qwen2 architecture."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from ..base.dynamic_model import BaseDynamicModel
from ..base.router import BaseRouter


class MoDRouter(BaseRouter):
    """Router for MoD token selection based on learned importance scores."""

    def __init__(self, config, layer_idx: int):
        # MoD uses 12.5% capacity by default
        capacity = getattr(config, 'mod_capacity', 0.125)
        super().__init__(capacity)

        self.layer_idx = layer_idx

        # Simple linear router as per MoD paper: r_i = w^T * x_i
        self.router = nn.Linear(config.hidden_size, 1, bias=False)

        # Auxiliary loss weight for load balancing
        self.aux_loss_weight = getattr(config, 'mod_aux_loss_weight', 0.01)

    def compute_routing_scores(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """Compute importance scores for each token.

        Args:
            hidden_states: [B, T, D]

        Returns:
            scores: Router scores [B, T]
            aux_loss: Load balancing loss
            stats: Routing statistics
        """
        B, T, D = hidden_states.shape

        # Compute routing scores: r_i = w^T * x_i
        router_logits = self.router(hidden_states).squeeze(-1)  # [B, T]

        # Compute auxiliary loss for load balancing during training
        aux_loss = None
        if self.training:
            # Encourage uniform routing across tokens
            k = max(1, int(T * self.capacity))
            target_load = k / T  # Expected fraction of tokens selected

            # Actual load (using sigmoid as soft selection)
            actual_load = torch.sigmoid(router_logits).mean()

            # MSE loss between actual and target load
            aux_loss = self.aux_loss_weight * ((actual_load - target_load) ** 2)

        stats = {
            'layer_idx': self.layer_idx,
            'capacity': self.capacity,
            'mean_score': router_logits.mean().item(),
            'std_score': router_logits.std().item(),
        }

        return router_logits, aux_loss, stats


class MoDLayer(nn.Module):
    """MoD layer that routes tokens to computation or residual connection."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Ensure config has attention implementation
        if not hasattr(config, '_attn_implementation'):
            config._attn_implementation = 'eager'

        # Standard Qwen2 decoder block
        self.block = Qwen2DecoderLayer(config, layer_idx)

        # MoD router
        self.router = MoDRouter(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict, Optional[Tuple], Optional[torch.Tensor]]:
        """Forward pass with conditional computation.

        Selected tokens go through the transformer block,
        others bypass via residual connection.
        """
        B, T, D = hidden_states.shape

        # Get routing decision
        scores, aux_loss, stats = self.router.compute_routing_scores(hidden_states)

        # Select top-k tokens
        selected_hidden, batch_idx, token_idx, selected_scores = self.router.select_tokens(
            scores, hidden_states
        )

        # Track statistics
        stats['selected_tokens'] = batch_idx.numel()
        stats['total_tokens'] = B * T

        # If no tokens selected, skip computation
        if batch_idx.numel() == 0:
            return (
                hidden_states,
                aux_loss,
                stats,
                past_key_values if use_cache else None,
                None
            )

        # Reshape for processing (add batch dimension)
        num_selected = selected_hidden.shape[0]
        selected_hidden = selected_hidden.unsqueeze(0)  # [1, num_selected, D]

        # Create attention mask for selected tokens if needed
        selected_attn_mask = None
        if attention_mask is not None and num_selected > 0:
            selected_attn_mask = _prepare_4d_causal_attention_mask(
                None, (1, num_selected), selected_hidden, 0
            )

        # Gather position information for selected tokens
        selected_pos_ids = None
        if position_ids is not None and num_selected > 0:
            pos_2d = position_ids.reshape(-1)
            flat_idx = batch_idx * T + token_idx
            selected_pos_ids = pos_2d[flat_idx].unsqueeze(0)  # [1, num_selected]

        # Gather position embeddings for selected tokens
        selected_pos_emb = None
        if position_embeddings is not None and num_selected > 0:
            cos, sin = position_embeddings
            gathered_cos = cos[batch_idx, token_idx].unsqueeze(0)  # [1, num_selected, head_dim]
            gathered_sin = sin[batch_idx, token_idx].unsqueeze(0)
            selected_pos_emb = (gathered_cos, gathered_sin)

        # Process selected tokens through transformer block
        layer_outputs = self.block(
            selected_hidden,
            attention_mask=selected_attn_mask,
            position_ids=selected_pos_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            position_embeddings=selected_pos_emb,
        )

        # Handle different return formats from Qwen2DecoderLayer
        if isinstance(layer_outputs, tuple):
            processed = layer_outputs[0].squeeze(0)  # [num_selected, D]
            cache = layer_outputs[1] if len(layer_outputs) > 1 else None
            attn_weights = layer_outputs[2] if len(layer_outputs) > 2 else None
        else:
            processed = layer_outputs.squeeze(0)  # [num_selected, D]
            cache = None
            attn_weights = None

        # Apply gating based on router scores (soft selection)
        gate_values = torch.sigmoid(selected_scores).unsqueeze(-1)  # [num_selected, 1]

        # Gated update: mix processed and original based on gate
        selected_hidden_flat = selected_hidden.squeeze(0)  # [num_selected, D]
        gated_processed = gate_values * processed + (1 - gate_values) * selected_hidden_flat

        # Scatter processed tokens back
        output = self.router.scatter_tokens(
            gated_processed,
            hidden_states,
            batch_idx,
            token_idx
        )

        return output, aux_loss, stats, cache, attn_weights


class MoDForCausalLM(BaseDynamicModel):
    """MoD (Mixture-of-Depths) model for causal language modeling."""

    def __init__(self, config):
        super().__init__(config)
        self.total_aux_loss_weight = getattr(config, 'mod_total_aux_loss_weight', 0.01)
        self._setup_layers()

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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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

        total_aux_loss = 0.0
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

        if not return_dict:
            outputs = (logits,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_attentions,)
            if use_cache:
                outputs += (next_decoder_cache,)
            return ((loss,) + outputs) if loss is not None else outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    def get_trainable_parameters(self):
        """Get parameter groups with different learning rates."""
        base_params = []
        router_params = []

        for name, param in self.named_parameters():
            if 'router' in name:
                router_params.append(param)
            else:
                base_params.append(param)

        groups = []
        if base_params:
            groups.append({'params': base_params, 'lr_scale': 1.0, 'name': 'base'})
        if router_params:
            groups.append({'params': router_params, 'lr_scale': 10.0, 'name': 'router'})

        return groups