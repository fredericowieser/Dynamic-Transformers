"""Dynamic Transformer (DTF) model implementation using Qwen2 architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm

from ..base.dynamic_model import BaseDynamicModel
from ..base.router import BaseRouter


class DTFPriorNetwork(nn.Module):
    """Lightweight network for computing prior predictions.

    Implements the change hypothesis by predicting the posterior state
    using minimal computation (SwiGLU with reduced intermediate dim).
    """

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        # Use 25% of hidden size for intermediate layer as per paper
        intermediate_size = int(hidden_size * getattr(config, 'prior_ffn_intermediate_size_factor', 0.25))

        self.norm = Qwen2RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply lightweight transformation to predict posterior state."""
        # Residual connection with normalized FFN
        return x + self.down_proj(self.act(self.up_proj(self.norm(x))))


class DTFRouter(BaseRouter):
    """Predictive router for DTF token selection based on surprise metrics.

    Implements soft VPR criteria using expected (CE) and unexpected (CU) change
    to determine which tokens need computational updates.
    """

    def __init__(self, config, layer_idx: int):
        # DTF uses 12.5% capacity by default
        capacity = getattr(config, 'dtf_capacity', 0.125)
        super().__init__(capacity)

        self.layer_idx = layer_idx

        # Learnable parameters for routing criteria
        self.beta_ce = nn.Parameter(torch.tensor(getattr(config, 'beta_ce_init', -0.5)))
        self.beta_cu = nn.Parameter(torch.tensor(getattr(config, 'beta_cu_init', -0.8)))
        self.cu_detection_multiplier = nn.Parameter(
            torch.tensor(getattr(config, 'cu_detection_multiplier_init', 1.2))
        )
        self.ce_criterion_offset = nn.Parameter(
            torch.tensor(getattr(config, 'ce_criterion_offset_init', 1.0))
        )

    def _get_capacity(self, config) -> float:
        """DTF uses γ (gamma) for capacity, typically 50%."""
        return getattr(config, 'capacity_gamma', 0.5)

    def compute_routing_scores(
        self,
        hidden_states: torch.Tensor,
        original: torch.Tensor,
        posterior: torch.Tensor,
        prior: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Compute surprise-based routing scores.

        Args:
            hidden_states: Current hidden states (not used, for API consistency)
            original: Original input state (static hypothesis)
            posterior: Output from full transformer block (ground truth)
            prior: Prediction from prior network (change hypothesis)

        Returns:
            scores: Combined CE + CU routing scores
            aux_loss: None (prior loss computed separately)
            stats: Routing statistics
        """
        B, T, D = original.shape

        # Compute surprise metrics (MSE as proxy for KL divergence)
        # Static surprise: error of assuming no change
        cu = (original - posterior).norm(dim=-1)  # [B, T]
        # Change surprise: error of prior prediction
        ce = (posterior - prior).norm(dim=-1)  # [B, T]

        # Compute routing criteria with learnable parameters
        # CE criterion: Is prior prediction better than static?
        cu_criterion = self.beta_cu * cu
        ce_criterion = self.beta_ce * (ce + self.ce_criterion_offset)

        # Combined routing score
        scores = cu_criterion + ce_criterion  # [B, T]

        # Compute gating signal for soft selection
        gate_signal = torch.sigmoid(scores)

        stats = {
            'layer_idx': self.layer_idx,
            'capacity': self.capacity,
            'avg_cu': cu.mean().item(),
            'avg_ce': ce.mean().item(),
            'avg_gate': gate_signal.mean().item(),
        }

        return scores, None, stats


class DTFDecisionLayer(nn.Module):
    """Decision layer that computes the three states needed for routing.

    Processes input through both a standard transformer block (posterior)
    and a lightweight prior network, providing all states for routing decisions.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Ensure config has attention implementation
        if not hasattr(config, '_attn_implementation'):
            config._attn_implementation = 'eager'

        # Standard transformer block for posterior
        self.block = Qwen2DecoderLayer(config, layer_idx)
        # Lightweight prior network for change prediction
        self.prior_network = DTFPriorNetwork(config)

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
    ) -> Dict[str, Any]:
        """Compute original, posterior, and prior states.

        Returns dictionary containing all three states plus auxiliary loss.
        """
        # Store original input
        original = hidden_states

        # Process through standard transformer layer to get posterior
        layer_outputs = self.block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
        )

        # Handle different return formats
        if isinstance(layer_outputs, tuple):
            posterior = layer_outputs[0]
            cache = layer_outputs[1] if len(layer_outputs) > 1 else None
            attn_weights = layer_outputs[2] if len(layer_outputs) > 2 else None
        else:
            posterior = layer_outputs
            cache = None
            attn_weights = None

        # Compute prior prediction (lightweight)
        prior = self.prior_network(hidden_states)

        # Compute prior loss for training (teach prior to predict posterior)
        prior_loss = None
        if self.training:
            prior_loss = F.mse_loss(prior, posterior.detach())

        return {
            'original': original,
            'posterior': posterior,
            'prior': prior,
            'prior_loss': prior_loss,
            'past_key_value': cache,
            'attention_weights': attn_weights,
        }


class DTFDynamicLayer(nn.Module):
    """Dynamic layer that processes selected tokens based on surprise.

    Uses routing decisions from decision layer to conditionally process
    tokens through a second transformer block.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Ensure config has attention implementation
        if not hasattr(config, '_attn_implementation'):
            config._attn_implementation = 'eager'

        # Standard transformer block for processing selected tokens
        self.block = Qwen2DecoderLayer(config, layer_idx)
        # Router for token selection
        self.router = DTFRouter(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        decision_output: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any], Optional[Tuple], Optional[torch.Tensor]]:
        """Process selected tokens through transformer block.

        Args:
            hidden_states: Current hidden states (from decision layer posterior)
            decision_output: Dictionary with original, posterior, prior states
            Other args: Standard transformer layer arguments

        Returns:
            Updated hidden states, routing stats, cache, attention weights
        """
        B, T, D = hidden_states.shape

        # Extract states from decision layer
        original = decision_output['original']
        posterior = decision_output['posterior']
        prior = decision_output['prior']

        # Ensure all have batch dimension
        if original.dim() == 2:
            original = original.unsqueeze(0)
        if posterior.dim() == 2:
            posterior = posterior.unsqueeze(0)
        if prior.dim() == 2:
            prior = prior.unsqueeze(0)

        # Compute routing scores based on surprise
        scores, _, stats = self.router.compute_routing_scores(
            hidden_states, original, posterior, prior
        )

        # Select top-k tokens
        selected_hidden, batch_idx, token_idx, selected_scores = self.router.select_tokens(
            scores, hidden_states
        )

        # Track statistics
        stats['selected_tokens'] = batch_idx.numel()
        stats['total_tokens'] = B * T

        # If no tokens selected, return input unchanged
        if batch_idx.numel() == 0:
            return hidden_states, stats, None, None

        # Reshape for processing
        num_selected = selected_hidden.shape[0]
        selected_hidden = selected_hidden.unsqueeze(0)  # [1, num_selected, D]

        # Create attention mask for selected tokens
        selected_attn_mask = None
        if attention_mask is not None and num_selected > 0:
            selected_attn_mask = _prepare_4d_causal_attention_mask(
                None, (1, num_selected), selected_hidden, 0
            )

        # Gather position information
        selected_pos_ids = None
        if position_ids is not None and num_selected > 0:
            pos_2d = position_ids.reshape(-1)
            flat_idx = batch_idx * T + token_idx
            selected_pos_ids = pos_2d[flat_idx].unsqueeze(0)

        # Gather position embeddings
        selected_pos_emb = None
        if position_embeddings is not None and num_selected > 0:
            cos, sin = position_embeddings
            gathered_cos = cos[batch_idx, token_idx].unsqueeze(0)
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

        # Handle return format
        if isinstance(layer_outputs, tuple):
            processed = layer_outputs[0].squeeze(0)  # [num_selected, D]
            cache = layer_outputs[1] if len(layer_outputs) > 1 else None
            attn_weights = layer_outputs[2] if len(layer_outputs) > 2 else None
        else:
            processed = layer_outputs.squeeze(0)
            cache = None
            attn_weights = None

        # Compute gate values based on surprise scores
        gate_values = torch.sigmoid(selected_scores).unsqueeze(-1)

        # Apply gated residual: new = original + (processed - original) * gate
        selected_hidden_flat = selected_hidden.squeeze(0)
        updated_tokens = selected_hidden_flat + (processed - selected_hidden_flat) * gate_values

        # Scatter back
        output = self.router.scatter_tokens(
            updated_tokens,
            hidden_states,
            batch_idx,
            token_idx
        )

        return output, stats, cache, attn_weights


class DTFForCausalLM(BaseDynamicModel):
    """DTF (Dynamic Transformer) model for causal language modeling.

    Alternates between decision layers and dynamic layers to implement
    surprise-based conditional computation.
    """

    def __init__(self, config):
        super().__init__(config)
        self.prior_loss_weight = getattr(config, 'prior_loss_weight', 0.05)
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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

        total_prior_loss = 0.0
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
                hidden_states, stats, cache, attn_weights = layer(
                    hidden_states,
                    decision_output,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    position_embeddings=position_embeddings,
                )

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
        prior_params = []

        for name, param in self.named_parameters():
            if 'router' in name:
                router_params.append(param)
            elif 'prior_network' in name:
                prior_params.append(param)
            else:
                base_params.append(param)

        groups = []
        if base_params:
            groups.append({'params': base_params, 'lr_scale': 1.0, 'name': 'base'})
        if router_params:
            groups.append({'params': router_params, 'lr_scale': 10.0, 'name': 'router'})
        if prior_params:
            groups.append({'params': prior_params, 'lr_scale': 10.0, 'name': 'prior'})

        return groups