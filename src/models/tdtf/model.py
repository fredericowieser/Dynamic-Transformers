"""Temporal Dynamic Transformer (TDTF) model implementation using Qwen2 architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any
import math

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm

from ..base.dynamic_model import BaseDynamicModel
from ..base.router import BaseRouter


class TDTFTransitionNetwork(nn.Module):
    """Transition Network (TPN) for predicting residual updates.

    Implements the change hypothesis by predicting the residual update
    of the current token using the final output state of the previous token.
    """

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        # Use reduced intermediate size for lightweight computation
        intermediate_size = int(hidden_size * getattr(config, 'tpn_intermediate_size_factor', 0.25))

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict residual update from previous token's output state."""
        return self.down_proj(self.act(self.up_proj(x)))


class TDTFPredictiveRouter(nn.Module):
    """Non-causal Predictive Router for training (teacher model).

    Uses actual residual and predicted residual to calculate continuous gate values
    based on static and change surprise metrics with VPR event criteria.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Learnable parameters for VPR criteria (initialized as per spec)
        self.o_ce = nn.Parameter(torch.tensor(getattr(config, 'o_ce_init', 1.025)))
        self.m_cu = nn.Parameter(torch.tensor(getattr(config, 'm_cu_init', 1.1)))
        self.beta_ce = nn.Parameter(torch.tensor(getattr(config, 'beta_ce_init', -0.3)))
        self.beta_cu = nn.Parameter(torch.tensor(getattr(config, 'beta_cu_init', -0.6)))

        # Capacity for TopK selection
        self.capacity = getattr(config, 'tdtf_capacity', 0.5)  # γ parameter

        # Moving average window for CU detection
        self.ma_window = getattr(config, 'ma_window', 100)
        self.register_buffer('static_surprise_history', torch.zeros(self.ma_window))
        self.register_buffer('history_pointer', torch.tensor(0))

    def compute_surprise_metrics(self, actual_residual: torch.Tensor, predicted_residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute static and change surprise metrics.

        Args:
            actual_residual: Actual residual update [B, T, D]
            predicted_residual: TPN predicted residual [B, T, D]

        Returns:
            D_st: Static surprise (magnitude of actual update)
            D_ch: Change surprise (TPN prediction error)
        """
        B, T, D = actual_residual.shape

        # Static surprise: magnitude of the actual update
        D_st = (actual_residual.norm(dim=-1) ** 2) / D  # [B, T]

        # Change surprise: TPN's prediction error
        D_ch = ((actual_residual - predicted_residual).norm(dim=-1) ** 2) / D  # [B, T]

        return D_st, D_ch

    def update_moving_average(self, D_st: torch.Tensor):
        """Update moving average of static surprise for CU detection."""
        if not self.training:
            return

        # Flatten across batch and time
        D_st_flat = D_st.flatten()

        for val in D_st_flat:
            idx = self.history_pointer % self.ma_window
            self.static_surprise_history[idx] = val.item()
            self.history_pointer += 1

    def get_moving_average(self, D_st: torch.Tensor) -> torch.Tensor:
        """Get moving average for CU detection."""
        if self.history_pointer < self.ma_window:
            # Not enough history, use current mean
            return D_st.mean()
        else:
            return self.static_surprise_history.mean()

    def compute_vpr_criteria(self, D_st: torch.Tensor, D_ch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute VPR event criteria.

        Args:
            D_st: Static surprise [B, T]
            D_ch: Change surprise [B, T]

        Returns:
            CE: Expected change criteria [B, T]
            CU: Unexpected change criteria [B, T]
        """
        # Expected Event (CE): D_st - (D_ch - log(o_ce + eps))
        CE = D_st - (D_ch - torch.log(self.o_ce + 1e-10))

        # Unexpected Event (CU): D_st - (m_cu * MA(D_st))
        ma_d_st = self.get_moving_average(D_st)
        CU = D_st - (self.m_cu * ma_d_st)

        return CE, CU

    def compute_continuous_gate(self, CE: torch.Tensor, CU: torch.Tensor) -> torch.Tensor:
        """Convert criteria to continuous gate values using sigmoid and probabilistic OR.

        Args:
            CE: Expected change criteria [B, T]
            CU: Unexpected change criteria [B, T]

        Returns:
            Continuous gate values g_t^(l) ∈ [0,1]
        """
        # Convert to probabilities using learnable inverse temperatures
        beta_ce_pos = F.softplus(self.beta_ce)
        beta_cu_pos = F.softplus(self.beta_cu)

        S_CE = torch.sigmoid(beta_ce_pos * CE)
        S_CU = torch.sigmoid(beta_cu_pos * CU)

        # Probabilistic OR: P(A or B) = P(A) + P(B) - P(A)P(B)
        g_continuous = S_CE + S_CU - (S_CE * S_CU)

        return g_continuous

    def forward(self, actual_residual: torch.Tensor, predicted_residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute continuous gate values and binary targets.

        Args:
            actual_residual: Actual residual from TF block [B, T, D]
            predicted_residual: TPN predicted residual [B, T, D]

        Returns:
            g_continuous: Continuous gate values [B, T]
            binary_targets: TopK binary mask for causal router training [B, T]
        """
        B, T, D = actual_residual.shape

        # Compute surprise metrics
        D_st, D_ch = self.compute_surprise_metrics(actual_residual, predicted_residual)

        # Update moving average (training only)
        if self.training:
            self.update_moving_average(D_st)

        # Compute VPR criteria
        CE, CU = self.compute_vpr_criteria(D_st, D_ch)

        # Compute continuous gate values
        g_continuous = self.compute_continuous_gate(CE, CU)

        # Generate binary targets by TopK selection
        k = max(1, int(T * self.capacity))
        _, topk_idx = g_continuous.topk(k, dim=-1)  # [B, k]

        binary_targets = torch.zeros_like(g_continuous)  # [B, T]
        batch_idx = torch.arange(B, device=g_continuous.device).unsqueeze(1)  # [B, 1]
        binary_targets[batch_idx, topk_idx] = 1.0

        return g_continuous, binary_targets


class TDTFCausalRouter(BaseRouter):
    """Causal Router for inference (student model).

    Simple linear layer that makes causal routing decisions using only
    pre-computation states from current and previous tokens.
    """

    def __init__(self, config, layer_idx: int):
        capacity = getattr(config, 'tdtf_capacity', 0.5)
        super().__init__(capacity)

        self.layer_idx = layer_idx
        hidden_size = config.hidden_size

        # Linear layer for causal prediction: input is [x_t^(l-1) || x_{t-1}^(l-1)]
        self.router_linear = nn.Linear(2 * hidden_size, 1, bias=True)

        # Initialize to reasonable values
        nn.init.normal_(self.router_linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.router_linear.bias)

    def compute_routing_scores(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Compute causal routing scores.

        Args:
            hidden_states: Current token states [B, T, D]

        Returns:
            scores: Routing logits [B, T]
            aux_loss: None
            stats: Routing statistics
        """
        B, T, D = hidden_states.shape

        # Prepare causal input: [x_t^(l-1) || x_{t-1}^(l-1)]
        # For t=1, x_0^(l-1) should be zero vector
        prev_states = torch.cat([
            torch.zeros(B, 1, D, device=hidden_states.device, dtype=hidden_states.dtype),
            hidden_states[:, :-1, :]  # Shift right by 1
        ], dim=1)  # [B, T, D]

        # Concatenate current and previous states
        causal_input = torch.cat([hidden_states, prev_states], dim=-1)  # [B, T, 2*D]

        # Compute routing logits
        logits = self.router_linear(causal_input).squeeze(-1)  # [B, T]

        # Convert to probabilities for statistics
        probs = torch.sigmoid(logits)

        stats = {
            'layer_idx': self.layer_idx,
            'capacity': self.capacity,
            'avg_prob': probs.mean().item(),
            'max_prob': probs.max().item(),
            'min_prob': probs.min().item(),
        }

        return logits, None, stats


class TDTFLayer(nn.Module):
    """Single TDTF layer implementing teacher-student training paradigm.

    During training: Uses both TPN + Predictive Router (teacher) and Causal Router (student)
    During inference: Uses only Causal Router for token gating decisions
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.training_mode = True  # Track if we're in training vs inference mode

        # Ensure config has attention implementation
        if not hasattr(config, '_attn_implementation'):
            config._attn_implementation = 'eager'

        # Standard transformer block
        self.transformer_block = Qwen2DecoderLayer(config, layer_idx)

        # Training components (teacher model)
        self.transition_network = TDTFTransitionNetwork(config)
        self.predictive_router = TDTFPredictiveRouter(config, layer_idx)

        # Inference component (student model)
        self.causal_router = TDTFCausalRouter(config, layer_idx)

        # Loss weights
        self.tpn_loss_weight = getattr(config, 'tpn_loss_weight', 1.0)
        self.causal_loss_weight = getattr(config, 'causal_loss_weight', 1.0)

    def forward_training(
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
        """Forward pass during training (teacher mode)."""
        B, T, D = hidden_states.shape

        # Store input state (original)
        x_original = hidden_states

        # Compute TF block output (posterior/ground truth)
        layer_outputs = self.transformer_block(
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
            x_posterior = layer_outputs[0]
            cache = layer_outputs[1] if len(layer_outputs) > 1 else None
            attn_weights = layer_outputs[2] if len(layer_outputs) > 2 else None
        else:
            x_posterior = layer_outputs
            cache = None
            attn_weights = None

        # Compute actual residual
        actual_residual = x_posterior - x_original

        # TPN prediction using previous token's final state
        # For t=1, use zero vector; for t>1, use x_{t-1}^(l)
        prev_final_states = torch.cat([
            torch.zeros(B, 1, D, device=hidden_states.device, dtype=hidden_states.dtype),
            x_posterior[:, :-1, :]  # Use final states from previous positions
        ], dim=1)  # [B, T, D]

        predicted_residual = self.transition_network(prev_final_states)

        # Compute TPN loss
        tpn_loss = F.mse_loss(predicted_residual, actual_residual.detach())

        # Predictive router (teacher)
        g_continuous, binary_targets = self.predictive_router(actual_residual, predicted_residual)

        # Causal router (student) training
        causal_scores, _, causal_stats = self.causal_router.compute_routing_scores(x_original)
        causal_probs = torch.sigmoid(causal_scores)

        # Causal router loss (BCE with binary targets from teacher)
        causal_loss = F.binary_cross_entropy(causal_probs, binary_targets.detach())

        return {
            'hidden_states': x_posterior,  # Use posterior as output
            'tpn_loss': tpn_loss,
            'causal_loss': causal_loss,
            'g_continuous': g_continuous,
            'binary_targets': binary_targets,
            'causal_probs': causal_probs,
            'past_key_value': cache,
            'attention_weights': attn_weights,
            'router_stats': causal_stats,
        }

    def forward_inference(
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
        """Forward pass during inference (student mode)."""
        B, T, D = hidden_states.shape

        # Get causal routing decisions
        causal_scores, _, causal_stats = self.causal_router.compute_routing_scores(hidden_states)

        # Binarize decisions (simple thresholding)
        # TODO: Could also implement TopK of generated prefix for better control
        routing_decisions = (torch.sigmoid(causal_scores) > 0.5).float()  # [B, T]

        # Initialize output
        output_states = hidden_states.clone()
        cache = None
        attn_weights = None
        processed_tokens = 0

        # Process each token based on routing decision
        for t in range(T):
            for b in range(B):
                if routing_decisions[b, t] == 1.0:
                    # Process this token through TF block
                    token_input = hidden_states[b:b+1, t:t+1, :]  # [1, 1, D]

                    # Prepare attention mask for single token
                    token_mask = None
                    if attention_mask is not None:
                        token_mask = attention_mask[b:b+1, :, t:t+1, :t+1]  # Causal mask up to position t

                    # Prepare position info
                    token_pos_ids = None
                    if position_ids is not None:
                        token_pos_ids = position_ids[b:b+1, t:t+1]

                    token_pos_emb = None
                    if position_embeddings is not None:
                        cos, sin = position_embeddings
                        token_pos_emb = (cos[b:b+1, t:t+1], sin[b:b+1, t:t+1])

                    # Process through TF block
                    layer_outputs = self.transformer_block(
                        token_input,
                        attention_mask=token_mask,
                        position_ids=token_pos_ids,
                        past_key_value=past_key_values,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        position_embeddings=token_pos_emb,
                    )

                    if isinstance(layer_outputs, tuple):
                        processed_token = layer_outputs[0]  # [1, 1, D]
                        if cache is None and layer_outputs[1] is not None:
                            cache = layer_outputs[1]
                        if attn_weights is None and len(layer_outputs) > 2:
                            attn_weights = layer_outputs[2]
                    else:
                        processed_token = layer_outputs

                    # Update output
                    output_states[b, t, :] = processed_token[0, 0, :]
                    processed_tokens += 1
                # else: token passes through unchanged (residual connection)

        causal_stats['processed_tokens'] = processed_tokens
        causal_stats['total_tokens'] = B * T
        causal_stats['processing_ratio'] = processed_tokens / (B * T) if B * T > 0 else 0.0

        return {
            'hidden_states': output_states,
            'routing_decisions': routing_decisions,
            'past_key_value': cache,
            'attention_weights': attn_weights,
            'router_stats': causal_stats,
        }

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
        """Forward pass - delegates to training or inference mode."""
        if self.training:
            return self.forward_training(
                hidden_states, attention_mask, position_ids, past_key_values,
                use_cache, output_attentions, position_embeddings, **kwargs
            )
        else:
            return self.forward_inference(
                hidden_states, attention_mask, position_ids, past_key_values,
                use_cache, output_attentions, position_embeddings, **kwargs
            )


class TDTFForCausalLM(BaseDynamicModel):
    """TDTF (Temporal Dynamic Transformer) model for causal language modeling.

    Implements student-teacher framework with training-time predictive router
    and inference-time causal router for conditional computation.
    """

    def __init__(self, config):
        super().__init__(config)

        # Loss weights
        self.tpn_loss_weight = getattr(config, 'tpn_loss_weight', 1.0)
        self.causal_loss_weight = getattr(config, 'causal_loss_weight', 1.0)

        self._setup_layers()

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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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
        tpn_params = []
        predictive_router_params = []
        causal_router_params = []

        for name, param in self.named_parameters():
            if 'transition_network' in name:
                tpn_params.append(param)
            elif 'predictive_router' in name:
                predictive_router_params.append(param)
            elif 'causal_router' in name:
                causal_router_params.append(param)
            else:
                base_params.append(param)

        groups = []
        if base_params:
            groups.append({'params': base_params, 'lr_scale': 1.0, 'name': 'base'})
        if tpn_params:
            groups.append({'params': tpn_params, 'lr_scale': 10.0, 'name': 'tpn'})
        if predictive_router_params:
            groups.append({'params': predictive_router_params, 'lr_scale': 10.0, 'name': 'predictive_router'})
        if causal_router_params:
            groups.append({'params': causal_router_params, 'lr_scale': 10.0, 'name': 'causal_router'})

        return groups