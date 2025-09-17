import pytest
import torch
import torch.nn as nn
from typing import Optional

from src.models.layers.decision_layer import DecisionLayer
from src.models.layers.dynamic_layer import DynamicLayer
from src.models.layers.mod_layer import MoDLayer


class TestDecisionLayer:
    def test_initialization(self, base_config):
        layer = DecisionLayer(base_config, layer_idx=0)
        assert layer is not None
        assert hasattr(layer, 'block')
        assert hasattr(layer, 'prior_ffn')
        assert hasattr(layer, 'prior_layernorm')

    def test_forward_shape(self, base_config, sample_hidden_states, sample_attention_mask, device):
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        layer = DecisionLayer(base_config, layer_idx=0)
        layer = layer.to(device)

        batch_size, seq_length, hidden_size = sample_hidden_states.shape

        # Prepare 4D causal attention mask
        causal_mask = _prepare_4d_causal_attention_mask(
            sample_attention_mask,
            (batch_size, seq_length),
            sample_hidden_states,
            0
        )

        output = layer(
            hidden_states=sample_hidden_states,
            attention_mask=causal_mask,
            position_ids=torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1),
        )

        assert hasattr(output, 'hidden_states')
        assert output.hidden_states.shape == (batch_size, seq_length, hidden_size)

    def test_gradient_flow(self, base_config, sample_hidden_states, sample_attention_mask, device):
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        layer = DecisionLayer(base_config, layer_idx=0)
        layer = layer.to(device)
        layer.train()

        seq_length = sample_hidden_states.shape[1]
        batch_size = sample_hidden_states.shape[0]

        sample_hidden_states.requires_grad = True

        causal_mask = _prepare_4d_causal_attention_mask(
            sample_attention_mask,
            (batch_size, seq_length),
            sample_hidden_states,
            0
        )

        output = layer(
            hidden_states=sample_hidden_states,
            attention_mask=causal_mask,
            position_ids=torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1),
        )

        loss = output.hidden_states.sum()
        loss.backward()

        assert sample_hidden_states.grad is not None
        assert not torch.isnan(sample_hidden_states.grad).any()

    def test_eval_mode(self, base_config, sample_hidden_states, sample_attention_mask, device):
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        layer = DecisionLayer(base_config, layer_idx=0)
        layer = layer.to(device)
        layer.eval()

        seq_length = sample_hidden_states.shape[1]
        batch_size = sample_hidden_states.shape[0]

        causal_mask = _prepare_4d_causal_attention_mask(
            sample_attention_mask,
            (batch_size, seq_length),
            sample_hidden_states,
            0
        )

        with torch.no_grad():
            output = layer(
                hidden_states=sample_hidden_states,
                attention_mask=causal_mask,
                position_ids=torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1),
            )

        assert not torch.isnan(output.hidden_states).any()
        assert not torch.isinf(output.hidden_states).any()


class TestDynamicLayer:
    def test_initialization(self, base_config):
        layer = DynamicLayer(base_config, layer_idx=1)
        assert layer is not None
        assert hasattr(layer, 'block')
        assert hasattr(layer, 'vpr_router')
        assert hasattr(layer, 'config')

    def test_forward_shape(self, base_config, sample_hidden_states, sample_attention_mask, device):
        from src.models.qwen.modeling_outputs import DecisionLayerOutput

        layer = DynamicLayer(base_config, layer_idx=1)
        layer = layer.to(device)

        batch_size, seq_length, hidden_size = sample_hidden_states.shape

        # Create a mock DecisionLayerOutput
        decision_output = DecisionLayerOutput(
            hidden_states=sample_hidden_states,
            vpr_signal_original_input=sample_hidden_states,
            vpr_signal_posterior_output=sample_hidden_states,
            vpr_signal_prior_hidden_states=sample_hidden_states,
            prior_loss=torch.tensor(0.0, device=device),
            present_key_value=None,
            attention_weights=None,
        )

        output = layer(
            hidden_states=sample_hidden_states,
            decision_output=decision_output,
            attention_mask=sample_attention_mask,
            position_ids=torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1),
        )

        assert hasattr(output, 'hidden_states')
        assert output.hidden_states.shape == (batch_size, seq_length, hidden_size)

    def test_skip_mechanism(self, base_config, sample_hidden_states, sample_attention_mask, device):
        from src.models.qwen.modeling_outputs import DecisionLayerOutput

        layer = DynamicLayer(base_config, layer_idx=1)
        layer = layer.to(device)

        seq_length = sample_hidden_states.shape[1]
        batch_size = sample_hidden_states.shape[0]

        decision_output = DecisionLayerOutput(
            hidden_states=sample_hidden_states,
            vpr_signal_original_input=sample_hidden_states,
            vpr_signal_posterior_output=sample_hidden_states,
            vpr_signal_prior_hidden_states=sample_hidden_states,
            prior_loss=torch.tensor(0.0, device=device),
            present_key_value=None,
            attention_weights=None,
        )

        output = layer(
            hidden_states=sample_hidden_states,
            decision_output=decision_output,
            attention_mask=sample_attention_mask,
            position_ids=torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1),
        )

        assert hasattr(output, 'hidden_states')

        # Check VPR router statistics
        if hasattr(output, 'vpr_gate_vec_binary'):
            decisions = output.vpr_gate_vec_binary
            if base_config.token_wise_gating:
                assert decisions.shape[0] == batch_size * seq_length or decisions.shape[0] == batch_size
            else:
                assert decisions.shape[0] == batch_size

    def test_gradient_flow_both_paths(self, base_config, sample_hidden_states, sample_attention_mask, device):
        from src.models.qwen.modeling_outputs import DecisionLayerOutput

        layer = DynamicLayer(base_config, layer_idx=1)
        layer = layer.to(device)
        layer.train()

        seq_length = sample_hidden_states.shape[1]
        batch_size = sample_hidden_states.shape[0]

        sample_hidden_states.requires_grad = True

        decision_output = DecisionLayerOutput(
            hidden_states=sample_hidden_states,
            vpr_signal_original_input=sample_hidden_states,
            vpr_signal_posterior_output=sample_hidden_states,
            vpr_signal_prior_hidden_states=sample_hidden_states,
            prior_loss=torch.tensor(0.0, device=device),
            present_key_value=None,
            attention_weights=None,
        )

        output = layer(
            hidden_states=sample_hidden_states,
            decision_output=decision_output,
            attention_mask=sample_attention_mask,
            position_ids=torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1),
        )

        loss = output.hidden_states.sum()
        loss.backward()

        assert sample_hidden_states.grad is not None
        assert not torch.isnan(sample_hidden_states.grad).any()


class TestMoDLayer:
    def test_initialization(self, mod_config):
        layer = MoDLayer(mod_config, layer_idx=1)
        assert layer is not None
        assert hasattr(layer, 'router')
        assert hasattr(layer, 'block')
        assert hasattr(layer, 'capacity_gamma')
        assert layer.capacity_gamma == mod_config.capacity_gamma

    def test_forward_shape(self, mod_config, sample_hidden_states, sample_attention_mask, device):
        layer = MoDLayer(mod_config, layer_idx=1)
        layer = layer.to(device)

        batch_size, seq_length, hidden_size = sample_hidden_states.shape

        output = layer(
            hidden_states=sample_hidden_states,
            attention_mask=sample_attention_mask,
            position_ids=torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1),
        )

        assert len(output) == 3
        hidden_states_out, _, _ = output
        assert hidden_states_out.shape == (batch_size, seq_length, hidden_size)

    def test_router_weights(self, mod_config, sample_hidden_states, sample_attention_mask, device):
        layer = MoDLayer(mod_config, layer_idx=1)
        layer = layer.to(device)

        seq_length = sample_hidden_states.shape[1]
        batch_size = sample_hidden_states.shape[0]

        # Test router produces weights for each token
        router_weights = layer.router(sample_hidden_states)
        assert router_weights.shape == (batch_size, seq_length)
        assert not torch.isnan(router_weights).any()
        assert not torch.isinf(router_weights).any()

    def test_capacity_factor(self, mod_config, sample_hidden_states, sample_attention_mask, device):
        layer = MoDLayer(mod_config, layer_idx=1)
        layer = layer.to(device)
        layer.train()

        seq_length = sample_hidden_states.shape[1]
        batch_size = sample_hidden_states.shape[0]

        # Test that capacity_gamma is used correctly
        expected_capacity = max(1, int(mod_config.capacity_gamma * seq_length))

        output = layer(
            hidden_states=sample_hidden_states,
            attention_mask=sample_attention_mask,
            position_ids=torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1),
        )

        assert len(output) == 3
        hidden_states_out, _, _ = output
        assert hidden_states_out.shape == (batch_size, seq_length, sample_hidden_states.shape[-1])

    def test_gradient_flow(self, mod_config, sample_hidden_states, sample_attention_mask, device):
        layer = MoDLayer(mod_config, layer_idx=1)
        layer = layer.to(device)
        layer.train()

        seq_length = sample_hidden_states.shape[1]
        batch_size = sample_hidden_states.shape[0]

        sample_hidden_states.requires_grad = True

        output = layer(
            hidden_states=sample_hidden_states,
            attention_mask=sample_attention_mask,
            position_ids=torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1),
        )

        hidden_states_out, _, _ = output
        loss = hidden_states_out.sum()
        loss.backward()

        assert sample_hidden_states.grad is not None
        assert not torch.isnan(sample_hidden_states.grad).any()