import pytest
import torch
import torch.nn as nn

from src.models.blocks.vpr_router import VPRRouter
from src.models.blocks.mod_router import MoDTokenRouter


class TestVPRRouter:
    def test_initialization(self, base_config):
        router = VPRRouter(base_config, layer_idx=0)
        assert router is not None
        assert router.layer_idx == 0
        assert router.hidden_size == base_config.hidden_size
        assert router.token_wise_gating == base_config.token_wise_gating

    def test_learnable_parameters(self, base_config):
        config = base_config
        config.learn_beta_ce = True
        config.learn_beta_cu = True
        config.learn_cu_multiplier = True
        config.learn_ce_offset = True

        router = VPRRouter(config, layer_idx=0)

        assert isinstance(router.beta_ce, nn.Parameter)
        assert isinstance(router.beta_cu, nn.Parameter)
        assert isinstance(router.cu_detection_multiplier, nn.Parameter)
        assert isinstance(router.ce_criterion_offset, nn.Parameter)

    def test_fixed_parameters(self, base_config):
        config = base_config
        config.learn_beta_ce = False
        config.learn_beta_cu = False
        config.learn_cu_multiplier = False
        config.learn_ce_offset = False

        router = VPRRouter(config, layer_idx=0)

        assert not isinstance(router.beta_ce, nn.Parameter)
        assert not isinstance(router.beta_cu, nn.Parameter)
        assert not isinstance(router.cu_detection_multiplier, nn.Parameter)
        assert not isinstance(router.ce_criterion_offset, nn.Parameter)

    def test_forward_decision_layer(self, base_config, sample_hidden_states, device):
        router = VPRRouter(base_config, layer_idx=0)
        router = router.to(device)

        batch_size, seq_length, hidden_size = sample_hidden_states.shape

        # VPRRouter returns a tuple when called
        output = router(
            sample_hidden_states,  # positional argument
            posterior_full_path_output=sample_hidden_states,
            prior_hidden_states=sample_hidden_states,
            capacity_gamma=base_config.capacity_gamma,
            is_training=router.training
        )

        assert isinstance(output, tuple)
        assert len(output) >= 11  # VPR router returns multiple statistics

    def test_forward_dynamic_layer(self, base_config, sample_hidden_states, device):
        router = VPRRouter(base_config, layer_idx=1)
        router = router.to(device)

        batch_size, seq_length, hidden_size = sample_hidden_states.shape

        output = router(
            sample_hidden_states,  # positional argument
            posterior_full_path_output=sample_hidden_states,
            prior_hidden_states=sample_hidden_states,
            capacity_gamma=base_config.capacity_gamma,
            is_training=router.training
        )

        assert isinstance(output, tuple)
        assert len(output) >= 11

    def test_token_wise_gating(self, base_config, sample_hidden_states, device):
        config = base_config
        config.token_wise_gating = True

        router = VPRRouter(config, layer_idx=1)
        router = router.to(device)

        batch_size, seq_length, hidden_size = sample_hidden_states.shape

        output = router(
            sample_hidden_states,  # positional argument
            posterior_full_path_output=sample_hidden_states,
            prior_hidden_states=sample_hidden_states,
            capacity_gamma=config.capacity_gamma,
            is_training=router.training
        )

        assert isinstance(output, tuple)
        gate_vec_binary = output[0]  # First element is the binary gating vector

        if config.token_wise_gating:
            assert gate_vec_binary.shape == (batch_size, seq_length)
        else:
            assert gate_vec_binary.shape == (batch_size, 1)

    def test_batch_wise_gating(self, base_config, sample_hidden_states, device):
        config = base_config
        config.token_wise_gating = False

        router = VPRRouter(config, layer_idx=1)
        router = router.to(device)

        batch_size, seq_length, hidden_size = sample_hidden_states.shape

        output = router(
            sample_hidden_states,  # positional argument
            posterior_full_path_output=sample_hidden_states,
            prior_hidden_states=sample_hidden_states,
            capacity_gamma=config.capacity_gamma,
            is_training=router.training
        )

        assert isinstance(output, tuple)
        gate_vec_binary = output[0]  # First element is the binary gating vector
        assert gate_vec_binary.shape == (batch_size, 1)  # Batch-wise gating


class TestMoDTokenRouter:
    def test_initialization(self, mod_config):
        router = MoDTokenRouter(mod_config.hidden_size)
        assert router is not None
        assert hasattr(router, 'gate')
        assert router.gate.in_features == mod_config.hidden_size
        assert router.gate.out_features == 1

    def test_gate_dimensions(self, mod_config, sample_hidden_states, device):
        router = MoDTokenRouter(mod_config.hidden_size)
        router = router.to(device)

        batch_size, seq_length, hidden_size = sample_hidden_states.shape

        gate_output = router.gate(sample_hidden_states)
        assert gate_output.shape == (batch_size, seq_length, 1)

    def test_forward_shape(self, mod_config, sample_hidden_states, device):
        router = MoDTokenRouter(mod_config.hidden_size)
        router = router.to(device)

        batch_size, seq_length, hidden_size = sample_hidden_states.shape

        output = router(sample_hidden_states)

        assert output.shape == (batch_size, seq_length)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_flow(self, mod_config, sample_hidden_states, device):
        router = MoDTokenRouter(mod_config.hidden_size)
        router = router.to(device)
        router.train()

        sample_hidden_states.requires_grad = True

        output = router(sample_hidden_states)
        loss = output.sum()
        loss.backward()

        assert sample_hidden_states.grad is not None
        assert not torch.isnan(sample_hidden_states.grad).any()

        for param in router.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()