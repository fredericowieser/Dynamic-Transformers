import pytest
import torch
import torch.nn as nn

from src.models.blocks.qwen_block import Qwen2Block
from src.models.blocks.prior_ffn import PriorFeedForward


class TestQwen2Block:
    def test_initialization(self, base_config):
        block = Qwen2Block(base_config, layer_idx=0)
        assert block is not None
        assert hasattr(block, 'self_attn')
        assert hasattr(block, 'mlp')
        assert hasattr(block, 'input_layernorm')
        assert hasattr(block, 'post_attention_layernorm')

    def test_forward_shape(self, base_config, sample_hidden_states, sample_attention_mask, device):
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        block = Qwen2Block(base_config, layer_idx=0)
        block = block.to(device)

        batch_size, seq_length, hidden_size = sample_hidden_states.shape

        # Prepare 4D causal attention mask
        causal_mask = _prepare_4d_causal_attention_mask(
            sample_attention_mask,
            (batch_size, seq_length),
            sample_hidden_states,
            0
        )

        output = block(
            hidden_states=sample_hidden_states,
            attention_mask=causal_mask,
            position_ids=torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1),
        )

        assert len(output) >= 1
        hidden_states_out = output[0]
        assert hidden_states_out.shape == (batch_size, seq_length, hidden_size)
        if len(output) > 1:
            attn_weights = output[1]
            assert attn_weights is None or isinstance(attn_weights, (tuple, torch.Tensor))

    def test_residual_connections(self, base_config, sample_hidden_states, sample_attention_mask, device):
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        block = Qwen2Block(base_config, layer_idx=0)
        block = block.to(device)

        batch_size, seq_length, hidden_size = sample_hidden_states.shape
        initial_hidden = sample_hidden_states.clone()

        # Prepare 4D causal attention mask
        causal_mask = _prepare_4d_causal_attention_mask(
            sample_attention_mask,
            (batch_size, seq_length),
            sample_hidden_states,
            0
        )

        output = block(
            hidden_states=sample_hidden_states,
            attention_mask=causal_mask,
            position_ids=torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1),
        )

        hidden_states_out, _ = output

        assert not torch.allclose(hidden_states_out, initial_hidden, atol=1e-6)

    def test_attention_mask_effect(self, base_config, sample_hidden_states, device):
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        block = Qwen2Block(base_config, layer_idx=0)
        block = block.to(device)

        batch_size, seq_length, hidden_size = sample_hidden_states.shape

        full_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)
        partial_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)
        partial_mask[:, seq_length//2:] = 0

        # Prepare 4D causal attention masks
        full_causal_mask = _prepare_4d_causal_attention_mask(
            full_mask,
            (batch_size, seq_length),
            sample_hidden_states,
            0
        )

        partial_causal_mask = _prepare_4d_causal_attention_mask(
            partial_mask,
            (batch_size, seq_length),
            sample_hidden_states,
            0
        )

        output_full = block(
            hidden_states=sample_hidden_states,
            attention_mask=full_causal_mask,
            position_ids=torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1),
        )

        output_partial = block(
            hidden_states=sample_hidden_states,
            attention_mask=partial_causal_mask,
            position_ids=torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1),
        )

        hidden_full, _ = output_full
        hidden_partial, _ = output_partial

        assert not torch.allclose(hidden_full, hidden_partial, atol=1e-5)

    def test_gradient_flow(self, base_config, sample_hidden_states, sample_attention_mask, device):
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        block = Qwen2Block(base_config, layer_idx=0)
        block = block.to(device)
        block.train()

        seq_length = sample_hidden_states.shape[1]
        batch_size = sample_hidden_states.shape[0]
        sample_hidden_states.requires_grad = True

        # Prepare 4D causal attention mask
        causal_mask = _prepare_4d_causal_attention_mask(
            sample_attention_mask,
            (batch_size, seq_length),
            sample_hidden_states,
            0
        )

        output = block(
            hidden_states=sample_hidden_states,
            attention_mask=causal_mask,
            position_ids=torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1),
        )

        hidden_states_out, _ = output
        loss = hidden_states_out.sum()
        loss.backward()

        assert sample_hidden_states.grad is not None
        assert not torch.isnan(sample_hidden_states.grad).any()
        assert not torch.isinf(sample_hidden_states.grad).any()

    def test_cache_usage(self, base_config, sample_hidden_states, sample_attention_mask, device):
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        block = Qwen2Block(base_config, layer_idx=0)
        block = block.to(device)
        block.eval()

        batch_size, seq_length, hidden_size = sample_hidden_states.shape

        # Prepare 4D causal attention mask
        causal_mask = _prepare_4d_causal_attention_mask(
            sample_attention_mask,
            (batch_size, seq_length),
            sample_hidden_states,
            0
        )

        with torch.no_grad():
            output = block(
                hidden_states=sample_hidden_states,
                attention_mask=causal_mask,
                position_ids=torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1),
                use_cache=True,
            )

        assert len(output) == 3 or len(output) == 2
        if len(output) == 3:
            hidden_states_out, attn_weights, cache = output
            assert cache is not None


class TestPriorFeedForward:
    def test_initialization(self, base_config):
        ffn = PriorFeedForward(base_config)
        assert ffn is not None
        assert hasattr(ffn, 'w1')
        assert hasattr(ffn, 'w2')
        assert hasattr(ffn, 'w3')
        assert hasattr(ffn, 'act')
        assert hasattr(ffn, 'dropout')

    def test_forward_shape(self, base_config, sample_hidden_states, device):
        ffn = PriorFeedForward(base_config)
        ffn = ffn.to(device)

        batch_size, seq_length, hidden_size = sample_hidden_states.shape

        output = ffn(sample_hidden_states)

        assert output.shape == (batch_size, seq_length, hidden_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_intermediate_size(self, base_config, sample_hidden_states, device):
        ffn = PriorFeedForward(base_config)
        ffn = ffn.to(device)

        # PriorFeedForward uses a factor of 2 by default
        expected_intermediate = max(2, base_config.hidden_size * 2)
        # Ensure it's even
        if expected_intermediate % 2 != 0:
            expected_intermediate += 1

        assert ffn.w1.out_features == expected_intermediate
        assert ffn.w3.out_features == expected_intermediate
        assert ffn.w2.in_features == expected_intermediate
        assert ffn.w2.out_features == base_config.hidden_size

    def test_activation_function(self, base_config, sample_hidden_states, device):
        ffn = PriorFeedForward(base_config)
        ffn = ffn.to(device)

        batch_size, seq_length, hidden_size = sample_hidden_states.shape

        small_input = torch.randn(1, 1, hidden_size, device=device) * 0.01
        large_input = torch.randn(1, 1, hidden_size, device=device) * 10.0

        small_output = ffn(small_input)
        large_output = ffn(large_input)

        assert not torch.allclose(small_output, large_output)
        assert torch.abs(large_output).mean() > torch.abs(small_output).mean()

    def test_gradient_flow(self, base_config, sample_hidden_states, device):
        ffn = PriorFeedForward(base_config)
        ffn = ffn.to(device)
        ffn.train()

        sample_hidden_states.requires_grad = True

        output = ffn(sample_hidden_states)
        loss = output.sum()
        loss.backward()

        assert sample_hidden_states.grad is not None
        assert not torch.isnan(sample_hidden_states.grad).any()
        assert not torch.isinf(sample_hidden_states.grad).any()

        for param in ffn.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    def test_dropout_effect(self, base_config, sample_hidden_states, device):
        config = base_config
        if not hasattr(config, 'ffn_dropout'):
            config.ffn_dropout = 0.5

        ffn = PriorFeedForward(config)
        ffn = ffn.to(device)

        ffn.train()
        outputs_train = []
        for _ in range(10):
            output = ffn(sample_hidden_states)
            outputs_train.append(output)

        ffn.eval()
        with torch.no_grad():
            output_eval_1 = ffn(sample_hidden_states)
            output_eval_2 = ffn(sample_hidden_states)

        assert torch.allclose(output_eval_1, output_eval_2, atol=1e-6)