import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.models.qwen.causal_lm import DynamicQwenForCausalLM
from src.models.qwen.config import DynamicQwenConfig


class TestDynamicQwenForCausalLM:
    def test_vpr_initialization(self, base_config):
        model = DynamicQwenForCausalLM(base_config)
        assert model is not None
        assert hasattr(model, 'model')
        assert hasattr(model, 'lm_head')
        assert len(model.model.layers) == base_config.num_hidden_layers

        for i, layer in enumerate(model.model.layers):
            if i % 2 == 0:
                assert layer.__class__.__name__ == 'DecisionLayer'
            else:
                assert layer.__class__.__name__ == 'DynamicLayer'

    def test_mod_initialization(self, mod_config):
        model = DynamicQwenForCausalLM(mod_config)
        assert model is not None
        assert hasattr(model, 'model')
        assert hasattr(model, 'lm_head')
        assert len(model.model.layers) == mod_config.num_hidden_layers

        for i, layer in enumerate(model.model.layers):
            if (i + 1) % 2 == 0:
                assert layer.__class__.__name__ == 'MoDLayer'
            else:
                assert layer.__class__.__name__ == 'Qwen2Block'

    def test_forward_shape(self, base_config, sample_input_ids, sample_attention_mask, device):
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.eval()

        batch_size, seq_length = sample_input_ids.shape

        with torch.no_grad():
            output = model(
                input_ids=sample_input_ids,
                attention_mask=sample_attention_mask,
            )

        assert hasattr(output, 'logits')
        assert output.logits.shape == (batch_size, seq_length, base_config.vocab_size)

    def test_loss_computation(self, base_config, sample_input_ids, sample_attention_mask, device):
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.train()

        batch_size, seq_length = sample_input_ids.shape
        labels = sample_input_ids.clone()

        output = model(
            input_ids=sample_input_ids,
            attention_mask=sample_attention_mask,
            labels=labels,
        )

        assert hasattr(output, 'loss')
        assert output.loss is not None
        assert output.loss.shape == torch.Size([])
        assert output.loss.item() > 0

    def test_gradient_flow(self, base_config, sample_input_ids, sample_attention_mask, device):
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.train()

        labels = sample_input_ids.clone()

        output = model(
            input_ids=sample_input_ids,
            attention_mask=sample_attention_mask,
            labels=labels,
        )

        loss = output.loss
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"

    def test_generation_capability(self, base_config, device):
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3]], device=device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=0,
                eos_token_id=2,
            )

        assert output_ids.shape[0] == 1
        assert output_ids.shape[1] > input_ids.shape[1]
        assert output_ids.shape[1] <= input_ids.shape[1] + 10

    def test_auxiliary_losses_vpr(self, base_config, sample_input_ids, sample_attention_mask, device):
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.train()

        labels = sample_input_ids.clone()

        output = model(
            input_ids=sample_input_ids,
            attention_mask=sample_attention_mask,
            labels=labels,
        )

        if hasattr(output, 'aux_losses'):
            assert output.aux_losses is not None
            assert isinstance(output.aux_losses, dict)

    def test_auxiliary_losses_mod(self, mod_config, sample_input_ids, sample_attention_mask, device):
        model = DynamicQwenForCausalLM(mod_config)
        model = model.to(device)
        model.train()

        labels = sample_input_ids.clone()

        output = model(
            input_ids=sample_input_ids,
            attention_mask=sample_attention_mask,
            labels=labels,
        )

        if hasattr(output, 'aux_losses'):
            assert output.aux_losses is not None
            if isinstance(output.aux_losses, dict) and 'moe_aux_loss' in output.aux_losses:
                moe_loss = output.aux_losses['moe_aux_loss']
                assert moe_loss.item() >= 0

    def test_freeze_main_transformer_blocks(self, base_config, device):
        config = base_config
        config.freeze_main_transformer_blocks = True

        model = DynamicQwenForCausalLM(config)
        model = model.to(device)

        for name, param in model.named_parameters():
            if 'main_block' in name:
                assert not param.requires_grad, f"Main block parameter {name} should be frozen"

    def test_model_size_and_parameters(self, base_config):
        model = DynamicQwenForCausalLM(base_config)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def test_attention_mask_causal(self, base_config, device):
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.eval()

        seq_length = 10
        batch_size = 2

        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        assert output.logits.shape == (batch_size, seq_length, base_config.vocab_size)

        for t in range(1, seq_length):
            future_masked_input = input_ids.clone()
            future_masked_input[:, t:] = 0

            future_masked_attention = attention_mask.clone()
            future_masked_attention[:, t:] = 0

            with torch.no_grad():
                masked_output = model(
                    input_ids=future_masked_input,
                    attention_mask=future_masked_attention
                )

            assert not torch.allclose(
                output.logits[:, :t],
                masked_output.logits[:, :t],
                atol=1e-5
            )