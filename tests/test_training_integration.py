import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_scheduler, DataCollatorForLanguageModeling
from accelerate import Accelerator
import tempfile
import os

from src.models.qwen.causal_lm import DynamicQwenForCausalLM
from src.models.qwen.config import DynamicQwenConfig
from src.models.utils.training import set_seed, calculate_metrics


class TestTrainingIntegration:
    def create_dummy_dataset(self, vocab_size, seq_length, num_samples):
        input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
        attention_mask = torch.ones(num_samples, seq_length, dtype=torch.long)
        labels = input_ids.clone()
        return TensorDataset(input_ids, attention_mask, labels)

    def test_training_step_vpr(self, base_config, device):
        set_seed(42)
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        batch_size = 2
        seq_length = 128
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)
        labels = input_ids.clone()

        initial_loss = None
        for step in range(5):
            optimizer.zero_grad()

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = output.loss
            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

        assert loss.item() < initial_loss, "Loss should decrease during training"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"

    def test_training_step_mod(self, mod_config, device):
        set_seed(42)
        model = DynamicQwenForCausalLM(mod_config)
        model = model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        batch_size = 2
        seq_length = 128
        input_ids = torch.randint(0, mod_config.vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)
        labels = input_ids.clone()

        initial_loss = None
        for step in range(5):
            optimizer.zero_grad()

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = output.loss
            if hasattr(output, 'aux_losses') and 'moe_aux_loss' in output.aux_losses:
                total_loss = loss + mod_config.mod_aux_loss_weight * output.aux_losses['moe_aux_loss']
            else:
                total_loss = loss

            if initial_loss is None:
                initial_loss = total_loss.item()

            total_loss.backward()
            optimizer.step()

        assert total_loss.item() < initial_loss, "Loss should decrease during training"
        assert not torch.isnan(total_loss), "Loss should not be NaN"
        assert not torch.isinf(total_loss), "Loss should not be infinite"

    def test_gradient_accumulation(self, base_config, device):
        set_seed(42)
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        accumulation_steps = 4

        batch_size = 1
        seq_length = 64
        total_loss = 0

        for step in range(accumulation_steps):
            input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_length), device=device)
            attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)
            labels = input_ids.clone()

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = output.loss / accumulation_steps
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        assert total_loss > 0, "Total loss should be positive"

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is None or torch.allclose(param.grad, torch.zeros_like(param.grad), atol=1e-7), \
                    f"Gradients should be cleared after optimizer step for {name}"

    def test_learning_rate_scheduler(self, base_config, device):
        set_seed(42)
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=100,
        )

        initial_lr = optimizer.param_groups[0]['lr']

        batch_size = 2
        seq_length = 64
        for step in range(20):
            input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_length), device=device)
            attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)
            labels = input_ids.clone()

            optimizer.zero_grad()

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = output.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        assert current_lr != initial_lr, "Learning rate should change with scheduler"

    def test_mixed_precision_training(self, base_config):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")

        accelerator = Accelerator(mixed_precision="fp16")
        device = accelerator.device

        set_seed(42)
        model = DynamicQwenForCausalLM(base_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        model, optimizer = accelerator.prepare(model, optimizer)

        batch_size = 2
        seq_length = 64
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)
        labels = input_ids.clone()

        for step in range(3):
            optimizer.zero_grad()

            with accelerator.autocast():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = output.loss

            accelerator.backward(loss)
            optimizer.step()

        assert not torch.isnan(loss), "Loss should not be NaN in mixed precision"
        assert not torch.isinf(loss), "Loss should not be infinite in mixed precision"

    def test_checkpoint_saving_loading(self, base_config, device):
        set_seed(42)
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        batch_size = 2
        seq_length = 64
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)
        labels = input_ids.clone()

        model.train()
        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss_before = output.loss
        loss_before.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_before.item(),
            }, checkpoint_path)

            new_model = DynamicQwenForCausalLM(base_config)
            new_model = new_model.to(device)
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)

            checkpoint = torch.load(checkpoint_path, map_location=device)
            new_model.load_state_dict(checkpoint['model_state_dict'])
            new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            new_model.eval()
            with torch.no_grad():
                output_loaded = new_model(input_ids=input_ids, attention_mask=attention_mask)

            model.eval()
            with torch.no_grad():
                output_original = model(input_ids=input_ids, attention_mask=attention_mask)

            assert torch.allclose(output_loaded.logits, output_original.logits, atol=1e-5), \
                "Loaded model should produce same output as original"

    def test_training_with_dataloader(self, base_config, device):
        set_seed(42)
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.train()

        dataset = self.create_dummy_dataset(
            vocab_size=base_config.vocab_size,
            seq_length=64,
            num_samples=10
        )

        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        losses = []
        for epoch in range(2):
            epoch_loss = 0
            for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = output.loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)
            losses.append(avg_epoch_loss)

        assert losses[1] < losses[0], "Average loss should decrease across epochs"

    def test_metrics_calculation(self, base_config, device):
        set_seed(42)
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.eval()

        batch_size = 4
        seq_length = 32
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)
        labels = input_ids.clone()

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

        metrics = calculate_metrics(model, batch, global_step=0)

        assert 'perplexity' in metrics
        assert 'lm_loss' in metrics
        assert 'total_loss' in metrics
        assert metrics['perplexity'] > 0
        assert metrics['lm_loss'] > 0