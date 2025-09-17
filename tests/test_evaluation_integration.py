import pytest
import torch
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.models.qwen.causal_lm import DynamicQwenForCausalLM
from src.models.qwen.config import DynamicQwenConfig
from transformers import AutoTokenizer


class TestEvaluationIntegration:
    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        tokenizer.vocab_size = 32000
        tokenizer.encode = lambda text, return_tensors=None: [1, 2, 3, 4, 5] if return_tensors is None else torch.tensor([[1, 2, 3, 4, 5]])
        tokenizer.decode = lambda ids: "Sample decoded text"
        tokenizer.batch_encode_plus = lambda texts, **kwargs: {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]] * len(texts)),
            'attention_mask': torch.ones(len(texts), 5, dtype=torch.long)
        }
        return tokenizer

    def test_model_generation_evaluation(self, base_config, device):
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.eval()

        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(1, base_config.vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=0,
                eos_token_id=2,
            )

        assert generated_ids.shape[0] == batch_size
        assert generated_ids.shape[1] > seq_length
        assert generated_ids.shape[1] <= seq_length + 20

        assert torch.all(generated_ids[:, :seq_length] == input_ids)

    def test_beam_search_generation(self, base_config, device):
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3]], device=device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            beam_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=15,
                num_beams=4,
                num_return_sequences=2,
                do_sample=False,
                pad_token_id=0,
                eos_token_id=2,
            )

        assert beam_output.shape[0] == 2
        assert beam_output.shape[1] > input_ids.shape[1]

    def test_top_k_top_p_sampling(self, base_config, device):
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3]], device=device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output_top_k = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                do_sample=True,
                top_k=50,
                temperature=0.8,
                pad_token_id=0,
                eos_token_id=2,
            )

            output_top_p = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                do_sample=True,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=0,
                eos_token_id=2,
            )

        assert output_top_k.shape[1] > input_ids.shape[1]
        assert output_top_p.shape[1] > input_ids.shape[1]

    def test_batch_generation_consistency(self, base_config, device):
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.eval()

        torch.manual_seed(42)

        single_input = torch.tensor([[1, 2, 3]], device=device)
        batch_input = torch.tensor([[1, 2, 3], [1, 2, 3]], device=device)

        single_mask = torch.ones_like(single_input)
        batch_mask = torch.ones_like(batch_input)

        with torch.no_grad():
            torch.manual_seed(42)
            single_output = model.generate(
                input_ids=single_input,
                attention_mask=single_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=0,
                eos_token_id=2,
            )

            torch.manual_seed(42)
            batch_output = model.generate(
                input_ids=batch_input,
                attention_mask=batch_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=0,
                eos_token_id=2,
            )

        assert torch.allclose(single_output[0], batch_output[0]), \
            "Same input should produce same output in batch"

    def test_perplexity_calculation(self, base_config, device):
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.eval()

        batch_size = 4
        seq_length = 32
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

        loss = output.loss
        perplexity = torch.exp(loss)

        assert perplexity > 0
        assert not torch.isnan(perplexity)
        assert not torch.isinf(perplexity)

    def test_logits_masking(self, base_config, device):
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.eval()

        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_length), device=device)

        full_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)
        partial_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)
        partial_mask[:, seq_length//2:] = 0

        with torch.no_grad():
            output_full = model(input_ids=input_ids, attention_mask=full_mask)
            output_partial = model(input_ids=input_ids, attention_mask=partial_mask)

        valid_positions = seq_length // 2
        assert not torch.allclose(
            output_full.logits[:, :valid_positions],
            output_partial.logits[:, :valid_positions],
            atol=1e-5
        )

    def test_model_export_import(self, base_config, device):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = DynamicQwenForCausalLM(base_config)
            model = model.to(device)

            save_path = os.path.join(tmpdir, "model")
            model.save_pretrained(save_path)
            base_config.save_pretrained(save_path)

            loaded_config = DynamicQwenConfig.from_pretrained(save_path)
            loaded_model = DynamicQwenForCausalLM.from_pretrained(save_path)
            loaded_model = loaded_model.to(device)

            assert loaded_config.hidden_size == base_config.hidden_size
            assert loaded_config.num_hidden_layers == base_config.num_hidden_layers
            assert loaded_config.dynamic_architecture == base_config.dynamic_architecture

            input_ids = torch.tensor([[1, 2, 3]], device=device)
            model.eval()
            loaded_model.eval()

            with torch.no_grad():
                original_output = model(input_ids=input_ids)
                loaded_output = loaded_model(input_ids=input_ids)

            assert torch.allclose(original_output.logits, loaded_output.logits, atol=1e-5)

    def test_routing_statistics_collection(self, base_config, device):
        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.eval()

        batch_size = 4
        seq_length = 32
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)

        routing_stats = {'layer_decisions': {}}

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            for idx, layer in enumerate(model.model.layers):
                if hasattr(layer, 'router') and hasattr(layer.router, 'routing_decisions'):
                    if layer.router.routing_decisions is not None:
                        decisions = layer.router.routing_decisions
                        routing_stats['layer_decisions'][f'layer_{idx}'] = {
                            'mean': decisions.float().mean().item(),
                            'std': decisions.float().std().item(),
                            'num_active': decisions.sum().item(),
                            'total': decisions.numel(),
                        }

        if routing_stats['layer_decisions']:
            for layer_name, stats in routing_stats['layer_decisions'].items():
                assert 0 <= stats['mean'] <= 1
                assert stats['num_active'] >= 0
                assert stats['num_active'] <= stats['total']

    def test_inference_speed_benchmark(self, base_config, device):
        import time

        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.eval()

        batch_sizes = [1, 2, 4]
        seq_length = 128
        num_iterations = 10

        timing_results = {}

        for batch_size in batch_sizes:
            input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_length), device=device)
            attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()

            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()

            avg_time = (end_time - start_time) / num_iterations
            timing_results[f'batch_{batch_size}'] = {
                'avg_time': avg_time,
                'throughput': batch_size * seq_length / avg_time
            }

        for batch_key, results in timing_results.items():
            assert results['avg_time'] > 0
            assert results['throughput'] > 0

    def test_memory_efficiency(self, base_config, device):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory test")

        torch.cuda.reset_peak_memory_stats()

        model = DynamicQwenForCausalLM(base_config)
        model = model.to(device)
        model.eval()

        batch_size = 2
        seq_length = 256
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)

        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

        peak_memory = torch.cuda.max_memory_allocated()
        model_params = sum(p.numel() * p.element_size() for p in model.parameters())

        memory_ratio = peak_memory / model_params
        assert memory_ratio > 0, "Memory usage should be positive"

        print(f"Peak memory: {peak_memory / 1024**2:.2f} MB")
        print(f"Model params size: {model_params / 1024**2:.2f} MB")
        print(f"Memory ratio: {memory_ratio:.2f}x")