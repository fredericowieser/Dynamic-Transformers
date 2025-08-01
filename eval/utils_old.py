# src.eval/utils.py
"""Utility functions for evaluation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Callable, Dict, Any
import numpy as np

def load_model_and_tokenizer(model_path: str, ce_bias: float, dynamic_k: float):
    """Load the model and tokenizer, applying CE bias and dynamic K."""
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.set_ce_bias(ce_bias)  # From your DynamicLlama code
    model.set_dynamic_k(dynamic_k)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def compute_average_activation(model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> float:
    """Compute average gate activation during inference."""
    with torch.no_grad():
        outputs = model(**inputs)
        # Assuming gate activations are accessible, e.g., from model.get_last_gate_means()
        gate_means = model.get_last_gate_means()  # From your code
        if gate_means is not None:
            return torch.mean(torch.tensor(gate_means)).item()
    return np.nan  # Fallback if not available

def run_benchmark(benchmark_fn: Callable, model, tokenizer, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single benchmark and return results."""
    results = benchmark_fn(model, tokenizer, config)
    avg_activation = compute_average_activation(model, results.get("inputs", {}))
    results["average_activation"] = avg_activation
    return results

