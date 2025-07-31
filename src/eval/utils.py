# utils.py
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)


def load_model_and_tokenizer(
        model_path: str,
        device: str,
        is_instruct: bool = False
    ):
    """Load model and tokenizer with fixes for config issues."""
    config = AutoConfig.from_pretrained(model_path)
    if isinstance(config.pad_token_id, (list, tuple)):
        config.pad_token_id = int(config.pad_token_id[0])
    if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict) and "type" not in config.rope_scaling:
        config.rope_scaling = None

    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, device_map="auto" if device == "cuda" else None
    )
    model.eval()  # Ensure model is in eval mode

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if isinstance(tokenizer.pad_token_id, (list, tuple)):
        tokenizer.pad_token_id = int(tokenizer.pad_token_id[0])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Apply CE bias and dynamic K if provided (overrides config defaults)
    if hasattr(model.config, "ce_bias"):
        model.config.ce_bias = ce_bias  # To be passed from main
    if hasattr(model.config, "dynamic_k"):
        model.config.dynamic_k = dynamic_k  # To be passed from main

    return model, tokenizer

def compute_average_metric(scores: list, metric_name: str):
    """Compute average of a metric (e.g., accuracy) and return with stats."""
    if not scores:
        return {"average": 0.0, "metric": metric_name}
    average = sum(scores) / len(scores)
    return {"average": average, "std_dev": (sum((x - average) ** 2 for x in scores) / len(scores)) ** 0.5, "metric": metric_name}