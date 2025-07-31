# utils.py
import torch
from transformers import AutoTokenizer
from src.models.d_llama_config import DynamicLlamaConfig  # Adjust path if needed
from src.models.d_llama_causal_lm import DynamicLlamaForCausalLM  # Adjust path if needed

def load_model_and_tokenizer(model_path: str, device: str, is_instruct: bool = False, ce_bias: float = None, dynamic_k: float = None):
    """Load model and tokenizer, handling custom DynamicLlama architecture."""
    config = DynamicLlamaConfig.from_pretrained(model_path)  # Directly use custom config
    
    # Apply overrides for CE bias and dynamic K
    if ce_bias is not None:
        config.ce_bias = ce_bias
    if dynamic_k is not None:
        config.dynamic_k = dynamic_k
    
    model = DynamicLlamaForCausalLM.from_pretrained(
        model_path, config=config, device_map="auto" if device == "cuda" else None
    )
    model.eval()  # Set to evaluation mode
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if isinstance(tokenizer.pad_token_id, (list, tuple)):
        tokenizer.pad_token_id = int(tokenizer.pad_token_id[0])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id  # Ensure consistency
    
    return model, tokenizer

def compute_average_metric(scores: list, metric_name: str):
    """Compute average of a metric (e.g., accuracy) and return with stats."""
    if not scores:
        return {"average": 0.0, "std_dev": 0.0, "metric": metric_name}
    average = sum(scores) / len(scores)
    std_dev = (sum((x - average) ** 2 for x in scores) / len(scores)) ** 0.5
    return {"average": average, "std_dev": std_dev, "metric": metric_name}