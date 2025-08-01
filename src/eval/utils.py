import torch
import logging
from transformers import AutoTokenizer
from src.models.d_llama_config import DynamicLlamaConfig
from src.models.d_llama_causal_lm import DynamicLlamaForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(
    model_path: str, device: str, is_instruct: bool = False, ce_bias: float = None, dynamic_k: float = None
) -> tuple:
    """
    Load the model and tokenizer, applying custom parameters if provided.

    Args:
        model_path (str): Path to the model.
        device (str): Device to use (e.g., 'cuda', 'cpu').
        is_instruct (bool): Whether the model is instruct-tuned.
        ce_bias (float, optional): Override for CE bias.
        dynamic_k (float, optional): Override for dynamic K.

    Returns:
        tuple: Loaded model and tokenizer.
    """
    config = DynamicLlamaConfig.from_pretrained(model_path)
    
    if ce_bias is not None:
        config.ce_bias = ce_bias
    if dynamic_k is not None:
        config.dynamic_k = dynamic_k
    
    model = DynamicLlamaForCausalLM.from_pretrained(
        model_path, config=config, device_map="auto" if device == "cuda" else None
    )
    model.eval()  # Set to evaluation mode
    
    # Set attributes if methods are available
    if hasattr(model, "set_dynamic_k") and dynamic_k is not None:
        model.set_dynamic_k(dynamic_k)
        logger.info(f"Set dynamic_k to {dynamic_k}")
    if hasattr(model, "set_ce_bias") and ce_bias is not None:
        model.set_ce_bias(ce_bias)
        logger.info(f"Set ce_bias to {ce_bias}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if isinstance(tokenizer.pad_token_id, (list, tuple)):
        tokenizer.pad_token_id = int(tokenizer.pad_token_id[0])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def compute_average_metric(scores: list[float], metric_name: str) -> dict:
    """
    Compute the average and standard deviation of scores.

    Args:
        scores (list[float]): List of scores.
        metric_name (str): Name of the metric.

    Returns:
        dict: Dictionary with average, std_dev, and metric name.
    """
    if not scores:
        return {"average": 0.0, "std_dev": 0.0, "metric": metric_name}
    average = sum(scores) / len(scores)
    std_dev = (sum((x - average) ** 2 for x in scores) / len(scores)) ** 0.5
    return {"average": average, "std_dev": std_dev, "metric": metric_name}

def manual_generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """
    Manually generate text using the model, with robust error handling.

    Args:
        model: The model instance.
        tokenizer: The tokenizer instance.
        prompt (str): Input prompt.
        max_new_tokens (int): Maximum tokens to generate.

    Returns:
        str: Generated text or an empty string on error.
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(next(model.parameters()).device)
    try:
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return ""