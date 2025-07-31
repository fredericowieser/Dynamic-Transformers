import json
from transformers import AutoConfig

def load_model_config(model_path: str):
    """Load model configuration and extract relevant parameters."""
    config = AutoConfig.from_pretrained(model_path)
    return {
        "ce_bias": getattr(config, "ce_bias", 0.0),  # Default to 0.0 if not set
        "dynamic_k": getattr(config, "dynamic_k", 0.5),  # Default to 0.5 if not set
        "other_params": config.to_dict(),  # For reference
    }

def save_results(output_file: str, results: dict):
    """Save results to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)