# src/eval/config.py
import json
from transformers import AutoConfig
from src.models.d_llama_config import DynamicLlamaConfig  # Ensure this import matches your structure

def load_model_config(model_path: str):
    """Load model configuration, with support for custom types like dynamic_llama."""
    try:
        # First, try the standard AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        return {
            "ce_bias": getattr(config, "ce_bias", 0.0),  # Default if not set
            "dynamic_k": getattr(config, "dynamic_k", 0.5),  # Default if not set
            "config_object": config,  # Return the full config object
        }
    except ValueError as e:
        if "dynamic_llama" in str(e).lower():  # Check for your custom type
            print(f"Detected custom model type 'dynamic_llama'. Loading with DynamicLlamaConfig...")
            custom_config = DynamicLlamaConfig.from_pretrained(model_path)  # Load your custom config
            return {
                "ce_bias": getattr(custom_config, "ce_bias", 0.0),
                "dynamic_k": getattr(custom_config, "dynamic_k", 0.5),
                "config_object": custom_config,
            }
        else:
            raise e  # Re-raise if it's not the expected error

def save_results(output_file: str, results: dict):
    """Save results to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)