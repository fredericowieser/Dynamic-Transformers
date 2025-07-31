import json
from transformers import AutoConfig
from src.models.d_llama_config import DynamicLlamaConfig

def load_model_config(model_path: str):
    try:
        config = AutoConfig.from_pretrained(model_path)
        return {
            "ce_bias": getattr(config, "ce_bias", 0.0),
            "dynamic_k": getattr(config, "dynamic_k", 0.5),
            "config_object": config,
        }
    except ValueError as e:
        if "dynamic_llama" in str(e).lower():
            custom_config = DynamicLlamaConfig.from_pretrained(model_path)
            return {
                "ce_bias": getattr(custom_config, "ce_bias", 0.0),
                "dynamic_k": getattr(custom_config, "dynamic_k", 0.5),
                "config_object": custom_config,
            }
        raise e

def save_results(output_file: str, results: dict):
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)