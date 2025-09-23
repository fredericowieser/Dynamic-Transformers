import argparse
import os
import torch
import logging
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Import custom models to make them available for AutoModelForCausalLM to find.
# This allows `trust_remote_code=True` to work correctly.
from src.models.mod.model import MoDForCausalLM
from src.models.sdt.model import SDTForCausalLM
from src.models.stt.model import STTForCausalLM
from src.models.standard.model import StandardTransformerForCausalLM

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Convert a custom .pt checkpoint to a standard Hugging Face model format.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the directory containing the model.pt and config.json files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the converted, standard-format model.")
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen2.5-0.5B", help="Name or path of the tokenizer to save with the model.")
    args = parser.parse_args()

    log.info(f"Starting conversion for checkpoint at: {args.checkpoint_path}")

    # 1. Load model configuration
    log.info("Loading model config...")
    config = AutoConfig.from_pretrained(args.checkpoint_path, trust_remote_code=True)

    # 2. Instantiate the correct model class based on the config
    model_type = getattr(config, "model_type", "standard")
    model_class_map = {
        "standard": StandardTransformerForCausalLM,
        "mod": MoDForCausalLM,
        "sdt": SDTForCausalLM,
        "stt": STTForCausalLM,
    }
    model_class = model_class_map.get(model_type)
    if not model_class:
        raise ValueError(f"Unknown model type '{model_type}' in config.")

    log.info(f"Instantiating model of type: {model_type} ({model_class.__name__})")
    
    # Pass the config as a dictionary to the model constructor
    model_kwargs = config.to_dict()
    model = model_class(config, model_type=model_type, **model_kwargs)

    # 3. Load the state dictionary from the custom model.pt file
    state_dict_path = os.path.join(args.checkpoint_path, "model.pt")
    log.info(f"Loading state dict from: {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location="cpu")

    # 4. Load the state dict into the model
    model.load_state_dict(state_dict)
    log.info("Successfully loaded state dict into the model.")

    # 5. Save the model in the standard Hugging Face format
    log.info(f"Saving model in standard format to: {args.output_path}")
    model.save_pretrained(args.output_path)

    # 6. Load and save the correct tokenizer
    log.info(f"Loading and saving tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.save_pretrained(args.output_path)

    log.info("\nConversion complete!")
    log.info(f"Your model is now ready for evaluation at: {args.output_path}")

if __name__ == "__main__":
    main()
