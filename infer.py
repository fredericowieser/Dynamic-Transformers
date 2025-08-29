"""
A simple, self-contained script to run inference with a trained DynamicQwen model.

This script can load a model from a local path or a Hugging Face Hub repository
and generate text based on a user-provided prompt.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

# --- Pre-flight Check: Ensure project root is in the Python path ---
# This allows for consistent absolute imports from the 'src' package.
try:
    project_root = Path(__file__).parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.models.qwen.causal_lm import DynamicQwenForCausalLM
    from src.models.qwen.config import DynamicQwenConfig
    from src.models.qwen.tokenizer import DynamicQwenTokenizer
except ImportError as e:
    print("❌ Error: Could not import custom model classes from 'src'.")
    print(f"   (Details: {e})")
    print("Please ensure you run this script from the root of your project directory.")
    sys.exit(1)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def run_inference(model_path: str, prompt: str, max_new_tokens: int):
    """Loads the model and tokenizer, then generates and prints text."""
    log.info(f"Starting inference for model: {model_path}")

    # --- 1. Register the custom architecture ---
    log.info("Registering custom 'dynamic_qwen' architecture...")
    AutoConfig.register("dynamic_qwen", DynamicQwenConfig)
    AutoModelForCausalLM.register(DynamicQwenConfig, DynamicQwenForCausalLM)
    log.info("✅ Architecture registered successfully.")

    # --- 2. Load model and tokenizer ---
    try:
        log.info("Loading model and tokenizer...")
        # It's crucial to use trust_remote_code=True for custom architectures
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",  # Use bfloat16 if available
            device_map="auto",   # Automatically use GPU if available
        )
        # Use the custom tokenizer class to ensure any special tokens are handled
        tokenizer = DynamicQwenTokenizer.from_pretrained(model_path)
        log.info("✅ Model and tokenizer loaded.")
    except Exception as e:
        log.error(f"❌ Failed to load model/tokenizer. Error: {e}")
        return

    # --- 3. Generate text ---
    try:
        log.info(f"Generating {max_new_tokens} tokens for prompt: '{prompt}'")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate text using the model
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("\n" + "="*25 + " Generated Text " + "="*25)
        print(generated_text)
        print("="*68 + "\n")

    except Exception as e:
        log.error(f"❌ An error occurred during text generation: {e}")


def main():
    """Parses command-line arguments and calls the inference function."""
    parser = argparse.ArgumentParser(
        description="Run inference with a custom DynamicQwen model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the local model directory or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="The text prompt to generate from.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="The maximum number of new tokens to generate.",
    )
    args = parser.parse_args()
    run_inference(args.model_path, args.prompt, args.max_new_tokens)


if __name__ == "__main__":
    main()
