import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Ensure project root is in Python path for imports
try:
    project_root = Path(__file__).parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import custom model classes from the new repo structure
    from src.models.dtf.causalLM import DTFForCausalLM
    from src.models.mod.causalLM import MoDForCausalLM
    from src.models.tdtf.causalLM import TDTFForCausalLM
    from src.models.standard.causalLM import StandardTransformerForCausalLM
    from transformers import Qwen2Config # Use Qwen2Config as the base config
except ImportError as e:
    print("❌ Error: Could not import custom model classes from 'src'.")
    print(f"   (Details: {e})")
    print("Please ensure you run this script from the root of your project directory.")
    sys.exit(1)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Register custom architectures with transformers Auto classes
# This is crucial for AutoModelForCausalLM.from_pretrained to correctly load custom models
AutoConfig.register("dtf", Qwen2Config) # Register DTF with Qwen2Config
AutoModelForCausalLM.register(Qwen2Config, DTFForCausalLM)

AutoConfig.register("mod", Qwen2Config) # Register MoD with Qwen2Config
AutoModelForCausalLM.register(Qwen2Config, MoDForCausalLM)

AutoConfig.register("tdtf", Qwen2Config) # Register TDTF with Qwen2Config
AutoModelForCausalLM.register(Qwen2Config, TDTFForCausalLM)

AutoConfig.register("standard", Qwen2Config) # Register Standard with Qwen2Config
AutoModelForCausalLM.register(Qwen2Config, StandardTransformerForCausalLM)


def run_inference(model_path: str, prompt: str, max_new_tokens: int):
    """Loads the model and tokenizer, then generates and prints text."""
    log.info(f"Starting inference for model: {model_path}")

    # Load model and tokenizer
    try:
        log.info("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        log.info("✅ Model and tokenizer loaded.")
    except Exception as e:
        log.error(f"❌ Failed to load model/tokenizer. Error: {e}", exc_info=True)
        return

    # Generate text
    try:
        log.info(f"Generating {max_new_tokens} tokens for prompt: '{prompt}'")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generation_kwargs = {"max_new_tokens": max_new_tokens}
        
        # The new models (DTF, MoD, TDTF) are designed to handle use_cache internally
        # with their causal routers. So, we don't explicitly disable it here.
        # If the model's config has use_cache=False, it will be respected.

        outputs = model.generate(**inputs, **generation_kwargs)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("\n" + "="*25 + " Generated Text " + "="*25)
        print(generated_text)
        print("="*68 + "\n")

    except Exception as e:
        log.error(f"❌ An error occurred during text generation: {e}", exc_info=True)

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
