import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig # Import AutoConfig
import os # Import os for path operations

# --- IMPORTANT: MODEL PATH CONFIGURATION ---
# This path is extracted directly from your screenshot's log output.
# It points to the directory where your fine-tuned model and tokenizer are saved.
# If you run this script in the same environment, this path should be correct.
MODEL_PATH = "outputs/llama3.2-1b-dynamic-finetune-openassistant_guanaco-2025-07-21_13-00-37/final_model" 

# --- Your prompt to test the model ---
prompt = "The quick brown fox jumps over the lazy"

# --- Model Loading ---
print(f"Loading model from: {MODEL_PATH}...")
try:
    # Load the configuration first
    config = AutoConfig.from_pretrained(MODEL_PATH)
    # print(f"Loaded config: {config.to_dict()}") # Uncomment to inspect the full config if needed

    # --- PATCH 1: Handle potential list for max_position_embeddings ---
    # The error "list < int" suggests max_position_embeddings might be a list.
    if hasattr(config, "max_position_embeddings") and isinstance(config.max_position_embeddings, list):
        # Assuming it should be a single integer, take the maximum value from the list
        # or the first element if it's typically a single-element list.
        # For max_position_embeddings, taking the max is a safer default.
        original_max_pos_embed = config.max_position_embeddings
        config.max_position_embeddings = max(original_max_pos_embed)
        print(f"WARNING: `max_position_embeddings` was a list ({original_max_pos_embed}), patched to: {config.max_position_embeddings}")
    
    # --- PATCH 2: Apply rope_scaling fix consistently (from trainer.py) ---
    # This ensures config used for inference is robust, same as training config.
    if not hasattr(config, "rope_scaling") or config.rope_scaling is None or not isinstance(config.rope_scaling, dict):
        original_rope_scaling_value = getattr(config, "rope_scaling", "N/A")
        print(
            f"WARNING: Config `rope_scaling` is missing or not a dict (was: {original_rope_scaling_value}). "
            "Initializing `rope_scaling` with default 'linear' type and factor 1.0."
        )
        config.rope_scaling = {"type": "linear", "factor": 1.0}
    
    if "rope_type" in config.rope_scaling and config.rope_scaling["rope_type"] is not None:
        if "type" not in config.rope_scaling:
            print(
                f"WARNING: Config `rope_scaling` has 'rope_type' but no 'type'. "
                f"Setting `rope_scaling.type` to be consistent with 'rope_type': {config.rope_scaling['rope_type']}."
            )
            config.rope_scaling["type"] = config.rope_scaling["rope_type"]
    else:
        if "type" not in config.rope_scaling:
            print(
                f"WARNING: Config `rope_scaling` lacks 'rope_type' (or it's None) and 'type'. "
                "Setting `rope_scaling.type` to 'linear' and `factor` to 1.0."
            )
            config.rope_scaling["type"] = "linear"
            config.rope_scaling["factor"] = config.rope_scaling.get("factor", 1.0) # Ensure factor is also set
        
        if "rope_type" not in config.rope_scaling or config.rope_scaling["rope_type"] is None:
            config.rope_scaling["rope_type"] = config.rope_scaling["type"]

    if "factor" not in config.rope_scaling:
        print(
            f"WARNING: Config `rope_scaling` lacks 'factor'. Setting to 1.0."
        )
        config.rope_scaling["factor"] = 1.0
    # --- END PATCHES ---

    # Now load the model with the (potentially patched) config
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=config, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

except Exception as e:
    print(f"\nERROR: Failed to load model or tokenizer from '{MODEL_PATH}'.")
    print(f"Please ensure the path is correct and the directory contains `config.json`, `pytorch_model.bin`, and `tokenizer.json` (or similar).")
    print(f"Details: {type(e).__name__}: {e}") # Print exception type for better debugging
    exit(1) 

# Set pad_token_id if not already set (common for generation)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)
model.eval() # Set model to evaluation mode

# --- Text Generation ---
print(f"\n--- Input Prompt ---\n'{prompt}'")

input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output_tokens = model.generate(
        input_ids,
        max_new_tokens=50,       # Generate up to 50 new tokens
        temperature=0.7,         # Controls randomness (higher = more random)
        top_p=0.9,               # Nucleus sampling
        do_sample=True,          # Enable sampling
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(output_tokens[0, input_ids.shape[1]:], skip_special_tokens=True)

# --- Output ---
print(f"\n--- Generated Completion ---\n{generated_text.strip()}")
print("\n" + "="*50)
print("Vibe check complete!")
print("="*50)