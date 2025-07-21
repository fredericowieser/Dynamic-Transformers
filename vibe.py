import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig # Import AutoConfig
import os # Import os for path operations

# --- IMPORTANT: MODEL PATH CONFIGURATION ---
# This path is extracted directly from your screenshot's log output.
# It points to the directory where your fine-tuned model and tokenizer are saved.
# No changes are needed if your environment is set up similarly to the screenshot's log.
MODEL_PATH = "outputs/llama3.2-1b-dynamic-finetune-openassistant_guanaco-2025-07-21_13-00-37/final_model" 

# --- Your prompt to test the model ---
prompt = "The quick brown fox jumps over the lazy"

# --- Model Loading ---
print(f"Loading model from: {MODEL_PATH}...")
try:
    # Load the configuration first, so we can modify it if needed
    config = AutoConfig.from_pretrained(MODEL_PATH)
    
    # --- CRITICAL PATCH: Ensure max_position_embeddings is an int ---
    # The error "list < int" suggests max_position_embeddings might be a list.
    if hasattr(config, "max_position_embeddings"):
        if isinstance(config.max_position_embeddings, list):
            original_value = config.max_position_embeddings
            # Assume it should be a single integer; take the maximum if it's a list of options.
            config.max_position_embeddings = max(original_value) 
            print(f"WARNING: `max_position_embeddings` was a list ({original_value}), patched to: {config.max_position_embeddings}")
        # Ensure it's treated as an int even if it was a float, which can happen.
        elif not isinstance(config.max_position_embeddings, int):
            original_value = config.max_position_embeddings
            config.max_position_embeddings = int(original_value)
            print(f"WARNING: `max_position_embeddings` was type {type(original_value).__name__} ({original_value}), patched to int: {config.max_position_embeddings}")
    else:
        # If it's entirely missing, provide a reasonable default (e.g., Llama2 default context length)
        print("WARNING: `max_position_embeddings` not found in config, setting to 4096 as a default.")
        config.max_position_embeddings = 4096


    # --- SECONDARY PATCH: Apply rope_scaling fix consistently (from trainer.py) ---
    # This block ensures that the rope_scaling config is well-formed,
    # as this model family sometimes has issues here too.
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
            config.rope_scaling["factor"] = config.rope_scaling.get("factor", 1.0) 
        
        if "rope_type" not in config.rope_scaling or config.rope_scaling["rope_type"] is None:
            config.rope_scaling["rope_type"] = config.rope_scaling["type"]

    if "factor" not in config.rope_scaling:
        print(
            f"WARNING: Config `rope_scaling` lacks 'factor'. Setting to 1.0."
        )
        config.rope_scaling["factor"] = 1.0
    # --- END PATCHES ---

    # DEBUG: Print the critical config attributes *after* patches, *before* model instatiation
    print(f"DEBUG: Config.max_position_embeddings value after patch: {config.max_position_embeddings} (type: {type(config.max_position_embeddings)})")
    print(f"DEBUG: Config.rope_scaling value after patch: {config.rope_scaling} (type: {type(config.rope_scaling)})")


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
model.eval() # Set model to evaluation mode for inference

# --- Text Generation ---
print(f"\n--- Input Prompt ---\n'{prompt}'")

# Encode the prompt into token IDs
# `add_special_tokens=True` ensures BOS token is added if applicable for the model
input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)

# Generate text. Adjust parameters as needed for desired output style.
with torch.no_grad(): # Disable gradient calculations for faster inference and less memory
    generated_output = model.generate(
        input_ids,
        max_new_tokens=50,       # Maximum number of *new* tokens to generate
        temperature=0.7,         # Controls randomness. Lower = more deterministic, Higher = more creative/random.
        top_p=0.9,               # Nucleus sampling: only consider tokens that cumulatively sum to top_p probability
        do_sample=True,          # If False, greedy decoding is used (no temperature/top_p)
        pad_token_id=tokenizer.pad_token_id, # Essential for batching and stopping generation
        eos_token_id=tokenizer.eos_token_id, # Model stops when this token is generated
    )

# Decode the generated tokens back to text.
# We slice `[0, input_ids.shape[1]:]` to get only the newly generated text, excluding the prompt.
generated_text = tokenizer.decode(generated_output[0, input_ids.shape[1]:], skip_special_tokens=True)

# --- Output the result ---
print(f"\n--- Generated Completion ---\n{generated_text.strip()}") # .strip() removes leading/trailing whitespace
print("\n" + "="*50)
print("Vibe check complete! Experiment with different prompts and generation parameters.")
print("="*50)