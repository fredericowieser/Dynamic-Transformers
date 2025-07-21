import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import json # Import json for pretty printing config

# --- IMPORTANT: MODEL PATH CONFIGURATION ---
# This path is extracted directly from your screenshot's log output.
# It points to the directory where your fine-tuned model and tokenizer are saved.
MODEL_PATH = "outputs/llama3.2-1b-dynamic-finetune-openassistant_guanaco-2025-07-21_13-00-37/final_model" 

# --- Your prompt to test the model ---
prompt = "The quick brown fox jumps over the lazy"

# --- Model Loading ---
print(f"Loading model from: {MODEL_PATH}...")
try:
    # Load the configuration first, so we can modify it if needed
    config = AutoConfig.from_pretrained(MODEL_PATH)
    
    # --- DEBUG: Print ORIGINAL config dictionary ---
    print("\n--- ORIGINAL CONFIG (full dictionary) ---")
    print(json.dumps(config.to_dict(), indent=2))
    print("-------------------------------------------\n")

    # List of common integer-like/float-like attributes in LlamaConfig that might sometimes be lists or wrong types
    # This list is more exhaustive for common culprits
    numeric_attributes_to_check = [
        "max_position_embeddings",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "head_dim",
        "pretraining_tp",
        "chunk_size_feed_forward",
        # Token IDs - can sometimes be lists for multi-token EOS, BOS, PAD
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "decoder_start_token_id",
        # Generation-related parameters that might be in config.json
        "max_length",
        "min_length",
        "num_beams",
        "num_beam_groups",
        "diversity_penalty",
        "temperature",
        "top_k",
        "top_p",
        "typical_p",
        "repetition_penalty",
        "length_penalty",
        "no_repeat_ngram_size",
        "encoder_no_repeat_ngram_size",
        "num_return_sequences",
        "exponential_decay_length_penalty",
    ]

    # --- AGGRESSIVE PATCH: Ensure common numerical attributes are correctly scalar types ---
    for attr_name in numeric_attributes_to_check:
        if hasattr(config, attr_name):
            current_value = getattr(config, attr_name)
            if isinstance(current_value, list):
                if len(current_value) > 0:
                    # For token IDs (like eos_token_id), taking the first is usually correct.
                    # For sizes/other numerical params, taking max is safer.
                    if any(sub in attr_name for sub in ["token_id", "pretraining_tp"]):
                        patched_value = current_value[0]
                    else:
                        patched_value = max(current_value) 
                    
                    print(f"WARNING: `{attr_name}` was a list ({current_value}), patched to: {patched_value}")
                    setattr(config, attr_name, patched_value)
                else:
                    # Handle empty list case by setting to None (for IDs) or 0 (for sizes)
                    print(f"WARNING: `{attr_name}` was an empty list, setting to None/0. You might need to adjust this default.")
                    setattr(config, attr_name, None if "token_id" in attr_name else 0)
            elif not isinstance(current_value, (int, float, type(None))): # Allow None, int, float
                # Attempt to convert other scalar types (e.g., bool) to int/float
                try:
                    if isinstance(current_value, bool): 
                        patched_value = int(current_value) # Convert bool (True=1, False=0)
                    else:
                        patched_value = int(current_value) # Try int first
                    print(f"WARNING: `{attr_name}` was type {type(current_value).__name__} ({current_value}), patched to int: {patched_value}")
                    setattr(config, attr_name, patched_value)
                except (ValueError, TypeError):
                    try: # Fallback to float if int conversion fails
                        patched_value = float(current_value)
                        print(f"WARNING: `{attr_name}` was type {type(current_value).__name__} ({current_value}), patched to float: {patched_value}")
                        setattr(config, attr_name, patched_value)
                    except (ValueError, TypeError):
                        print(f"WARNING: Could not convert `{attr_name}` ({current_value}) to int/float. Keeping original value. This might cause issues.")
        # else:
        #     # You might consider setting sensible defaults for missing critical attributes here
        #     # For simplicity, we only patch existing ones.
        
    # --- PATCH: Apply rope_scaling fix consistently ---
    # This block ensures that the rope_scaling config is well-formed,
    # as this model family sometimes has issues here too.
    if not hasattr(config, "rope_scaling") or config.rope_scaling is None or not isinstance(config.rope_scaling, dict):
        original_rope_scaling_value = getattr(config, "rope_scaling", "N/A")
        print(
            f"WARNING: Config `rope_scaling` is missing or not a dict (was: {original_rope_scaling_value}). "
            "Initializing `rope_scaling` with default 'linear' type and factor 1.0."
        )
        config.rope_scaling = {"type": "linear", "factor": 1.0}
    
    # Ensure 'type' and 'rope_type' are consistent if one exists, otherwise set defaults
    if "rope_type" in config.rope_scaling and config.rope_scaling["rope_type"] is not None:
        if "type" not in config.rope_scaling:
            config.rope_scaling["type"] = config.rope_scaling["rope_type"]
            print(f"WARNING: Setting `rope_scaling.type` to be consistent with 'rope_type': {config.rope_scaling['rope_type']}.")
    else: # 'rope_type' is missing or None
        if "type" not in config.rope_scaling:
            config.rope_scaling["type"] = "linear"
            print("WARNING: `rope_scaling` lacks 'rope_type' (or it's None) and 'type'. Setting `type` to 'linear'.")
        # Make `rope_type` consistent with `type` if it's still missing or None
        if "rope_type" not in config.rope_scaling or config.rope_scaling["rope_type"] is None:
            config.rope_scaling["rope_type"] = config.rope_scaling["type"]
            print(f"WARNING: Setting `rope_scaling.rope_type` to be consistent with 'type': {config.rope_scaling['type']}.")

    # Ensure 'factor' is present for scaling
    if "factor" not in config.rope_scaling:
        config.rope_scaling["factor"] = 1.0
        print(f"WARNING: Config `rope_scaling` lacks 'factor'. Setting to 1.0.")

    # --- END PATCHES ---

    # --- DEBUG: Print config details AFTER patching ---
    print("\n--- CONFIG AFTER PATCHING (full dictionary) ---")
    print(json.dumps(config.to_dict(), indent=2))
    print("-----------------------------------------------\n")


    # Now load the model with the (potentially patched) config
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=config, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

except Exception as e:
    print(f"\nERROR: Failed to load model or tokenizer from '{MODEL_PATH}'.")
    print(f"Please ensure the path is correct and the directory contains `config.json`, `pytorch_model.bin`, and `tokenizer.json` (or similar).")
    print(f"Details: {type(e).__name__}: {e}") 
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

input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)

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

generated_text = tokenizer.decode(generated_output[0, input_ids.shape[1]:], skip_special_tokens=True)

# --- Output the result ---
print(f"\n--- Generated Completion ---\n{generated_text.strip()}")
print("\n" + "="*50)
print("Vibe check complete! Experiment with different prompts and generation parameters.")
print("="*50)