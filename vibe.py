import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os

MODEL_PATH = "outputs/llama3.2-1b-dynamic-finetune-openassistant_guanaco-2025-07-21_13-00-37/final_model" 
prompt = "The quick brown fox jumps over the lazy"

print(f"Loading model from: {MODEL_PATH}...")
try:
    config = AutoConfig.from_pretrained(MODEL_PATH)
    
    # List of common integer-like attributes in LlamaConfig that might sometimes be lists
    int_attributes_to_check = [
        "max_position_embeddings",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "head_dim", # Llama 3 sometimes defines this directly
        "pretraining_tp", # Tensor parallelism rank
        "chunk_size_feed_forward",
    ]

    # --- AGGRESSIVE PATCH: Ensure common integer attributes are actually integers ---
    for attr_name in int_attributes_to_check:
        if hasattr(config, attr_name):
            current_value = getattr(config, attr_name)
            if isinstance(current_value, list):
                if len(current_value) > 0:
                    patched_value = max(current_value) # Take max as a safe default
                    print(f"WARNING: `{attr_name}` was a list ({current_value}), patched to: {patched_value}")
                    setattr(config, attr_name, patched_value)
                else:
                    # Handle empty list case (e.g., set to 0 or a reasonable default based on context)
                    print(f"WARNING: `{attr_name}` was an empty list, setting to 0. You might need to adjust this default.")
                    setattr(config, attr_name, 0)
            elif not isinstance(current_value, int):
                # Attempt to convert to int if it's float or another scalar type
                try:
                    patched_value = int(current_value)
                    print(f"WARNING: `{attr_name}` was type {type(current_value).__name__} ({current_value}), patched to int: {patched_value}")
                    setattr(config, attr_name, patched_value)
                except (ValueError, TypeError):
                    print(f"WARNING: Could not convert `{attr_name}` ({current_value}) to int. Keeping original value. This might cause issues.")


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

    # --- DEBUG: Print config details after patching ---
    # This will help identify if the problem persists and which value causes it.
    print("\n--- CONFIG AFTER PATCHING (relevant attributes) ---")
    for attr in int_attributes_to_check:
        if hasattr(config, attr):
            print(f"  {attr}: {getattr(config, attr)} (type: {type(getattr(config, attr))})")
    print(f"  rope_scaling: {config.rope_scaling} (type: {type(config.rope_scaling)})")
    print("--------------------------------------------------\n")


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

input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

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