import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import json 

MODEL_PATH = "outputs/llama3.2-1b-dynamic-finetune-openassistant_guanaco-2025-07-21_13-00-37/final_model" 
prompt = "The quick brown fox jumps over the lazy"


print(f"Loading model from: {MODEL_PATH}...")
try:
    config = AutoConfig.from_pretrained(MODEL_PATH)
    
    # --- DEBUG: Print ORIGINAL config dictionary ---
    print("\n--- ORIGINAL CONFIG (full dictionary) ---")
    print(json.dumps(config.to_dict(), indent=2))
    print("-------------------------------------------\n")

    numeric_attributes_to_check = [
        "max_position_embeddings", "hidden_size", "intermediate_size", 
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads", 
        "vocab_size", "head_dim", "pretraining_tp", "chunk_size_feed_forward",
        "pad_token_id", "bos_token_id", "eos_token_id", "decoder_start_token_id",
        "max_length", "min_length", "num_beams", "num_beam_groups", "diversity_penalty", 
        "temperature", "top_k", "top_p", "typical_p", "repetition_penalty", 
        "length_penalty", "no_repeat_ngram_size", "encoder_no_repeat_ngram_size", 
        "num_return_sequences", "exponential_decay_length_penalty",
    ]

    for attr_name in numeric_attributes_to_check:
        if hasattr(config, attr_name):
            current_value = getattr(config, attr_name)
            if isinstance(current_value, list):
                if len(current_value) > 0:
                    if any(sub in attr_name for sub in ["token_id", "pretraining_tp"]):
                        patched_value = current_value[0]
                    else:
                        patched_value = max(current_value) 
                    print(f"WARNING: `{attr_name}` was a list ({current_value}), patched to: {patched_value}")
                    setattr(config, attr_name, patched_value)
                else:
                    print(f"WARNING: `{attr_name}` was an empty list, setting to None/0. You might need to adjust this default.")
                    setattr(config, attr_name, None if "token_id" in attr_name else 0)
            elif not isinstance(current_value, (int, float, type(None))):
                try:
                    if isinstance(current_value, bool): 
                        patched_value = int(current_value)
                    else:
                        patched_value = int(current_value) 
                    print(f"WARNING: `{attr_name}` was type {type(current_value).__name__} ({current_value}), patched to int: {patched_value}")
                    setattr(config, attr_name, patched_value)
                except (ValueError, TypeError):
                    try:
                        patched_value = float(current_value)
                        print(f"WARNING: `{attr_name}` was type {type(current_value).__name__} ({current_value}), patched to float: {patched_value}")
                        setattr(config, attr_name, patched_value)
                    except (ValueError, TypeError):
                        print(f"WARNING: Could not convert `{attr_name}` ({current_value}) to int/float. Keeping original value. This might cause issues.")
        
    if not hasattr(config, "rope_scaling") or config.rope_scaling is None or not isinstance(config.rope_scaling, dict):
        original_rope_scaling_value = getattr(config, "rope_scaling", "N/A")
        print(
            f"WARNING: Config `rope_scaling` is missing or not a dict (was: {original_rope_scaling_value}). "
            "Initializing `rope_scaling` with default 'linear' type and factor 1.0."
        )
        config.rope_scaling = {"type": "linear", "factor": 1.0}
    
    if "rope_type" in config.rope_scaling and config.rope_scaling["rope_type"] is not None:
        if "type" not in config.rope_scaling:
            config.rope_scaling["type"] = config.rope_scaling["rope_type"]
            print(f"WARNING: Setting `rope_scaling.type` to be consistent with 'rope_type': {config.rope_scaling['rope_type']}."
            )
    else:
        if "type" not in config.rope_scaling:
            config.rope_scaling["type"] = "linear"
            print("WARNING: `rope_scaling` lacks 'rope_type' (or it's None) and 'type'. Setting `type` to 'linear'.")
        if "rope_type" not in config.rope_scaling or config.rope_scaling["rope_type"] is None:
            config.rope_scaling["rope_type"] = config.rope_scaling["type"]
            print(f"WARNING: Setting `rope_scaling.rope_type` to be consistent with 'type': {config.rope_scaling['type']}."
            )

    if "factor" not in config.rope_scaling:
        config.rope_scaling["factor"] = 1.0
        print(f"WARNING: Config `rope_scaling` lacks 'factor'. Setting to 1.0.")

    print("\n--- CONFIG AFTER PATCHING (full dictionary) ---")
    print(json.dumps(config.to_dict(), indent=2))
    print("-----------------------------------------------\n")

    # Load the model with the (potentially patched) config and specified dtype
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=config, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

except Exception as e:
    print(f"\nERROR: Failed to load model or tokenizer from '{MODEL_PATH}'.")
    print(f"Please ensure the path is correct and the directory contains `config.json`, `pytorch_model.bin` (or `model.safetensors` files), and `tokenizer.json` (or similar).")
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

print(f"\n--- Input Prompt ---\n'{prompt}'")

input_encoding = tokenizer.encode_plus(
    prompt, 
    return_tensors="pt", 
    add_special_tokens=True,
    return_attention_mask=True
)
input_ids = input_encoding["input_ids"].to(device)
attention_mask = input_encoding["attention_mask"].to(device)

# FIX: Use torch.autocast for mixed precision inference
# This matches the 'bf16-mixed' precision used during training,
# preventing dtype mismatches during internal generation operations.
with torch.no_grad():
    with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16): # device.split(':')[0] handles 'cuda:0' vs 'cuda'
        generated_output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,       
            temperature=0.7,         
            top_p=0.9,               
            do_sample=True,          
            pad_token_id=tokenizer.pad_token_id, 
            eos_token_id=tokenizer.eos_token_id, 
        )

generated_text = tokenizer.decode(generated_output[0, input_ids.shape[1]:], skip_special_tokens=True)

# Output the result
print(f"\n--- Generated Completion ---\n{generated_text.strip()}")
print("\n" + "="*50)
print("Vibe check complete! Experiment with different prompts and generation parameters.")
print("="*50)