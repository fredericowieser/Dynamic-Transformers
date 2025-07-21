import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os # Import os to join path parts safely

# --- IMPORTANT: MODEL PATH CONFIGURATION ---
# This path is extracted directly from your screenshot's log output.
# It points to the directory where your fine-tuned model and tokenizer are saved.
# No changes are needed if your environment is set up similarly to the screenshot's log.
MODEL_PATH = "outputs/llama3.2-1b-dynamic-finetune-openassistant_guanaco-2025-07-21_13-00-37/final_model" 

# --- Your prompt to test the model ---
prompt = "Explain the concept of quantum entanglement in simple terms:"

# --- Model Loading ---
print(f"Loading model from: {MODEL_PATH}...")
try:
    # Load the model and tokenizer. Use bfloat16 for the model if that's what was used in training.
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"\nERROR: Failed to load model or tokenizer from '{MODEL_PATH}'.")
    print(f"Please ensure the path is correct and the directory contains `config.json`, `pytorch_model.bin`, and `tokenizer.json` (or similar).")
    print(f"Details: {e}")
    exit(1) # Exit with an error code if loading fails

# Set pad_token_id if it's not explicitly set, common for text generation
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

# Determine device (GPU if available, otherwise CPU)
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
        max_new_tokens=150,      # Maximum number of *new* tokens to generate
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