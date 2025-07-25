import argparse
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import sys

def _patch_pad_token_id(config):
    """Fixes pad_token_id if it's a list, which causes errors."""
    pad_val = getattr(config, "pad_token_id", None)
    if isinstance(pad_val, (list, tuple)):
        patched = pad_val[0] if len(pad_val) > 0 else None
        print(f"INFO: Patching config.pad_token_id from {pad_val} -> {patched}", file=sys.stderr)
        config.pad_token_id = patched

def _patch_rope_scaling(config):
    """Ensures rope_scaling is a valid dict to prevent loading errors."""
    rs = getattr(config, "rope_scaling", None)
    if rs is None or not isinstance(rs, dict):
        print(f"INFO: Patching missing/invalid rope_scaling (was {rs!r}) -> {{'type':'linear', ...}}", file=sys.stderr)
        config.rope_scaling = {"type": "linear", "rope_type": "linear", "factor": 1.0}
        return

    # Ensure 'type' exists, falling back to 'rope_type' or 'linear'
    if "type" not in rs or rs["type"] is None:
        new_type = rs.get("rope_type") or "linear"
        print(f"INFO: Patching rope_scaling['type'] from {rs.get('type')!r} -> {new_type!r}", file=sys.stderr)
        rs["type"] = new_type

    # Ensure 'rope_type' exists, falling back to 'type'
    if "rope_type" not in rs or rs["rope_type"] is None:
        print(f"INFO: Patching rope_scaling['rope_type'] from {rs.get('rope_type')!r} -> {rs['type']!r}", file=sys.stderr)
        rs["rope_type"] = rs["type"]

    # Ensure 'factor' exists
    if "factor" not in rs or rs["factor"] is None:
        print(f"INFO: Patching rope_scaling['factor'] from {rs.get('factor')!r} -> 1.0", file=sys.stderr)
        rs["factor"] = 1.0

    config.rope_scaling = rs

def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a DynamicLlama model and inspect gate activations."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the saved model directory (e.g., outputs/.../final_model).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The input prompt for the model.",
    )
    parser.add_argument(
        "--dynamic_k",
        type=float,
        default=None,
        help="Override the model's default dynamic_k gating threshold.",
    )
    parser.add_argument(
        "--print_gates",
        action="store_true",
        help="Enable and print the per-layer gate activations after generation.",
    )
    args = parser.parse_args()

    print("--- Loading Model ---", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", file=sys.stderr)

    # 1. Load and patch the configuration BEFORE loading the model
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    _patch_pad_token_id(config)
    _patch_rope_scaling(config)

    # 2. Load the tokenizer and the model using the patched config
    #    trust_remote_code=True is essential for loading your custom class
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Ensure tokenizer pad token is set for open-ended generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id

    model.to(device).eval()
    print("Model loaded successfully.", file=sys.stderr)

    # 3. Configure the dynamic model for inference
    if hasattr(model, "set_dynamic_k"):
        if args.dynamic_k is not None:
            model.set_dynamic_k(args.dynamic_k)
            print(f"Set dynamic_k to: {args.dynamic_k}", file=sys.stderr)
    
    if args.print_gates:
        if hasattr(model, "enable_gate_logging"):
            model.enable_gate_logging(True)
            print("Gate activation logging is enabled.", file=sys.stderr)
        else:
            print("WARNING: --print_gates was passed, but model has no 'enable_gate_logging' method.", file=sys.stderr)

    # 4. Tokenize prompt and generate completion
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad(), torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    completion = tokenizer.decode(output_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    print("\n--- Completion ---")
    print(completion.strip())

    # 5. Report gate activations if requested
    if args.print_gates and hasattr(model, "get_last_gate_means"):
        gate_means = model.get_last_gate_means()
        if gate_means:
            print("\n--- Per-Layer Mean Gate Activations (during generation) ---")
            for i, mean_val in enumerate(gate_means):
                print(f"Layer {i:2d}: {mean_val:.4f}")
        else:
            print("\n--- No gate activations were recorded. ---", file=sys.stderr)

if __name__ == "__main__":
    main()