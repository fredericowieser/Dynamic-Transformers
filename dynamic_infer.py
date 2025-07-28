import argparse
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
import sys
from src.models.dynamic_llama_causal import DynamicLlamaForCausalLM


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
    if "type" not in rs or rs["type"] is None:
        new_type = rs.get("rope_type") or "linear"
        print(f"INFO: Patching rope_scaling['type'] from {rs.get('type')!r} -> {new_type!r}", file=sys.stderr)
        rs["type"] = new_type
    if "rope_type" not in rs or rs["rope_type"] is None:
        print(f"INFO: Patching rope_scaling['rope_type'] from {rs.get('rope_type')!r} -> {rs['type']!r}", file=sys.stderr)
        rs["rope_type"] = rs["type"]
    if "factor" not in rs or rs["factor"] is None:
        print(f"INFO: Patching rope_scaling['factor'] from {rs.get('factor')!r} -> 1.0", file=sys.stderr)
        rs["factor"] = 1.0
    config.rope_scaling = rs

def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a DynamicLlama model and inspect gate activations."
    )
    parser.add_argument("model_path", type=str, help="Path to the saved model directory.")
    parser.add_argument("--prompt", type=str, required=True, help="The input prompt for the model.")
    parser.add_argument("--dynamic_k", type=float, default=None, help="Override the model's default dynamic_k.")
    parser.add_argument("--print_gates", action="store_true", help="Enable and print per-layer gate activations.")
    args = parser.parse_args()

    print("--- Loading Model ---", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", file=sys.stderr)

    config = AutoConfig.from_pretrained(args.model_path)
    _patch_pad_token_id(config)
    _patch_rope_scaling(config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = DynamicLlamaForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id

    model.to(device).eval()

    # ─────────── NEW: Set inference gating params ───────────
    model.config.dynamic_k = args.dynamic_k if args.dynamic_k is not None else 0.5
    model.config.gate_warmup_iters = 0  # disable warm-up bias
    model.config.ce_bias = 0.0          # no CE bias
    # ────────────────────────────────────────────────────────

    # --- CRITICAL CHECK ---
    # Verify that the correct custom model class was actually loaded.
    if not isinstance(model, DynamicLlamaForCausalLM):
        print("\n--- FATAL ERROR ---", file=sys.stderr)
        print("The loaded model is NOT the custom 'DynamicLlamaForCausalLM'.", file=sys.stderr)
        print("This means the custom architecture was not found and the extra layers were discarded.", file=sys.stderr)
        print("\nTo fix this, please install your project in editable mode:", file=sys.stderr)
        print("  uv pip install -e .", file=sys.stderr)
        print("\nThen, re-run this script.", file=sys.stderr)
        sys.exit(1)

    print("Model loaded successfully as DynamicLlamaForCausalLM.", file=sys.stderr)

    if hasattr(model, "set_dynamic_k") and args.dynamic_k is not None:
        model.set_dynamic_k(args.dynamic_k)
        print(f"Set dynamic_k to: {args.dynamic_k}", file=sys.stderr)
    
    if args.print_gates and hasattr(model, "enable_gate_logging"):
        model.enable_gate_logging(True)
        print("Gate activation logging is enabled.", file=sys.stderr)

    num_layers = model.config.num_hidden_layers
    accum_gate_means = [[] for _ in range(num_layers)]  # List of lists: accum_gate_means[i] = means for layer i across all tokens

    # Monkey-patch model's forward to collect gate means after each call (each generated token)
    original_forward = model.forward
    def wrapped_forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        if model._log_gates:
            last_means = model.get_last_gate_means()
            if last_means:  # Ensure something was recorded
                for i, mean_val in enumerate(last_means):
                    accum_gate_means[i].append(mean_val)
        return out
    model.forward = wrapped_forward

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

    if args.print_gates:
            print(f"\n--- Per-Layer Gate Stats (mean ± std across all {len(accum_gate_means[0]) if accum_gate_means else 0} generated tokens) ---")
            for i in range(num_layers):
                layer_means = accum_gate_means[i]
                if layer_means:
                    # Convert to tensor for mean/std
                    layer_tensor = torch.tensor(layer_means)
                    overall_mean = layer_tensor.mean().item()
                    overall_std = layer_tensor.std().item() if len(layer_tensor) > 1 else 0.0
                    print(f"Layer {i:2d}: {overall_mean:.4f} ± {overall_std:.4f}")
                else:
                    print(f"Layer {i:2d}: No data (logging may have failed)", file=sys.stderr)

if __name__ == "__main__":
    main()