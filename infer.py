import argparse
import sys

import torch
from transformers import AutoConfig, AutoTokenizer

from src.models.d_llama_causal_lm import DynamicLlamaForCausalLM
from src.utils.llamam_config_utils import fix_pad_token_id, fix_rope_scaling


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a DynamicLlama model and inspect gate activations."
    )
    parser.add_argument(
        "model_path", type=str, help="Path to the saved model directory."
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="The input prompt for the model."
    )
    parser.add_argument(
        "--dynamic_k",
        type=float,
        default=None,
        help="Override the model's default dynamic_k.",
    )
    parser.add_argument(
        "--ce_bias",
        type=float,
        default=0.0,
        help="Override the model's default CE‐bias.",
    )
    parser.add_argument(
        "--print_gates",
        action="store_true",
        help="Enable and print per-layer gate activations.",
    )
    args = parser.parse_args()

    print("--- Loading Model ---", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", file=sys.stderr)

    # Load & patch config
    config = AutoConfig.from_pretrained(args.model_path)
    config = fix_pad_token_id(config)
    config = fix_rope_scaling(config)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = config.pad_token_id or tokenizer.eos_token_id

    # Model
    model = DynamicLlamaForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=(
            "auto" if device == "cuda" else None
        ),  # Add this line to handle multi-GPU/CPU loading
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id

    # Inference‐time gating params
    if args.dynamic_k is not None:
        model.config.dynamic_k = args.dynamic_k
        print(f"Set dynamic_k to: {args.dynamic_k}", file=sys.stderr)
    # Always disable warmup at inference
    model.config.gate_warmup_iters = 0
    # CE bias override
    model.config.ce_bias = args.ce_bias
    print(f"Set ce_bias to: {args.ce_bias}", file=sys.stderr)

    # Sanity check
    if not isinstance(model, DynamicLlamaForCausalLM):
        print("\n--- FATAL ERROR ---", file=sys.stderr)
        print(
            "The loaded model is NOT the custom 'DynamicLlamaForCausalLM'.",
            file=sys.stderr,
        )
        print(
            "Make sure you `pip install -e .` so that `trust_remote_code` finds your class.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.print_gates and hasattr(model, "enable_gate_logging"):
        model.enable_gate_logging(True)
        print("Gate activation logging is enabled.", file=sys.stderr)

    # Prepare to accumulate gate means across ALL tokens
    num_layers = model.config.num_hidden_layers
    accum_gate_means = [[] for _ in range(num_layers)]

    # Monkey-patch forward to record per-layer gate means each call
    original_forward = model.forward

    def wrapped_forward(*f_args, **f_kwargs):
        out = original_forward(*f_args, **f_kwargs)
        if model._log_gates:
            last = model.get_last_gate_means()
            if last:
                for i, v in enumerate(last):
                    accum_gate_means[i].append(v)
        return out

    model.forward = wrapped_forward

    # Tokenize & generate
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16),
    ):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    # Decode
    completion = tokenizer.decode(
        output_ids[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    print("\n--- Completion ---")
    print(completion.strip())

    # Print overall gate‐stats
    if args.print_gates:
        n_tokens = len(accum_gate_means[0]) if accum_gate_means else 0
        print(f"\n--- Per-Layer Gate Stats (mean ± std over {n_tokens} tokens) ---")
        for i, layer_vals in enumerate(accum_gate_means):
            if layer_vals:
                t = torch.tensor(layer_vals)
                mean, std = t.mean().item(), t.std(unbiased=False).item()
                print(f"Layer {i:2d}: {mean:.4f} ± {std:.4f}")
            else:
                print(f"Layer {i:2d}: No data recorded", file=sys.stderr)


if __name__ == "__main__":
    main()
