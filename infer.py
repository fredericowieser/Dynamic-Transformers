#!/usr/bin/env python3
"""
Inference script for DynamicLlamaForCausalLM w/ manual weight loading
that handles sharded safetensors.
"""
import argparse
import glob
import os
import sys

import torch
from transformers import AutoTokenizer

from src.models.d_llama_config import DynamicLlamaConfig
from src.models.d_llama_causal_lm import DynamicLlamaForCausalLM

try:
    from safetensors.torch import load_file as safe_load
except ImportError:
    safe_load = None


def load_weights(model, model_dir, device):
    """
    Load model weights from:
      - one or more .safetensors shards
      - or a single pytorch_model.safetensors
      - or a single pytorch_model.bin
    """
    state_dict = {}
    # 1) sharded safetensors: model-*.safetensors
    if safe_load:
        shard_paths = sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors")))
        if shard_paths:
            for shard in shard_paths:
                sd_part = safe_load(shard, device=device)
                state_dict.update(sd_part)
            model.load_state_dict(state_dict, strict=False)
            return

        # 2) single-file pytorch_model.safetensors
        single = os.path.join(model_dir, "pytorch_model.safetensors")
        if os.path.isfile(single):
            sd = safe_load(single, device=device)
            model.load_state_dict(sd, strict=False)
            return

    # 3) single-file pytorch_model.bin
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        sd = torch.load(bin_path, map_location=device)
        model.load_state_dict(sd, strict=False)
        return

    raise FileNotFoundError(
        f"No weights found in {model_dir}. "
        "Expected sharded '*.safetensors' or 'pytorch_model.safetensors' or 'pytorch_model.bin'."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Infer with DynamicLlamaForCausalLM (manual load w/ shards)"
    )
    parser.add_argument(
        "model_dir", help="Path to the saved `final_model` folder"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time,",
        help="Text prompt",
    )
    parser.add_argument(
        "--dynamic_k",
        type=float,
        default=None,
        help="Override config.dynamic_k",
    )
    parser.add_argument(
        "--ce_bias",
        type=float,
        default=None,
        help="Override config.ce_bias",
    )
    parser.add_argument(
        "--gate_warmup_iters",
        type=int,
        default=0,
        help="Override config.gate_warmup_iters",
    )
    parser.add_argument(
        "--print_gates",
        action="store_true",
        help="Log per-layer gate activations",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling",
    )
    parser.add_argument(
        "--do_sample",
        type=bool,
        default=False,
        choices=[True, False],
        help="Enable sampling; otherwise greedy",
    )
    args = parser.parse_args()

    # 1) Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", file=sys.stderr)

    # 2) Load config + tokenizer
    config = DynamicLlamaConfig.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = config.pad_token_id or tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id

    # 3) Build model & load weights manually
    model = DynamicLlamaForCausalLM(config)
    model.to(device)
    model.eval()

    print("🔄 Loading weights...", file=sys.stderr)
    load_weights(model, args.model_dir, device)
    print("✅ Weights loaded.", file=sys.stderr)

    # 4) Override dynamic params (always define these attributes)
    dyn_k = args.dynamic_k if args.dynamic_k is not None else config.dynamic_k
    if dyn_k is None:
        raise ValueError("dynamic_k must be set in config.json or via --dynamic_k")
    model.set_dynamic_k(dyn_k)
    print(f"-> dynamic_k = {model.dynamic_k}", file=sys.stderr)

    gw = args.gate_warmup_iters
    model.set_gate_warmup_iters(gw)
    print(f"-> gate_warmup_iters = {model.gate_warmup_iters}", file=sys.stderr)

    cb = args.ce_bias if args.ce_bias is not None else getattr(config, "ce_bias", 0.0)
    model.set_ce_bias(cb)
    print(f"-> ce_bias = {model.ce_bias}", file=sys.stderr)

    # 5) Optional gate logging
    accum_means = []
    if args.print_gates and hasattr(model, "enable_gate_logging"):
        model.enable_gate_logging(True)
        original_forward = model.forward

        def wrapped_forward(*f_args, **f_kwargs):
            out = original_forward(*f_args, **f_kwargs)
            if getattr(model, "_log_gates", False):
                last = model.get_last_gate_means()
                if last:
                    accum_means.append(last.copy())
            return out

        model.forward = wrapped_forward
        print("✅ Gate logging enabled", file=sys.stderr)

    # 6) Tokenize prompt
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    # 7) Generate
    with torch.no_grad(), torch.autocast(
        device_type=device.split(":")[0], dtype=torch.bfloat16
    ):
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    # 8) Decode only new tokens
    gen = outputs[0, inputs.input_ids.shape[1] :].tolist()
    completion = tokenizer.decode(gen, skip_special_tokens=True)
    print("\n--- Completion ---")
    print(completion.strip())

    # 9) Print gate stats if requested
    if args.print_gates and accum_means:
        arr = torch.tensor(accum_means)  # (steps, layers)
        m = arr.mean(dim=0)
        s = arr.std(dim=0, unbiased=False)
        print(f"\n--- Gate activations over {len(arr)} steps ---")
        for i, (mi, si) in enumerate(zip(m.tolist(), s.tolist())):
            print(f"Layer {i:2d}: {mi:.4f} ± {si:.4f}")


if __name__ == "__main__":
    main()