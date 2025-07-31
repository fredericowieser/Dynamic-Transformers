import argparse
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
    Try to load weights from safetensors first, then .bin.
    """
    # 1) safetensors
    if safe_load:
        path = os.path.join(model_dir, "pytorch_model.safetensors")
        if os.path.isfile(path):
            sd = safe_load(path, device=device)
            model.load_state_dict(sd, strict=False)
            return
    # 2) fallback to .bin
    path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.isfile(path):
        sd = torch.load(path, map_location=device)
        model.load_state_dict(sd, strict=False)
        return

    raise FileNotFoundError(
        "No `pytorch_model.safetensors` or `pytorch_model.bin` in " + model_dir
    )


def main():
    parser = argparse.ArgumentParser(
        description="Infer with DynamicLlamaForCausalLM (manual load)"
    )
    parser.add_argument(
        "model_dir", help="Path to saved `final_model` folder"
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
        action="store_true",
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

    load_weights(model, args.model_dir, device)
    print("✅ Model weights loaded", file=sys.stderr)

    # 4) Override dynamic params
    if args.dynamic_k is not None:
        model.set_dynamic_k(args.dynamic_k)
        print(f"-> dynamic_k = {args.dynamic_k}", file=sys.stderr)
    model.set_gate_warmup_iters(args.gate_warmup_iters)
    if args.ce_bias is not None:
        model.set_ce_bias(args.ce_bias)
        print(f"-> ce_bias = {args.ce_bias}", file=sys.stderr)

    # 5) Optional gate logging
    accum_means = []
    if args.print_gates and hasattr(model, "enable_gate_logging"):
        model.enable_gate_logging(True)
        original_forward = model.forward

        def wrapped_forward(*f_args, **f_kwargs):
            out = original_forward(*f_args, **f_kwargs)
            if model._log_gates:
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
            use_cache=False,  # ensure full forward each step
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