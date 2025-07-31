import argparse
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.models.d_llama_config import DynamicLlamaConfig
from src.models.d_llama_causal_lm import DynamicLlamaForCausalLM


def main():
    parser = argparse.ArgumentParser(
        description="Inference for DynamicLlamaForCausalLM"
    )
    parser.add_argument(
        "model_path", type=str, help="Path to the saved model directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time,",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--dynamic_k",
        type=float,
        default=None,
        help="Override model.config.dynamic_k",
    )
    parser.add_argument(
        "--ce_bias",
        type=float,
        default=None,
        help="Override model.config.ce_bias",
    )
    parser.add_argument(
        "--gate_warmup_iters",
        type=int,
        default=0,
        help="Number of warmup iters (set to 0 to disable at inference)",
    )
    parser.add_argument(
        "--print_gates",
        action="store_true",
        help="Enable and print per-layer gate activation stats",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate",
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
        help="Nucleus sampling top-p",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling (otherwise greedy)",
    )
    args = parser.parse_args()

    # device & dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", file=sys.stderr)

    # register custom config/model so Auto* picks them up
    AutoConfig.register("dynamic_llama", DynamicLlamaConfig)
    AutoModelForCausalLM.register(DynamicLlamaConfig, DynamicLlamaForCausalLM)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # model (do NOT pass config= here)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    # override dynamic params
    if args.dynamic_k is not None:
        model.set_dynamic_k(args.dynamic_k)
    model.set_gate_warmup_iters(args.gate_warmup_iters)
    if args.ce_bias is not None:
        model.set_ce_bias(args.ce_bias)

    # optional gate logging: accumulate per-layer means each forward call
    accum = []
    if args.print_gates and hasattr(model, "enable_gate_logging"):
        model.enable_gate_logging(True)
        orig_fwd = model.forward

        def wrapped_forward(*f_args, **f_kwargs):
            out = orig_fwd(*f_args, **f_kwargs)
            if model._log_gates:
                means = model.get_last_gate_means()
                if means:
                    accum.append(means.copy())
            return out

        model.forward = wrapped_forward
        print("Gate logging enabled.", file=sys.stderr)

    # tokenize prompt
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    # generation (disable use_cache so gating runs each step)
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.split(":")[0], dtype=torch.bfloat16
    ):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    # decode only the newly generated tokens
    gen_tokens = output_ids[0, inputs.input_ids.shape[1] :].tolist()
    completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    print("\n--- Completion ---")
    print(completion.strip())

    # print gate stats if requested
    if args.print_gates and accum:
        arr = torch.tensor(accum)  # shape: (num_steps, num_layers)
        means = arr.mean(dim=0)
        stds = arr.std(dim=0, unbiased=False)
        print(f"\n--- Gate Activations (over {len(accum)} steps) ---")
        for i, (m, s) in enumerate(zip(means.tolist(), stds.tolist())):
            print(f"Layer {i:2d}: {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()