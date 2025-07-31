import argparse
import glob
import os
import sys
import threading

import torch
from transformers import AutoTokenizer, TextIteratorStreamer

from src.models.d_llama_config import DynamicLlamaConfig
from src.models.d_llama_causal_lm import DynamicLlamaForCausalLM

try:
    from safetensors.torch import load_file as safe_load
except ImportError:
    safe_load = None


def load_weights(model, model_dir, device):
    state_dict = {}
    if safe_load:
        shard_paths = sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors")))
        if shard_paths:
            for shard in shard_paths:
                sd_part = safe_load(shard, device=device)
                state_dict.update(sd_part)
            model.load_state_dict(state_dict, strict=False)
            return

        single = os.path.join(model_dir, "pytorch_model.safetensors")
        if os.path.isfile(single):
            sd = safe_load(single, device=device)
            model.load_state_dict(sd, strict=False)
            return
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
    p = argparse.ArgumentParser(description="Chat w/ DynamicLlamaForCausalLM")
    p.add_argument("model_dir", help="Path to `final_model` folder")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument(
        "--no_sample", action="store_true",
        help="Use greedy decoding instead of sampling"
    )
    p.add_argument("--dynamic_k", type=float, default=None,
                   help="Override config.dynamic_k")
    p.add_argument("--ce_bias", type=float, default=None,
                   help="Override config.ce_bias")
    p.add_argument("--gate_warmup_iters", type=int, default=0,
                   help="Override config.gate_warmup_iters")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", file=sys.stderr)

    # load config + tokenizer
    config = DynamicLlamaConfig.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = config.pad_token_id or tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id

    # build model & load weights
    model = DynamicLlamaForCausalLM(config).to(device).eval()
    print("Loading weights…", file=sys.stderr)
    load_weights(model, args.model_dir, device)
    print("✅ Weights loaded", file=sys.stderr)

    # override dynamic parameters
    dyn_k = args.dynamic_k if args.dynamic_k is not None else config.dynamic_k
    if dyn_k is None:
        raise ValueError("dynamic_k must be set in config or via --dynamic_k")
    model.set_dynamic_k(dyn_k)
    model.set_gate_warmup_iters(args.gate_warmup_iters)
    cb = args.ce_bias if args.ce_bias is not None else config.ce_bias or 0.0
    model.set_ce_bias(cb)

    # chat loop
    sep = "\n"
    history = ""
    print("=== Chat ready (type 'exit' or Ctrl-C to quit) ===", file=sys.stderr)
    while True:
        try:
            user_in = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…", file=sys.stderr)
            break
        if not user_in:
            continue
        if user_in.lower() in ("exit", "quit"):
            print("Bye!", file=sys.stderr)
            break

        history += f"User: {user_in}{sep}Assistant: "
        prompt = history

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            streamer=streamer,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.no_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
        thread = threading.Thread(
            target=model.generate, kwargs=gen_kwargs, daemon=True
        )
        thread.start()

        # print & accumulate streamed tokens
        assistant_out = ""
        for chunk in streamer:
            print(chunk, end="", flush=True)
            assistant_out += chunk
        thread.join()
        print()  # newline after assistant finishes

        history += f"{assistant_out}{sep}"


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()