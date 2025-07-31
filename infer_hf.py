#!/usr/bin/env python3
"""
Terminal chat for any HF causal LM with streaming,
letting the model decide when to stop (via its EOS token).
"""
import argparse
import threading
import sys

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)


def main():
    parser = argparse.ArgumentParser(
        description="Chat with an HF causal LM (streaming)"
    )
    parser.add_argument(
        "model_name",
        help=(
            "Model name or path, e.g. 'gpt2', "
            "'EleutherAI/gpt-j-6B', or 'meta-llama/Llama-3.2-1B'"
        ),
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help=(
            "Hard cap on new tokens. "
            "If omitted, uses tokenizer.model_max_length - prompt_length."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling p")
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Use greedy decoding (otherwise sampling)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", file=sys.stderr)

    # load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map="auto" if device.startswith("cuda") else None,
    )
    model.to(device).eval()
    print("✅ Model loaded", file=sys.stderr)

    # history + system prompt
    system = "System: You are a helpful assistant.\n"
    history = system
    sep = "\n"
    print("=== Chat ready (type 'exit' or Ctrl-C to quit) ===", file=sys.stderr)

    while True:
        try:
            user_in = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!", file=sys.stderr)
            break

        if not user_in or user_in.lower() in ("exit", "quit"):
            print("Goodbye!", file=sys.stderr)
            break

        # append to history
        history += f"User: {user_in}{sep}Assistant: "
        prompt = history

        # tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # compute max_new_tokens
        prompt_len = inputs["input_ids"].shape[1]
        if args.max_new_tokens is not None:
            max_new = args.max_new_tokens
        else:
            max_total = getattr(tokenizer, "model_max_length", 2048)
            max_new = max_total - prompt_len
            if max_new <= 0:
                max_new = 256  # fallback cap

        # prepare streamer
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # generation kwargs
        gen_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            streamer=streamer,
            do_sample=not args.no_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
            max_new_tokens=max_new,
        )

        # run generate in background thread
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        # stream out tokens
        assistant_out = ""
        for chunk in streamer:
            print(chunk, end="", flush=True)
            assistant_out += chunk
        thread.join()
        print()  # newline

        # update history
        history += assistant_out + sep


if __name__ == "__main__":
    main()