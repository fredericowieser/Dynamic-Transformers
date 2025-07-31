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
    p = argparse.ArgumentParser(description="Chat with an HF causal LM (streaming)")
    p.add_argument(
        "model_name",
        help="Model name or path (e.g. gpt2, EleutherAI/gpt-j-6B, /path/to/ckpt)",
    )
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument(
        "--no_sample",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", file=sys.stderr)

    # Load HF model + tokenizer
    print(f"Loading model `{args.model_name}`…", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        # some models lack pad_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map="auto" if device.startswith("cuda") else None,
    )
    model.to(device).eval()
    print("✅ Model loaded.", file=sys.stderr)

    # Chat history with a tiny system prompt
    system = "System: You are a helpful assistant.\n"
    history = system
    sep = "\n"
    print("=== Chat ready (CTRL-C or 'exit' to quit) ===", file=sys.stderr)

    while True:
        try:
            user_in = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.", file=sys.stderr)
            break

        if not user_in:
            continue
        if user_in.lower() in ("exit", "quit"):
            print("Goodbye!", file=sys.stderr)
            break

        # Append user turn & prepare prompt
        history += f"User: {user_in}{sep}Assistant: "
        prompt = history

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Set up streamer
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Generation kwargs
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

        # Fire off generation on a background thread
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        # Stream out tokens as they arrive
        assistant_text = ""
        for chunk in streamer:
            print(chunk, end="", flush=True)
            assistant_text += chunk
        thread.join()
        print()  # newline

        # Append assistant reply to history
        history += assistant_text + sep


if __name__ == "__main__":
    main()