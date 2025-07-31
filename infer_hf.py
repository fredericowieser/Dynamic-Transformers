import argparse
import sys
import threading

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from src.utils.llamam_config_utils import fix_rope_scaling, fix_pad_token_id


def main():
    parser = argparse.ArgumentParser(
        description="Chat with an HF causal LM (streaming, multi-turn)"
    )
    parser.add_argument(
        "model_name",
        help="HF model name or local path (e.g. meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Nucleus sampling top-p"
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", file=sys.stderr)

    # 1) Load config, fix rope-scaling & pad-token
    config = AutoConfig.from_pretrained(args.model_name)
    if hasattr(config, "rope_scaling"):
        config = fix_rope_scaling(config)
    config = fix_pad_token_id(config)

    # 2) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = config.pad_token_id

    # 3) Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map="auto" if device.startswith("cuda") else None,
    )
    model.to(device).eval()
    print("✅ Model loaded", file=sys.stderr)

    # 4) Chat loop
    system_prompt = "System: You are a helpful assistant.\n"
    history = system_prompt
    sep = "\n"
    print("=== Chat ready (type 'exit' or Ctrl-C to quit) ===", file=sys.stderr)

    while True:
        try:
            user_in = input("User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!", file=sys.stderr)
            break

        if not user_in:
            continue
        if user_in.lower() in ("exit", "quit"):
            print("Goodbye!", file=sys.stderr)
            break

        # append user turn
        history += f"User: {user_in}{sep}Assistant: "
        prompt = history

        # tokenize + prepare streamer
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
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
            use_cache=False,  # force full forward each step
        )

        # launch generation in background thread
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        # stream out tokens as they arrive
        assistant_out = ""
        for chunk in streamer:
            print(chunk, end="", flush=True)
            assistant_out += chunk
        thread.join()
        print()  # newline after turn

        # add assistant reply to history
        history += assistant_out + sep


if __name__ == "__main__":
    main()