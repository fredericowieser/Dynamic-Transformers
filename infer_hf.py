#!/usr/bin/env python3
"""
Terminal chat for any HF causal LM with streaming,
using a proper chat template for true multi-turn conversation.
"""

import argparse
import threading
import sys
import torch
import re

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from src.utils.llama_config_utils import fix_rope_scaling, fix_pad_token_id

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids  # List of token ID sequences to stop on
    
    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_ids:
            if len(input_ids[0]) >= len(stop_id) and torch.equal(input_ids[0][-len(stop_id):], stop_id):
                return True
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Chat with an HF causal LM (streaming) with LLaMA chat template"
    )
    parser.add_argument(
        "model_name",
        help="Model name or path, e.g. 'meta-llama/Llama-3.2-1B'"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Hard cap on new tokens"
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

    # Load & patch config BEFORE you instantiate the model
    config = AutoConfig.from_pretrained(args.model_name)
    config = fix_rope_scaling(config)
    config = fix_pad_token_id(config)

    # Load the tokenizer and set a proper chat template
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set the chat template for LLaMA-3.2 style
    tokenizer.chat_template = (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'system' %}"
        "<|start_header_id|>system<|end_header_id|>\n{{ message['content'].strip() }}<|eot_id|>"
        "{%- elif message['role'] == 'user' %}"
        "<|start_header_id|>user<|end_header_id|>\n{{ message['content'].strip() }}<|eot_id|>"
        "{%- elif message['role'] == 'assistant' %}"
        "<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'].strip() }}<|eot_id|>"
        "{% endif %}"
        "{% endfor %}"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.to(device).eval()

    # Patch out the tiny default max_length in generation_config.json
    if hasattr(model, "generation_config"):
        max_len = getattr(tokenizer, "model_max_length", None)
        if max_len is None:
            max_len = min(model.config.max_position_embeddings, 4096)  # Cap based on Llama practices
        model.generation_config.max_length = max_len

    print("✅ Model loaded", file=sys.stderr)
    system_prompt = "You are a helpful assistant."
    history = []  # List of dicts for messages

    print("=== Chat ready (type 'exit' or Ctrl-C to quit) ===", file=sys.stderr)
    while True:
        try:
            user_in = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!", file=sys.stderr)
            sys.exit(0)
        
        if not user_in or user_in.lower() in ("exit", "quit"):
            print("Goodbye!", file=sys.stderr)
            sys.exit(0)

        history.append({"role": "user", "content": user_in})
        messages = [{"role": "system", "content": system_prompt}] + history
        
        # Generate prompt using the chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        prompt_len = inputs["input_ids"].shape[1]
        if args.max_new_tokens is not None:
            max_new = args.max_new_tokens
        else:
            max_total = getattr(tokenizer, "model_max_length", 2048)
            max_new = max_total - prompt_len
            if max_new <= 0:
                max_new = 256  # Fallback cap
        
        stop_ids = [tokenizer.encode("<|eot_id|>", add_special_tokens=False)]
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])
        
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        
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
            stopping_criteria=stopping_criteria,
        )
        
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        
        assistant_out = ""
        for chunk in streamer:
            if "<|eot_id|>" in chunk:
                # Stop and clean up
                cleaned_chunk = chunk.split("<|eot_id|>")[0].strip()
                assistant_out += cleaned_chunk
                print(cleaned_chunk, end="", flush=True)
                break
            assistant_out += chunk
            print(chunk, end="", flush=True)
        thread.join()
        print()  # Newline after response
        
        # Clean and append to history, removing any template artifacts
        cleaned_response = re.sub(r"<\|[^>]+>", "", assistant_out).strip()  # Remove any <|tokens|>
        history.append({"role": "assistant", "content": cleaned_response})

if __name__ == "__main__":
    main()