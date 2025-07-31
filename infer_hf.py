import argparse
import threading
import sys
import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

# import your fixes
from src.utils.llama_config_utils import fix_rope_scaling, fix_pad_token_id

def build_prompt(system_prompt, history, tokenizer):
    """
    Build a single prompt string using LLaMA’s chat template.

    <s>[INST] <<SYS>>
    {system_prompt}
    <</SYS>>

    {user1} [/INST]
    {assistant1}
    [INST] {user2} [/INST]
    {assistant2}
    …
    [INST] {last_user} [/INST]
    """
    bos = tokenizer.bos_token or "<s>"
    inst = "[INST]"
    inst_end = "[/INST]"
    sys_open = "<<SYS>>"
    sys_close = "<</SYS>>"

    # start with BOS + system block
    prompt = f"{bos}{inst} {sys_open}\n{system_prompt}\n{sys_close}"

    # interleave user / assistant turns
    for msg in history:
        role, content = msg["role"], msg["content"].strip()
        if role == "user":
            # open a new user instruction
            prompt += f"\n\n{content} {inst_end}"
        else:  # assistant
            # assistant reply follows immediately
            prompt += f"\n\n{content}"

    # model will generate the assistant reply to the last user
    return prompt

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

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Finally load the model with the patched config
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map="auto" if device.startswith("cuda") else None,
    )
    model.to(device).eval()
    print("✅ Model loaded", file=sys.stderr)
    system_prompt = "You are a helpful assistant."
    history = []

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

        history.append({"role": "user", "content": user_in})
        prompt = build_prompt(system_prompt, history, tokenizer)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # compute max_new_tokens, create streamer, etc… (as before)

        # generate & stream
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
        )
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        assistant_out = ""
        for chunk in streamer:
            print(chunk, end="", flush=True)
            assistant_out += chunk
        thread.join()
        print()
        history.append({"role": "assistant", "content": assistant_out})


if __name__ == "__main__":
    main()