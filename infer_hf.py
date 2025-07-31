import argparse
import threading
import sys
from typing import List
import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# import your fixes
from src.utils.llama_config_utils import fix_rope_scaling, fix_pad_token_id

class StopOnSequence(StoppingCriteria):
    def __init__(self, stop_ids: List[List[int]]):
        super().__init__()
        # stop_ids is a list of token‐id sequences (each a List[int])
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        for seq in self.stop_ids:
            if len(input_ids[0]) >= len(seq) and \
               input_ids[0, -len(seq) :].tolist() == seq:
                return True
        return False

# your existing build_prompt, but ensure you open [INST] on every user turn:
def build_prompt(system_prompt, history, tokenizer):
    bos = tokenizer.bos_token or "<s>"
    inst, inst_end = "[INST]", "[/INST]"
    sys_open, sys_close = "<<SYS>>", "<</SYS>>"
    prompt = f"{bos}{inst} {sys_open}\n{system_prompt}\n{sys_close}\n\n"

    # history is a list of dicts {role:"user"/"assistant", content:str}
    for msg in history:
        if msg["role"] == "user":
            # open a fresh INST block for every user
            prompt += f"{inst} {msg['content']} {inst_end}\n"
        else:
            # assistant text follows, then two newlines
            prompt += f"{msg['content']}\n\n"

    # At this point, history should end in a user message
    # so the model will generate the assistant response
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.to(device).eval()

    # Patch out the tiny default max_length (20) in generation_config.json
    if hasattr(model, "generation_config"):
        # prefer tokenizer.model_max_length, fall back to config.max_position_embeddings
        max_len = getattr(tokenizer, "model_max_length", None)
        if max_len is None:
            max_len = model.config.max_position_embeddings
        model.generation_config.max_length = max_len

    print("✅ Model loaded", file=sys.stderr)
    system_prompt = "You are a helpful assistant."
    history = []

    stop_ids = [
        tokenizer.encode("[/INST]", add_special_tokens=False)
    ]
    stopper = StoppingCriteriaList([StopOnSequence(stop_ids)])

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

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
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
            #max_new_tokens=args.max_new_tokens,
            stopping_criteria=stopper,
        )
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        # stream + prune out the first appearance of "[/INST]"
        assistant_out = ""
        for chunk in streamer:
            # if the model ever emits the literal "[/INST]", stop here
            if "[/INST]" in chunk:
                idx = chunk.index("[/INST]")
                assistant_out += chunk[:idx]
                print(chunk[:idx], end="", flush=True)
                break
            assistant_out += chunk
            print(chunk, end="", flush=True)
        thread.join()
        print()  # newline after the assistant reply

        # strip _all_ template markers before appending to history
        clean = re.sub(r"\[\/?INST\]", "", assistant_out).strip()
        history.append({"role": "assistant", "content": clean})

if __name__ == "__main__":
    main()