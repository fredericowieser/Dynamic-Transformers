import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str,
                        help="local folder or HF id with dynamic checkpoint")
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--dynamic_k", type=float, default=0.9,
                        help="override the gate threshold k")
    parser.add_argument("--print_gates", action="store_true",
                        help="print per-layer mean gate activations")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mod = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        trust_remote_code=True).to(device).eval()

    # --- dynamic-k & logging ----------------------------------------------
    if hasattr(mod, "set_dynamic_k"):
        mod.set_dynamic_k(args.dynamic_k)
    if args.print_gates and hasattr(mod, "enable_gate_logging"):
        mod.enable_gate_logging(True)

    # --- generation --------------------------------------------------------
    input_ids = tok(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad(), torch.autocast(device_type=device.split(":")[0],
                                         dtype=torch.bfloat16):
        out = mod.generate(
            **input_ids,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
        )

    print("\n--- Completion ---")
    print(tok.decode(out[0], skip_special_tokens=True))

    # --- gate report -------------------------------------------------------
    if args.print_gates:
        gates = mod.get_last_gate_means()
        if gates is not None:
            print("\n--- Per-layer mean gate activations ---")
            for i, g in enumerate(gates):
                print(f"Layer {i:2d}: {g:.3f}")
        else:
            print("Gate logging was not enabled.")


if __name__ == "__main__":
    main()