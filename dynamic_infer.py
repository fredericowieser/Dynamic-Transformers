import argparse
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)

def _patch_pad_token_id(config):
    pad_val = getattr(config, "pad_token_id", None)
    if isinstance(pad_val, (list, tuple)):
        patched = pad_val[0] if len(pad_val) > 0 else None
        print(f"Patching config.pad_token_id from {pad_val} → {patched}")
        config.pad_token_id = patched

def _patch_rope_scaling(config):
    rs = getattr(config, "rope_scaling", None)
    if rs is None or not isinstance(rs, dict):
        print(f"Patching missing/invalid rope_scaling (was {rs!r}) → "
              "{'type':'linear','rope_type':'linear','factor':1.0}")
        config.rope_scaling = {
            "type": "linear", "rope_type": "linear", "factor": 1.0
        }
        return

    # Ensure 'type'
    if "type" not in rs or rs["type"] is None:
        new_type = rs.get("rope_type") or "linear"
        print(f"Patching rope_scaling['type'] from {rs.get('type')!r} → {new_type!r}")
        rs["type"] = new_type

    # Ensure 'rope_type'
    if "rope_type" not in rs or rs["rope_type"] is None:
        print(f"Patching rope_scaling['rope_type'] from {rs.get('rope_type')!r} → {rs['type']!r}")
        rs["rope_type"] = rs["type"]

    # Ensure 'factor'
    if "factor" not in rs or rs["factor"] is None:
        print(f"Patching rope_scaling['factor'] from {rs.get('factor')!r} → 1.0")
        rs["factor"] = 1.0

    config.rope_scaling = rs

def main():
    parser = argparse.ArgumentParser(
        description="Inference with DynamicLlama; patches pad_token_id & rope_scaling"
    )
    parser.add_argument(
        "model",
        type=str,
        help="Path to local model folder or HF model ID",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The input prompt to complete",
    )
    parser.add_argument(
        "--dynamic_k",
        type=float,
        default=None,
        help="Override the gating threshold k (if omitted, uses config value)",
    )
    parser.add_argument(
        "--print_gates",
        action="store_true",
        help="Print per-layer mean gate activations after generation",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load & patch config
    config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=True
    )
    _patch_pad_token_id(config)
    _patch_rope_scaling(config)

    # 2) Load tokenizer + model with patched config
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id

    model.to(device).eval()

    # 3) Override dynamic_k and enable gate logging
    if args.dynamic_k is not None and hasattr(model, "set_dynamic_k"):
        model.set_dynamic_k(args.dynamic_k)
    if args.print_gates and hasattr(model, "enable_gate_logging"):
        model.enable_gate_logging(True)

    # 4) Tokenize & generate
    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)

    with torch.no_grad():
        # match training bf16 setup
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    completion = tokenizer.decode(
        output_ids[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    print("\n--- Completion ---\n" + completion.strip())

    # 5) Optionally print gate activations
    if args.print_gates and hasattr(model, "get_last_gate_means"):
        means = model.get_last_gate_means()
        if means is not None:
            print("\n--- Per-layer mean gate activations ---")
            for i, m in enumerate(means):
                print(f"Layer {i:2d}: {m:.3f}")
        else:
            print("Gate logging was not enabled or no activations recorded.")

if __name__ == "__main__":
    main()