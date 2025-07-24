import argparse
from transformers import AutoConfig, AutoModelForCausalLM


def count_model_params(model_dir: str) -> None:
    """
    Loads a causal‐LM from `model_dir`, patches list‐valued pad_token_id
    to an int (if necessary), and prints:
      • total parameters
      • trainable parameters
      • non‐trainable parameters
    """
    # 1) Load & patch config
    config = AutoConfig.from_pretrained(model_dir)
    pad_val = config.pad_token_id
    if isinstance(pad_val, (list, tuple)):
        patched = pad_val[0] if len(pad_val) > 0 else None
        print(f"Patching config.pad_token_id from {pad_val} to {patched}")
        config.pad_token_id = patched

    # 2) Load model using the patched config
    model = AutoModelForCausalLM.from_pretrained(model_dir, config=config)

    # 3) Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    print(f"Model directory:       {model_dir}")
    print(f"Total parameters:      {total:,}")
    print(f"Trainable parameters:  {trainable:,}")
    print(f"Non‐trainable params:  {non_trainable:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Count parameters of a HuggingFace causal LM."
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to your saved `final_model` folder "
             "(must contain config.json & model weights).",
    )
    args = parser.parse_args()
    count_model_params(args.model_dir)


if __name__ == "__main__":
    main()