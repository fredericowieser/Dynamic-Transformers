import argparse
from transformers import AutoConfig, AutoModelForCausalLM


def count_model_params(model_name_or_path: str) -> None:
    """
    Loads a causal-LM from `model_name_or_path` (local folder or HF ID),
    patches list-valued pad_token_id to an int if necessary, then
    prints total / trainable / non-trainable parameter counts.
    """
    # 1) Load & patch config
    config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    pad_val = config.pad_token_id
    if isinstance(pad_val, (list, tuple)):
        patched = pad_val[0] if len(pad_val) > 0 else None
        print(f"Patching config.pad_token_id from {pad_val} to {patched}")
        config.pad_token_id = patched

    # 2) Load model with the patched config
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, config=config, trust_remote_code=True
    )

    # 3) Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    print(f"\nModel: {model_name_or_path}")
    print(f"Total parameters:      {total:,}")
    print(f"Trainable parameters:  {trainable:,}")
    print(f"Non-trainable params:  {non_trainable:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Count parameters of a HF causal LM (local or remote)."
    )
    parser.add_argument(
        "model",
        type=str,
        help=(
            "Path to a local model folder (with config.json & weights) "
            "or a HuggingFace model ID (e.g. 'gpt2')."
        ),
    )
    args = parser.parse_args()
    count_model_params(args.model)


if __name__ == "__main__":
    main()