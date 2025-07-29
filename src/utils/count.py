import argparse

from transformers import AutoConfig, AutoModelForCausalLM


def _patch_rope_scaling(config):
    """
    Ensure config.rope_scaling is a dict containing keys
    'type', 'rope_type', and 'factor', so LlamaRotaryEmbedding
    never KeyErrors.
    """
    rs = getattr(config, "rope_scaling", None)
    if rs is None or not isinstance(rs, dict):
        print(
            f"Patching missing/invalid rope_scaling "
            f"(was: {rs!r}) → "
            "{'type':'linear','rope_type':'linear','factor':1.0}"
        )
        config.rope_scaling = {"type": "linear", "rope_type": "linear", "factor": 1.0}
        return

    # Ensure 'type'
    if "type" not in rs or rs["type"] is None:
        if "rope_type" in rs and rs["rope_type"] is not None:
            patched = rs["rope_type"]
        else:
            patched = "linear"
        print(f"Patching rope_scaling['type'] from {rs.get('type')!r} → {patched!r}")
        rs["type"] = patched

    # Ensure 'rope_type'
    if "rope_type" not in rs or rs["rope_type"] is None:
        print(
            f"Patching rope_scaling['rope_type'] from {rs.get('rope_type')!r} → {rs['type']!r}"
        )
        rs["rope_type"] = rs["type"]

    # Ensure 'factor'
    if "factor" not in rs or rs["factor"] is None:
        print(f"Patching rope_scaling['factor'] from {rs.get('factor')!r} → 1.0")
        rs["factor"] = 1.0

    config.rope_scaling = rs


def count_model_params(model_name_or_path: str) -> None:
    """
    Loads a causal-LM from `model_name_or_path` (local folder or HF ID),
    patches list-valued pad_token_id and rope_scaling, then prints
    total/trainable/non-trainable parameter counts.
    """
    # 1) Load & patch config
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    # Patch pad_token_id if it's a list
    pad_val = getattr(config, "pad_token_id", None)
    if isinstance(pad_val, (list, tuple)):
        patched = pad_val[0] if len(pad_val) > 0 else None
        print(f"Patching pad_token_id from {pad_val!r} → {patched!r}")
        config.pad_token_id = patched

    # Patch rope_scaling for LlamaRotaryEmbedding
    _patch_rope_scaling(config)

    # 2) Load model with the patched config
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
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
            "Local model folder (with config.json & weights) "
            "or HuggingFace ID (e.g. 'gpt2')."
        ),
    )
    args = parser.parse_args()
    count_model_params(args.model)


if __name__ == "__main__":
    main()
