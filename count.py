import argparse
import torch
from transformers import AutoModelForCausalLM


def count_model_params(model_dir: str) -> None:
    """
    Loads a causal LM from `model_dir` and prints:
      - total parameters
      - trainable parameters
      - non-trainable parameters
    """
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    print(f"Model directory: {model_dir}")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count parameters of a Hugging Face causal LM."
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to the model folder (e.g. outputs/.../final_model)",
    )
    args = parser.parse_args()
    count_model_params(args.model_dir)