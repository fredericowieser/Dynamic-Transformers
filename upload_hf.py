"""
A robust script to upload a custom DynamicQwen model, tokenizer, and a
generated model card to the Hugging Face Hub.

This script handles the registration of the custom architecture, generation
of a detailed README.md from training and evaluation artifacts, and the
final upload process.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from huggingface_hub import HfApi, HfFolder
from transformers import AutoConfig, AutoModelForCausalLM

# --- Pre-flight Check: Ensure project root is in the Python path ---
# This allows for consistent absolute imports from the 'src' package.
try:
    # Get the absolute path of the project root (the directory containing this script)
    project_root = Path(__file__).parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.models.qwen.causal_lm import DynamicQwenForCausalLM
    from src.models.qwen.config import DynamicQwenConfig
    from src.models.qwen.tokenizer import DynamicQwenTokenizer
except ImportError as e:
    print("❌ Error: Could not import custom model classes from 'src'.")
    print(f"   (Details: {e})")
    print("Please ensure you run this script from the root of your project directory.")
    sys.exit(1)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def _register_custom_architecture():
    """Registers the custom DynamicQwen model with the AutoModel classes."""
    log.info("Registering custom 'dynamic_qwen' architecture...")
    AutoConfig.register("dynamic_qwen", DynamicQwenConfig)
    AutoModelForCausalLM.register(DynamicQwenConfig, DynamicQwenForCausalLM)
    log.info("✅ Architecture registered successfully.")


def _generate_eval_section(eval_results: Dict[str, Any]) -> str:
    """Generates the evaluation results section for the model card."""
    if not eval_results:
        return ""

    header = "\n## Evaluation\nResults on standard benchmarks:\n\n"
    table_header = "| Task | Metric | Value |\n|---|---|---|\n"
    table_rows = []
    for task, metrics in sorted(eval_results.items()):
        for metric, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                table_rows.append(f"| {task} | {metric} | {value:.4f} |")
    return header + table_header + "\n".join(table_rows)


def _generate_training_section(training_config: Dict[str, Any]) -> str:
    """Generates the training details section for the model card."""
    if not training_config:
        return ""

    header = "\n## Training Details\n"
    # Safely access nested keys
    datasets = training_config.get("data", {}).get("dataset_configs", [])
    dataset_info = ""
    if datasets:
        dataset_info += "The model was trained on a mixture of the following datasets:\n"
        for ds in datasets:
            name = ds.get("dataset_name", "N/A")
            subset = ds.get("train_subset_ratio", 1.0) * 100
            dataset_info += f"- `{name}` ({subset:.0f}% of its training data)\n"

    # Extract a relevant subset of the config to display
    rel_config = {
        "run_name": training_config.get("run", {}).get("name"),
        "model": training_config.get("model", {}).get("model_cfg"),
        "training": training_config.get("training"),
    }
    config_yaml = yaml.dump(rel_config, indent=2, sort_keys=False, default_flow_style=False)
    config_section = f"\n### Training Configuration\nKey parameters:\n```yaml\n{config_yaml}```\n"

    return header + dataset_info + config_section


def generate_model_card(
    model_config: Dict[str, Any],
    repo_id: str,
    training_config: Optional[Dict[str, Any]] = None,
    eval_results: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generates a complete README.md model card from available artifacts.
    """
    arch = model_config.dynamic_architecture.upper()
    gamma = model_config.capacity_gamma
    base_model = model_config._name_or_path

    return f"""---
license: apache-2.0
tags:
- qwen
- dynamic-transformer
- {model_config.dynamic_architecture}
---

# Model Card for {repo_id.split('/')[-1]}

This model is a dynamically-computed version of `{base_model}`, fine-tuned
using the **{arch}** architecture.

- **Dynamic Architecture**: `{arch}`
- **Capacity Gamma (γ)**: `{gamma}`

The `{arch}` architecture enables the model to conditionally skip parts of its
computation, aiming for improved efficiency. The `capacity_gamma` parameter
controls the portion of tokens processed by the dynamic components.

## How to Use

This model requires `trust_remote_code=True` to load the custom architecture.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# It is recommended to load in bfloat16 for efficiency
model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Example usage
prompt = "The capital of the United Kingdom is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
{_generate_training_section(training_config)}
{_generate_eval_section(eval_results)}
"""


def main():
    """Main function to drive the model upload process."""
    parser = argparse.ArgumentParser(
        description="Upload a DynamicQwen model to the Hugging Face Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_path", type=Path, help="Path to the local saved model directory.")
    parser.add_argument("repo_name", type=str, help="Desired repository name on the Hub.")
    parser.add_argument("--hf_username", type=str, required=True, help="Your Hugging Face username.")
    parser.add_argument("--eval_results", type=Path, help="Optional: Path to evaluation results JSON file.")
    parser.add_argument("--commit_message", type=str, default="Upload dynamic Qwen model", help="Commit message.")
    parser.add_argument("--private", action="store_true", help="Create a private repository.")
    args = parser.parse_args()

    if not HfFolder.get_token():
        log.error("❌ Not logged in to Hugging Face. Please run `huggingface-cli login` first.")
        return

    _register_custom_architecture()
    repo_id = f"{args.hf_username}/{args.repo_name}"

    try:
        # --- 1. Load Artifacts ---
        log.info(f"Loading artifacts from {args.model_path}...")
        if not args.model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {args.model_path}")

        model_config = AutoConfig.from_pretrained(args.model_path)
        
        # Find training config (expected in parent's .hydra subdir)
        training_config_path = args.model_path.parent / ".hydra" / "config.yaml"
        training_config = yaml.safe_load(training_config_path.read_text()) if training_config_path.exists() else None
        if not training_config:
            log.warning("Could not find training config. Training details will be omitted.")

        # Load eval results if provided
        eval_results = json.loads(args.eval_results.read_text()) if args.eval_results and args.eval_results.exists() else None
        if args.eval_results and not eval_results:
            log.warning(f"Evaluation results file not found at: {args.eval_results}")

        # --- 2. Generate and Save Model Card ---
        log.info("Generating model card...")
        model_card = generate_model_card(model_config, repo_id, training_config, eval_results)
        (args.model_path / "README.md").write_text(model_card, encoding="utf-8")
        log.info("✅ Model card created successfully.")

        # --- 3. Upload to Hub ---
        api = HfApi()
        log.info(f"🚀 Creating repository '{repo_id}' and uploading files...")
        repo_url = api.create_repo(repo_id=repo_id, exist_ok=True, private=args.private)
        api.upload_folder(folder_path=str(args.model_path), repo_id=repo_id, commit_message=args.commit_message)
        
        log.info(f"✨ Upload complete! Access your model at: {repo_url}")

    except FileNotFoundError as e:
        log.error(f"❌ File not found: {e}")
    except Exception as e:
        log.error(f"❌ An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
