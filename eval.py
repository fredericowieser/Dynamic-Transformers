import argparse
import json
import logging
import os
import torch
import numpy as np
import wandb
from lm_eval import simple_evaluate

from transformers import AutoConfig, AutoModelForCausalLM
from src.models.qwen.causal_lm import DynamicQwenForCausalLM
from src.models.qwen.config import DynamicQwenConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Register custom architecture with transformers
log.info("Registering custom 'dynamic_qwen' architecture with transformers.")
AutoConfig.register("dynamic_qwen", DynamicQwenConfig)
AutoModelForCausalLM.register(DynamicQwenConfig, DynamicQwenForCausalLM)

# Define task groups for easy evaluation
TASK_SUITES = {
    "general": ["arc_challenge", "hellaswag", "mmlu", "winogrande", "truthfulqa_mc2"],
    "math": ["gsm8k", "math_qa"],
    "code": ["humaneval", "mbpp"],
    "quick_test": ["arc_challenge", "hellaswag"]
}

def _make_json_serializable(obj):
    """
    Recursively traverses a dictionary or list to convert non-serializable
    types (like torch.dtype, torch.Tensor, numpy types) to JSON-friendly formats.
    """
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(elem) for elem in obj]
    elif isinstance(obj, torch.dtype):
        return str(obj)  # Convert dtype to string, e.g., "torch.bfloat16"
    elif isinstance(obj, torch.Tensor):
        return obj.tolist() # Convert tensor to list
    elif isinstance(obj, np.ndarray):
        return obj.tolist() # Convert numpy array to list
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item() # Convert numpy numbers to standard Python numbers
    return obj

def get_wandb_run_dir(model_path: str) -> str | None:
    """
    Tries to find the wandb run directory by reading wandb_info.json,
    resuming the run, and getting its local directory.
    """
    wandb_info_path = os.path.join(model_path, "wandb_info.json")
    if not os.path.exists(wandb_info_path):
        log.warning("wandb_info.json not found. Results will be saved locally in --output_dir.")
        return None
    with open(wandb_info_path, "r") as f:
        wandb_info = json.load(f)
    try:
        # Resume the run to get access to its properties and filesystem
        run = wandb.init(
            project=wandb_info["project"],
            entity=wandb_info["entity"],
            id=wandb_info["run_id"],
            resume="must"
        )
        log.info(f"Successfully resumed wandb run '{run.name}' (ID: {run.id}).")
        return run.dir
    except Exception as e:
        log.error(f"Failed to resume wandb run. Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on a custom Qwen model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model directory.")
    parser.add_argument("--tasks", type=str, default="quick_test", help=f"Tasks or suites. Available: {list(TASK_SUITES.keys())}")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Fallback directory to save results.")
    args = parser.parse_args()

    # Expand task suites
    task_names = sorted(list(set(
        task for suite in args.tasks.split(',') for task in TASK_SUITES.get(suite, [suite])
    )))
    log.info(f"Running evaluation on tasks: {task_names}")

    # Conditionally enable Flash Attention
    try:
        model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        use_flash_attention = getattr(model_config, "use_flash_attention_2", False)
    except Exception:
        use_flash_attention = False

    model_args_list = [f"pretrained={args.model_path}", "trust_remote_code=True"]
    if use_flash_attention:
        log.info("Enabling Flash Attention 2 for evaluation based on model config.")
        model_args_list.extend(["attn_implementation='flash_attention_2'", "torch_dtype='bfloat16'"])
    model_args_str = ",".join(model_args_list)

    # Run evaluation
    results = simple_evaluate(
        model="hf", model_args=model_args_str, tasks=task_names,
        batch_size=args.batch_size, device="cuda:0"
    )
    final_results = results.get("results", {})
    serializable_results = _make_json_serializable(final_results)

    output_dir = get_wandb_run_dir(args.model_path)
    if output_dir is None:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    model_name = os.path.basename(args.model_path.rstrip('/'))
    output_filename = f"{model_name}_eval_results.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    log.info("--- Evaluation Results ---")
    print(json.dumps(serializable_results, indent=2))
    log.info(f"Results saved to {output_path}")

    # If in a resumed wandb run, upload the results and finish
    if wandb.run is not None:
        log.info("Uploading results to wandb as an artifact...")
        wandb.save(output_path)
        wandb.finish()

if __name__ == "__main__":
    main()