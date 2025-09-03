import argparse
import json
import logging
import os
import torch
import numpy as np
import wandb
from lm_eval import simple_evaluate

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from src.models.qwen.causal_lm import DynamicQwenForCausalLM
from src.models.qwen.config import DynamicQwenConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Register custom architecture
log.info("Registering custom 'dynamic_qwen' architecture with transformers.")
AutoConfig.register("dynamic_qwen", DynamicQwenConfig)
AutoModelForCausalLM.register(DynamicQwenConfig, DynamicQwenForCausalLM)

# Define task groups
TASK_SUITES = {
    "general": ["arc_challenge", "hellaswag", "mmlu", "winogrande", "truthfulqa_mc2"],
    "math": ["gsm8k", "math_qa"],
    "code": ["humaneval", "mbpp"],
    "quick_test": ["arc_challenge", "hellaswag"]
}

def _make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(elem) for elem in obj]
    elif isinstance(obj, torch.dtype):
        return str(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    return obj

def get_wandb_run_dir(model_path: str) -> str | None:
    wandb_info_path = os.path.join(model_path, "wandb_info.json")
    if not os.path.exists(wandb_info_path):
        log.warning("wandb_info.json not found.")
        return None
    with open(wandb_info_path, "r") as f:
        wandb_info = json.load(f)
    try:
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
    
    task_names = sorted(list(set(
        task for suite in args.tasks.split(',') for task in TASK_SUITES.get(suite, [suite])
    )))
    log.info(f"Running evaluation on tasks: {task_names}")

    # --- FINALIZED MODEL LOADING SECTION ---
    # Build the arguments for lm-eval to load the model automatically.
    # The key "pretrained" is the standard for this library.
    model_args_dict = {
        "pretrained": args.model_path,
        "trust_remote_code": True,
    }
    
    try:
        model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        if getattr(model_config, "use_flash_attention_2", False):
            log.info("Enabling Flash Attention 2 for evaluation based on model config.")
            model_args_dict["attn_implementation"] = "flash_attention_2"
            model_args_dict["torch_dtype"] = "bfloat16"
    except Exception as e:
        log.warning(f"Could not determine Flash Attention support from config: {e}")
    # --- END OF LOADING SECTION ---

    # Define the number of shots for each specific task.
    shot_counts = {
        "mmlu": 5,
        "arc_challenge": 25,
        "truthfulqa_mc2": 0,
        "winogrande": 5,
        "hellaswag": 10,
    }

    all_results = {}
    for task_name in task_names:
        num_fewshot = shot_counts.get(task_name, 0)
        log.info(f"--> Running task '{task_name}' with {num_fewshot} shots...")

        # Let lm-eval handle the model loading automatically.
        results = simple_evaluate(
            model="hf",
            model_args=model_args_dict,
            tasks=[task_name],
            num_fewshot=num_fewshot,
            batch_size=args.batch_size,
            device="cuda:0",
        )
        all_results.update(results.get("results", {}))
    
    serializable_results = _make_json_serializable(all_results)

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

    if wandb.run is not None:
        log.info("Uploading results to wandb as an artifact...")
        wandb.save(output_path)
        wandb.finish()

if __name__ == "__main__":
    main()