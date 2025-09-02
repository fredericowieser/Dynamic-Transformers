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

    # --- START OF IMPROVED SECTION: Manual Model Loading ---

    # --- Manual Model Loading ---
    log.info(f"Loading model and tokenizer from: {args.model_path}")

    model_load_kwargs = {"trust_remote_code": True}
    try:
        model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        if getattr(model_config, "use_flash_attention_2", False):
            log.info("Enabling Flash Attention 2 for evaluation based on model config.")
            model_load_kwargs["attn_implementation"] = "flash_attention_2"
            model_load_kwargs["torch_dtype"] = torch.bfloat16
    except Exception as e:
        log.warning(f"Could not determine Flash Attention support from config: {e}")

    # --- START OF FIX ---
    # Explicitly name the `pretrained_model_name_or_path` argument. This is more
    # robust and can prevent subtle argument parsing bugs inside the library.
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        **model_load_kwargs
    )
    # --- END OF FIX ---
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.to("cuda:0")
    model.eval()

    # --- END OF IMPROVED SECTION ---

    # Define the number of shots for each specific task.
    shot_counts = {
        "mmlu": 5,
        #"arc_challenge": 25,
        "truthfulqa_mc2": 0,
        "winogrande": 5,
        #"hellaswag": 10,
    }

    # Run evaluation for each task individually.
    all_results = {}
    for task_name in task_names:
        num_fewshot = shot_counts.get(task_name, 0)
        log.info(f"--> Running task '{task_name}' with {num_fewshot} shots...")

        # Pass the pre-loaded model and tokenizer objects directly
        results = simple_evaluate(
            model=model,
            tokenizer=tokenizer,
            tasks=[task_name],
            num_fewshot=num_fewshot,
            batch_size=args.batch_size,
        )
        all_results.update(results.get("results", {}))
    
    serializable_results = _make_json_serializable(all_results)

    # Determine output directory (wandb run dir or local fallback)
    output_dir = get_wandb_run_dir(args.model_path)
    if output_dir is None:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    model_name = os.path.basename(args.model_path.rstrip('/'))
    output_filename = f"{model_name}_eval_results.json"
    output_path = os.path.join(output_dir, output_filename)

    # Save and print results
    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    log.info("--- Evaluation Results ---")
    print(json.dumps(serializable_results, indent=2))
    log.info(f"Results saved to {output_path}")

    # If in a resumed wandb run, upload the results as an artifact and finish
    if wandb.run is not None:
        log.info("Uploading results to wandb as an artifact...")
        wandb.save(output_path)
        wandb.finish()

if __name__ == "__main__":
    main()