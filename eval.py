import argparse
import json
import logging
import os
import torch
import numpy as np
import wandb
from lm_eval import simple_evaluate

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.models.qwen.causal_lm import DynamicQwenForCausalLM
from src.models.qwen.config import DynamicQwenConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Registering the custom architecture is still good practice
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
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved LoRA adapter directory.")
    parser.add_argument("--tasks", type=str, default="quick_test", help=f"Tasks or suites. Available: {list(TASK_SUITES.keys())}")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Fallback directory to save results.")
    args = parser.parse_args()
    
    task_names = sorted(list(set(
        task for suite in args.tasks.split(',') for task in TASK_SUITES.get(suite, [suite])
    )))
    log.info(f"Running evaluation on tasks: {task_names}")

    # --- BUG NEUTRALIZATION: Direct PEFT Loading ---
    # We bypass the faulty AutoModel factory by loading the base model and adapters in two explicit steps.
    
    log.info(f"Loading configuration from: {args.model_path}")
    config = DynamicQwenConfig.from_pretrained(args.model_path, trust_remote_code=True)

    # Configure Flash Attention if supported
    if getattr(config, "use_flash_attention_2", False):
        log.info("Enabling Flash Attention 2 for evaluation based on model config.")
        config.attn_implementation = "flash_attention_2"
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16 # or whatever is appropriate

    log.info("Instantiating base model from config...")
    base_model = DynamicQwenForCausalLM(config)

    log.info(f"Loading LoRA adapters from {args.model_path} onto the base model...")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model = model.merge_and_unload() # Merge adapters for evaluation performance
    
    model.to("cuda:0", dtype=torch_dtype)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # --- END OF BUG NEUTRALIZATION ---

    shot_counts = {"mmlu": 5, "truthfulqa_mc2": 0, "winogrande": 5}

    all_results = {}
    for task_name in task_names:
        num_fewshot = shot_counts.get(task_name, 0)
        log.info(f"--> Running task '{task_name}' with {num_fewshot} shots...")

        results = simple_evaluate(
            model=model,
            tokenizer=tokenizer,
            tasks=[task_name],
            num_fewshot=num_fewshot,
            batch_size=args.batch_size,
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