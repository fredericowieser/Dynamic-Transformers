import argparse
import json
import logging
import os
from lm_eval import simple_evaluate

from transformers import AutoConfig
from src.models.qwen.causal_lm import DynamicQwenForCausalLM
from src.models.qwen.config import DynamicQwenConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Explicitly register the custom model and config with the Auto classes.
# This allows lm-evaluation-harness to find and use your custom architecture.
log.info("Registering custom 'dynamic_qwen' architecture with transformers.")
AutoConfig.register("dynamic_qwen", DynamicQwenConfig)
AutoModelForCausalLM.register(DynamicQwenConfig, DynamicQwenForCausalLM)

# Define task groups for easy evaluation
TASK_SUITES = {
    "general": ["arc_challenge", "hellaswag", "mmlu", "winogrande", "truthfulqa_mc2"],
    "math": ["gsm8k", "math_qa"],
    "code": ["humaneval", "mbpp"],
    "quick_test": ["arc_challenge", "hellaswag"] # A smaller set for quick checks
}

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on a custom Qwen model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved fine-tuned model directory.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="quick_test",
        help=f"Comma-separated list of tasks or task suites. Available suites: {list(TASK_SUITES.keys())}",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save the evaluation results JSON.",
    )
    args = parser.parse_args()

    # Expand task suites into a list of individual tasks
    task_names = []
    for task in args.tasks.split(','):
        if task in TASK_SUITES:
            task_names.extend(TASK_SUITES[task])
        else:
            task_names.append(task)
    
    # Remove duplicates
    task_names = sorted(list(set(task_names)))
    log.info(f"Running evaluation on the following tasks: {task_names}")

    log.info(f"Loading configuration from {args.model_path}...")
    try:
        model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        use_flash_attention = getattr(model_config, "use_flash_attention_2", False)
    except Exception as e:
        log.warning(f"Could not read 'use_flash_attention_2' from config: {e}. Defaulting to False.")
        use_flash_attention = False

    # Conditionally build the model_args string for lm-evaluation-harness
    model_args_list = [
        f"pretrained={args.model_path}",
        "trust_remote_code=True"
    ]
    if use_flash_attention:
        log.info("Found 'use_flash_attention_2=True' in model config. Enabling for evaluation.")
        model_args_list.append("attn_implementation='flash_attention_2'")
        model_args_list.append("torch_dtype='bfloat16'") # Match dtype for consistency

    model_args_str = ",".join(model_args_list)
    log.info(f"Using model_args for lm-eval: '{model_args_str}'")

    results = simple_evaluate(
        model="hf",
        model_args=model_args_str,
        tasks=task_names,
        batch_size=args.batch_size,
        device="cuda:0",
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip('/'))
    output_path = os.path.join(args.output_dir, f"{model_name}_results.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info("--- Evaluation Results ---")
    print(json.dumps(results, indent=2))
    log.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()