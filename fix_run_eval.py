import argparse
import json
import logging
import os
import torch
import numpy as np
from lm_eval import simple_evaluate
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Import custom models to make them available for AutoModelForCausalLM to find.
# This allows `trust_remote_code=True` to work correctly.
from src.models.mod.model import MoDForCausalLM
from src.models.sdt.model import SDTForCausalLM
from src.models.stt.model import STTForCausalLM
from src.models.standard.model import StandardTransformerForCausalLM

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Task groups
TASK_SUITES = {
    "general": ["arc_challenge", "hellaswag", "mmlu", "winogrande", "truthfulqa_mc2"],
    "math": ["gsm8k", "math_qa"],
    "code": ["humaneval", "mbpp"],
    "quick_test": ["arc_challenge", "hellaswag"]
}

def _make_json_serializable(obj):
    """Recursively convert non-serializable types to serializable types."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(elem) for elem in obj]
    elif isinstance(obj, torch.dtype):
        return str(obj)
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    return obj

def print_summary(results_dict):
    """Prints a formatted summary table of the main evaluation metrics, including error bars."""
    summary_lines = ["--- Final Benchmark Summary ---"]
    headers = f"| {'Task':<20} | {'Metric':<15} | {'Value':<23} |"
    separator = "|-" + "-"*22 + "|-" + "-"*17 + "|-" + "-"*25 + "|"
    summary_lines.append(headers)
    summary_lines.append(separator)

    for task_name, metrics in sorted(results_dict.get("results", {}).items()):
        primary_metric, primary_value, primary_stderr = None, None, None
        
        metric_priority = ["acc_norm", "acc", "f1", "bleu"]
        found_metrics = [m for m in metrics.keys() if "stderr" not in m and isinstance(metrics[m], float)]

        for m in metric_priority:
            if m in found_metrics:
                primary_metric = m
                primary_value = metrics[m]
                stderr_key = f"{m}_stderr"
                if stderr_key in metrics:
                    primary_stderr = metrics[stderr_key]
                break
        
        if not primary_metric and found_metrics:
            primary_metric = found_metrics[0]
            primary_value = metrics[primary_metric]
            stderr_key = f"{primary_metric}_stderr"
            if stderr_key in metrics:
                primary_stderr = metrics[stderr_key]

        if primary_metric is not None:
            if primary_stderr is not None:
                value_str = f"{primary_value:.4f} ± {primary_stderr:.4f}"
            else:
                value_str = f"{primary_value:.4f}"
            
            line = f"| {task_name:<20} | {primary_metric:<15} | {value_str:<23} |"
            summary_lines.append(line)

    summary_lines.append("-" * len(separator))
    log.info("\n".join(summary_lines))



def main():
    parser = argparse.ArgumentParser(description="Run lm-eval benchmarks on a trained model with a separate tokenizer.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model directory.")
    # FIX: Add a separate argument for the tokenizer path
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen2.5-0.5B", help="Path or Hub name of the tokenizer to use.")
    parser.add_argument("--tasks", type=str, default="general", help=f"Comma-separated list of tasks or suites. Available suites: {list(TASK_SUITES.keys())}")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    args = parser.parse_args()
    
    task_names = sorted(list(set(
        task for suite in args.tasks.split(',') for task in TASK_SUITES.get(suite, [suite])
    )))
    log.info(f"Running evaluation on tasks: {task_names}")

    # FIX: Use the separate tokenizer path in the model arguments for lm-eval
    model_args_dict = {
        "pretrained": args.model_path,
        "tokenizer": args.tokenizer_path,
        "trust_remote_code": True,
    }

    # Only set torch_dtype if not specified in the model's config.json
    # to avoid a multiple-values-for-keyword-argument error.
    try:
        config = AutoConfig.from_pretrained(args.model_path)
        if not hasattr(config, "torch_dtype") or config.torch_dtype is None:
            model_args_dict["torch_dtype"] = "auto"
    except Exception:
        model_args_dict["torch_dtype"] = "auto" # Fallback
    
    # Shot counts to align with official Qwen 2.5 evaluations
    shot_counts = {
        "mmlu": 5,
        "arc_challenge": 25,
        "truthfulqa_mc2": 0,
        "winogrande": 5,
        "hellaswag": 10,
    }

    # If ONLY quick_test is specified, override to use zero-shot for its tasks
    if args.tasks == "quick_test":
        log.info("Quick test specified, overriding to use zero-shot for all tasks in the suite.")
        shot_counts["arc_challenge"] = 0
        shot_counts["hellaswag"] = 0

    all_results = {}
    for task_name in task_names:
        # Get the task-specific shot count, defaulting to 0 if not specified
        num_fewshot = shot_counts.get(task_name, 0)
        log.info(f"--> Running task '{task_name}' with {num_fewshot} shots...")

        results = simple_evaluate(
            model="hf",
            model_args=model_args_dict,
            tasks=[task_name],
            num_fewshot=num_fewshot,
            batch_size=args.batch_size,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        
        if "results" in results:
            all_results.update(results["results"])
    
    # Re-create the top-level structure that the rest of the script expects
    final_results_structure = {"results": all_results}
    serializable_results = _make_json_serializable(final_results_structure)

    # Print summary table to stderr for console logging
    log.info("--- Final Benchmark Summary ---")
    print_summary(serializable_results)

    # Print final JSON results to stdout for capture by the parent process
    print(json.dumps(serializable_results))

if __name__ == "__main__":
    main()
