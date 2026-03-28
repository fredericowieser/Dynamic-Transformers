import argparse
import json
import logging
import os
import warnings

import numpy as np
import torch

# Suppress annoying library warnings
warnings.filterwarnings("ignore")
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.models import (
    MoDForCausalLM,
    SDTForCausalLM,
    StandardTransformerForCausalLM,
    STTForCausalLM,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Task groups
TASK_SUITES = {
    "general": ["arc_challenge", "hellaswag", "mmlu", "winogrande", "truthfulqa_mc2"],
    "math": ["gsm8k", "math_qa"],
    "code": ["humaneval", "mbpp"],
    "quick_test": ["truthfulqa_mc2"],
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
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def print_summary(results_dict):
    """Prints a formatted summary table of the main evaluation metrics, including error bars."""
    summary_lines = ["--- Final Benchmark Summary ---"]
    headers = f"| {'Task':<20} | {'Metric':<15} | {'Value':<23} |"
    separator = "|-" + "-" * 22 + "|-" + "-" * 17 + "|-" + "-" * 25 + "|"
    summary_lines.append(headers)
    summary_lines.append(separator)

    for task_name, metrics in sorted(results_dict.get("results", {}).items()):
        primary_metric, primary_value, primary_stderr = None, None, None

        metric_priority = ["acc_norm", "acc", "f1", "bleu"]
        found_metrics = [
            m for m in metrics.keys() if "stderr" not in m and isinstance(metrics[m], float)
        ]

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
    parser = argparse.ArgumentParser(description="Run lm-eval benchmarks on a trained model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model directory (output from save_pretrained).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="quick_test",
        help=f"Comma-separated list of tasks or suites. Available suites: {list(TASK_SUITES.keys())}",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument(
        "--use_causal_router",
        action="store_true",
        help="If set, uses the causal router during evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results. Defaults to model_path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples per task (useful for quick testing).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.model_path

    task_names = sorted(
        list(
            set(task for suite in args.tasks.split(",") for task in TASK_SUITES.get(suite, [suite]))
        )
    )
    log.info(f"Running evaluation on tasks: {task_names}")

    # Explicitly load the model using the correct custom class to prevent loading a standard
    # model with missing weights. This is the most robust way to handle custom architectures.
    log.info(f"Loading model and tokenizer from: {args.model_path}")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        # Fallback to CPU on MPS to avoid autocast error in lm-eval
        device = "cpu"
    else:
        device = "cpu"

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if args.use_causal_router:
        log.info("Overriding use_causal_router_in_validation to True for evaluation.")
        config.use_causal_router_in_validation = True
    
    model_type = getattr(config, "model_type", "standard")
    model_class_map = {
        "standard": StandardTransformerForCausalLM,
        "mod": MoDForCausalLM,
        "sdt": SDTForCausalLM,
        "stt": STTForCausalLM,
    }
    model_class = model_class_map.get(model_type)
    if not model_class:
        raise ValueError(f"Unknown model type '{model_type}' in config.")

    log.info(f"Explicitly loading model class: {model_class.__name__}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Determine if we should use model parallelism
    num_gpus = torch.cuda.device_count()
    parallelize = num_gpus > 1
    
    if parallelize:
        log.info(f"Detected {num_gpus} GPUs. Enabling model parallelization for faster evaluation.")
        # When using parallelize=True, we don't pass 'device' to HFLM
        hflm_device = None
    else:
        hflm_device = device

    # Passing the path instead of the model object helps lm-eval manage memory and 
    # reduces 'not a string' warnings. We pass the modified config.
    # Note: AutoModelForCausalLM.register MUST be called before this, 
    # which is handled by 'from src.models import ...'
    adaptor = HFLM(
        pretrained=args.model_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=hflm_device,
        parallelize=parallelize,
        trust_remote_code=True,
        config=config
    )

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
        shot_counts["truthfulqa_mc2"] = 0

    from tqdm import tqdm
    all_results = {}
    suffix = "causal" if args.use_causal_router else "non_causal"
    output_path = os.path.join(output_dir, f"eval_results_{suffix}.json")

    pbar = tqdm(task_names, desc="Evaluating tasks")
    for task_name in pbar:
        pbar.set_description(f"Evaluating {task_name}")
        # Get the task-specific shot count, defaulting to 0 if not specified
        num_fewshot = shot_counts.get(task_name, 0)
        log.info(f"--> Running task '{task_name}' with {num_fewshot} shots...")

        results = simple_evaluate(
            model=adaptor,
            tasks=[task_name],
            num_fewshot=num_fewshot,
            batch_size=args.batch_size,
            limit=args.limit,
        )

        if "results" in results:
            task_results = results["results"]
            all_results.update(task_results)
            
            # Immediate logging of results for this task
            task_serializable = _make_json_serializable({"results": task_results})
            log.info(f"--- Results for {task_name} ---")
            print_summary(task_serializable)
            
            # Save intermediate results to file
            serializable_results = _make_json_serializable({"results": all_results})
            with open(output_path, "w") as f:
                json.dump(serializable_results, f, indent=2)
            log.info(f"Updated evaluation results saved to {output_path}")

    # Final results summary
    serializable_results = _make_json_serializable({"results": all_results})
    log.info("--- Final Full Benchmark Summary ---")
    print_summary(serializable_results)


if __name__ == "__main__":
    main()
