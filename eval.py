#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys

import torch
from transformers import pipeline

# Add the parent directory of 'src' to the Python path
# This assumes 'evaluate.py' is in the project root alongside 'src' and 'eval'.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import from the new modular structure
from eval.benchmarks import BENCHMARKS
from eval.dynamic_llama_utils import load_dynamic_llama_model_and_tokenizer
from eval.runners import (
    run_generative_benchmark,
    run_humaneval_benchmark,
    run_multiple_choice_benchmark,
    run_perplexity_benchmark,
    run_mmlu_benchmark, # NEW: Dedicated MMLU runner
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DynamicLlama models on various benchmarks."
    )
    parser.add_argument(
        "--model_path", required=True, help="Path or HF ID of your trained model"
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to use (e.g., 'cpu', 'cuda')"
    )
    parser.add_argument(
        "--max_eval_samples", type=int, default=512, help="Samples per benchmark"
    )
    parser.add_argument(
        "--is_instruct", action="store_true", help="Model is instruct-tuned"
    )
    parser.add_argument(
        "--output_file", default="results.json", help="Output JSON file"
    )
    # DynamicLlama specific parameters (will override config values if provided)
    parser.add_argument(
        "--dynamic_k",
        type=float,
        default=None,
        help="Override config.dynamic_k for dynamic layers",
    )
    parser.add_argument(
        "--ce_bias",
        type=float,
        default=None,
        help="Override config.ce_bias for dynamic layers",
    )
    parser.add_argument(
        "--gate_warmup_iters",
        type=int,
        default=None,
        help="Override config.gate_warmup_iters for dynamic layers",
    )
    parser.add_argument(
        "--token_wise",
        type=bool,
        default=None,
        help="Override config.token_wise for dynamic layers (True/False)",
    )

    args = parser.parse_args()

    device = args.device
    num_samples = args.max_eval_samples

    log.info(f"Loading model and tokenizer from {args.model_path} on {device}...")
    try:
        model, tokenizer = load_dynamic_llama_model_and_tokenizer(
            model_path=args.model_path,
            device=device,
            is_instruct=args.is_instruct,
            dynamic_k=args.dynamic_k,
            ce_bias=args.ce_bias,
            gate_warmup_iters=args.gate_warmup_iters,
            token_wise=args.token_wise,
        )
        log.info("Model and tokenizer loaded successfully.")
    except Exception as e:
        log.error(f"Failed to load model and tokenizer: {e}")
        sys.exit(1)

    # Initialize a text generation pipeline for generative tasks once
    # Ensure device is handled correctly for the pipeline (e.g., device=model.device.index if CUDA)
    pipe_device = -1 # Default to CPU
    if str(model.device).startswith("cuda"):
        pipe_device = model.device.index if model.device.index is not None else 0

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
        device=pipe_device,
    )

    results = {}
    
    for benchmark_name, config in BENCHMARKS.items():
        log.info(f"\n--- Running {benchmark_name} ({config['type']} benchmark) ---")
        try:
            if config["type"] == "ppl":
                score = run_perplexity_benchmark(
                    model, tokenizer, config, num_samples, device
                )
                results[benchmark_name] = score
            elif config["type"] == "mc":
                score = run_multiple_choice_benchmark(
                    model, tokenizer, config, num_samples, device, args.is_instruct
                )
                results[benchmark_name] = score
            elif config["type"] == "mc_multi_subject": # New type for MMLU
                score = run_mmlu_benchmark(
                    model, tokenizer, config, num_samples, device, args.is_instruct
                )
                results[benchmark_name] = score
            elif config["type"] == "generative":
                score = run_generative_benchmark(
                    gen_pipe, model, config, num_samples
                )
                results[benchmark_name] = score
            elif config["type"] == "humaneval":
                score = run_humaneval_benchmark(
                    model, tokenizer, config, device
                )
                results[benchmark_name] = score
            else:
                log.warning(f"Unknown benchmark type: {config['type']} for {benchmark_name}. Skipping.")

        except Exception as e:
            log.error(f"Error running {benchmark_name}: {type(e).__name__}: {e}")
            results[benchmark_name] = f"ERROR: {type(e).__name__}: {e}"
            
    # Save final results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nEvaluation complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()