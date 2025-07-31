# benchmarks.py
import evaluate as hf_evaluate
from lm_eval import evaluator, tasks  # Eleuther AI LM eval
from datasets import load_dataset
from transformers import pipeline
import torch
from tqdm import tqdm

def run_benchmark_lm_eval(model, tokenizer, benchmark_name: str, num_samples: int, is_instruct: bool):
    """Run benchmarks using LM eval harness."""
    task = tasks.get_task(benchmark_name)
    results = evaluator.simple_evaluate(
        model=model,
        tokenizer=tokenizer,
        tasks=[benchmark_name],
        num_fewshot=5 if "mmlu" in benchmark_name else 0,  # Adjust based on benchmark
        batch_size=8,
        limit=num_samples,
    )
    scores = results['results'][benchmark_name]['acc']  # Example for accuracy; adjust as needed
    return compute_average_metric(scores, benchmark_name)

def run_custom_benchmark(model, tokenizer, benchmark_name: str, dataset_name: str, num_samples: int, is_instruct: bool):
    """Run custom benchmarks not in LM eval."""
    ds = load_dataset(dataset_name, split=f"test[:{num_samples}]")
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0.0, do_sample=False)

    if benchmark_name == "IFEval":
        # Simplified IFEval (expand as needed)
        metric = hf_evaluate.load("accuracy")
        for ex in tqdm(ds, desc=benchmark_name):
            prompt = ex["prompt"]
            out = gen_pipe(prompt)[0]["generated_text"]
            metric.add(prediction=out, reference=ex["expected_output"])
        scores = [metric.compute()["accuracy"]]  # Single score for average
    elif benchmark_name == "TLDR9+":
        rouge = hf_evaluate.load("rouge")
        for ex in tqdm(ds, desc=benchmark_name):
            prompt = "Summarize: " + ex["input_text"]
            out = gen_pipe(prompt)[0]["generated_text"]
            rouge.add(prediction=out, reference=ex["target_text"])
        scores = [rouge.compute()["rougeL"]]  # RougeL score
    # Add more custom benchmarks as needed (e.g., Open-rewrite)
    else:
        scores = []  # Fallback

    return compute_average_metric(scores, benchmark_name)

def run_all_benchmarks(model, tokenizer, num_samples: int, is_instruct: bool):
    benchmarks = {
        "MMLU": "mmlu",
        "ARC-C": "arc_challenge",
        "GPQA": "gpqa",  # If available in LM eval
        "HellaSwag": "hellaswag",
        "GSM8K": "gsm8k",  # Custom for CoT
        "MATH": "math",  # Custom
        "IFEval": "livecodebench/ifeval",  # Custom
        "TLDR9+": "pszemraj/long-t5-tglobal-large-16384-pubmed-3k_steps",  # Custom
        # Add others like InfiniteBench, etc., as per LM eval tasks
    }

    results = {}
    for name, ds_name in benchmarks.items():
        try:
            if name in ["MMLU", "ARC-C", "HellaSwag"]:  # Use LM eval
                avg_metric = run_benchmark_lm_eval(model, tokenizer, ds_name, num_samples, is_instruct)
            else:  # Custom
                avg_metric = run_custom_benchmark(model, tokenizer, name, ds_name, num_samples, is_instruct)
            results[name] = avg_metric
        except Exception as e:
            results[name] = {"error": str(e)}
    return results