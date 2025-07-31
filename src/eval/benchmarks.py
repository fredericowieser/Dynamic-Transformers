# benchmarks.py
import evaluate as hf_evaluate
from lm_eval import evaluator, tasks  # Eleuther AI LM eval
from datasets import load_dataset
from transformers import pipeline
import torch
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_benchmark_lm_eval(model, tokenizer, benchmark_name: str, num_samples: int, is_instruct: bool):
    """Run benchmarks using LM eval harness."""
    try:
        task = tasks.get_task(benchmark_name)  # This might fail if task doesn't exist
        results = evaluator.simple_evaluate(
            model=model,
            tokenizer=tokenizer,
            tasks=[benchmark_name],
            num_fewshot=5 if "mmlu" in benchmark_name.lower() else 0,
            batch_size=8,
            limit=num_samples,
        )
        scores = results['results'].get(benchmark_name, {}).get('acc', [])  # Fallback to empty list
        return compute_average_metric(scores, benchmark_name)
    except AttributeError as e:
        logger.error(f"LM eval error for {benchmark_name}: {e}")
        return {"error": f"module 'lm_eval.tasks' has no attribute 'get_task' for {benchmark_name}"}
    except Exception as e:
        logger.error(f"Error in LM eval for {benchmark_name}: {e}")
        return {"error": str(e)}

def run_custom_benchmark(model, tokenizer, benchmark_name: str, dataset_name: str, num_samples: int, is_instruct: bool):
    """Run custom benchmarks with proper error handling."""
    try:
        ds = load_dataset(dataset_name, split=f"test[:{num_samples}]")
        gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0.0, do_sample=False)

        if benchmark_name == "IFEval":
            metric = hf_evaluate.load("accuracy")
            for ex in tqdm(ds, desc=benchmark_name):
                prompt = ex.get("prompt", "")
                out = gen_pipe(prompt)[0]["generated_text"]
                ref = ex.get("expected_output", "")
                metric.add(prediction=out, reference=ref)
            scores = [metric.compute()["accuracy"]]  # Single score
        elif benchmark_name == "TLDR9+":
            rouge = hf_evaluate.load("rouge")
            for ex in tqdm(ds, desc=benchmark_name):
                prompt = "Summarize: " + ex.get("input_text", "")[:512]
                out = gen_pipe(prompt)[0]["generated_text"]
                ref = ex.get("target_text", "")
                rouge.add(prediction=out, reference=ref)
            scores = [rouge.compute()["rougeL"]]
        elif benchmark_name == "GSM8K":
            metric = hf_evaluate.load("accuracy")
            for ex in tqdm(ds, desc=benchmark_name):
                prompt = ex["question"] + "\nLet's think step by step."
                out = gen_pipe(prompt)[0]["generated_text"]
                ans = extract_boxed_answer(out)  # Assume this function exists
                metric.add(prediction=ans, reference=ex["answer"])
            scores = [metric.compute()["accuracy"]]
        elif benchmark_name == "MATH":
            metric = hf_evaluate.load("accuracy")
            for ex in tqdm(ds, desc=benchmark_name):
                prompt = ex["problem"] + "\nLet's think step by step."
                out = gen_pipe(prompt)[0]["generated_text"]
                ans = extract_boxed_answer(out)
                metric.add(prediction=ans, reference=ex["solution"])
            scores = [metric.compute()["accuracy"]]
        else:
            scores = []
        return compute_average_metric(scores, benchmark_name)
    except Exception as e:
        logger.error(f"Error loading or running {benchmark_name}: {e}")
        return {"error": str(e)}

def run_all_benchmarks(model, tokenizer, num_samples: int, is_instruct: bool):
    benchmarks = {
        "MMLU": "mmlu",  # lm_eval task
        "ARC-C": "arc_challenge",
        "GPQA": "stanford-gpqa/gpqa",  # Full dataset name
        "HellaSwag": "hellaswag",
        "GSM8K": "gsm8k/main",
        "MATH": "hendrycks/competition_math",
        "IFEval": "lukaemon/ifeval",  # Verified name; adjust if needed
        "TLDR9+": "tldr/tldr",  # Example; confirm exact name
    }

    results = {}
    for name, ds_name in benchmarks.items():
        if name in ["MMLU", "ARC-C", "HellaSwag"]:  # Use LM eval
            results[name] = run_benchmark_lm_eval(model, tokenizer, ds_name, num_samples, is_instruct)
        else:  # Custom
            results[name] = run_custom_benchmark(model, tokenizer, name, ds_name, num_samples, is_instruct)
    return results

# Assume extract_boxed_answer is defined here or imported
def extract_boxed_answer(text):
    import re
    match = re.search(r"\\boxed{([^}]*)}", text)
    return match.group(1) if match else text.strip().split("\n")[-1].strip()