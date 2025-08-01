import evaluate as hf_evaluate
from lm_eval import evaluator, tasks  # Import tasks for registry access
from datasets import load_dataset
from transformers import pipeline  # Not used now, but kept for reference
import torch
from tqdm import tqdm
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def manual_generate(model, tokenizer, prompt, max_new_tokens=256):
    """Manual text generation for unsupported models."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_benchmark_lm_eval(model, tokenizer, benchmark_name: str, num_samples: int, is_instruct: bool):
    try:
        task_dict = tasks.get_task_dict([benchmark_name])  # Pass the required argument
        if benchmark_name not in task_dict:
            raise AttributeError(f"Task '{benchmark_name}' not found in lm_eval tasks.")
        task = task_dict[benchmark_name]
        
        results = evaluator.simple_evaluate(
            model=model,
            tokenizer=tokenizer,
            tasks=[benchmark_name],
            num_fewshot=5 if "mmlu" in benchmark_name.lower() else 0,
            batch_size=8,
            limit=num_samples,
        )
        scores = results['results'].get(benchmark_name, {}).get('acc', [])
        return compute_average_metric(scores, benchmark_name)
    except Exception as e:
        logger.error(f"LM eval error for {benchmark_name}: {e}")
        return {"error": str(e)}

def run_custom_benchmark(model, tokenizer, benchmark_name: str, dataset_name: str, num_samples: int, is_instruct: bool):
    try:
        if benchmark_name == "GSM8K":
            ds = load_dataset(dataset_name, "main", split=f"test[:{num_samples}]")
        elif benchmark_name == "IFEval":
            ds = load_dataset("openai/ifeval", split=f"test[:{num_samples}]")  # Fallback to openai if available
        else:
            ds = load_dataset(dataset_name, split=f"test[:{num_samples}]")
        
        if benchmark_name == "IFEval":
            metric = hf_evaluate.load("accuracy")
            for ex in tqdm(ds, desc=benchmark_name):
                prompt = ex.get("prompt", "")
                out = manual_generate(model, tokenizer, prompt)
                ref = ex.get("expected_output", "")
                metric.add(prediction=out, reference=ref)
            scores = [metric.compute()["accuracy"]]
        elif benchmark_name == "TLDR9+":
            rouge = hf_evaluate.load("rouge")
            for ex in tqdm(ds, desc=benchmark_name):
                prompt = "Summarize: " + ex.get("article", "")[:512]  # Adjusted for cnn_dailymail
                out = manual_generate(model, tokenizer, prompt)
                ref = ex.get("highlights", "")
                rouge.add(prediction=out, reference=ref)
            scores = [rouge.compute()["rougeL"]]
        elif benchmark_name == "GSM8K":
            metric = hf_evaluate.load("accuracy")
            for ex in tqdm(ds, desc=benchmark_name):
                prompt = ex["question"] + "\nLet's think step by step."
                out = manual_generate(model, tokenizer, prompt)
                ans = extract_boxed_answer(out)
                metric.add(prediction=ans, reference=ex["answer"])
            scores = [metric.compute()["accuracy"]]
        elif benchmark_name == "MATH":
            metric = hf_evaluate.load("accuracy")
            for ex in tqdm(ds, desc=benchmark_name):
                prompt = ex["problem"] + "\nLet's think step by step."
                out = manual_generate(model, tokenizer, prompt)
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
        "MMLU": "mmlu",
        "ARC-C": "arc_challenge",
        "GPQA": "hendrycks/gpqa",
        "HellaSwag": "hellaswag",
        "GSM8K": "gsm8k",
        "MATH": "hendrycks/competition_math",
        "IFEval": "openai/ifeval",  # Updated fallback
        "TLDR9+": "cnn_dailymail",  # Standard summarization dataset
    }

    results = {}
    for name, ds_name in benchmarks.items():
        if name in ["MMLU", "ARC-C", "HellaSwag"]:
            results[name] = run_benchmark_lm_eval(model, tokenizer, ds_name, num_samples, is_instruct)
        else:
            results[name] = run_custom_benchmark(model, tokenizer, name, ds_name, num_samples, is_instruct)
    return results

def extract_boxed_answer(text):
    match = re.search(r"\\boxed{([^}]*)}", text)
    return match.group(1) if match else text.strip().split("\n")[-1].strip()