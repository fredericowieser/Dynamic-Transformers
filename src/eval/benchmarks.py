import evaluate as hf_evaluate
from datasets import load_dataset
import torch
from tqdm import tqdm
import logging
import re
from src.eval.utils import compute_average_metric, manual_generate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_benchmark(
    model, tokenizer, benchmark_name: str, dataset_name: str, num_samples: int, is_instruct: bool
) -> dict:
    """
    Run a single benchmark using Hugging Face tools.

    Args:
        model: The model instance.
        tokenizer: The tokenizer instance.
        benchmark_name (str): Name of the benchmark.
        dataset_name (str): Hugging Face dataset name.
        num_samples (int): Maximum number of samples.
        is_instruct (bool): Whether the model is instruct-tuned.

    Returns:
        dict: Computed average metric.
    """
    try:
        ds = load_dataset(dataset_name, split=f"test[:{num_samples}]")
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        return {"error": str(e)}

    if benchmark_name == "MMLU":
        metric = hf_evaluate.load("accuracy")
        for ex in tqdm(ds, desc=benchmark_name):
            prompt = ex["question"]
            out = manual_generate(model, tokenizer, prompt)
            # Simplified: Assume choices and answer are in the dataset
            pred = ex["choices"][ex["choices"].index(out)]  # Mock prediction
            metric.add(prediction=pred, reference=ex["answer"])
        scores = [metric.compute()["accuracy"]]
    elif benchmark_name == "ARC-C":
        metric = hf_evaluate.load("accuracy")
        for ex in tqdm(ds, desc=benchmark_name):
            prompt = ex["question"]
            out = manual_generate(model, tokenizer, prompt)
            metric.add(prediction=out, reference=ex["answerKey"])
        scores = [metric.compute()["accuracy"]]
    elif benchmark_name == "GPQA":
        metric = hf_evaluate.load("accuracy")
        for ex in tqdm(ds, desc=benchmark_name):
            prompt = ex["question"]
            out = manual_generate(model, tokenizer, prompt)
            metric.add(prediction=out, reference=ex["answer"])
        scores = [metric.compute()["accuracy"]]
    elif benchmark_name == "HellaSwag":
        metric = hf_evaluate.load("accuracy")
        for ex in tqdm(ds, desc=benchmark_name):
            prompt = ex["ctx"]
            out = manual_generate(model, tokenizer, prompt)
            metric.add(prediction=out, reference=ex["ending"])
        scores = [metric.compute()["accuracy"]]
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
    elif benchmark_name == "IFEval":
        metric = hf_evaluate.load("accuracy")
        for ex in tqdm(ds, desc=benchmark_name):
            prompt = ex["prompt"]
            out = manual_generate(model, tokenizer, prompt)
            metric.add(prediction=out, reference=ex["expected_output"])
        scores = [metric.compute()["accuracy"]]
    elif benchmark_name == "TLDR9+":
        rouge = hf_evaluate.load("rouge")
        for ex in tqdm(ds, desc=benchmark_name):
            prompt = "Summarize: " + ex["article"][:512]
            out = manual_generate(model, tokenizer, prompt)
            rouge.add(prediction=out, reference=ex["highlights"])
        scores = [rouge.compute()["rougeL"]]
    # Add more benchmarks as needed (e.g., for Nexus, InfiniteBench, etc.)
    else:
        scores = []
    
    return compute_average_metric(scores, benchmark_name)

def run_all_benchmarks(model, tokenizer, num_samples: int, is_instruct: bool) -> dict:
    """
    Run all benchmarks.

    Args:
        model: The model instance.
        tokenizer: The tokenizer instance.
        num_samples (int): Maximum samples per benchmark.
        is_instruct (bool): Whether the model is instruct-tuned.

    Returns:
        dict: Results for all benchmarks.
    """
    benchmarks = {
        "MMLU": "cais/mmlu",
        "ARC-C": "ai2_arc/ARC-Challenge",
        "GPQA": "gpqa",  # Use standard or verified name
        "HellaSwag": "hellaswag",
        "GSM8K": "gsm8k",
        "MATH": "hendrycks/competition_math",
        "IFEval": "lukaemon/ifeval",
        "TLDR9+": "cnn_dailymail",
        # Add others as needed
    }

    results = {}
    for name, ds_name in benchmarks.items():
        logger.info(f"Running benchmark: {name}")
        results[name] = run_benchmark(model, tokenizer, name, ds_name, num_samples, is_instruct)
    return results

def extract_boxed_answer(text: str) -> str:
    """
    Extract the boxed answer from generated text.

    Args:
        text (str): The generated text.

    Returns:
        str: The extracted answer or fallback.
    """
    match = re.search(r"\\boxed{([^}]*)}", text)
    return match.group(1) if match else text.strip().split("\n")[-1].strip()