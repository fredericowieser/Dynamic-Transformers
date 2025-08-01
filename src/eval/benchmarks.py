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
    Run a single benchmark using Hugging Face tools, with robust field checking.

    Args:
        model: The model instance.
        tokenizer: The tokenizer instance.
        benchmark_name (str): Name of the benchmark.
        dataset_name (str): Hugging Face dataset name or path.
        num_samples (int): Maximum number of samples.
        is_instruct (bool): Whether the model is instruct-tuned.

    Returns:
        dict: Computed average metric or error details.
    """
    try:
        if benchmark_name == "MMLU":
            # MMLU requires a subset; default to one for simplicity
            ds = load_dataset(dataset_name, "abstract_algebra", split=f"test[:{num_samples}]")  # Can be made configurable
        elif benchmark_name == "ARC-C":
            ds = load_dataset("ai2_arc", "ARC-Challenge", split=f"test[:{num_samples}]")
        elif benchmark_name == "GPQA":
            ds = load_dataset("lukaemon/gpqa", split=f"test[:{num_samples}]")  # Fallback; log if it fails
        elif benchmark_name == "HellaSwag":
            ds = load_dataset("hellaswag", split=f"test[:{num_samples}]")
        elif benchmark_name == "GSM8K":
            ds = load_dataset("gsm8k", "main", split=f"test[:{num_samples}]")
        elif benchmark_name == "MATH":
            ds = load_dataset("hendrycks/competition_math", split=f"test[:{num_samples}]")
        elif benchmark_name == "IFEval":
            ds = load_dataset("lukaemon/ifeval", split=f"test[:{num_samples}]")
        elif benchmark_name == "TLDR9+":
            ds = load_dataset("cnn_dailymail", "3.0.0", split=f"test[:{num_samples}]")
        else:
            ds = load_dataset(dataset_name, split=f"test[:{num_samples}]")
    except Exception as e:
        logger.error(f"Error loading dataset for {benchmark_name}: {e}")
        return {"error": str(e)}

    metric = hf_evaluate.load("accuracy")  # Default metric; override as needed
    scores = []

    if benchmark_name == "MMLU":
        for ex in tqdm(ds, desc=benchmark_name):
            prompt = ex.get("question", "")
            out = manual_generate(model, tokenizer, prompt)
            if "answer" in ex:  # Check for field existence
                metric.add(prediction=out, reference=ex["answer"])
        scores = [metric.compute()["accuracy"]]
    elif benchmark_name == "ARC-C":
        for ex in tqdm(ds, desc=benchmark_name):
            prompt = ex.get("question", "")
            out = manual_generate(model, tokenizer, prompt)
            if "answerKey" in ex:
                metric.add(prediction=out, reference=ex["answerKey"])
        scores = [metric.compute()["accuracy"]]
    elif benchmark_name == "GPQA":
        for ex in tqdm(ds, desc=benchmark_name):
            prompt = ex.get("question", "")
            out = manual_generate(model, tokenizer, prompt)
            if "answer" in ex:
                metric.add(prediction=out, reference=ex["answer"])
        scores = [metric.compute()["accuracy"]]
    elif benchmark_name == "HellaSwag":
        for ex in tqdm(ds, desc=benchmark_name):
            prompt = ex.get("ctx", "")
            out = manual_generate(model, tokenizer, prompt)
            if "endings" in ex and "label" in ex:  # Correct field handling
                predicted_index = ex["endings"].index(out.strip())  # Simplified; adjust as needed
                metric.add(prediction=predicted_index, reference=ex["label"])
        scores = [metric.compute()["accuracy"]]
    elif benchmark_name == "GSM8K":
        for ex in tqdm(ds, desc=benchmark_name):
            prompt = ex.get("question", "") + "\nLet's think step by step."
            out = manual_generate(model, tokenizer, prompt)
            ans = extract_boxed_answer(out)
            if "answer" in ex:
                metric.add(prediction=ans, reference=ex["answer"])
        scores = [metric.compute()["accuracy"]]
    elif benchmark_name == "MATH":
        for ex in tqdm(ds, desc=benchmark_name):
            prompt = ex.get("problem", "") + "\nLet's think step by step."
            out = manual_generate(model, tokenizer, prompt)
            ans = extract_boxed_answer(out)
            if "solution" in ex:
                metric.add(prediction=ans, reference=ex["solution"])
        scores = [metric.compute()["accuracy"]]
    elif benchmark_name == "IFEval":
        for ex in tqdm(ds, desc=benchmark_name):
            prompt = ex.get("prompt", "")
            out = manual_generate(model, tokenizer, prompt)
            if "expected_output" in ex:
                metric.add(prediction=out, reference=ex["expected_output"])
        scores = [metric.compute()["accuracy"]]
    elif benchmark_name == "TLDR9+":
        rouge = hf_evaluate.load("rouge")
        for ex in tqdm(ds, desc=benchmark_name):
            prompt = "Summarize: " + ex.get("article", "")[:512]
            out = manual_generate(model, tokenizer, prompt)
            if "highlights" in ex:
                rouge.add(prediction=out, reference=ex["highlights"])
        scores = [rouge.compute()["rougeL"]]
    
    return compute_average_metric(scores, benchmark_name)

def run_all_benchmarks(model, tokenizer, num_samples: int, is_instruct: bool) -> dict:
    """
    Run all benchmarks using Hugging Face tools.

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
        "ARC-C": "ai2_arc",
        "GPQA": "lukaemon/gpqa",
        "HellaSwag": "hellaswag",
        "GSM8K": "gsm8k",
        "MATH": "hendrycks/competition_math",
        "IFEval": "lukaemon/ifeval",
        "TLDR9+": "cnn_dailymail",
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
        str: The extracted answer or a fallback.
    """
    match = re.search(r"\\boxed{([^}]*)}", text)
    return match.group(1) if match else text.strip().split("\n")[-1].strip()