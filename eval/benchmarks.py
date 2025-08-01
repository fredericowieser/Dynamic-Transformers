# eval/benchmarks.py
import os
import itertools
from datasets import load_dataset
from human_eval.data import read_problems


# Helper functions for MMLU and other dataset preprocessing
def mmlu_preprocess(example, dev_shot_data):
    """
    Preprocesses MMLU examples for few-shot evaluation.
    'dev_shot_data' contains the few-shot examples (e.g., first 5 from dev split).
    This function will be passed to dataset.map() with `with_indices=False` if dev_shot_data is provided separately.
    It constructs the prompt by prepending few-shot examples.
    """
    ctx = ""
    for s_ex in dev_shot_data:
        # This prompt format for MMLU is typical for 5-shot CoT
        ctx += f"Question: {s_ex['question']}\nAnswer: {s_ex['choices'][s_ex['answer']]}\n\n"
    
    # We build the prompt directly, the is_instruct flag will be handled by mc_accuracy for chat template
    return {
        "prompt": ctx + f"Question: {example['question']}\nAnswer:",
        "choices": example["choices"],
        "gold": example["answer"],
    }


# Define all benchmarks
BENCHMARKS = {
    "wikitext_ppl": {
        "name": "Wikitext-2-raw-v1",
        "type": "ppl",
        "dataset_id": "wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "split": "validation", # No slicing here for streaming
        "fetch_texts_fn": lambda ds, N: (ds["text"] if not ds.streaming else [t["text"] for t in itertools.islice(ds, N)]), # Handles both
    },
    "pile_ppl": {
        "name": "Pile (streaming)",
        "type": "ppl",
        "dataset_id": "EleutherAI/pile",
        "split": "test", # No slicing here
        "streaming": True,
        "trust_remote_code": True, # Many streaming datasets need this
        "fetch_texts_fn": lambda ds, N: [ex["text"] for ex in itertools.islice(ds, N)],
    },
    "c4_ppl": {
        "name": "C4-en (streaming)",
        "type": "ppl",
        "dataset_id": "allenai/c4",
        "dataset_config": "en",
        "split": "validation", # No slicing here
        "streaming": True,
        "fetch_texts_fn": lambda ds, N: [ex["text"] for ex in itertools.islice(ds, N)],
    },
    "hellaswag_acc": {
        "name": "HellaSwag",
        "type": "mc",
        "dataset_id": "hellaswag",
        "dataset_config": "default",
        "split": "validation",
        "prompt_key": "ctx",
        "choices_key": "endings",
        "label_key": "label",
    },
    "winogrande_acc": {
        "name": "Winogrande",
        "type": "mc",
        "dataset_id": "winogrande",
        "dataset_config": "winogrande_xl",
        "split": "validation",
        "preprocess_fn": lambda ex: {
            "prompt": ex["sentence"],
            "choices": [ex["option1"], ex["option2"]],
            "gold": int(ex["answer"]) - 1,
        },
        "prompt_key": "prompt",
        "choices_key": "choices",
        "label_key": "gold",
    },
    "piqa_acc": {
        "name": "PIQA",
        "type": "mc",
        "dataset_id": "piqa",
        "split": "validation",
        "trust_remote_code": True, # FIX: Added for PIQA
        "preprocess_fn": lambda ex: {
            "prompt": ex["goal"],
            "choices": [ex["sol1"], ex["sol2"]],
            "gold": ex["label"],
        },
        "prompt_key": "prompt",
        "choices_key": "choices",
        "label_key": "gold",
    },
    "commonsenseqa_acc": {
        "name": "CommonsenseQA",
        "type": "mc",
        "dataset_id": "commonsense_qa",
        "split": "validation",
        "preprocess_fn": lambda ex: {
            "prompt": ex["question"],
            "choices": ex["choices"]["text"],
            "gold": ex["choices"]["label"].index(ex["answerKey"]),
        },
        "prompt_key": "prompt",
        "choices_key": "choices",
        "label_key": "gold",
    },
    "arc_challenge_acc": {
        "name": "ARC-Challenge",
        "type": "mc",
        "dataset_id": "ai2_arc",
        "dataset_config": "ARC-Challenge",
        "split": "test",
        "preprocess_fn": lambda ex: {
            "prompt": ex["question"],
            "choices": ex["choices"]["text"],
            "gold": ex["choices"]["label"].index(ex["answerKey"]),
        },
        "prompt_key": "prompt",
        "choices_key": "choices",
        "label_key": "gold",
    },
    "gpqa_acc": {
        "name": "GPQA",
        "type": "mc",
        "dataset_id": "Idavidrein/gpqa",
        "dataset_config": "gpqa_diamond",
        "split": "main",
        "gated": True, # NOTE: This dataset is gated and requires access
        "preprocess_fn": lambda ex: {
            "prompt": ex["Question"],
            "choices": ex["choices"],
            "gold": ex["correct_answer"],
        },
        "prompt_key": "prompt",
        "choices_key": "choices",
        "label_key": "gold",
    },
    "mmlu_mean": { # NEW: Now has its own runner type
        "name": "MMLU",
        "type": "mc_multi_subject", 
        "dataset_id": "cais/mmlu",
        "split": "test", # Test split for evaluation
        "dev_split": "dev", # Dev split for few-shot examples
        "num_shot": 5, # 5-shot
        "preprocess_fn": mmlu_preprocess, # Will be called with few-shot data
        "prompt_key": "prompt",
        "choices_key": "choices",
        "label_key": "gold",
        "subjects": [ # List of MMLU subjects
            "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
            "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
            "college_medicine", "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
            "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
            "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology", "high_school_statistics",
            "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
            "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management",
            "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
            "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
            "professional_medicine", "professional_psychology", "public_relations", "security_studies",
            "sociology", "us_foreign_policy", "virology", "world_religions",
        ]
    },
    "squad_v2": { # Generative, returns EM and F1
        "name": "SQuAD v2",
        "type": "generative_qa", # NEW: Specific type for QA that returns EM/F1
        "dataset_id": "squad_v2",
        "split": "validation",
        "question_key": "question",
        "context_key": "context",
        "answers_key": "answers",
    },
    "gsm8k_acc": {
        "name": "GSM8K",
        "type": "generative",
        "dataset_id": "gsm8k",
        "dataset_config": "main",
        "split": "test",
        "prompt_builder_fn": lambda ex: ex["question"] + "\nLet's think step by step.",
        "answer_extractor_fn": "extract_boxed_answer", # Refers to function in utils.py
        "reference_key": "answer",
        "metric": "accuracy",
        "generation_kwargs": {"max_new_tokens": 256, "do_sample": False},
    },
    "math_acc": {
        "name": "MATH",
        "type": "generative",
        "dataset_id": "hendrycks/competition_math",
        "split": "test",
        "trust_remote_code": True, # FIX: Added for MATH
        "prompt_builder_fn": lambda ex: ex["problem"] + "\nLet's think step by step.",
        "answer_extractor_fn": "extract_boxed_answer",
        "reference_key": "solution",
        "metric": "accuracy",
        "generation_kwargs": {"max_new_tokens": 256, "do_sample": False},
    },
    "humaneval_pass1": {
        "name": "HumanEval",
        "type": "humaneval",
        "read_problems_fn": read_problems, # Function from human_eval library
        "k": [1],
        "n_workers": 4,
        "timeout": 3.0,
    },
    "ifeval_acc": {
        "name": "IFEval",
        "type": "generative",
        "dataset_id": "livecodebench/ifeval",
        "split": "test",
        "trust_remote_code": True, # FIX: Added for IFEval
        "prompt_builder_fn": lambda ex: ex["prompt"],
        "answer_extractor_fn": lambda text: text, # No specific extraction, use full generated text
        "reference_key": "expected_output",
        "metric": "accuracy", # Will use standard EM/accuracy for now, full IFEval is more complex
        "generation_kwargs": {"max_new_tokens": 128},
    },
    "tldr_rougeL": {
        "name": "TLDR9+",
        "type": "generative",
        "dataset_id": "abdelkader/tldr_news", # FIX: Changed from model ID to dataset
        "split": "test",
        "prompt_builder_fn": lambda ex: "Summarize: " + ex["article"][:512], # Assuming 'article' key
        "answer_extractor_fn": lambda text: text, # Expecting direct summary
        "reference_key": "summary", # Assuming 'summary' key
        "metric": "rouge",
        "generation_kwargs": {"max_new_tokens": 128},
    },
    "open_rewrite_rougeL": {
        "name": "Open-rewrite",
        "type": "generative",
        "dataset_id": "tasksource/openai-rewrite",
        "split": "test",
        "trust_remote_code": True, # FIX: Added for openai-rewrite
        "prompt_builder_fn": lambda ex: "Rewrite: " + ex["original"],
        "answer_extractor_fn": lambda text: text,
        "reference_key": "rewritten",
        "metric": "rouge",
        "generation_kwargs": {"max_new_tokens": 128},
    },
    "infinitebench_qa_f1": {
        "name": "InfiniteBench/En.QA",
        "type": "generative",
        "dataset_id": "akariasai/InfiniteBench",
        "dataset_config": "en_qa",
        "split": "test",
        "trust_remote_code": True, # FIX: Added for InfiniteBench
        "prompt_builder_fn": lambda ex: ex["input"],
        "answer_extractor_fn": lambda text: text,
        "reference_key": "output",
        "metric": "f1",
        "generation_kwargs": {"max_new_tokens": 128},
    },
    "infinitebench_mc_acc": {
        "name": "InfiniteBench/En.MC",
        "type": "mc",
        "dataset_id": "akariasai/InfiniteBench",
        "dataset_config": "en_mc",
        "split": "test",
        "trust_remote_code": True, # FIX: Added for InfiniteBench
        "prompt_key": "input",
        "choices_key": "choices",
        "label_key": "answer",
    },
    "mgsm_em": {
        "name": "MGSM",
        "type": "generative",
        "dataset_id": "juletxara/mgsm",
        "dataset_config": "en", # FIX: Added English config for MGSM
        "split": "test",
        "prompt_builder_fn": lambda ex: ex["question"] + "\nLet's think step by step.",
        "answer_extractor_fn": "extract_boxed_answer",
        "reference_key": "answer",
        "metric": "accuracy",
        "generation_kwargs": {"max_new_tokens": 128},
    },
    "bfcl_v2_acc": {
        "name": "BFCL V2",
        "type": "mc",
        "dataset_id": "livecodebench/bfcl",
        "split": "test",
        "trust_remote_code": True, # FIX: Added for BFCL
        "prompt_key": "prompt",
        "choices_key": "choices",
        "label_key": "answer",
    },
}