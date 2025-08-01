# eval/runners.py
import evaluate
import logging
from datasets import load_dataset
from human_eval.evaluation import evaluate_functional_correctness
from tqdm import tqdm

from eval.utils import (
    compute_ppl_from_texts,
    extract_boxed_answer,
    generative_exact_match,
    mc_accuracy,
)

log = logging.getLogger(__name__)


def _load_and_prepare_dataset(config, num_samples):
    """Loads a dataset from Hugging Face and applies slicing/preprocessing."""
    log.info(f"Loading dataset: {config.get('dataset_id')} ({config.get('split')})...")
    load_kwargs = {
        "path": config["dataset_id"],
        "name": config.get("dataset_config"),
        "split": f"{config['split']}[:{num_samples}]",
        "streaming": config.get("streaming", False),
    }
    if config.get("trust_remote_code"):
        load_kwargs["trust_remote_code"] = True

    dataset = load_dataset(**load_kwargs)
    
    # Handle streaming datasets needing explicit iteration
    if config.get("streaming"):
        if "fetch_texts_fn" in config: # For PPL streaming
            dataset_content = config["fetch_texts_fn"](dataset, num_samples)
        else: # For other streaming tasks, convert to list if needed
            dataset_content = list(dataset.take(num_samples))
    else:
        dataset_content = dataset

    if "preprocess_fn" in config:
        log.info("Applying dataset preprocessing function...")
        # Map with a simple lambda to pass is_instruct if needed, or pass other args.
        # For MMLU specifically, preprocess_fn needs dev_shot_data.
        # This will be handled in the run_multiple_choice_benchmark logic for mmlu_mean.
        if config.get("type") != "mc_multi_subject": # Don't map yet for MMLU subjects
            dataset_content = dataset_content.map(config["preprocess_fn"])
    
    log.info(f"Loaded {len(dataset_content)} samples.")
    return dataset_content


def run_perplexity_benchmark(model, tokenizer, config, num_samples, device):
    """Runs a perplexity evaluation benchmark."""
    try:
        if config.get("streaming"):
            # Streaming datasets are handled by fetch_texts_fn
            texts = _load_and_prepare_dataset(config, num_samples)
        else:
            ds = _load_and_prepare_dataset(config, num_samples)
            texts = config["fetch_texts_fn"](ds, num_samples)
        
        ppl = compute_ppl_from_texts(
            model, tokenizer, texts, device, **config.get("ppl_kwargs", {})
        )
        log.info(f"  Result: PPL = {ppl:.2f}")
        return ppl
    except Exception as e:
        log.error(f"Error in perplexity benchmark '{config['name']}': {e}")
        raise


def run_multiple_choice_benchmark(
    model, tokenizer, config, num_samples, device, is_instruct
):
    """Runs a multiple-choice evaluation benchmark."""
    if config.get("type") == "mc_multi_subject":  # Special handling for MMLU
        mmlu_accs = []
        for subj in tqdm(config["subjects"], desc="MMLU subjects"):
            try:
                # Load test split
                ds_test = load_dataset(config["dataset_id"], subj, split=f"{config['split']}[:{num_samples}]")
                # Load dev split for few-shot examples
                ds_dev = load_dataset(config["dataset_id"], subj, split=config["dev_split"])
                dev_shot = ds_dev.select(range(min(config["num_shot"], len(ds_dev))))

                # Apply preprocessing for each subject's test set
                processed_ds = ds_test.map(lambda ex: config["preprocess_fn"](ex, dev_shot, is_instruct))
                
                acc = mc_accuracy(
                    model,
                    tokenizer,
                    processed_ds,
                    device,
                    prompt_key=config["prompt_key"],
                    choices_key=config["choices_key"],
                    label_key=config["label_key"],
                    is_instruct=is_instruct,
                )
                mmlu_accs.append(acc)
                log.info(f"    MMLU/{subj} → {acc*100:.2f}%")
            except Exception as e:
                log.warning(f"      ! Skipping MMLU/{subj}: {type(e).__name__}: {e}")

        if mmlu_accs:
            mean_mmlu = sum(mmlu_accs) / len(mmlu_accs)
            log.info(
                f"  Result: MMLU (mean over {len(mmlu_accs)} subjects) = {mean_mmlu*100:.2f}%"
            )
            return {"mean_accuracy": mean_mmlu, "individual_accuracies": {s: a for s, a in zip(config["subjects"], mmlu_accs)}}
        else:
            log.info("  No MMLU subjects loaded; skipping mean calculation.")
            return {"mean_accuracy": 0.0, "individual_accuracies": {}}
    else:  # Standard MC benchmark
        try:
            ds = _load_and_prepare_dataset(config, num_samples)
            acc = mc_accuracy(
                model,
                tokenizer,
                ds,
                device,
                prompt_key=config["prompt_key"],
                choices_key=config["choices_key"],
                label_key=config["label_key"],
                is_instruct=is_instruct,
            )
            log.info(f"  Result: {acc*100:.2f}%")
            return acc
        except Exception as e:
            log.error(f"Error in multiple-choice benchmark '{config['name']}': {e}")
            raise


def run_generative_benchmark(gen_pipe_obj, config, num_samples):
    """Runs a generative evaluation benchmark (e.g., SQuAD, GSM8K)."""
    try:
        ds = _load_and_prepare_dataset(config, num_samples)

        if config["metric"] == "accuracy":
            metric = evaluate.load("accuracy")
        elif config["metric"] == "rouge":
            metric = evaluate.load("rouge")
        elif config["metric"] == "f1":
            metric = evaluate.load("f1")
        else:
            raise ValueError(f"Unsupported metric: {config['metric']}")

        results = {}

        if config["metric"] in ["exact_match", "f1"] and config.get("answers_key"):
            em_f1_scores = generative_exact_match(
                gen_pipe_obj,
                ds,
                config["question_key"],
                config["context_key"],
                config["answers_key"],
            )
            results["exact_match"] = em_f1_scores[0]["exact_match"]
            results["f1"] = em_f1_scores[1]["f1"]
            log.info(
                f"  Result: EM = {results['exact_match']:.2f}%, F1 = {results['f1']:.2f}%"
            )
        else:
            # General generative evaluation loop
            answer_extractor = (
                getattr(eval.utils, config["answer_extractor_fn"])
                if isinstance(config["answer_extractor_fn"], str)
                else config["answer_extractor_fn"]
            )
            
            for ex in tqdm(ds, desc=f"Generating for {config['name']}"):
                prompt_text = config["prompt_builder_fn"](ex)
                gen_kwargs = config.get("generation_kwargs", {})
                
                # Ensure pad_token_id and eos_token_id are passed to generation pipeline if not already handled
                # The pipeline usually inherits this from its tokenizer/model, but explicit can be safer.
                # Assuming pipeline itself uses the model/tokenizer's config and tokens.
                
                out = gen_pipe_obj(prompt_text, **gen_kwargs)[0]["generated_text"]
                
                # Remove prompt from output
                if out.startswith(prompt_text):
                    generated_only = out[len(prompt_text):].strip()
                else: # Fallback if prompt isn't perfectly at start
                    generated_only = out.strip()

                prediction = answer_extractor(generated_only)
                
                # Ensure reference is in expected format (list for ROUGE, single string for accuracy)
                reference = ex[config["reference_key"]]
                if config["metric"] == "rouge":
                    if not isinstance(reference, list):
                        reference = [str(reference)]
                else: # For accuracy/f1
                    reference = str(reference)
                
                metric.add(prediction=prediction, reference=reference)

            computed_metric = metric.compute()
            if config["metric"] == "rouge":
                results["rougeL"] = computed_metric["rougeL"]
                log.info(f"  Result: RougeL = {results['rougeL']:.2f}")
            elif config["metric"] == "accuracy":
                results["accuracy"] = computed_metric["accuracy"] * 100
                log.info(f"  Result: Accuracy = {results['accuracy']:.2f}%")
            elif config["metric"] == "f1":
                results["f1"] = computed_metric["f1"] * 100
                log.info(f"  Result: F1 = {results['f1']:.2f}%")
            
        return results
    except Exception as e:
        log.error(f"Error in generative benchmark '{config['name']}': {e}")
        raise


def run_humaneval_benchmark(model, tokenizer, config, device):
    """Runs the HumanEval benchmark."""
    try:
        problems = config["read_problems_fn"]()
        
        # HumanEval's evaluate_functional_correctness needs specific model/tokenizer args.
        # It's better to pass them directly, as it manages generation internally.
        # This requires the model to be on the correct device already.
        pass_metrics = evaluate_functional_correctness(
            problems=problems,
            k=config["k"],
            n_workers=config["n_workers"],
            timeout=config["timeout"],
            model=model,      # Passed directly
            tokenizer=tokenizer, # Passed directly
            device=device,
        )
        log.info(f"  Result: pass@1 = {pass_metrics['pass@1']*100:.2f}%")
        return pass_metrics["pass@1"]
    except Exception as e:
        log.error(f"Error in HumanEval benchmark '{config['name']}': {e}")
        raise