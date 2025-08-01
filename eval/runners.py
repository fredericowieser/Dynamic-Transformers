# eval/runners.py
import evaluate
import logging
from datasets import load_dataset
from human_eval.evaluation import evaluate_functional_correctness
from tqdm import tqdm

import eval.utils as utils # Import as alias to fix 'builtin_function_or_method' error


log = logging.getLogger(__name__)


def _load_and_prepare_dataset(config, num_samples):
    """Loads a dataset from Hugging Face and applies slicing/preprocessing."""
    log.info(f"Loading dataset: {config.get('dataset_id')} ({config.get('split')})...")
    load_kwargs = {
        "path": config["dataset_id"],
        "name": config.get("dataset_config"),
        "split": config["split"], # Load full split for streaming, slice later
    }
    # Add trust_remote_code if specified in benchmark config
    if config.get("trust_remote_code"):
        load_kwargs["trust_remote_code"] = True

    dataset = load_dataset(**load_kwargs)
    
    # Handle streaming datasets needing explicit iteration
    if config.get("streaming"):
        # For streaming, apply islice directly after loading the stream
        dataset_content = list(itertools.islice(dataset, num_samples))
    else:
        # For non-streaming, apply slicing here
        dataset_content = dataset.select(range(min(num_samples, len(dataset))))

    # Apply general preprocessing if a preprocess_fn is provided
    # Note: MMLU's preprocess_fn needs dev_shot_data and is handled in run_mmlu_benchmark.
    if "preprocess_fn" in config and config.get("type") != "mc_multi_subject":
        log.info("Applying dataset preprocessing function...")
        dataset_content = dataset_content.map(config["preprocess_fn"])
    
    log.info(f"Loaded {len(dataset_content)} samples.")
    return dataset_content


def _log_gate_stats_for_benchmark(model, benchmark_name):
    """Helper to log gate activation means after a benchmark run."""
    if hasattr(model, "get_last_gate_means") and model._log_gates:
        avg_gate_activation = model.get_last_gate_means() # This is now a single float
        if avg_gate_activation is not None:
            log.info(f"    Average gate activation for {benchmark_name}: {avg_gate_activation:.4f}")
        else:
            log.info(f"    No gate activation data recorded for {benchmark_name}.")
        # Reset for next benchmark
        model._last_gate_means = None


def run_perplexity_benchmark(model, tokenizer, config, num_samples, device):
    """Runs a perplexity evaluation benchmark."""
    try:
        # fetch_texts_fn now handles both streaming and non-streaming slicing logic
        texts = _load_and_prepare_dataset(config, num_samples) # This returns a list of strings
        
        ppl = utils.compute_ppl_from_texts(
            model, tokenizer, texts, device, **config.get("ppl_kwargs", {})
        )
        log.info(f"  Result: PPL = {ppl:.2f}")
        _log_gate_stats_for_benchmark(model, config["name"]) # Log here after benchmark
        return ppl
    except Exception as e:
        log.error(f"Error in perplexity benchmark '{config['name']}': {e}")
        raise


def run_multiple_choice_benchmark(
    model, tokenizer, config, num_samples, device, is_instruct
):
    """Runs a standard multiple-choice evaluation benchmark."""
    try:
        ds = _load_and_prepare_dataset(config, num_samples)
        acc = utils.mc_accuracy( # Call mc_accuracy from utils
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
        _log_gate_stats_for_benchmark(model, config["name"]) # Log here after benchmark
        return acc
    except Exception as e:
        log.error(f"Error in multiple-choice benchmark '{config['name']}': {e}")
        raise


def run_mmlu_benchmark(model, tokenizer, config, num_samples, device, is_instruct):
    """Runs the MMLU multiple-choice benchmark with 5-shot setup."""
    mmlu_accs = []
    # Accumulate gate activation means per layer for the entire MMLU run
    model_gate_means_history_mmlu = []
    original_forward = model.forward # Save original forward

    if hasattr(model, "enable_gate_logging") and model._log_gates:
        def wrapped_forward_mmlu(*f_args, **f_kwargs):
            out = original_forward(*f_args, **f_kwargs)
            if hasattr(model, "_log_gates") and model._log_gates and hasattr(model, "get_last_gate_means"):
                last = model.get_last_gate_means()
                if last:
                    model_gate_means_history_mmlu.append(last)
            return out
        model.forward = wrapped_forward_mmlu # Apply hook for MMLU

    try:
        for subj in tqdm(config["subjects"], desc="MMLU subjects"):
            try:
                # Load test split for current subject
                ds_test = load_dataset(config["dataset_id"], subj, split=f"{config['split']}[:{num_samples}]")
                # Load dev split for few-shot examples
                ds_dev = load_dataset(config["dataset_id"], subj, split=config["dev_split"])
                dev_shot_data = ds_dev.select(range(min(config["num_shot"], len(ds_dev))))

                # Apply preprocessing for each subject's test set with the few-shot data
                # Pass dev_shot_data directly to preprocess_fn
                processed_ds = ds_test.map(lambda ex: config["preprocess_fn"](ex, dev_shot_data))
                
                # Now, run mc_accuracy on this preprocessed dataset
                acc = utils.mc_accuracy( # Call mc_accuracy from utils
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
            # Combine all individual gate means collected during MMLU for overall MMLU average
            if model_gate_means_mmlu:
                all_layer_activations = [item for sublist in model_gate_means_mmlu for item in sublist]
                overall_mmlu_gate_activation = sum(all_layer_activations) / len(all_layer_activations) if all_layer_activations else None
                if overall_mmlu_gate_activation is not None:
                    log.info(f"    Average gate activation for MMLU_mean: {overall_mmlu_gate_activation:.4f}")

            return {"mean_accuracy": mean_mmlu, "individual_accuracies": {s: a for s, a in zip(config["subjects"], mmlu_accs)}}
        else:
            log.info("  No MMLU subjects loaded; skipping mean calculation.")
            return {"mean_accuracy": 0.0, "individual_accuracies": {}}
    finally:
        # Restore original forward method after MMLU run
        if hasattr(model, "enable_gate_logging") and model._log_gates:
            model.forward = original_forward
            model._last_gate_means = None # Reset global gate stats after this multi-stage benchmark


def run_generative_benchmark(gen_pipe_obj, model, config, num_samples):
    """Runs a generative evaluation benchmark."""
    
    # Accumulate gate activation means per layer for this benchmark
    model_gate_means_history = []
    original_forward = model.forward # Save original forward

    if hasattr(model, "enable_gate_logging") and model._log_gates:
        def wrapped_forward(*f_args, **f_kwargs):
            out = original_forward(*f_args, **f_kwargs)
            if hasattr(model, "_log_gates") and model._log_gates and hasattr(model, "get_last_gate_means"):
                last = model.get_last_gate_means()
                if last:
                    model_gate_means_history.append(last)
            return out
        model.forward = wrapped_forward # Apply hook
    
    try:
        ds = _load_and_prepare_dataset(config, num_samples)

        results = {}

        # NEW: Handle SQuAD-like QA separately, as it uses generative_exact_match
        if config.get("type") == "generative_qa":
            em_f1_scores = utils.generative_exact_match(
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
        else: # Standard generative evaluation loop for other metrics (accuracy, rouge, f1)
            if config["metric"] == "accuracy":
                metric = evaluate.load("accuracy")
            elif config["metric"] == "rouge":
                metric = evaluate.load("rouge")
            elif config["metric"] == "f1":
                metric = evaluate.load("f1")
            else:
                raise ValueError(f"Unsupported metric: {config['metric']}")

            answer_extractor = (
                getattr(utils, config["answer_extractor_fn"])
                if isinstance(config["answer_extractor_fn"], str)
                else config["answer_extractor_fn"]
            )
            
            for ex in tqdm(ds, desc=f"Generating for {config['name']}"):
                prompt_text = config["prompt_builder_fn"](ex)
                gen_kwargs = config.get("generation_kwargs", {})
                
                out = gen_pipe_obj(prompt_text, **gen_kwargs)[0]["generated_text"]
                
                # Remove prompt from output for cleaner evaluation
                if out.startswith(prompt_text):
                    generated_only = out[len(prompt_text):].strip()
                else: # Fallback if prompt isn't perfectly at start
                    generated_only = out.strip()

                prediction = answer_extractor(generated_only)
                
                # Ensure reference is in expected format (list for ROUGE, single string for accuracy/f1)
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
    finally:
        # Restore original forward method
        if hasattr(model, "enable_gate_logging") and model._log_gates:
            model.forward = original_forward
            # Set the overall mean for this benchmark in the model's internal state
            if model_gate_means_history:
                # model_gate_means_history is list of lists, each inner list is per-layer activation for one forward pass
                # Flatten the list of lists and calculate mean
                all_layer_activations = [item for sublist in model_gate_means_history for item in sublist]
                if all_layer_activations:
                    model._last_gate_means = sum(all_layer_activations) / len(all_layer_activations)
                else:
                    model._last_gate_means = None
            else:
                model._last_gate_means = None
        _log_gate_stats_for_benchmark(model, config["name"]) # Log here after benchmark


def run_humaneval_benchmark(model, tokenizer, config, device):
    """Runs the HumanEval benchmark."""
    # Accumulate gate activation means per layer for this benchmark
    model_gate_means_history = []
    original_forward = model.forward # Save original forward

    if hasattr(model, "enable_gate_logging") and model._log_gates:
        def wrapped_forward(*f_args, **f_kwargs):
            out = original_forward(*f_args, **f_kwargs)
            if hasattr(model, "_log_gates") and model._log_gates and hasattr(model, "get_last_gate_means"):
                last = model.get_last_gate_means()
                if last:
                    model_gate_means_history.append(last)
            return out
        model.forward = wrapped_forward # Apply hook

    try:
        problems = config["read_problems_fn"]()
        
        # HumanEval's evaluate_functional_correctness expects problems dict as first positional argument.
        pass_metrics = evaluate_functional_correctness( # FIX: removed problems=
            problems,
            k=config["k"],
            n_workers=config["n_workers"],
            timeout=config["timeout"],
            model=model,      # Passed directly
            tokenizer=tokenizer, # Passed directly
            device=device,
        )
        log.info(f"  Result: pass@1 = {pass_metrics['pass@1']*100:.2f}%")
        return pass_metrics["pass@1"]
    finally:
        # Restore original forward method
        if hasattr(model, "enable_gate_logging") and model._log_gates:
            model.forward = original_forward
            if model_gate_means_history:
                all_layer_activations = [item for sublist in model_gate_means_history for item in sublist]
                if all_layer_activations:
                    model._last_gate_means = sum(all_layer_activations) / len(all_layer_activations)
                else:
                    model._last_gate_means = None
            else:
                model._last_gate_means = None
        _log_gate_stats_for_benchmark(model, config["name"]) # Log here after benchmark