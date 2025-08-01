import math
import re
import itertools
import logging

import evaluate
import torch
from tqdm import tqdm
from transformers import pipeline

log = logging.getLogger(__name__)


def compute_ppl_from_texts(
    model, tokenizer, texts, device, block_size=1024, batch_size=8
):
    """Computes perplexity for a list of texts."""
    total_loss = 0.0
    n_batches = 0
    # Ensure model is on the correct device for PPL calculation
    model.to(device)
    for i in tqdm(range(0, len(texts), batch_size), desc="PPL batches"):
        batch = texts[i : i + batch_size]
        if not batch: # Handle empty batches or end of list
            continue
        
        # Filter out empty strings from the batch if any
        batch = [text for text in batch if text.strip()]
        if not batch:
            continue

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=block_size,
        ).to(model.device)
        
        if enc.input_ids.numel() == 0: # Check if batch became empty after tokenization
            log.warning(f"Skipping an empty tokenized batch during PPL computation.")
            continue

        with torch.no_grad():
            # For PPL, we need to ensure labels are shifted within the model.
            # For causal LMs, passing labels=input_ids with attention_mask usually handles this internally.
            out = model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                labels=enc.input_ids, # Labels are typically input_ids for PPL
            )
        
        # NEW: Check if out.loss is not None
        if out.loss is not None:
            total_loss += out.loss.item()
            n_batches += 1
        else:
            log.warning(f"Model output loss was None for a batch during PPL computation. Skipping this batch.")

    return math.exp(total_loss / n_batches) if n_batches > 0 else float("inf")


def mc_accuracy(
    model,
    tokenizer,
    dataset, # Renamed from 'ds' for clarity
    device,
    prompt_key,
    choices_key,
    label_key,
    is_instruct: bool = False,
):
    """
    Multiple-choice accuracy: normalized logprob of choice tokens.
    For instruct models, use chat template; for base, raw prompt.
    """
    correct = 0
    total = 0

    # Ensure model is on the correct device for MC calculation
    model.to(device)

    # Accumulate gate activation means per layer for this benchmark
    if hasattr(model, "enable_gate_logging") and model._log_gates:
        model_gate_means_history = []
        # Temporarily wrap model's forward to capture gate means per call
        original_forward = model.forward
        def wrapped_forward(*f_args, **f_kwargs):
            out = original_forward(*f_args, **f_kwargs)
            if hasattr(model, "_log_gates") and model._log_gates and hasattr(model, "get_last_gate_means"):
                last = model.get_last_gate_means()
                if last:
                    model_gate_means_history.append(last) # last is a list of floats
            return out
        model.forward = wrapped_forward
    else:
        model_gate_means_history = [] # Will remain empty if not logging

    try:
        for ex in tqdm(dataset, desc="MC eval"):
            prompt = ex[prompt_key]
            choices = ex[choices_key]
            gold_idx = ex[label_key]  # integer index

            best_score = float("-inf")
            predicted_idx = None

            for i, choice_text in enumerate(choices):
                if not isinstance(choice_text, str) or not choice_text.strip(): # Skip non-string or empty choices
                    continue

                # Construct the full text to get logprobs over the choice
                if is_instruct:
                    # Apply chat template for prompt only to get prompt token length in instruction format
                    # Then apply for full conversation to get full tokenized input
                    prompt_only_conv = [{"role": "user", "content": prompt}]
                    prompt_only_text_for_len = tokenizer.apply_chat_template(prompt_only_conv, tokenize=False, add_generation_prompt=True)
                    prompt_len = tokenizer(prompt_only_text_for_len, return_tensors="pt").input_ids.shape[-1]
                    
                    # Full conversation
                    full_conv = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": choice_text},
                    ]
                    full_text = tokenizer.apply_chat_template(full_conv, tokenize=False, add_generation_prompt=False)
                    full_encoded = tokenizer(full_text, return_tensors="pt").to(model.device)
                    
                else: # Base model
                    # For base models, ensure a space is included to separate prompt and choice,
                    # mimicking how the model would likely generate.
                    prompt_encoded = tokenizer(prompt + " ", return_tensors="pt").to(model.device)
                    prompt_len = prompt_encoded.input_ids.shape[-1]
                    
                    full_text = prompt + " " + choice_text
                    full_encoded = tokenizer(full_text, return_tensors="pt").to(model.device)


                input_ids = full_encoded.input_ids
                attention_mask = full_encoded.attention_mask

                if input_ids.shape[1] <= prompt_len:
                    # Choice text is empty or too short / results in no new tokens after prompt
                    score = float("-inf")
                else:
                    with torch.no_grad():
                        # Using the model.forward wrapper for gate logging
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                        logprobs = torch.log_softmax(logits, dim=-1) # (batch_size, seq_len, vocab_size)

                        # Extract logprobs for the choice tokens
                        # logprobs[0, k] is the logprob of token at (k+1) given tokens 0..k
                        # So, to score tokens from index `prompt_len` onwards (i.e., `input_ids[0, prompt_len:]`),
                        # we need logprobs from logits at `prompt_len-1` onwards.
                        
                        # choice_token_ids_in_full_sequence are input_ids[0, prompt_len:]
                        # logprobs_for_choice_tokens are logprobs[0, prompt_len-1 : input_ids.shape[1]-1]
                        
                        choice_logprobs_all_vocab = logprobs[0, prompt_len - 1 : -1] # Logprobs for all vocab at each position that *predicts* a choice token
                        choice_ids_actual = input_ids[0, prompt_len:] # Actual token IDs for the choice text
                        
                        if choice_ids_actual.numel() == 0:
                            score = float("-inf")
                        else:
                            # Gather the logprobs corresponding to the actual choice tokens
                            gathered_logprobs = torch.gather(
                                choice_logprobs_all_vocab, dim=1, index=choice_ids_actual.unsqueeze(-1)
                            ).squeeze(-1) # (num_choice_tokens,)

                            # Calculate mean logprob over the choice tokens
                            score = gathered_logprobs.mean().item()

                if score > best_score:
                    best_score = score
                    predicted_idx = i

            if predicted_idx is not None and predicted_idx == gold_idx:
                correct += 1
            total += 1
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


    return correct / total if total > 0 else 0.0


def generative_exact_match(gen_pipe_obj: pipeline, dataset, question_key, context_key, answers_key):
    """Generates answers and computes Exact Match and F1 scores."""
    em = evaluate.load("exact_match")
    f1 = evaluate.load("f1")

    # Accumulate gate activation means per layer for this benchmark
    model = gen_pipe_obj.model # Access model from pipeline
    if hasattr(model, "enable_gate_logging") and model._log_gates:
        model_gate_means_history = []
        original_forward = model.forward
        def wrapped_forward(*f_args, **f_kwargs):
            out = original_forward(*f_args, **f_kwargs)
            if hasattr(model, "_log_gates") and model._log_gates and hasattr(model, "get_last_gate_means"):
                last = model.get_last_gate_means()
                if last:
                    model_gate_means_history.append(last)
            return out
        model.forward = wrapped_forward
    else:
        model_gate_means_history = []

    try:
        for ex in tqdm(dataset, desc="Generative EM/F1"):
            prompt_text = f"Question: {ex[question_key]}\nContext: {ex[context_key]}\nAnswer:"
            # Use the pipeline object passed in
            pred = (
                gen_pipe_obj(prompt_text, max_new_tokens=32, do_sample=False)[0]["generated_text"]
                .split("Answer:")[-1]
                .strip()
            )
            # Ensure references are always a list of strings
            refs = ex[answers_key]
            if isinstance(refs, dict) and "text" in refs:
                refs = refs["text"]
            if not isinstance(refs, list):
                refs = [str(refs)] # Convert to list of strings for evaluation

            em.add(prediction=pred, references=refs)
            f1.add(prediction=pred, references=refs)
        return em.compute(), f1.compute()
    finally:
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


def extract_boxed_answer(text):
    """Extracts answer from \\boxed{} or last line."""
    match = re.search(r"\\boxed{([^}]*)}", text)
    if match:
        return match.group(1).strip()
    else:
        # Fallback to last line for cases where \\boxed{} is not used
        # This might need refinement based on specific dataset output formats
        lines = text.strip().split("\n")
        if lines:
            # Attempt to find common answer patterns if not boxed, e.g., "The answer is X."
            # For simplicity, returning the last non-empty line
            for line in reversed(lines):
                if line.strip():
                    return line.strip()
        return text.strip() # If no lines, return original stripped text