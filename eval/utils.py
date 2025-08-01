# src.eval/utils.py
"""Utility functions for evaluation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Callable, Dict, Any
import numpy as np

import math
import re
import itertools

import evaluate
from tqdm import tqdm
from transformers import pipeline

def load_model_and_tokenizer(model_path: str, ce_bias: float, dynamic_k: float):
    """Load the model and tokenizer, applying CE bias and dynamic K."""
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.set_ce_bias(ce_bias)  # From your DynamicLlama code
    model.set_dynamic_k(dynamic_k)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def compute_average_activation(model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> float:
    """Compute average gate activation during inference."""
    with torch.no_grad():
        outputs = model(**inputs)
        # Assuming gate activations are accessible, e.g., from model.get_last_gate_means()
        gate_means = model.get_last_gate_means()  # From your code
        if gate_means is not None:
            return torch.mean(torch.tensor(gate_means)).item()
    return np.nan  # Fallback if not available

def run_benchmark(benchmark_fn: Callable, model, tokenizer, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single benchmark and return results."""
    results = benchmark_fn(model, tokenizer, config)
    avg_activation = compute_average_activation(model, results.get("inputs", {}))
    results["average_activation"] = avg_activation
    return results

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
        if not batch: # Handle empty batches
            continue
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=block_size,
        ).to(model.device)
        with torch.no_grad():
            # For PPL, we need to ensure labels are shifted within the model
            # For causal LMs, labels=input_ids with attention_mask usually handles this.
            out = model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                labels=enc.input_ids, # Labels are typically input_ids for PPL
            )
        total_loss += out.loss.item()
        n_batches += 1
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

    for ex in tqdm(dataset, desc="MC eval"):
        prompt = ex[prompt_key]
        choices = ex[choices_key]
        gold_idx = ex[label_key]  # integer index

        best_score = float("-inf")
        predicted_idx = None

        for i, choice_text in enumerate(choices):
            if not choice_text: # Skip empty choices
                continue

            # Construct the full text to get logprobs over the choice
            if is_instruct:
                # Assuming the assistant's turn is only the choice text
                conv = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": choice_text},
                ]
                full_text = tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=False
                )
                # Need to find where the assistant's content starts to get choice logprobs
                # A common instruct format might be "PROMPT [SEP] ASSISTANT_RESPONSE"
                # We need to find the token ID corresponding to the start of the assistant's content.
                # A simple way for logprob calculation is to tokenize prompt and choice separately
                # then combine their token IDs and calculate logprobs over the choice tokens.
                
                # Tokenize prompt and choice separately to identify choice tokens accurately
                # Note: This is a robust way to ensure we only score the `choice_text` part
                prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
                choice_input_ids = tokenizer(choice_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
                
                # Concatenate, adding BOS if needed by tokenizer (handled by AutoTokenizer generally)
                # Ensure prompt_input_ids doesn't already end with a token that should be part of choice.
                # Simplest way is to just feed the full chat template output and slice logprobs
                full_encoded = tokenizer(full_text, return_tensors="pt").to(model.device)
                
                # Determine prompt length in tokens from the full chat template applied to only the prompt
                prompt_only_conv = [{"role": "user", "content": prompt}]
                prompt_only_text = tokenizer.apply_chat_template(prompt_only_conv, tokenize=False, add_generation_prompt=True)
                # The add_generation_prompt=True here is crucial if the chat template adds tokens for the assistant's turn beginning.
                prompt_len = tokenizer(prompt_only_text, return_tensors="pt").input_ids.shape[-1]
                
            else: # Base model
                full_text = prompt + " " + choice_text
                full_encoded = tokenizer(full_text, return_tensors="pt").to(model.device)
                prompt_len = tokenizer(prompt + " ", return_tensors="pt").input_ids.shape[-1]
                # Adjust prompt_len if tokenizer adds a leading space token or BOS token in a different way.
                # A more precise way would be to encode (prompt) and (choice_text) separately
                # and then align their token IDs in the full sequence.
                # For simplicity, sticking to the original logic.


            input_ids = full_encoded.input_ids
            attention_mask = full_encoded.attention_mask

            if input_ids.shape[1] <= prompt_len:
                # Choice text is empty or too short, score is irrelevant or -inf
                score = float("-inf")
            else:
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    logprobs = torch.log_softmax(logits, dim=-1) # (batch_size, seq_len, vocab_size)

                    # Get logprobs only for the choice tokens
                    # The token at prompt_len-1 is the token *before* the first choice token (or the last prompt token)
                    # We want logprobs for tokens from index `prompt_len` onwards, predicted by logits at `prompt_len-1` onwards.
                    # No, this is wrong. logprobs[0, k] is the logprob of token k+1 given tokens 0..k.
                    # So, to get the logprobs of `choice_input_ids`, we need to look at `logprobs` from `prompt_len-1` up to `input_ids.shape[1]-2`.
                    # And use `input_ids` from `prompt_len` up to `input_ids.shape[1]-1`.

                    # This is more robust: calculate sum of logprobs of each token in choice_text given previous context.
                    # Slice logprobs from the position *after* the prompt tokens.
                    # And slice input_ids from the position *of* the first choice token.
                    
                    # Original slicing was `prompt_len - 1 : -1`, which scores the token *after* the last prompt token,
                    # and then all subsequent tokens until the last token in `input_ids` (which has no logprob for next token).
                    # This seems correct for scoring the tokens that make up the `choice` part.
                    choice_logprobs_all_vocab = logprobs[0, prompt_len - 1 : -1] # Logprobs for all vocab at each choice token position
                    choice_ids_actual = input_ids[0, prompt_len:] # Actual token IDs for the choice text
                    
                    if choice_ids_actual.numel() == 0:
                        score = float("-inf")
                    else:
                        # Gather the logprobs corresponding to the actual choice tokens
                        gathered_logprobs = torch.gather(
                            choice_logprobs_all_vocab, dim=1, index=choice_ids_actual.unsqueeze(-1)
                        ).squeeze(-1) # (num_choice_tokens,)

                        # Normalize by length or sum
                        score = gathered_logprobs.mean().item() # Mean logprob (normalized)

            if score > best_score:
                best_score = score
                predicted_idx = i

        if predicted_idx is not None and predicted_idx == gold_idx:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def generative_exact_match(gen_pipe_obj: pipeline, dataset, question_key, context_key, answers_key):
    """Generates answers and computes Exact Match and F1 scores."""
    em = evaluate.load("exact_match")
    f1 = evaluate.load("f1")
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
            return lines[-1].strip()
        return text.strip() # If no lines, return original stripped text