#!/usr/bin/env python3
# evaluate_model.py
# A streamlined evaluation script for Llama-based models, solving all prior bugs:
# - Patches pad_token_id if it's a list
# - Uses streaming for large datasets to avoid huge downloads
# - Correct column names and mappings for each dataset
# - Wraps MC prompts in chat template for instruct-tuned models
# - Skips or falls back gracefully on errors (e.g., MMLU not found)
# - Samples N examples per task via --max_eval_samples

import argparse
import math
import torch
import itertools
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from datasets import load_dataset
import evaluate

# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------
def load_model_and_tokenizer(model_path: str, device: str):
    # 1) Load config and patch pad_token_id if needed
    config = AutoConfig.from_pretrained(model_path)
    if isinstance(config.pad_token_id, (list, tuple)):
        config.pad_token_id = int(config.pad_token_id[0])

    # 2) Load model with the fixed config
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config)
    model.to(device).eval()

    # 3) Load tokenizer and patch its pad_token_id
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if isinstance(tokenizer.pad_token_id, (list, tuple)):
        tokenizer.pad_token_id = int(tokenizer.pad_token_id[0])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4) Sync model.config to tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

def compute_ppl_from_texts(model, tokenizer, texts, device,
                           block_size=1024, batch_size=8):
    """Per-batch tokenization + PPL over a list of raw strings."""
    total_loss = 0.0
    n_batches = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=block_size,
        ).to(device)
        with torch.no_grad():
            out = model(input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask,
                        labels=enc.input_ids)
        total_loss += out.loss.item()
        n_batches += 1
    return math.exp(total_loss / n_batches) if n_batches > 0 else float("inf")

def mc_accuracy(
    model,
    tokenizer,
    ds,
    device,
    prompt_key,
    choices_key,
    label_key,
):
    """
    Multiple-choice accuracy: pick the choice with highest log-likelihood.
    Wraps in chat template for instruct-tuned models.
    """
    correct = 0
    total = 0

    for ex in ds:
        prompt = ex[prompt_key]
        choices = ex[choices_key]
        gold = ex[label_key]  # integer index

        best_score = None
        best_idx = None

        for i, choice in enumerate(choices):
            # Wrap in chat template
            conv = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": choice},
            ]
            text = tokenizer.apply_chat_template(conv, tokenize=False)

            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(device)

            with torch.no_grad():
                out = model(enc.input_ids, labels=enc.input_ids)
            score = -out.loss.item()  # higher is better

            if best_score is None or score > best_score:
                best_score = score
                best_idx = i

        if best_idx == gold:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0

def generative_exact_match(
    qa_pipeline, ds, question_key, context_key, answers_key
):
    """
    Run HF QA pipeline and measure EM/F1.
    """
    em = evaluate.load("exact_match")
    f1 = evaluate.load("f1")
    for ex in ds:
        pred = qa_pipeline(
            question=ex[question_key],
            context=ex[context_key],
            max_length=256,
            truncation=True,
        )["answer"]
        refs = ex[answers_key]["text"] if isinstance(ex[answers_key], dict) else ex[answers_key]
        em.add(prediction=pred, references=refs)
        f1.add(prediction=pred, references=refs)
    return em.compute(), f1.compute()

# -----------------------------------------------------------------------------
# MAIN BENCHMARK SUITE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path or HF ID of your trained model")
    parser.add_argument("--device", default="cuda",
                        help="cpu, cuda, or mps")
    parser.add_argument("--max_eval_samples", type=int, default=512,
                        help="How many examples to sample per benchmark")
    args = parser.parse_args()

    device = args.device
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)

    N = args.max_eval_samples
    print("\n1) Perplexity benchmarks (N =", N, "per split)")

    # -- 1a) Wikitext-2-raw-v1 (tiny) --
    try:
        wt2 = load_dataset("wikitext", "wikitext-2-raw-v1",
                           split=f"validation[:{N}]")
        wt2_texts = wt2["text"]
        wt2_ppl = compute_ppl_from_texts(model, tokenizer, wt2_texts, device)
        print(f"   • Wikitext-2-raw-v1[0:{N}] → PPL = {wt2_ppl:.2f}")
    except Exception as e:
        print(f"   ! Skipping Wikitext-2: {type(e).__name__}: {e}")

    # -- 1b) Pile: streaming sample N examples (no full download) --
    try:
        pile_stream = load_dataset(
            "EleutherAI/pile",
            split="test",
            streaming=True,
            trust_remote_code=True,
        )
        pile_texts = [ex["text"] for ex in itertools.islice(pile_stream, N)]
        pile_ppl = compute_ppl_from_texts(model, tokenizer, pile_texts, device)
        print(f"   • Pile[stream first {N}] → PPL = {pile_ppl:.2f}")
    except Exception as e:
        print(f"   ! Skipping Pile streaming ({type(e).__name__}): {e}")

    # -- 1c) C4 fallback, streaming mode --
    try:
        c4_stream = load_dataset(
            "allenai/c4", "en",
            split="validation",
            streaming=True,
        )
        c4_texts = [ex["text"] for ex in itertools.islice(c4_stream, N)]
        c4_ppl = compute_ppl_from_texts(model, tokenizer, c4_texts, device)
        print(f"   • C4-en[stream first {N}] → PPL = {c4_ppl:.2f}")
    except Exception as e:
        print(f"   ! Skipping C4: {type(e).__name__}: {e}")

    print("\n2) Multiple-choice reasoning")

    # HellaSwag
    try:
        hs = load_dataset("hellaswag", "default",
                          split=f"validation[:{N}]")
        acc_hs = mc_accuracy(model, tokenizer, hs, device,
                             prompt_key="ctx",
                             choices_key="endings",
                             label_key="label")
        print(f"  • HellaSwag → {acc_hs*100:.2f}%")
    except Exception as e:
        print(f"  ! Skipping HellaSwag: {type(e).__name__}: {e}")

    # Winogrande
    try:
        wg = load_dataset("winogrande", "winogrande_xl",
                          split=f"validation[:{N}]")
        wg = wg.map(lambda ex: {
            "prompt": ex["sentence"],
            "choices": [ex["option1"], ex["option2"]],
            "gold": int(ex["answer"]) - 1,  # '1' or '2' → 0 or 1
        })
        acc_wg = mc_accuracy(model, tokenizer, wg, device,
                             prompt_key="prompt",
                             choices_key="choices",
                             label_key="gold")
        print(f"  • Winogrande → {acc_wg*100:.2f}%")
    except Exception as e:
        print(f"  ! Skipping Winogrande: {type(e).__name__}: {e}")

    # PIQA
    try:
        piqa = load_dataset("piqa", split=f"validation[:{N}]")
        piqa = piqa.map(lambda ex: {
            "prompt": ex["goal"],
            "choices": [ex["sol1"], ex["sol2"]],
            "gold": ex["label"],
        })
        acc_piqa = mc_accuracy(model, tokenizer, piqa, device,
                               prompt_key="prompt",
                               choices_key="choices",
                               label_key="gold")
        print(f"  • PIQA → {acc_piqa*100:.2f}%")
    except Exception as e:
        print(f"  ! Skipping PIQA: {type(e).__name__}: {e}")

    # CommonsenseQA
    try:
        csqa = load_dataset("commonsense_qa", split=f"validation[:{N}]")
        csqa = csqa.map(lambda ex: {
            "prompt": ex["question"],
            "choices": ex["choices"]["text"],
            "gold": ex["choices"]["label"].index(ex["answerKey"]),
        })
        acc_csqa = mc_accuracy(model, tokenizer, csqa, device,
                               prompt_key="prompt",
                               choices_key="choices",
                               label_key="gold")
        print(f"  • CommonsenseQA → {acc_csqa*100:.2f}%")
    except Exception as e:
        print(f"  ! Skipping CommonsenseQA: {type(e).__name__}: {e}")

    print("\n3) MMLU (5-shot)")
    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "computer_security", "econometrics",
        "global_facts", "high_school_chemistry",
    ]
    mmlu_accs = []
    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split=f"dev[:{N}]")  # use dev for small N
            # 5-shot: prepend first 5 examples to each test prompt
            shot = ds.select(range(min(5, len(ds))))
            def make_prompt(ex):
                ctx = ""
                for s in shot:
                    ctx += f"Question: {s['question']}\nAnswer: {s['choices'][s['answer']]}\n\n"
                return {"prompt": ctx + f"Question: {ex['question']}\nAnswer:"}
            test = ds.map(make_prompt)
            test = test.map(lambda ex: {
                "choices": ex["choices"],
                "gold": ex["answer"],
            })
            acc = mc_accuracy(
                model, tokenizer, test, device,
                prompt_key="prompt",
                choices_key="choices",
                label_key="gold",
            )
            mmlu_accs.append(acc)
            print(f"  • MMLU/{subj} → {acc*100:.2f}%")
        except Exception as e:
            print(f"  ! Skipping MMLU/{subj}: {type(e).__name__}: {e}")

    if mmlu_accs:
        mean_mmlu = sum(mmlu_accs) / len(mmlu_accs)
        print(f"  → MMLU (mean over {len(mmlu_accs)} subjects) = {mean_mmlu*100:.2f}%")
    else:
        print("  → No MMLU subjects could be loaded; skipping overall MMLU.")

    print("\n4) Reading comprehension")
    try:
        qa_pipe = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=0 if device.startswith("cuda") else -1,
        )
        squad = load_dataset("squad_v2", split=f"validation[:{N}]")
        em_squad, f1_squad = generative_exact_match(
            qa_pipe, squad, "question", "context", "answers"
        )
        print(f"  • SQuAD v2 → EM = {em_squad['exact_match']:.2f}%, F1 = {f1_squad['f1']:.2f}%")
    except Exception as e:
        print(f"  ! Skipping SQuAD v2: {type(e).__name__}: {e}")

    print("\n5) Math (GSM8K)")
    try:
        gsm = load_dataset("gsm8k", "main", split=f"test[:{N}]")
        metric = evaluate.load("accuracy")
        for ex in gsm:
            prompt = ex["question"] + "\nLet's think step by step."
            out = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device.startswith("cuda") else -1,
                max_new_tokens=64,
                temperature=0.0,
                do_sample=False,
            )([prompt])[0]["generated_text"]
            # extract final line as answer
            ans = out.strip().split("\n")[-1]
            metric.add(prediction=ans, reference=ex["answer"])
        print(f"  • GSM8K → {metric.compute()['accuracy']*100:.2f}%")
    except Exception as e:
        print(f"  ! Skipping GSM8K: {type(e).__name__}: {e}")

    print("\n6) HumanEval (pass@1)")
    try:
        from human_eval.data import HUMAN_EVAL
        from human_eval.evaluation import evaluate_functional_correctness
        # HumanEval requires the dataset file; download if needed
        import os
        if not os.path.exists(HUMAN_EVAL):
            os.system("wget https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz -O " + HUMAN_EVAL)
        pass1 = evaluate_functional_correctness(
            model=model,
            tokenizer=tokenizer,
            device=device,
            k=[1],
            n_samples=1,  # pass@1
            problems=HUMAN_EVAL,
        )["pass@1"]
        print(f"  • HumanEval pass@1 = {pass1*100:.2f}%")
    except ImportError as e:
        print(f"  ! Skipping HumanEval: {e} (pip install human-eval)")
    except Exception as e:
        print(f"  ! Skipping HumanEval: {type(e).__name__}: {e}")

    print("\nDone.")