#!/usr/bin/env python3
# evaluate_model.py
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
        # Rarely HF writes pad_token_id as a list; just grab the first slot
        config.pad_token_id = int(config.pad_token_id[0])

    # 2) Load model with your (now clean) config
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config)
    model.to(device).eval()

    # 3) Load tokenizer and patch its pad_token_id
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if isinstance(tokenizer.pad_token_id, (list, tuple)):
        tokenizer.pad_token_id = int(tokenizer.pad_token_id[0])
    if tokenizer.pad_token_id is None:
        # fallback to eos_token_id if pad is still unset
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4) Sync model.config to tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

def compute_ppl(model, tokenizer, ds, device, block_size=1024, batch_size=4):
    """
    Compute perplexity on a HF Dataset 'ds' with a 'text' column by
    tokenizing+padding each small batch manually to avoid collate errors.
    """
    total_loss = 0.0
    n_batches  = 0

    for i in range(0, len(ds), batch_size):
        # grab a slice of raw strings
        batch_texts = ds[i : i + batch_size]["text"]
        # tokenize+pad/truncate to block_size
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=block_size,
        ).to(device)

        with torch.no_grad():
            out = model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                labels=enc.input_ids,
            )
        total_loss += out.loss.item()
        n_batches  += 1

    avg_loss = total_loss / n_batches
    return math.exp(avg_loss)

def mc_accuracy(
    model, tokenizer, ds, device,
    prompt_key: str,
    choices_key: str,
    answer_key: str,
):
    """
    Multiple-choice accuracy: pick the choice with highest log-likelihood.
    """
    metric = evaluate.load("accuracy")
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for ex in ds:
        prompt = ex[prompt_key]
        best_score = None
        best_choice = None
        for choice in ex[choices_key]:
            text = prompt + " " + choice
            enc = tokenizer(text, return_tensors="pt").to(device)
            # we compute NLL on the *choice* portion only
            input_ids = enc.input_ids
            with torch.no_grad():
                out = model(input_ids, labels=input_ids)
            # out.loss is averaged over all tokens including prompt, so
            # we re-compute token-wise and mask prompt:
            logits = out.logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous()
            # mask out prompt tokens
            prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.size(1)
            labels[:, : prompt_len - 1] = -100
            token_loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )  # shape=(num_tokens,)
            nll = token_loss[labels.view(-1) != -100].sum().item()
            score = -nll  # higher is better
            if best_score is None or score > best_score:
                best_score = score
                best_choice = choice
        metric.add(prediction=best_choice, reference=ex[answer_key])
    return metric.compute()["accuracy"]

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
        # refs may be list
        em.add(prediction=pred, reference=refs)
        f1.add(prediction=pred, reference=refs)
    return em.compute(), f1.compute()

# -----------------------------------------------------------------------------
# MAIN BENCHMARK SUITE
# -----------------------------------------------------------------------------
def compute_ppl_from_texts(model, tokenizer, texts, device,
                           block_size=1024, batch_size=8):
    """Per‐batch tokenization+ppl over a list of raw strings."""
    total_loss = 0.0
    n_batches  = 0
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
        n_batches  += 1
    return math.exp(total_loss / n_batches)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_eval_samples", type=int, default=512,
                        help="How many examples to sample per benchmark")
    args = parser.parse_args()

    device = args.device
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)

    N = args.max_eval_samples
    print("1) Perplexity benchmarks (N =", N, "per split)")

    # -- 1a) Wikitext-2-raw-v1 (tiny) --
    wt2 = load_dataset("wikitext", "wikitext-2-raw-v1",
                       split=f"validation[:{N}]")
    wt2_texts = wt2["text"]
    wt2_ppl = compute_ppl_from_texts(model, tokenizer, wt2_texts, device)
    print(f"   • Wikitext-2-raw-v1[0:{N}] → PPL = {wt2_ppl:.2f}")

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

    # -- 1c) (Optional) C4 fallback, streaming mode --
    try:
        c4_stream = load_dataset(
            "allenai/c4", "en",
            split="validation",
            streaming=True,
        )
        c4_texts = [ex["text"] for ex in itertools.islice(c4_stream, N)]
        c4_ppl = compute_ppl_from_texts(model, tokenizer, c4_texts, device)
        print(f"   • C4-en[stream first {N}] → PPL = {c4_ppl:.2f}")
    except Exception:
        pass

    print("\n2) Multiple-choice reasoning")
    for task in [
        ("HellaSwag",      "hellaswag",      "validation", "context", "endings", "label"),
        ("Winogrande",     "winogrande",     "validation", "sentence", "options", "answer"),
        ("PIQA",           "piqa",           "validation", "goal", "sol1",    "label0"),  # label0/label1
        ("CSQA",           "commonsense_qa", "validation", "question", "choices", "answerKey"),
    ]:
        name, ds_id, split, qk, ck, ak = task
        ds = load_dataset(ds_id, split=split)
        acc = mc_accuracy(model, tokenizer, ds, device, qk, ck, ak)
        print(f"  • {name} → accuracy = {acc*100:.2f}%")

    print("\n3) MMLU (5-shot)")
    subjects = [
      # pick a handful or loop through all 57
      "abstract_algebra", "anatomy", "astronomy", "business_ethics",
      "clinical_knowledge", "computer_security", "econometrics",
      "global_facts", "high_school_chemistry",
    ]
    mmlu_accs = []
    for subj in subjects:
        ds = load_dataset("mmlu", subj, split="test")
        # 5-shot: prepend first 5 examples to each test prompt
        shot = ds.select(range(5))
        def make_prompt(i, ex):
            ctx = ""
            for j, s in enumerate(shot):
                ctx += f"Question: {s['question']}\nAnswer: {s['answer']}\n\n"
            return {"prompt": ctx + "Question: " + ex["question"] + "\nAnswer:"}
        test = ds.select(range(5, 105)).map(make_prompt)
        # mc
        acc = mc_accuracy(
            model, tokenizer, test.to_dict(), device,
            prompt_key="prompt",
            choices_key="options",
            answer_key="answer",
        )
        mmlu_accs.append(acc)
        print(f"  • MMLU/{subj} → {acc*100:.2f}%")
    print("  → MMLU (mean) =", sum(mmlu_accs)/len(mmlu_accs)*100, "%")

    print("\n4) Reading comprehension")
    qa_pipe = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        device=0 if device.startswith("cuda") else -1,
    )
    squad = load_dataset("squad_v2", split="validation")
    em_squad, f1_squad = generative_exact_match(
        qa_pipe, squad, "question", "context", "answers"
    )
    print(f"  • SQuAD v2 → EM = {em_squad['exact_match']:.2f}%, F1 = {f1_squad['f1']:.2f}%")

    # you can add QuAC, DROP, Needle-in-Haystack in the same way…

    print("\n5) Math (GSM8K)")
    gsm = load_dataset("gsm8k", "main", split="test")
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

    print("\n6) HumanEval (pass@1)")
    # requires git+https://github.com/openai/human-eval to be installed
    try:
        from human_eval.data import get_test_problems
        from human_eval.evaluator import evaluate_function
        problems = get_test_problems()
        pass1 = 0
        for fn_name, fn_spec in problems.items():
            prompt = fn_spec.prompt
            gen = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device.startswith("cuda") else -1,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
            )([prompt])[0]["generated_text"]
            # wrap into a def and exec
            success = evaluate_function(gen, fn_spec.entry_point)
            if success:
                pass1 += 1
        print(f"  • HumanEval pass@1 = {pass1}/{len(problems)} = {pass1/len(problems)*100:.2f}%")
    except ImportError:
        print("  • Skipping HumanEval (requires `human-eval` pip install)")

    print("\nDone.")