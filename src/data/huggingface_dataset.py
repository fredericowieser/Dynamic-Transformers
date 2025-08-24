import json
import logging
import re
from typing import Any
import os

import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import random_split
from transformers import PreTrainedTokenizerBase

log = logging.getLogger(__name__)

def _dict_list_to_chat(tokenizer, conv: list[dict[str, Any]]) -> dict[str, str]:
    """Takes [{'role': 'user', 'content': '…'}, …] -> {'text': '…'}"""
    norm = []
    for turn in conv:
        role = (turn.get("role") or turn.get("from") or "").lower()
        if role in {"human", "user"}:
            role = "user"
        elif role in {"assistant", "gpt", "model"}:
            role = "assistant"
        norm.append({"role": role, "content": turn.get("content") or turn.get("value") or ""})

    norm = [t for t in norm if t["content"] and t["content"].strip()]
    if not norm:
        return None

    try:
        return {"text": tokenizer.apply_chat_template(norm, tokenize=False)}
    except Exception:
        joined = "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in norm)
        return {"text": joined}

class HuggingFaceDataset:
    """
    A class to load, process, and prepare a single Hugging Face dataset for language model training.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_name: str,
        text_column: str,
        block_size: int,
        dataset_config: str = None,
        validation_split_percentage: int = 5,
        train_subset_ratio: float | None = None,
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.block_size = block_size
        self.dataset_config = dataset_config
        self.validation_split_percentage = validation_split_percentage
        self.train_subset_ratio = train_subset_ratio

    def _format_text(self, examples):
        """Extremely tolerant normaliser for various chat and instruction formats."""
        preferred = self.text_column
        raw = examples.get(preferred)

        if raw is None:
            for alt in ("messages", "conversation", "conversations", "prompt_response", "text", "chosen"):
                if alt in examples:
                    raw = examples[alt]
                    break
        
        if raw is None:
            q = examples.get("query") or examples.get("prompt")
            a = examples.get("response") or examples.get("answer")
            if q is not None and a is not None:
                return _dict_list_to_chat(self.tokenizer, [{"role": "user", "content": q}, {"role": "assistant", "content": a}])

        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            return _dict_list_to_chat(self.tokenizer, raw) or {"text": ""}

        if isinstance(raw, str):
            if raw.strip().startswith(("{", "[")):
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, list):
                        return _dict_list_to_chat(self.tokenizer, obj) or {"text": ""}
                except Exception:
                    pass
            
            blocks = re.split(r"###\s*|\n(?=\s*(Human|Assistant|User):)", raw.strip())
            conv = []
            for blk in blocks:
                if not isinstance(blk, str):
                    continue
                m = re.match(r"\s*(Human|Assistant|User)\s*:\s*(.*)", blk, flags=re.S)
                if m:
                    role = "user" if m.group(1) in {"Human", "User"} else "assistant"
                    content = m.group(2).strip()
                    if content:
                        conv.append({"role": role, "content": content})
            if conv:
                return _dict_list_to_chat(self.tokenizer, conv) or {"text": ""}

        return {"text": str(raw).strip() if raw is not None else ""}

    def _group_texts(self, examples):
        """Concatenate and group texts into blocks."""
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // self.block_size) * self.block_size
        result = {
            k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def load_and_process(self):
        """Main method to download, process, and split the dataset."""
        log.info(f"Loading and processing dataset: {self.dataset_name}")
        raw_datasets = load_dataset(self.dataset_name, self.dataset_config, trust_remote_code=True)

        if isinstance(raw_datasets, Dataset):
            raw_datasets = DatasetDict({"train": raw_datasets})
        if "train" not in raw_datasets:
            first_key = next(iter(raw_datasets.keys()))
            log.warning(f"No 'train' split found. Using '{first_key}' as the training split.")
            raw_datasets["train"] = raw_datasets.pop(first_key)

        num_proc = os.cpu_count()
        log.info(f"Using {num_proc} cores for data processing.")

        formatted_datasets = raw_datasets.map(self._format_text, batched=False, num_proc=num_proc)
        formatted_datasets = formatted_datasets.filter(lambda x: x.get("text") and len(x["text"]) > 10, num_proc=num_proc)
        
        tokenized_datasets = formatted_datasets.map(
            lambda e: self.tokenizer(e["text"]),
            batched=True,
            remove_columns=formatted_datasets["train"].column_names,
            num_proc=num_proc
        )
        
        lm_datasets = tokenized_datasets.map(self._group_texts, batched=True, num_proc=num_proc)
        full_dataset = lm_datasets["train"]

        if self.train_subset_ratio and 0.0 < self.train_subset_ratio < 1.0:
            num_samples = int(len(full_dataset) * self.train_subset_ratio)
            full_dataset = full_dataset.select(range(num_samples))
            log.info(f"Subsetting '{self.dataset_name}' to {num_samples} samples.")

        val_size = int(len(full_dataset) * (self.validation_split_percentage / 100))
        train_size = len(full_dataset) - val_size
        
        if train_size == 0 or val_size == 0 and len(full_dataset) > 1:
            log.warning("Train or validation split is zero. Adjusting to ensure both are non-empty.")
            val_size = max(1, val_size)
            train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        log.info(f"Finished processing '{self.dataset_name}': {len(train_dataset)} train, {len(val_dataset)} val samples.")
        
        return train_dataset, val_dataset
