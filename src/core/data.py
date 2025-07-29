import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase, DataCollatorForLanguageModeling
import logging
import json
import re
from typing import Optional, Any, Dict, List
import torch

log = logging.getLogger(__name__)

def _dict_list_to_chat(
    tokenizer, conv: List[Dict[str, Any]]
) -> Dict[str, str]:
    """Takes [{'role': 'user', 'content': '…'}, …] -> {'text': '…'}"""
    # Map role aliases
    norm = []
    for turn in conv:
        role = (turn.get("role") or turn.get("from") or "").lower()
        if role in {"human", "user"}:
            role = "user"
        elif role in {"assistant", "gpt", "model"}:
            role = "assistant"
        # Other roles (system, tool, critic, …) are kept as is
        norm.append({"role": role, "content": turn.get("content") or turn.get("value") or ""})

    # Drop empty turns
    norm = [t for t in norm if t["content"].strip()]
    if not norm:
        return None

    try:
        return {"text": tokenizer.apply_chat_template(norm, tokenize=False)}
    except Exception:
        # Fall back to a simple "Role: content" format if the tokenizer
        # has no chat template.
        joined = "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in norm)
        return {"text": joined}

class HuggingFaceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        block_size: int,
        batch_size: int,
        text_column: str,
        dataset_config: str = None,
        validation_split_percentage: int = 5,
        train_subset_ratio: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

    def prepare_data(self) -> None:
        log.info(f"Downloading dataset '{self.hparams.dataset_name}'...")
        load_dataset(
            self.hparams.dataset_name,
            self.hparams.dataset_config,
            trust_remote_code=True,
            # download_mode="force_redownload",
        )
        log.info("Dataset download complete.")


    def _format_text(self, examples):
        """
        Extremely tolerant normaliser:
        - list-of-dict chats               -> chat template
        - (query, response) / (prompt, completion) columns
        - strings with '### Human:' / 'Human:' / 'User:' markers
        - raw strings                         (fallback)
        """

        # -------------------------------------------------- #
        # 0️⃣  Pick the source field
        # -------------------------------------------------- #
        preferred = self.hparams.text_column
        raw = examples.get(preferred)

        # Common alternatives
        if raw is None:
            for alt in (
                "messages",
                "conversation",
                "conversations",
                "prompt_response",
                "text",
                "chosen",        # UltraFeedback
                "chosen_response",
                "response",
                "completion",
            ):
                if alt in examples:
                    raw = examples[alt]
                    break

        # (query / response) or (prompt / completion) dual-column pattern
        if raw is None:
            q = examples.get("query") or examples.get("prompt") or examples.get("instruction")
            a = examples.get("response") or examples.get("answer") or examples.get("completion")
            if q is not None and a is not None:
                return _dict_list_to_chat(
                    self.tokenizer,
                    [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                )

        # -------------------------------------------------- #
        # 1️⃣  list-of-dict chat already?
        # -------------------------------------------------- #
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            out = _dict_list_to_chat(self.tokenizer, raw)
            if out:
                return out

        # -------------------------------------------------- #
        # 2️⃣  Maybe the entry is a JSON string?
        # -------------------------------------------------- #
        if isinstance(raw, str) and raw.strip().startswith(("{", "[")):
            try:
                obj = json.loads(raw)
                if isinstance(obj, list):
                    out = _dict_list_to_chat(self.tokenizer, obj)
                    if out:
                        return out
            except Exception:
                pass  # fall through

        # -------------------------------------------------- #
        # 3️⃣  '### Human:' / 'Human:' style multi-turn string
        # -------------------------------------------------- #
        if isinstance(raw, str):
            s = raw.strip()

            # 3a. '###' markers
            if "###" in s:
                blocks = re.split(r"###\s*", s)
            else:
                # 3b. plain 'Human:/Assistant:' lines
                blocks = re.split(r"\n(?=\s*(Human|Assistant|User):)", s)

            conv = []
            for blk in blocks:
                m = re.match(r"\s*(Human|Assistant|User)\s*:\s*(.*)", blk, flags=re.S)
                if m:
                    role = "user" if m.group(1) in {"Human", "User"} else "assistant"
                    conv.append({"role": role, "content": m.group(2).strip()})

            if conv:
                out = _dict_list_to_chat(self.tokenizer, conv)
                if out:
                    return out

        # -------------------------------------------------- #
        # 4️⃣  Fallback: keep whatever we have as plain text
        # -------------------------------------------------- #
        if raw is None:
            raise KeyError(
                f"Could not locate usable text in example keys {list(examples.keys())}"
            )

        return {"text": str(raw).strip()}

    def setup(self, stage: str | None = None) -> None:
        """
        Loads and processes the dataset, creating train/val splits.
        """
        log.info(f"Setting up data for stage: {stage}")

        raw_datasets = load_dataset(
            self.hparams.dataset_name,
            self.hparams.dataset_config,
            trust_remote_code=True,
        )

        # ------------------------------------------------------------
        # 1️⃣  Guarantee we have a DatasetDict that contains a "train" split
        # ------------------------------------------------------------
        if isinstance(raw_datasets, Dataset):
            # single unnamed split → make it the train split
            raw_datasets = DatasetDict({"train": raw_datasets})

        if "train" not in raw_datasets:
            # take the very first available split as train
            first_key = next(iter(raw_datasets.keys()))
            log.warning(
                f"No 'train' split found in {self.hparams.dataset_name!r}. "
                f"Using '{first_key}' as the training split."
            )
            raw_datasets = DatasetDict({"train": raw_datasets[first_key]})

        # ------------------------------------------------------------
        # 2️⃣  Continue exactly as before
        # ------------------------------------------------------------
        formatted_datasets = raw_datasets.map(self._format_text, batched=False)

        # remove empty texts …
        formatted_datasets["train"] = formatted_datasets["train"].filter(
            lambda x: x.get("text") is not None and len(x["text"]) > 0
        )

        tokenized_datasets = formatted_datasets.map(
            lambda e: self.tokenizer(e["text"]),
            batched=True,
            remove_columns=formatted_datasets["train"].column_names,
        )

        # Group into blocks        
        # Filter out empty or problematic text entries after formatting
        for split in formatted_datasets:
            # Check if 'text' column exists and filter out empty strings
            if 'text' in formatted_datasets[split].column_names:
                formatted_datasets[split] = formatted_datasets[split].filter(lambda x: x["text"] is not None and len(x["text"]) > 0)
            else:
                log.warning(f"Split '{split}' does not have a 'text' column after formatting. This might indicate a problem.")

        tokenized_datasets = formatted_datasets.map(
            lambda e: self.tokenizer(e["text"]),
            batched=True,
            # Remove original columns, keep only 'input_ids', 'attention_mask' etc.
            remove_columns=formatted_datasets["train"].column_names,
        )

        # Group into blocks
        def group_texts(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated[list(examples.keys())[0]])
            total_length = (total_length // self.hparams.block_size) * self.hparams.block_size
            result = {
                k: [t[i : i + self.hparams.block_size] for i in range(0, total_length, self.hparams.block_size)]
                for k, t in concatenated.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        lm_datasets = tokenized_datasets.map(group_texts, batched=True)

        full_dataset = lm_datasets["train"]

        # Apply training data subsetting if specified
        if self.hparams.train_subset_ratio is not None and 0.0 < self.hparams.train_subset_ratio < 1.0:
            num_samples_to_use = int(len(full_dataset) * self.hparams.train_subset_ratio)
            if num_samples_to_use == 0 and len(full_dataset) > 0:
                num_samples_to_use = 1 # Ensure at least one sample if dataset is not empty
            
            # Randomly select indices for the subset
            # Using torch.randperm for reproducibility (if seed_everything is set)
            indices = torch.randperm(len(full_dataset))[:num_samples_to_use].tolist()
            full_dataset = full_dataset.select(indices)
            log.info(f"Reduced training dataset to {num_samples_to_use} samples ({self.hparams.train_subset_ratio*100:.2f}% of original).")

        # Create train/val splits
        val_size = int(len(full_dataset) * (self.hparams.validation_split_percentage / 100))
        train_size = len(full_dataset) - val_size
        
        # Ensure that validation_split_percentage doesn't result in empty split
        if train_size == 0 or val_size == 0:
            log.warning(f"Training or validation split size is zero. Train size: {train_size}, Val size: {val_size}. "
                        "Consider adjusting validation_split_percentage or batch_size if this is unexpected.")
            # If one is zero, make them equal if possible, or raise error.
            # For simplicity here, we'll allow it and let downstream parts handle potential empty dataloaders.
            # A common approach for small datasets is to use all data for train and test/val on a subset
            # of train or another dedicated test set if available.

        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        
        log.info(f"Train dataset size: {len(self.train_dataset)}")
        log.info(f"Validation dataset size: {len(self.val_dataset)}")

        # Use validation set for testing if no test set is available
        self.test_dataset = self.val_dataset
        log.info(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, collate_fn=self.data_collator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size,
            num_workers=4, pin_memory=True, collate_fn=self.data_collator,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.hparams.batch_size,
            num_workers=4, pin_memory=True, collate_fn=self.data_collator,
        )