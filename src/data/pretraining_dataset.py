import logging
import os
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import random_split
from transformers import PreTrainedTokenizerBase

log = logging.getLogger(__name__)


class PretrainingDataset:
    """
    A class to load and process a single Hugging Face dataset specifically for
    continued pre-training. It handles raw text without applying any chat
    or instruction formatting.
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

    def _prepare_text(self, examples: dict[str, Any]) -> dict[str, str]:
        """
        Extracts and cleans the text from the specified column.
        Unlike the SFT version, this does NOT apply chat templates.
        """
        raw_text = examples.get(self.text_column)
        return {"text": str(raw_text).strip() if raw_text is not None else ""}

    def _group_texts(self, examples: dict[str, list]) -> dict[str, list]:
        """Concatenate and group texts into blocks for next-token prediction."""
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // self.block_size) * self.block_size
        result = {
            k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def load_and_process(self) -> tuple[Dataset, Dataset]:
        """Main method to download, process, and split the dataset."""
        log.info(f"Loading pre-training dataset: {self.dataset_name}")
        raw_datasets = load_dataset(self.dataset_name, self.dataset_config, trust_remote_code=True)

        if isinstance(raw_datasets, Dataset):
            raw_datasets = DatasetDict({"train": raw_datasets})
        if "train" not in raw_datasets:
            first_key = next(iter(raw_datasets.keys()))
            log.warning(f"No 'train' split found. Using '{first_key}' as the training split.")
            raw_datasets["train"] = raw_datasets.pop(first_key)

        num_proc = os.cpu_count() or 1
        log.info(f"Using {num_proc} cores for data processing.")

        # Step 1: Prepare the raw text
        prepared_datasets = raw_datasets.map(
            self._prepare_text, batched=False, num_proc=num_proc
        )
        # Filter out empty or very short texts
        prepared_datasets = prepared_datasets.filter(
            lambda x: x.get("text") and len(x["text"]) > 10, num_proc=num_proc
        )

        # Step 2: Tokenize the text
        tokenized_datasets = prepared_datasets.map(
            lambda e: self.tokenizer(e["text"]),
            batched=True,
            remove_columns=prepared_datasets["train"].column_names,
            num_proc=num_proc,
        )

        # Step 3: Group into blocks
        lm_datasets = tokenized_datasets.map(
            self._group_texts, batched=True, num_proc=num_proc
        )
        full_dataset = lm_datasets["train"]

        if self.train_subset_ratio and 0.0 < self.train_subset_ratio < 1.0:
            num_samples = int(len(full_dataset) * self.train_subset_ratio)
            full_dataset = full_dataset.select(range(num_samples))
            log.info(f"Subsetting '{self.dataset_name}' to {num_samples} samples for pre-training.")

        if self.validation_split_percentage > 0:
            val_size = int(len(full_dataset) * (self.validation_split_percentage / 100))
            train_size = len(full_dataset) - val_size
            if train_size <= 0 or val_size <= 0 and len(full_dataset) > 1:
                val_size = max(1, val_size)
                train_size = len(full_dataset) - val_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        else:
            train_dataset = full_dataset
            val_dataset = [] # No validation set

        log.info(
            f"Finished processing '{self.dataset_name}': {len(train_dataset):,} train, {len(val_dataset):,} val samples."
        )

        return train_dataset, val_dataset
