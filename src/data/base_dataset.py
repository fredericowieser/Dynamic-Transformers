import logging
import os
from abc import ABC, abstractmethod
from typing import Dict

from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import random_split
from transformers import PreTrainedTokenizerBase

log = logging.getLogger(__name__)

class BaseDatasetHandler(ABC):
    """
    An abstract base class for handling Hugging Face datasets.

    It encapsulates the shared logic for loading, processing, and splitting,
    while delegating the specific text formatting to subclasses.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_name: str,
        text_column: str,
        block_size: int,
        dataset_config: str = None,
        validation_split_percentage: int = 5,
        train_subset_ratio: float = None,
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.block_size = block_size
        self.dataset_config = dataset_config
        self.validation_split_percentage = validation_split_percentage
        self.train_subset_ratio = train_subset_ratio

    @abstractmethod
    def _process_text_column(self, examples: Dict) -> Dict[str, str]:
        """
        Processes the raw text column into a standardized 'text' field.
        Subclasses must implement this to handle their specific data format
        (e.g., plain text vs. chat/instruction format).
        """
        pass

    def _group_texts(self, examples: Dict) -> Dict:
        """Concatenates and groups texts into fixed-size blocks."""
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
        """The main pipeline for loading and preparing the dataset."""
        log.info(f"Loading and processing dataset: {self.dataset_name}")
        raw_datasets = load_dataset(self.dataset_name, self.dataset_config, trust_remote_code=True)

        if isinstance(raw_datasets, Dataset):
            raw_datasets = DatasetDict({"train": raw_datasets})
        if "train" not in raw_datasets:
            first_key = next(iter(raw_datasets.keys()))
            log.warning(f"No 'train' split found. Using '{first_key}' as the training split.")
            raw_datasets["train"] = raw_datasets.pop(first_key)

        num_proc = os.cpu_count() or 1

        # 1. Format the text using the subclass-specific implementation
        formatted_datasets = raw_datasets.map(self._process_text_column, batched=False, num_proc=num_proc)
        
        # 2. Filter out short/empty examples
        filtered_datasets = formatted_datasets.filter(lambda x: x.get("text") and len(x["text"]) > 10, num_proc=num_proc)
        
        # 3. Tokenize
        tokenized_datasets = filtered_datasets.map(
            lambda e: self.tokenizer(e["text"]),
            batched=True,
            remove_columns=filtered_datasets["train"].column_names,
            num_proc=num_proc
        )
        
        # 4. Group into blocks
        lm_datasets = tokenized_datasets.map(self._group_texts, batched=True, num_proc=num_proc)
        full_dataset = lm_datasets["train"]

        # 5. Subset the training data if requested
        if self.train_subset_ratio and 0.0 < self.train_subset_ratio < 1.0:
            num_samples = int(len(full_dataset) * self.train_subset_ratio)
            full_dataset = full_dataset.select(range(num_samples))
            log.info(f"Subsetting '{self.dataset_name}' to {num_samples} samples.")

        # 6. Split into training and validation sets
        if self.validation_split_percentage > 0 and len(full_dataset) > 1:
            val_size = int(len(full_dataset) * (self.validation_split_percentage / 100))
            val_size = max(1, val_size) # Ensure at least one validation sample
            train_size = len(full_dataset) - val_size
            if train_size <= 0:
                raise ValueError("Dataset is too small to create a non-empty training split.")
            
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        else:
            train_dataset = full_dataset
            val_dataset = []

        log.info(f"Finished processing '{self.dataset_name}': {len(train_dataset):,} train, {len(val_dataset):,} val samples.")
        return train_dataset, val_dataset
