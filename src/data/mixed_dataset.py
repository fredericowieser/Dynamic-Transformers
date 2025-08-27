import logging
from typing import List

import hydra
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# --- START OF MODIFICATION ---
# Import both dataset handlers
from .huggingface_dataset import HuggingFaceDataset
from .pretraining_dataset import PretrainingDataset
# --- END OF MODIFICATION ---

log = logging.getLogger(__name__)

class MixedDataset:
    """
    A class to load, process, and combine multiple Hugging Face datasets.
    This class orchestrates multiple dataset handlers based on the specified
    dataset type (e.g., 'sft' for instruction tuning, 'pretrain' for continued
    pre-training).
    """
    def __init__(
        self,
        dataset_configs: List[DictConfig],
        tokenizer_name: str,
        block_size: int,
        batch_size: int, # Kept for Hydra instantiation compatibility, but not used here
        validation_split_percentage: int = 5,
        **kwargs,
    ):
        self.dataset_configs = dataset_configs
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.block_size = block_size
        self.validation_split_percentage = validation_split_percentage
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None) -> None:
        """
        Loads, processes, and concatenates all specified datasets using the
        appropriate handler for each.
        """
        log.info("Setting up mixed dataset...")
        all_train_datasets, all_val_datasets = [], []

        for cfg in self.dataset_configs:
            # --- START OF MODIFICATION ---
            # Dynamically select the dataset handler based on the config
            dataset_type = cfg.get("type", "sft") # Default to 'sft' for backward compatibility
            log.info(f"Processing dataset '{cfg['dataset_name']}' with handler: '{dataset_type}'")

            if dataset_type == "sft":
                handler_class = HuggingFaceDataset
            elif dataset_type == "pretrain":
                handler_class = PretrainingDataset
            else:
                raise ValueError(f"Unknown dataset type '{dataset_type}' in config for '{cfg['dataset_name']}'.")

            # The hydra instantiation is now more generic
            single_dataset_handler = handler_class(
                tokenizer=self.tokenizer,
                dataset_name=cfg["dataset_name"],
                text_column=cfg["text_column"],
                block_size=self.block_size,
                dataset_config=cfg.get("dataset_config"),
                validation_split_percentage=self.validation_split_percentage,
                train_subset_ratio=cfg.get("train_subset_ratio"),
            )
            # --- END OF MODIFICATION ---
            
            train_data, val_data = single_dataset_handler.load_and_process()

            if len(train_data) > 0:
                all_train_datasets.append(train_data)
            if len(val_data) > 0:
                all_val_datasets.append(val_data)

        self.train_dataset = ConcatDataset(all_train_datasets) if all_train_datasets else []
        self.val_dataset = ConcatDataset(all_val_datasets) if all_val_datasets else []
        self.test_dataset = self.val_dataset # Use val set for testing

        log.info(f"Total mixed training samples: {len(self.train_dataset):,}")
        log.info(f"Total mixed validation samples: {len(self.val_dataset):,}")
