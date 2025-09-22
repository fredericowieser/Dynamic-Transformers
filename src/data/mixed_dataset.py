import logging
from typing import List

import hydra
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Dataset handlers
from .huggingface_dataset import HuggingFaceDataset
from .pretraining_dataset import PretrainingDataset

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
        batch_size: int,  # Hydra compatibility
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
        Loads and concatenates all specified datasets using a handler mapping.
        """
        log.info("Setting up mixed dataset...")
        
        handler_map = {
            "sft": HuggingFaceDataset,
            "pretrain": PretrainingDataset
        }
        
        all_train_datasets, all_val_datasets = [], []

        for cfg in self.dataset_configs:
            dataset_type = cfg.get("type", "sft")
            handler_class = handler_map.get(dataset_type)

            if not handler_class:
                raise ValueError(f"Unknown dataset type '{dataset_type}' in config. "
                                 f"Available types are: {list(handler_map.keys())}")

            log.info(f"Processing dataset '{cfg.dataset_name}' with handler: '{dataset_type}'")

            handler = handler_class(
                tokenizer=self.tokenizer,
                dataset_name=cfg.dataset_name,
                text_column=cfg.text_column,
                block_size=self.block_size,
                dataset_config=cfg.get("dataset_config"),
                validation_split_percentage=self.validation_split_percentage,
                train_subset_ratio=cfg.get("train_subset_ratio"),
            )
            
            train_data, val_data = handler.load_and_process()

            if train_data: all_train_datasets.append(train_data)
            if val_data: all_val_datasets.append(val_data)

        self.train_dataset = ConcatDataset(all_train_datasets) if all_train_datasets else []
        self.val_dataset = ConcatDataset(all_val_datasets) if all_val_datasets else []

        log.info(f"Total mixed training samples: {len(self.train_dataset):,}")
        log.info(f"Total mixed validation samples: {len(self.val_dataset):,}")
