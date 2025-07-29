# src/core/data/mixed_datamodule.py
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, DataCollatorForLanguageModeling
from omegaconf import DictConfig
from typing import List, Optional
import hydra
import logging

log = logging.getLogger(__name__)

class MixedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_configs: List[DictConfig],
        tokenizer_name: str,
        block_size: int,
        batch_size: int,
        validation_split_percentage: int = 5,
        **unused_kwargs,
    ):
        """
        A DataModule that mixes multiple HuggingFace datasets.

        Args:
            dataset_configs: A list of Hydra configs, each for a HuggingFaceDataModule.
            tokenizer_name: The name of the tokenizer to use for all datasets.
            block_size: The block size for tokenization.
            batch_size: The final, overall batch size for the mixed data.
            validation_split_percentage: The percentage of data to use for validation.
        """
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

    def prepare_data(self) -> None:
        """
        Downloads all necessary datasets by instantiating their DataModules.
        """
        log.info(f"Preparing {len(self.hparams.dataset_configs)} datasets for mixing...")
        for config in self.hparams.dataset_configs:
            log.info(f"--- Preparing dataset: {config.dataset_name} ---")
            # We must provide all required __init__ arguments for instantiation,
            # even if prepare_data() itself doesn't use all of them.
            hydra.utils.instantiate(
                config,
                # Pass down the required parameters from the parent config
                tokenizer_name=self.hparams.tokenizer_name,
                block_size=self.hparams.block_size,
                batch_size=self.hparams.batch_size,
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Processes and concatenates all datasets.
        """
        log.info("Setting up mixed dataset...")
        all_train_datasets = []
        all_val_datasets = []

        for config in self.hparams.dataset_configs:
            log.info(f"--- Processing dataset: {config.dataset_name} ---")
            # We instantiate the single DataModule to handle all its specific logic
            # (text formatting, tokenizing, etc.)
            single_datamodule = hydra.utils.instantiate(
                config,
                tokenizer_name=self.hparams.tokenizer_name,
                block_size=self.hparams.block_size,
                # These are placeholders; the final batching is handled by this MixedDataModule
                batch_size=self.hparams.batch_size,
                validation_split_percentage=self.hparams.validation_split_percentage,
            )
            single_datamodule.setup(stage)

            if len(single_datamodule.train_dataset) > 0:
                all_train_datasets.append(single_datamodule.train_dataset)
                log.info(f"  + Added {len(single_datamodule.train_dataset):,} training samples.")
            if len(single_datamodule.val_dataset) > 0:
                all_val_datasets.append(single_datamodule.val_dataset)
                log.info(f"  + Added {len(single_datamodule.val_dataset):,} validation samples.")

        # Concatenate all processed datasets
        self.train_dataset = ConcatDataset(all_train_datasets)
        self.val_dataset = ConcatDataset(all_val_datasets)
        self.test_dataset = self.val_dataset # Use the combined validation set for testing

        log.info("-" * 50)
        log.info(f"Total mixed training samples: {len(self.train_dataset):,}")
        log.info(f"Total mixed validation samples: {len(self.val_dataset):,}")
        log.info("-" * 50)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size, # Use the overall batch size
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.data_collator,
        )