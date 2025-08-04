import collections.abc as cab
import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
)

log = logging.getLogger(__name__)


def _get_val(cfg, key, default="<unknown>"):
    """
    Helper that works for both DictConfig and plain dict produced when
    _convert_='partial'.
    """
    if isinstance(cfg, cab.Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class MixedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_configs: list[DictConfig],
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
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

    def prepare_data(self) -> None:
        log.info(
            f"Preparing {len(self.hparams.dataset_configs)} datasets for mixing..."
        )
        for cfg in self.hparams.dataset_configs:
            # cfg is DictConfig when _convert_ is "none", but a plain dict when
            # _convert_ is "partial".  Handle both.
            if isinstance(cfg, cab.Mapping):
                ds_name = cfg.get("dataset_name", cfg.get("path", "<unknown>"))
            else:  # still a DictConfig or Structured config
                ds_name = getattr(cfg, "dataset_name", "<unknown>")

            log.info(f"Preparing dataset: {ds_name}")

            hydra.utils.instantiate(
                cfg,
                tokenizer_name=self.hparams.tokenizer_name,
                block_size=self.hparams.block_size,
                batch_size=self.hparams.batch_size,
            )

    def setup(self, stage: str | None = None) -> None:
        """
        Processes and concatenates all datasets.
        """
        log.info("Setting up mixed dataset...")
        all_train_datasets, all_val_datasets = [], []

        for cfg in self.hparams.dataset_configs:
            ds_name = _get_val(cfg, "dataset_name", _get_val(cfg, "path"))
            log.info(f"Processing dataset: {ds_name}")

            single_dm = hydra.utils.instantiate(
                cfg,
                tokenizer_name=self.hparams.tokenizer_name,
                block_size=self.hparams.block_size,
                batch_size=self.hparams.batch_size,
                validation_split_percentage=self.hparams.validation_split_percentage,
            )
            single_dm.setup(stage)

            if len(single_dm.train_dataset):
                all_train_datasets.append(single_dm.train_dataset)
                log.info(
                    f"  + Added {len(single_dm.train_dataset):,} training samples."
                )
            if len(single_dm.val_dataset):
                all_val_datasets.append(single_dm.val_dataset)
                log.info(
                    f"  + Added {len(single_dm.val_dataset):,} validation samples."
                )

        self.train_dataset = ConcatDataset(all_train_datasets)
        self.val_dataset = ConcatDataset(all_val_datasets)
        self.test_dataset = self.val_dataset

        log.info(f"Total mixed training samples: {len(self.train_dataset):,}")
        log.info(f"Total mixed validation samples: {len(self.val_dataset):,}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
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
