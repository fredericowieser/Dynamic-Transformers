import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    DataCollatorForLanguageModeling,
)
import logging

log = logging.getLogger(__name__)


class HuggingFaceDataModule(pl.LightningDataModule):
    """
    For handling datasets from the Hugging Face Hub.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str,
        tokenizer_name: str,
        block_size: int,
        batch_size: int,
    ):
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
            f"Downloading dataset '{self.hparams.dataset_name}' ({self.hparams.dataset_config})..."
        )
        load_dataset(
            self.hparams.dataset_name, self.hparams.dataset_config, trust_remote_code=True
        )
        log.info("Dataset download complete.")

    def setup(self, stage: str | None = None) -> None:
        log.info(f"Setting up data for stage: {stage}")
        raw_datasets = load_dataset(
            self.hparams.dataset_name, self.hparams.dataset_config, trust_remote_code=True
        )
        for split in raw_datasets:
            raw_datasets[split] = raw_datasets[split].filter(lambda x: len(x["text"]) > 0)
        def tokenize_function(examples):
            return self.tokenizer(examples["text"])
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )
        def group_texts(examples):
            concatenated_examples = {
                k: sum(examples[k], []) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (
                total_length // self.hparams.block_size
            ) * self.hparams.block_size
            result = {
                k: [
                    t[i : i + self.hparams.block_size]
                    for i in range(0, total_length, self.hparams.block_size)
                ]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        lm_datasets = tokenized_datasets.map(group_texts, batched=True)
        if stage == "fit" or stage is None:
            self.train_dataset = lm_datasets["train"]
            self.val_dataset = lm_datasets["validation"]
            log.info(f"Train dataset size: {len(self.train_dataset)}")
            log.info(f"Validation dataset size: {len(self.val_dataset)}")
        if stage == "test" or stage is None:
            self.test_dataset = lm_datasets["test"]
            log.info(f"Test dataset size: {len(self.test_dataset)}")

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