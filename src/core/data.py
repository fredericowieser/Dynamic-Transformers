import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, DataCollatorForLanguageModeling
import logging
import json

log = logging.getLogger(__name__)

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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

    def prepare_data(self) -> None:
        log.info(f"Downloading dataset '{self.hparams.dataset_name}'...")
        load_dataset(self.hparams.dataset_name, self.hparams.dataset_config, trust_remote_code=True)
        log.info("Dataset download complete.")

    def _format_text(self, examples):
        # Handles different structures like SlimOrca's list of dicts
        raw_text = examples[self.hparams.text_column]
        if isinstance(raw_text, list) and all(isinstance(i, dict) for i in raw_text):
            # Apply chat template for conversational data
            try:
                return {"text": self.tokenizer.apply_chat_template(raw_text, tokenize=False)}
            except Exception as e:
                log.warning(f"Could not apply chat template: {e}. Falling back to JSON dump.")
                return {"text": json.dumps(raw_text)}
        # For simple text columns
        return {"text": raw_text}

    def setup(self, stage: str | None = None) -> None:
        log.info(f"Setting up data for stage: {stage}")
        raw_datasets = load_dataset(
            self.hparams.dataset_name, self.hparams.dataset_config, trust_remote_code=True
        )
        
        # Format and tokenize
        formatted_datasets = raw_datasets.map(self._format_text, batched=False)
        tokenized_datasets = formatted_datasets.map(
            lambda e: self.tokenizer(e["text"]),
            batched=True,

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

        # Create train/val splits
        val_size = int(len(full_dataset) * (self.hparams.validation_split_percentage / 100))
        train_size = len(full_dataset) - val_size
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