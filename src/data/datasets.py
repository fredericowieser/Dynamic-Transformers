import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import logging

log = logging.getLogger(__name__)


class HuggingFaceDataset(Dataset):
    """Dataset wrapper for HuggingFace datasets."""

    def __init__(self, config, tokenizer, split="train"):
        self.config = config
        self.tokenizer = tokenizer
        self.block_size = config.data.block_size

        # Load dataset
        dataset_name = config.data.dataset_name
        dataset_config = getattr(config.data, 'dataset_config', None)

        log.info(f"Loading dataset: {dataset_name}")

        if dataset_config:
            self.dataset = load_dataset(dataset_name, dataset_config, split=split)
        else:
            self.dataset = load_dataset(dataset_name, split=split)

        # Tokenize dataset
        self.tokenized = self._tokenize_dataset()

    def _tokenize_dataset(self):
        """Tokenize and chunk the dataset."""
        def tokenize_function(examples):
            # Handle different text field names
            text_field = "text"
            if "text" not in examples:
                # Try common alternatives
                for field in ["content", "document", "passage"]:
                    if field in examples:
                        text_field = field
                        break

            texts = examples[text_field]

            # Filter out empty strings
            texts = [t for t in texts if t and len(t.strip()) > 0]

            if not texts:
                # Return empty tensors if no valid text
                return {
                    "input_ids": [],
                    "attention_mask": []
                }

            return self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.block_size,
                return_special_tokens_mask=True
            )

        # Get number of workers
        num_proc = getattr(self.config.data, 'num_workers', 1)

        tokenized = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.dataset.column_names,
            num_proc=num_proc if num_proc > 1 else None
        )

        # Filter out empty samples
        tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)

        tokenized.set_format("torch")
        return tokenized

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, idx):
        item = self.tokenized[idx]
        item["labels"] = item["input_ids"].clone()
        return item


class MixedDataset(Dataset):
    """Dataset that mixes multiple data sources."""

    def __init__(self, config, tokenizer, split="train"):
        self.datasets = []
        self.weights = getattr(config.data, 'dataset_weights', None)

        log.info(f"Creating mixed dataset with {len(config.data.dataset_names)} sources")

        for i, dataset_name in enumerate(config.data.dataset_names):
            log.info(f"Loading dataset {i+1}/{len(config.data.dataset_names)}: {dataset_name}")

            # Create a temporary config for each dataset
            temp_config = type('Config', (), {})()
            temp_config.data = type('DataConfig', (), {})()
            temp_config.data.dataset_name = dataset_name
            temp_config.data.block_size = config.data.block_size
            temp_config.data.num_workers = getattr(config.data, 'num_workers', 1)

            try:
                dataset = HuggingFaceDataset(temp_config, tokenizer, split)
                if len(dataset) > 0:
                    self.datasets.append(dataset)
                else:
                    log.warning(f"Dataset {dataset_name} is empty, skipping")
            except Exception as e:
                log.warning(f"Failed to load {dataset_name}: {e}")

        if not self.datasets:
            raise ValueError("No datasets could be loaded")

        # Compute dataset sizes and sampling probabilities
        self.dataset_sizes = [len(d) for d in self.datasets]
        self.total_size = sum(self.dataset_sizes)

        if self.weights and len(self.weights) == len(self.datasets):
            self.probs = torch.tensor(self.weights[:len(self.datasets)], dtype=torch.float32)
            self.probs /= self.probs.sum()
        else:
            # Uniform sampling
            self.probs = torch.ones(len(self.datasets)) / len(self.datasets)

        log.info(f"Mixed dataset created: {len(self.datasets)} sources, {self.total_size} total samples")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Sample dataset based on weights
        dataset_idx = torch.multinomial(self.probs, 1).item()
        dataset = self.datasets[dataset_idx]

        # Sample random item from selected dataset
        item_idx = torch.randint(0, len(dataset), (1,)).item()
        return dataset[item_idx]


def get_dataloader(config, tokenizer, split="train"):
    """Get dataloader based on configuration."""

    # Check if mixed dataset
    is_mixed = getattr(config.data, 'mixed', False)

    if is_mixed:
        dataset = MixedDataset(config, tokenizer, split)
    else:
        dataset = HuggingFaceDataset(config, tokenizer, split)

    log.info(f"Created {split} dataset with {len(dataset)} samples")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=(split == "train"),
        collate_fn=collator,
        num_workers=getattr(config.data, 'num_workers', 0),
        pin_memory=False,  # Disable for Mac compatibility
        drop_last=True
    )