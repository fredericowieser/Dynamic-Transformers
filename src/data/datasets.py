import logging
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

log = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""

    def __init__(self, config, tokenizer, split="train"):
        """Initialize dataset.

        Args:
            config: Data configuration
            tokenizer: Tokenizer to use
            split: Dataset split (train/validation)
        """
        # Load dataset
        dataset_name = config.data.get('dataset', 'wikitext')
        dataset_config = config.data.get('dataset_config', 'wikitext-2-raw-v1')

        log.info(f"Loading {dataset_name} dataset...")
        raw_dataset = load_dataset(dataset_name, dataset_config)

        # Handle splits
        if split == "validation" and "validation" not in raw_dataset:
            split = "test"

        self.dataset = raw_dataset[split]

        # Tokenize
        self.tokenizer = tokenizer
        self.block_size = config.data.get('block_size', 512)

        # Process texts
        self._process_dataset()

    def _process_dataset(self):
        """Process and tokenize dataset."""
        # Filter empty texts
        texts = [text for text in self.dataset['text'] if text.strip()]

        # Tokenize all texts
        log.info(f"Tokenizing {len(texts)} texts...")
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.block_size,
            return_overflowing_tokens=True,
            return_length=True,
        )

        # Filter by length
        self.input_ids = []
        for ids, length in zip(tokenized['input_ids'], tokenized['length']):
            if length > 10:  # Skip very short sequences
                self.input_ids.append(torch.tensor(ids))

        log.info(f"Created {len(self.input_ids)} samples")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx]}


def get_dataloader(config, tokenizer, split="train"):
    """Create dataloader for training or evaluation.

    Args:
        config: Configuration object
        tokenizer: Tokenizer
        split: Dataset split

    Returns:
        DataLoader instance
    """
    dataset = TextDataset(config, tokenizer, split)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8
    )

    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=(split == "train"),
        collate_fn=collator,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False,
        drop_last=True
    )