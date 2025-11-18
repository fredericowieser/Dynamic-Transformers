import logging
import os
from collections import deque
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from .download import download_dataset

log = logging.getLogger(__name__)


class StreamingDataset(IterableDataset):
    """
    An iterable dataset that streams and tokenizes data from a directory of Parquet files.
    Inspired by nanochat's dataloader, this is designed for large-scale pretraining
    where pre-tokenizing the entire dataset is not feasible.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use.
        data_dir (str): Local directory where the dataset shards are stored.
        remote_name (str): Hugging Face repository name of the dataset (e.g., "karpathy/fineweb-edu-100b-shuffle").
        num_shards (int): The number of shards to download if not present locally.
        batch_size (int): The batch size for yielding token batches.
        block_size (int): The sequence length of the model.
        split (str): "train" or "val". The last shard is reserved for validation.
        num_workers_download (int): Number of parallel workers for downloading.
        seed (int): Random seed for shuffling shards.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_dir: str,
        remote_name: str,
        num_shards: int,
        batch_size: int,
        block_size: int,
        split: str = "train",
        num_workers_download: int = 4,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.data_dir = Path(data_dir)
        self.remote_name = remote_name
        self.num_shards = num_shards
        self.batch_size = batch_size
        self.block_size = block_size
        self.split = split
        self.num_workers_download = num_workers_download
        self.seed = seed

        # DDP settings
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))

        # Download data if it doesn't exist
        self._download_if_needed()

    def _download_if_needed(self):
        if not self.data_dir.exists() or not any(self.data_dir.iterdir()):
            if self.rank == 0:
                log.info(f"Dataset not found at {self.data_dir}. Downloading...")
                download_dataset(
                    data_dir=str(self.data_dir),
                    remote_name=self.remote_name,
                    num_shards=self.num_shards,
                    num_workers=self.num_workers_download,
                )
            # All ranks wait until the download is complete
            if self.world_size > 1:
                torch.distributed.barrier()
        else:
            log.info(f"Found dataset at {self.data_dir}. Skipping download.")

    def __iter__(self):
        # Get a list of all shard paths
        all_shards = sorted([p for p in self.data_dir.glob("*.parquet")])
        
        # Shuffle shards deterministically
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        shuffled_indices = torch.randperm(len(all_shards), generator=rng).tolist()
        all_shards = [all_shards[i] for i in shuffled_indices]

        # Split shards for train/val
        if len(all_shards) == 1:
            shards = all_shards
        elif self.split == "train":
            shards = all_shards[:-1]
        else:
            shards = all_shards[-1:]
        
        if not shards:
            log.warning(f"No shards found for split '{self.split}' in {self.data_dir}.")
            return

        # Each rank gets a subset of shards
        shards_for_rank = shards[self.rank :: self.world_size]
        if not shards_for_rank:
            log.warning(f"Rank {self.rank} has no shards to process for split '{self.split}'.")
            return

        token_buffer = deque()
        needed_tokens = self.batch_size * self.block_size + 1

        # Infinite loop over the data for this rank
        while True:
            for shard_path in shards_for_rank:
                pf = pq.ParquetFile(shard_path)
                for rg_idx in range(pf.num_row_groups):
                    rg = pf.read_row_group(rg_idx)
                    texts = rg.column("text").to_pylist()

                    # Tokenize texts and fill the buffer
                    for text in texts:
                        if text:
                            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                            token_buffer.extend(token_ids)

                    # Yield batches when enough tokens are available
                    while len(token_buffer) >= needed_tokens:
                        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
                        x = torch.tensor(tokens[:-1], dtype=torch.long).view(self.batch_size, self.block_size)
                        y = torch.tensor(tokens[1:], dtype=torch.long).view(self.batch_size, self.block_size)
                        yield x, y
