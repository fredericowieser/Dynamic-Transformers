import os
import time
import requests
import pyarrow.parquet as pq
import torch
import logging
from multiprocessing import Pool
from pathlib import Path
from torch.utils.data import IterableDataset

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# FineWeb-Edu 100BT Dataset Config
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 
index_to_filename = lambda index: f"shard_{index:05d}.parquet"

def download_single_file(index, data_dir):
    """ Downloads a single file index, with some backoff """
    os.makedirs(data_dir, exist_ok=True)
    filename = index_to_filename(index)
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return True

    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            return True
        except (requests.RequestException, IOError):
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try: os.remove(path)
                    except: pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False

def list_parquet_files(data_dir):
    """ Looks into a data dir and returns full paths to all parquet files. """
    if not os.path.exists(data_dir):
        return []
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

class FineWebDataloader(IterableDataset):
    def __init__(
        self, tokenizer, B, T, split, data_dir, num_shards=183,
        tokenizer_batch_size=128,
        device="cuda", resume_state_dict=None,
        buffer_size=1000, num_workers_download=8,
        rank=0, world_size=1
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.B = B
        self.T = T
        self.split = split
        self.data_dir = Path(data_dir)
        self.num_shards = num_shards
        self.device = device
        self.buffer_size = buffer_size
        self.tokenizer_batch_size = tokenizer_batch_size
        self.resume_state_dict = resume_state_dict
        
        # DDP settings
        self.world_size = world_size
        self.rank = rank

        # Ensure data is downloaded
        if self.rank == 0:
            log.info(f"Ensuring {num_shards} shards are downloaded to {data_dir}...")
            ids_to_download = list(range(num_shards))
            with Pool(processes=num_workers_download) as pool:
                results = pool.starmap(download_single_file, [(i, data_dir) for i in ids_to_download])
            successful = sum(1 for success in results if success)
            if successful < num_shards:
                log.warning(f"Only {successful}/{num_shards} shards downloaded successfully.")
        
        if self.world_size > 1:
            torch.distributed.barrier()

        self.bos_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        self.row_capacity = T + 1

    def _document_batches(self):
        parquet_paths = list_parquet_files(self.data_dir)
        if self.num_shards > 0:
            parquet_paths = parquet_paths[:self.num_shards]
        
        assert len(parquet_paths) != 0, f"No dataset parquet files found in {self.data_dir}"
        
        if self.split == "train":
            parquet_paths = parquet_paths[:-1] if len(parquet_paths) > 1 else parquet_paths
        else:
            parquet_paths = parquet_paths[-1:]

        resume_pq_idx = self.resume_state_dict["pq_idx"] if self.resume_state_dict is not None else 0
        resume_rg_idx = self.resume_state_dict["rg_idx"] if self.resume_state_dict is not None else None
        resume_epoch = self.resume_state_dict.get("epoch", 1) if self.resume_state_dict is not None else 1
        first_pass = True
        epoch = resume_epoch

        while True:
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(parquet_paths):
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                    base_idx = resume_rg_idx // self.world_size
                    base_idx += 1
                    rg_idx = base_idx * self.world_size + self.rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None
                else:
                    rg_idx = self.rank
                
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist()
                    for i in range(0, len(batch), self.tokenizer_batch_size):
                        yield batch[i:i+self.tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                    rg_idx += self.world_size
                pq_idx += 1
            first_pass = False
            epoch += 1

    def __iter__(self):
        batches = self._document_batches()
        doc_buffer = []
        
        # Internal state for progress tracking
        self.pq_idx, self.rg_idx, self.epoch = 0, 0, 1

        def refill_buffer():
            nonlocal doc_buffer
            doc_batch, (self.pq_idx, self.rg_idx, self.epoch) = next(batches)
            tokenized = self.tokenizer(doc_batch, add_special_tokens=False)["input_ids"]
            for tokens in tokenized:
                doc_buffer.append([self.bos_token] + tokens)

        row_buffer = torch.empty((self.B, self.row_capacity), dtype=torch.long)

        while True:
            for row_idx in range(self.B):
                pos = 0
                while pos < self.row_capacity:
                    while len(doc_buffer) < self.buffer_size:
                        refill_buffer()

                    remaining = self.row_capacity - pos
                    best_idx = -1
                    best_len = 0
                    for i, doc in enumerate(doc_buffer):
                        doc_len = len(doc)
                        if doc_len <= remaining and doc_len > best_len:
                            best_idx = i
                            best_len = doc_len

                    if best_idx >= 0:
                        doc = doc_buffer.pop(best_idx)
                        doc_len = len(doc)
                        row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                        pos += doc_len
                    else:
                        shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                        doc = doc_buffer.pop(shortest_idx)
                        row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                        pos += remaining

            x = row_buffer[:, :-1].clone()
            y = row_buffer[:, 1:].clone()
            yield x, y

    def state_dict(self):
        return {"pq_idx": self.pq_idx, "rg_idx": self.rg_idx, "epoch": self.epoch}
