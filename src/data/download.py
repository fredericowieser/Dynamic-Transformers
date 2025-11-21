import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path

import requests

log = logging.getLogger(__name__)

def download_shard(args):
    """Downloads a single shard of a dataset."""
    shard_index, data_dir, base_url = args
    filename = f"shard_{shard_index:05d}.parquet"
    filepath = data_dir / filename
    
    if filepath.exists():
        log.debug(f"Skipping {filepath} (already exists)")
        return True

    url = f"{base_url}/{filename}"
    log.info(f"Downloading {filename} to {data_dir}...")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            temp_path = filepath.with_suffix(".parquet.tmp")
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            
            os.rename(temp_path, filepath)
            log.info(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            log.warning(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                log.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                log.error(f"Failed to download {filename} after {max_attempts} attempts")
                return False
    return False

def download_dataset(data_dir: str, remote_name: str, num_shards: int, num_workers: int = 4):
    """
    Downloads a dataset composed of multiple shards from the Hugging Face Hub.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://huggingface.co/datasets/{remote_name}/resolve/main"
    
    shard_indices = list(range(num_shards))
    
    log.info(f"Downloading {len(shard_indices)} shards using {num_workers} workers...")
    log.info(f"Target directory: {data_dir}")

    pool_args = [(i, data_dir, base_url) for i in shard_indices]

    with Pool(processes=num_workers) as pool:
        results = pool.map(download_shard, pool_args)

    successful_downloads = sum(1 for success in results if success)
    log.info(f"Download complete. Successfully downloaded {successful_downloads}/{len(shard_indices)} shards to {data_dir}")

    if successful_downloads != len(shard_indices):
        log.warning("Some shards failed to download. Please check the logs.")

