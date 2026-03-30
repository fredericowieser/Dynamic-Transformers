import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer

# Make sure all custom models and the create_model util are available
from src.models import (
    MoDForCausalLM,
    SDTForCausalLM,
    StandardTransformerForCausalLM,
    STTForCausalLM,
)
from src.training.utils import create_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


class PerformanceMetrics:
    """A context manager to measure wall-clock time and peak VRAM usage."""

    def __init__(self, device: str):
        self.device = device
        self.start_time = 0.0
        self.end_time = 0.0
        self.peak_vram_mb = 0.0

    def __enter__(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device == "cuda":
            torch.cuda.synchronize()
            self.peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        self.end_time = time.perf_counter()

    @property
    def duration_ms(self) -> float:
        """Returns the duration of the context in milliseconds."""
        return (self.end_time - self.start_time) * 1000


def run_benchmark(
    model: torch.nn.Module,
    device: str,
    sequence_lengths: List[int],
    batch_size: int,
    num_runs: int,
    num_warmup_runs: int,
) -> Dict:
    """Runs performance benchmarks for a given model across multiple sequence lengths."""
    model.eval()
    model.to(device)

    results = {}

    log.info(
        f"Device: {device}, Batch Size: {batch_size}, Runs per length: {num_runs} (Warmup: {num_warmup_runs})"
    )

    for seq_len in sequence_lengths:
        log.info(f"Benchmarking sequence length: {seq_len}")

        try:
            # Generate dummy input
            input_ids = torch.randint(
                0, model.config.vocab_size, (batch_size, seq_len), device=device
            )

            total_duration_ms = 0.0
            peak_vram_mb = 0.0

            # Warmup runs
            for _ in range(num_warmup_runs):
                with torch.no_grad():
                    _ = model(input_ids)

            # Actual benchmark runs
            for i in range(num_runs):
                metrics = PerformanceMetrics(device)
                with metrics:
                    with torch.no_grad():
                        _ = model(input_ids)

                total_duration_ms += metrics.duration_ms
                if i == num_runs - 1:
                    peak_vram_mb = metrics.peak_vram_mb

            avg_duration_ms = total_duration_ms / num_runs

            results[seq_len] = {
                "avg_duration_ms": avg_duration_ms,
                "peak_vram_mb": peak_vram_mb,
                "throughput_tokens_per_sec": (batch_size * seq_len)
                / (avg_duration_ms / 1000)
                if avg_duration_ms > 0
                else 0,
            }
            log.info(f"  Avg Duration: {results[seq_len]['avg_duration_ms']:.2f} ms")
            log.info(f"  Throughput: {results[seq_len]['throughput_tokens_per_sec']:.2f} tokens/sec")

        except torch.cuda.OutOfMemoryError:
            log.error(f"  Out of Memory at sequence length {seq_len}. Skipping.")
            results[seq_len] = {
                "avg_duration_ms": float("inf"),
                "peak_vram_mb": float("inf"),
                "throughput_tokens_per_sec": 0,
            }
            continue
        except Exception as e:
            log.error(f"  An error occurred at sequence length {seq_len}: {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run performance benchmarks on randomly initialized dynamic transformer models."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="0.5B",
        help="Model size to benchmark (e.g., '10M', '0.5B', '1.5B').",
    )
    parser.add_argument(
        "--sequence_lengths",
        type=str,
        default="1024,2048,4096,8192,16384,32768",
        help="Comma-separated list of sequence lengths to benchmark.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for benchmarking."
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of benchmark runs for each setting.",
    )
    parser.add_argument(
        "--num_warmup_runs",
        type=int,
        default=2,
        help="Number of warmup runs before benchmarking.",
    )

    args = parser.parse_args()

    seq_lengths = sorted([int(s) for s in args.sequence_lengths.split(",")])

    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.bfloat16
    else:
        log.warning("CUDA not available, running on CPU.")
        device = "cpu"
        torch_dtype = torch.float32

    # Load base config
    cfg = OmegaConf.load("config/default.yaml")
    cfg.model.from_scratch = True
    cfg.model.size = args.model_size
    cfg.system.use_flash_attention = device == "cuda"
    cfg.model.attn_implementation = "flash_attention_2" if device == "cuda" else "eager"
    
    # Define benchmarks to run
    benchmarks = [
        {"name": "Dense Baseline", "type": "standard", "causal": False},
        {"name": "MoD Baseline", "type": "mod", "causal": False},
        {"name": "SDT (Causal)", "type": "sdt", "causal": True},
        {"name": "STT (Causal)", "type": "stt", "causal": True},
    ]

    all_results = {}

    for bench in benchmarks:
        log.info(f"\n--- Benchmarking {bench['name']} ({args.model_size}) ---")
        cfg.model.type = bench["type"]
        cfg.model.use_causal_router_in_validation = bench["causal"]

        model = create_model(bench["type"], cfg)
        model = model.to(dtype=torch_dtype)

        results = run_benchmark(
            model=model,
            device=device,
            sequence_lengths=seq_lengths,
            batch_size=args.batch_size,
            num_runs=args.num_runs,
            num_warmup_runs=args.num_warmup_runs,
        )
        all_results[bench["name"]] = results

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- Print Markdown Table ---
    print("\n\n### Performance Benchmark Results\n")
    
    # Header row
    header = "| Model Variant | "
    for sl in seq_lengths:
        header += f"{sl} ctx Latency (ms) \u2193 | {sl} ctx Throughput \u2191 | "
    print(header)
    
    # Separator row
    separator = "| :--- | "
    for _ in seq_lengths:
        separator += ":--- | :--- | "
    print(separator)
    
    # Data rows
    for name in [b["name"] for b in benchmarks]:
        row = f"| {name} | "
        results = all_results.get(name, {})
        for sl in seq_lengths:
            if sl in results:
                m = results[sl]
                lat = f"{m['avg_duration_ms']:.1f}" if m['avg_duration_ms'] != float("inf") else "OOM"
                thr = f"{m['throughput_tokens_per_sec']:.1f}" if m['throughput_tokens_per_sec'] > 0 else "0.0"
                row += f"{lat} | {thr} | "
            else:
                row += "N/A | N/A | "
        print(row)
    print("\n")


if __name__ == "__main__":
    main()
