import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict

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
    sequence_lengths: list[int],
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
                # VRAM is measured as the peak for the final run
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
            log.info(f"  Peak VRAM: {results[seq_len]['peak_vram_mb']:.2f} MB")
            log.info(
                f"  Throughput: {results[seq_len]['throughput_tokens_per_sec']:.2f} tokens/sec"
            )

        except torch.cuda.OutOfMemoryError:
            log.error(f"  Out of Memory at sequence length {seq_len}. Skipping.")
            results[seq_len] = {
                "avg_duration_ms": float("inf"),
                "peak_vram_mb": float("inf"),
                "throughput_tokens_per_sec": 0,
            }
            continue  # Skip to next sequence length
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
        help="Model size to benchmark (e.g., '10M', '0.5B', '1.5B'). Must match a key in scratch_config.",
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
    parser.add_argument(
        "--use_causal_router",
        action="store_true",
        help="If set, uses the causal router during inference instead of non-causal Top-K.",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        default="standard,mod,sdt,stt",
        help="Comma-separated list of model types to benchmark.",
    )

    args = parser.parse_args()

    seq_lengths = sorted([int(s) for s in args.sequence_lengths.split(",")])

    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.bfloat16
    else:
        log.warning(
            "CUDA not available, running on CPU. VRAM and wall-clock time benchmarks will not be meaningful."
        )
        device = "cpu"
        torch_dtype = torch.float32

    # Load base config and override for benchmarking
    try:
        cfg = OmegaConf.load("config/default.yaml")
    except FileNotFoundError:
        log.error("Could not find 'config/default.yaml'. Please run this script from the project root.")
        return
        
    cfg.model.from_scratch = True
    cfg.model.size = args.model_size
    cfg.system.use_flash_attention = device == "cuda"
    cfg.model.attn_implementation = "flash_attention_2" if device == "cuda" else "eager"
    cfg.model.use_causal_router_in_validation = args.use_causal_router
    
    all_results = {}
    model_types_to_benchmark = [mt.strip().lower() for mt in args.model_types.split(",")]

    for model_type in model_types_to_benchmark:
        log.info(f"\n--- Initializing and benchmarking {model_type.upper()} model ({args.model_size}) ---")
        cfg.model.type = model_type

        # Create a randomly initialized model
        model = create_model(model_type, cfg)
        model = model.to(dtype=torch_dtype)

        results = run_benchmark(
            model=model,
            device=device,
            sequence_lengths=seq_lengths,
            batch_size=args.batch_size,
            num_runs=args.num_runs,
            num_warmup_runs=args.num_warmup_runs,
        )
        all_results[model_type] = results

        # Free memory before loading the next model
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- Reporting ---
    if not all_results:
        log.error("No results were generated.")
        return

    dense_results = all_results.get("standard")

    print("\n" + "=" * 97)
    print(f"  Overall Performance Benchmark Summary (Model Size: {args.model_size})")
    print("=" * 97)
    header = f"| {'Seq Len':<10} | {'Model':<10} | {'Avg Duration (ms)':<20} | {'Speedup vs Dense':<20} | {'Peak VRAM (MB)':<18} |"
    print(header)
    print(f"|{'-' * 12}|{'-' * 12}|{'-' * 22}|{'-' * 22}|{'-' * 20}|")

    for seq_len in seq_lengths:
        for model_type, results in sorted(all_results.items()):
            if seq_len in results:
                metrics = results[seq_len]
                speedup_str = "N/A"
                if dense_results and seq_len in dense_results and model_type != "standard":
                    dense_duration = dense_results[seq_len]["avg_duration_ms"]
                    if metrics["avg_duration_ms"] > 0 and dense_duration < float("inf"):
                        speedup = dense_duration / metrics["avg_duration_ms"]
                        speedup_str = f"{speedup:.2f}x"
                elif model_type == "standard":
                    speedup_str = "1.00x (Baseline)"

                duration_str = f"{metrics['avg_duration_ms']:.2f}" if metrics["avg_duration_ms"] != float("inf") else "OOM"
                vram_str = f"{metrics['peak_vram_mb']:.2f}" if metrics['peak_vram_mb'] != float("inf") else "OOM"

                print(
                    f"| {seq_len:<10} | {model_type.upper():<10} | {duration_str:<20} | "
                    f"{speedup_str:<20} | {vram_str:<18} |"
                )
        if len(all_results) > 1 and seq_len != seq_lengths[-1]:
            print(f"|{'-' * 12}|{'-' * 12}|{'-' * 22}|{'-' * 22}|{'-' * 20}|")

    print("=" * 97)
    print("\nNote: FLOPs/MACs are not measured directly due to the complexity of dynamic operations and custom kernels.")
    print("      Speedup is calculated as (Dense Model Duration / Dynamic Model Duration).")
    print("\nTo run this script:")
    print(
        f"  python {os.path.basename(__file__)} --model_size {args.model_size} --sequence_lengths {args.sequence_lengths}"
    )
    print("")


if __name__ == "__main__":
    main()
