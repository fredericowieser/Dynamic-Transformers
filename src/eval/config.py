# src.eval/config.py
"""Configuration management for evaluation."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

@dataclass
class EvalConfig:
    """Dataclass for evaluation configuration."""
    # Benchmark settings from LLaMA paper
    benchmarks: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "MMLU": {"shots": 5, "metric": "macro_avg/acc"},
        "Open-rewrite eval": {"shots": 0, "metric": "micro_avg/rougeL"},
        "TLDR9+": {"shots": 1, "metric": "rougeL"},
        "IFEval": {"shots": 0, "metric": "Avg(Prompt/Instruction acc Loose/Strict)"},
        "GSM8K": {"shots": 8, "metric": "em_maj1@1"},
        "MATH": {"shots": 0, "metric": "final_em"},
        "ARC-C": {"shots": 0, "metric": "acc"},
        "GPQA": {"shots": 0, "metric": "acc"},
        "HellaSwag": {"shots": 0, "metric": "acc"},
        "BFCL V2": {"shots": 0, "metric": "acc"},
        "Nexus": {"shots": 0, "metric": "macro_avg/acc"},
        "InfiniteBench/En.QA": {"shots": 0, "metric": "longbook_qa/f1"},
        "InfiniteBench/En.MC": {"shots": 0, "metric": "longbook_choice/acc"},
        "NIH/Multi-needle": {"shots": 0, "metric": "recall"},
        "MGSM": {"shots": 0, "metric": "em"},
    })

    # Model parameters with defaults from training
    ce_bias: float = 0.0  # Default; override from trained model
    dynamic_k: float = 0.5  # Default; override from trained model
    model_path: str = "path/to/trained/model"  # Path to trained model
    output_dir: str = "./eval_results"  # Directory for outputs
    device: str = "cuda"  # Or "cpu"

    def load_from_model(self, model_path: str):
        """Load CE bias and dynamic K from trained model config."""
        try:
            # Assuming model config is saved via Hydra or similar
            cfg = hydra.utils.load_config_from_file(os.path.join(model_path, "config.yaml"))
            self.ce_bias = cfg.get("model", {}).get("model_cfg", {}).get("ce_bias", self.ce_bias)
            self.dynamic_k = cfg.get("model", {}).get("model_cfg", {}).get("dynamic_k", self.dynamic_k)
        except Exception as e:
            print(f"Warning: Could not load from model config. Using defaults. Error: {e}")

    def override_from_args(self, args: Dict[str, Any]):
        """Override configs from command-line args or input dict."""
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)