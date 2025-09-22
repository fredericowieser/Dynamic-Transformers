# Directory: `conf`

## File: `conf/base-mod.yaml`

```yaml
defaults:
  - base

model:
  model_cfg:
    # Architecture configuration
    dynamic_architecture: "mod"  # MoD architecture
    capacity_gamma: 0.5

    # VPR parameters (disabled for MoD)
    prior_loss_schedule: null
    learn_beta_ce: False
    learn_beta_cu: False
    learn_cu_multiplier: False
    learn_ce_offset: False
    beta_ce_init: null
    beta_cu_init: null
    cu_detection_multiplier_init: null
    ce_criterion_offset_init: null
```

## File: `conf/base.yaml`

```yaml
defaults:
  - _self_
  - data: pretrain_mix

run:
  name: "LARGER-PRIOR-TEST-qwen2.5-0.5B-${model.model_cfg.dynamic_architecture}-${data.name}-${now:%Y-%m-%d_%H-%M-%S}-gamma=${model.model_cfg.capacity_gamma}"
  output_dir: "outputs/${run.name}"
  seed: 42
  device: "auto"
  precision: "bf16"
  run_final_evaluation: True

data:
  name: ${hydra:runtime.choices.data}
  batch_size: 16
  tokenizer_name: "Qwen/Qwen2.5-0.5B"
  block_size: 1024
  validation_split_percentage: 2

peft:
  enabled: False
  config:
    _target_: peft.LoraConfig
    r: 16
    lora_alpha: 32
    target_modules:
      - "q_proj"      # Query projection
      - "k_proj"      # Key projection
      - "v_proj"      # Value projection
      - "o_proj"      # Output projection
      - "gate_proj"   # Gate projection (SwiGLU)
      - "up_proj"     # Up projection (SwiGLU)
      - "down_proj"   # Down projection
    modules_to_save:
      - "vpr_router"
      - "prior_ffn"
    lora_dropout: 0.05
    bias: "none"
    task_type: "CAUSAL_LM"

model:
  _target_: src.models.qwen.causal_lm.DynamicQwenForCausalLM.from_pretrained
  pretrained_model_name_or_path: "Qwen/Qwen2.5-0.5B"
  use_flash_attention_2: true

  model_cfg:
    # Architecture configuration
    dynamic_architecture: "vpr"  # Options: "vpr" or "mod"
    capacity_gamma: 0.5          # VPR: routing capacity, MoD: token percentage

    # VPR-specific parameters
    prior_loss_schedule:
      initial_weight: 0.05  # Initial weight for training
      final_weight: 0.05    # Final stable weight
      decay_steps: 0        # Decay duration (steps)

    learn_beta_ce: True
    learn_beta_cu: True
    learn_cu_multiplier: True
    learn_ce_offset: True

    beta_ce_init: -0.3
    beta_cu_init: -0.6
    cu_detection_multiplier_init: 1.1
    ce_criterion_offset_init: 1.025

    token_wise_gating: True
    moving_average_window_size: 100
    prior_ffn_intermediate_size_factor: 0.5  # Fraction of main FFN size

    # General training parameters
    freeze_main_transformer_blocks: False

training:
  # Training duration control - specify either num_epochs or max_steps
  # If max_steps > 0, it takes precedence over num_epochs
  num_epochs: 1
  max_steps: -1

  accumulate_grad_batches: 64
  eval_interval: 100

  use_gradient_clipping: False
  gradient_clip_val: 1.0

  optimizer:
    base_model_lr: 1.0e-5       # Base model learning rate
    prior_lr: 1.0e-3
    vpr_router_lr: 1.0e-2
    weight_decay: 0.01
    warmup_ratio: 0.01

logging:
  wandb:
    enabled: true
    project: "Dynamic-Transformers"
    entity: "huawei-noahs-ark"

```

# Directory: `conf/data`

## File: `conf/data/lang_mix.yaml`

```yaml
_target_: src.data.mixed_dataset.MixedDataset
_recursive_: false   # Don't instantiate children automatically
_convert_:  partial  # Pass nested structures as DictConfig/ListConfig

dataset_configs:

  # Category 1: Core Knowledge & Coherence (~45%)
  # High-quality English text foundation

  - _target_: src.data.huggingface_dataset.HuggingFaceDataset
    dataset_name: "wikitext"
    dataset_config: "wikitext-103-raw-v1"  # Clean benchmark dataset
    text_column: "text"
    train_subset_ratio: 1.0  # Use full dataset

  - _target_: src.data.huggingface_dataset.HuggingFaceDataset
    dataset_name: "cnn_dailymail"
    dataset_config: "3.0.0"
    text_column: "article"  # Full articles for long-form text
    train_subset_ratio: 0.2  # 20% for high-quality news

  - _target_: src.data.huggingface_dataset.HuggingFaceDataset
    dataset_name: "roneneldan/TinyStories"  # Narrative and causal learning
    text_column: "text"
    train_subset_ratio: 1.0

  # Category 2: Code & Technical Acumen (~25%)
  # High-quality code for HumanEval and MBPP performance

  - _target_: src.data.huggingface_dataset.HuggingFaceDataset
    dataset_name: "codeparrot/codeparrot-clean-valid"  # Validated Python code
    text_column: "content"
    train_subset_ratio: 1.0

  # - _target_: src.data.huggingface_dataset.HuggingFaceDataset
  #   dataset_name: "bigcode/the-stack-smol"
  #   text_column: "content" # The text column for this dataset is 'content'.
  #   train_subset_ratio: 0.2

  # Category 3: Mathematical & Scientific Reasoning (~20%)
  # Targets MATH, GSM8K, and science benchmarks

  - _target_: src.data.huggingface_dataset.HuggingFaceDataset
    dataset_name: "gsm8k"
    dataset_config: "main"
    text_column: "question"  # Question text for pre-training
    train_subset_ratio: 1.0  # Full dataset (small size)

  - _target_: src.data.huggingface_dataset.HuggingFaceDataset
    dataset_name: "sciq"  # Science questions for reasoning
    text_column: "support"  # Supporting evidence text
    train_subset_ratio: 1.0

  # Category 4: Multilingual Capabilities (~10%)
  # Targeted multilingual understanding and translation

  # - _target_: src.data.huggingface_dataset.HuggingFaceDataset
  #   # REPLACED: Switched to the stable UN Parallel Corpus for multilingual data.
  #   dataset_name: "un_pc"
  #   dataset_config: "en-fr" # English-French parallel text.
  #   text_column: "translation" # Contains a dictionary with 'en' and 'fr' keys.
  #   train_subset_ratio: 0.2 # Take a 20% slice to keep it fast.

```

## File: `conf/data/pretrain_mix.yaml`

```yaml
_target_: src.data.mixed_dataset.MixedDataset
_recursive_: false
_convert_: partial

dataset_configs:

  # Category 1: Foundational General Knowledge (~40%)
  # Diverse high-quality text to prevent catastrophic forgetting
  # and maintain broad language skills

  - type: "pretrain"
    dataset_name: "wikitext"
    dataset_config: "wikitext-103-raw-v1"
    text_column: "text"
    train_subset_ratio: 1.0  # ~1.8M docs, core text

  - type: "pretrain"
    dataset_name: "cnn_dailymail"
    dataset_config: "3.0.0"
    text_column: "article"
    train_subset_ratio: 0.2  # ~57k docs, news articles

  - type: "pretrain"
    dataset_name: "storytracer/US-PD-Books"
    text_column: "text"
    train_subset_ratio: 0.5  # ~327k docs, classic literature

  # Category 2: Academic & Scientific Reasoning (~40%)
  # Targets MMLU and ARC-Challenge with technical text

  - type: "pretrain"
    dataset_name: "HuggingFaceTB/cosmopedia"
    dataset_config: "openstax"  # OpenStax configuration
    text_column: "text"
    train_subset_ratio: 0.1  # ~3M docs, synthetic textbooks

  - type: "pretrain"
    dataset_name: "sciq"
    text_column: "support"
    train_subset_ratio: 1.0  # ~13k docs, science passages

  # Category 3: Logical & Commonsense Reasoning (~20%)
  # Improves Hellaswag and general reasoning performance

  - type: "pretrain"
    dataset_name: "codeparrot/codeparrot-clean-valid"
    text_column: "content"
    train_subset_ratio: 1.0  # ~18k docs, code structure

  - type: "pretrain"
    dataset_name: "roneneldan/TinyStories"
    text_column: "text"
    train_subset_ratio: 1.0  # ~2.1M docs, basic causality

```

## File: `conf/data/sft_mix.yaml`

```yaml
_target_: src.data.mixed_dataset.MixedDataset
_recursive_: false   # <- DO NOT instantiate children automatically
_convert_:  partial  # <- Pass nested lists/dicts as DictConfig/ListConfig

dataset_configs:
  
  # High-Quality General Instructions (The Foundation)
  - _target_: src.data.huggingface_dataset.HuggingFaceDataset
    dataset_name: "HuggingFaceH4/ultrafeedback_binarized"
    text_column: "chosen" # Contains the highest-rated response.
    # This dataset is huge and diverse. 20% is a very large, high-quality sample.
    train_subset_ratio: 0.20

  # Mathematical and Logical Reasoning
  - _target_: src.data.huggingface_dataset.HuggingFaceDataset
    dataset_name: "meta-math/MetaMathQA"
    text_column: "messages" # Formatted as a chat.
    # MetaMath is smaller but crucial for reasoning. We use a larger portion.
    train_subset_ratio: 0.50

  # Coding and Programming Instructions
  - _target_: src.data.huggingface_dataset.HuggingFaceDataset
    dataset_name: "WizardLMTeam/WizardLM_evol_instruct_70k"
    text_column: "conversations" # High-quality, complex code instructions.
    # Use the entire dataset to maximize coding ability.
    train_subset_ratio: 1.0

  # High-Quality Human-Generated Dialogue
  - _target_: src.data.huggingface_dataset.HuggingFaceDataset
    dataset_name: "HuggingFaceH4/no_robots"
    text_column: "prompt_response" # 10k high-quality human demonstrations.
    # Use all of it to improve the model's natural style.
    train_subset_ratio: 1.0

  # Safety and Helpfulness (Alignment)
  - _target_: src.data.huggingface_dataset.HuggingFaceDataset
    dataset_name: "Anthropic/hh-rlhf"
    text_column: "chosen" # The "helpful and harmless" response.
    # Use all of it to teach the model safe boundaries.
    train_subset_ratio: 1.0
```

## File: `conf/data/test.yaml`

```yaml
# This file defines a simple, single-dataset configuration for testing.
# It uses only the 'openassistant-guanaco' dataset.
# This does not use the MixedDataModule, as it's only one dataset.
#
# To use this config, run:
# python main.py data=test

_target_: src.data.huggingface_datamodule.HuggingFaceDataModule  # Updated: New path to HuggingFaceDataModule

dataset_name: "timdettmers/openassistant-guanaco"
dataset_config: null
text_column: "text" # This dataset has a single 'text' column.
tokenizer_name: "meta-llama/Llama-3.2-1B-instruct"
block_size: 1024
batch_size: 4
validation_split_percentage: 5
train_subset_ratio: null # Use the full dataset.
```

# Directory: `src/data`

## File: `src/data/gate_logging.py`

```python
import logging
from collections import deque

import torch

log = logging.getLogger(__name__)
ROLLING_WINDOW_SIZE = 100


class GateLogger:
    """
    Handles logging and rolling statistics for dynamic gate activations.
    """

    def __init__(self, num_layers: int):
        self.per_layer_gate_activation_rolling_history = [
            {
                "mean": deque(maxlen=ROLLING_WINDOW_SIZE),
                "std": deque(maxlen=ROLLING_WINDOW_SIZE),
            }
            for _ in range(num_layers)
        ]

    def update_rolling_history(self, per_layer_gate_stats: list[dict]):
        """Updates the rolling history with new stats from a training step."""
        for i, stats in enumerate(per_layer_gate_stats):
            # Ensure stats are tensors before calling .item()
            mean_val = stats["mean"].item() if isinstance(stats["mean"], torch.Tensor) else stats["mean"]
            std_val = stats["std"].item() if isinstance(stats["std"], torch.Tensor) else stats["std"]
            
            hist = self.per_layer_gate_activation_rolling_history[i]
            hist["mean"].append(mean_val)
            hist["std"].append(std_val)

    def log_rolling_history(self, global_step: int, log_interval: int):
        """Logs the current rolling average statistics to the console."""
        if global_step > 0 and global_step % log_interval == 0:
            lines = [
                f"--- Per-Layer Gate Activations (Rolling Avg over last {ROLLING_WINDOW_SIZE} steps) ---"
            ]
            for i, history in enumerate(self.per_layer_gate_activation_rolling_history):
                if history["mean"]:
                    rolling_mean = sum(history["mean"]) / len(history["mean"])
                    # Calculate std dev from the list of stds for a sense of variance
                    rolling_std_of_means = torch.tensor(list(history["mean"])).std().item()

                    lines.append(
                        f"  Layer {i:02d}: Mean Activation = {rolling_mean:.3f} (Std of Means = {rolling_std_of_means:.3f})"
                    )
            log.info("\n".join(lines))


```

## File: `src/data/huggingface_dataset.py`

```python
import json
import logging
import re
from typing import Any
import os

import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import random_split
from transformers import PreTrainedTokenizerBase

log = logging.getLogger(__name__)

def _dict_list_to_chat(tokenizer, conv: list[dict[str, Any]]) -> dict[str, str]:
    """Convert conversation list to formatted text."""
    norm = []
    for turn in conv:
        role = (turn.get("role") or turn.get("from") or "").lower()
        if role in {"human", "user"}:
            role = "user"
        elif role in {"assistant", "gpt", "model"}:
            role = "assistant"
        norm.append({"role": role, "content": turn.get("content") or turn.get("value") or ""})

    norm = [t for t in norm if t["content"] and t["content"].strip()]
    if not norm:
        return None

    try:
        return {"text": tokenizer.apply_chat_template(norm, tokenize=False)}
    except Exception:
        joined = "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in norm)
        return {"text": joined}

class HuggingFaceDataset:
    """
    A class to load, process, and prepare a single Hugging Face dataset for language model training.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_name: str,
        text_column: str,
        block_size: int,
        dataset_config: str = None,
        validation_split_percentage: int = 5,
        train_subset_ratio: float | None = None,
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.block_size = block_size
        self.dataset_config = dataset_config
        self.validation_split_percentage = validation_split_percentage
        self.train_subset_ratio = train_subset_ratio

    def _format_text(self, examples):
        """Normalize various chat and instruction formats."""
        preferred = self.text_column
        raw = examples.get(preferred)

        if raw is None:
            for alt in ("messages", "conversation", "conversations", "prompt_response", "text", "chosen"):
                if alt in examples:
                    raw = examples[alt]
                    break
        
        if raw is None:
            q = examples.get("query") or examples.get("prompt")
            a = examples.get("response") or examples.get("answer")
            if q is not None and a is not None:
                return _dict_list_to_chat(self.tokenizer, [{"role": "user", "content": q}, {"role": "assistant", "content": a}])

        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            return _dict_list_to_chat(self.tokenizer, raw) or {"text": ""}

        if isinstance(raw, str):
            if raw.strip().startswith(("{", "[")):
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, list):
                        return _dict_list_to_chat(self.tokenizer, obj) or {"text": ""}
                except Exception:
                    pass
            
            blocks = re.split(r"###\s*|\n(?=\s*(Human|Assistant|User):)", raw.strip())
            conv = []
            for blk in blocks:
                if not isinstance(blk, str):
                    continue
                m = re.match(r"\s*(Human|Assistant|User)\s*:\s*(.*)", blk, flags=re.S)
                if m:
                    role = "user" if m.group(1) in {"Human", "User"} else "assistant"
                    content = m.group(2).strip()
                    if content:
                        conv.append({"role": role, "content": content})
            if conv:
                return _dict_list_to_chat(self.tokenizer, conv) or {"text": ""}

        return {"text": str(raw).strip() if raw is not None else ""}

    def _group_texts(self, examples):
        """Group texts into fixed-size blocks."""
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // self.block_size) * self.block_size
        result = {
            k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def load_and_process(self):
        """Download, process, and split the dataset."""
        log.info(f"Loading and processing dataset: {self.dataset_name}")
        raw_datasets = load_dataset(self.dataset_name, self.dataset_config, trust_remote_code=True)

        if isinstance(raw_datasets, Dataset):
            raw_datasets = DatasetDict({"train": raw_datasets})
        if "train" not in raw_datasets:
            first_key = next(iter(raw_datasets.keys()))
            log.warning(f"No 'train' split found. Using '{first_key}' as the training split.")
            raw_datasets["train"] = raw_datasets.pop(first_key)

        num_proc = os.cpu_count()
        log.info(f"Using {num_proc} cores for data processing.")

        formatted_datasets = raw_datasets.map(self._format_text, batched=False, num_proc=num_proc)
        formatted_datasets = formatted_datasets.filter(lambda x: x.get("text") and len(x["text"]) > 10, num_proc=num_proc)
        
        tokenized_datasets = formatted_datasets.map(
            lambda e: self.tokenizer(e["text"]),
            batched=True,
            remove_columns=formatted_datasets["train"].column_names,
            num_proc=num_proc
        )
        
        lm_datasets = tokenized_datasets.map(self._group_texts, batched=True, num_proc=num_proc)
        full_dataset = lm_datasets["train"]

        if self.train_subset_ratio and 0.0 < self.train_subset_ratio < 1.0:
            num_samples = int(len(full_dataset) * self.train_subset_ratio)
            full_dataset = full_dataset.select(range(num_samples))
            log.info(f"Subsetting '{self.dataset_name}' to {num_samples} samples.")

        val_size = int(len(full_dataset) * (self.validation_split_percentage / 100))
        train_size = len(full_dataset) - val_size
        
        if train_size == 0 or val_size == 0 and len(full_dataset) > 1:
            log.warning("Train or validation split is zero. Adjusting to ensure both are non-empty.")
            val_size = max(1, val_size)
            train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        log.info(f"Finished processing '{self.dataset_name}': {len(train_dataset)} train, {len(val_dataset)} val samples.")
        
        return train_dataset, val_dataset

```

## File: `src/data/mixed_dataset.py`

```python
import logging
from typing import List

import hydra
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Dataset handlers
from .huggingface_dataset import HuggingFaceDataset
from .pretraining_dataset import PretrainingDataset

log = logging.getLogger(__name__)

class MixedDataset:
    """
    A class to load, process, and combine multiple Hugging Face datasets.
    This class orchestrates multiple dataset handlers based on the specified
    dataset type (e.g., 'sft' for instruction tuning, 'pretrain' for continued
    pre-training).
    """
    def __init__(
        self,
        dataset_configs: List[DictConfig],
        tokenizer_name: str,
        block_size: int,
        batch_size: int,  # Hydra compatibility
        validation_split_percentage: int = 5,
        **kwargs,
    ):
        self.dataset_configs = dataset_configs
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.block_size = block_size
        self.validation_split_percentage = validation_split_percentage
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None) -> None:
        """
        Loads, processes, and concatenates all specified datasets using the
        appropriate handler for each.
        """
        log.info("Setting up mixed dataset...")
        all_train_datasets, all_val_datasets = [], []

        for cfg in self.dataset_configs:
            # Select handler by type
            dataset_type = cfg.get("type", "sft")
            log.info(f"Processing dataset '{cfg['dataset_name']}' with handler: '{dataset_type}'")

            if dataset_type == "sft":
                handler_class = HuggingFaceDataset
            elif dataset_type == "pretrain":
                handler_class = PretrainingDataset
            else:
                raise ValueError(f"Unknown dataset type '{dataset_type}' in config for '{cfg['dataset_name']}'.")

            # Create handler instance
            single_dataset_handler = handler_class(
                tokenizer=self.tokenizer,
                dataset_name=cfg["dataset_name"],
                text_column=cfg["text_column"],
                block_size=self.block_size,
                dataset_config=cfg.get("dataset_config"),
                validation_split_percentage=self.validation_split_percentage,
                train_subset_ratio=cfg.get("train_subset_ratio"),
            )
            
            train_data, val_data = single_dataset_handler.load_and_process()

            if len(train_data) > 0:
                all_train_datasets.append(train_data)
            if len(val_data) > 0:
                all_val_datasets.append(val_data)

        self.train_dataset = ConcatDataset(all_train_datasets) if all_train_datasets else []
        self.val_dataset = ConcatDataset(all_val_datasets) if all_val_datasets else []
        self.test_dataset = self.val_dataset

        log.info(f"Total mixed training samples: {len(self.train_dataset):,}")
        log.info(f"Total mixed validation samples: {len(self.val_dataset):,}")

```

## File: `src/data/pretraining_dataset.py`

```python
import logging
import os
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import random_split
from transformers import PreTrainedTokenizerBase

log = logging.getLogger(__name__)


class PretrainingDataset:
    """
    A class to load and process a single Hugging Face dataset specifically for
    continued pre-training. It handles raw text without applying any chat
    or instruction formatting.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_name: str,
        text_column: str,
        block_size: int,
        dataset_config: str = None,
        validation_split_percentage: int = 5,
        train_subset_ratio: float | None = None,
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.block_size = block_size
        self.dataset_config = dataset_config
        self.validation_split_percentage = validation_split_percentage
        self.train_subset_ratio = train_subset_ratio

    def _prepare_text(self, examples: dict[str, Any]) -> dict[str, str]:
        """
        Extracts and cleans the text from the specified column.
        Unlike the SFT version, this does NOT apply chat templates.
        """
        raw_text = examples.get(self.text_column)
        return {"text": str(raw_text).strip() if raw_text is not None else ""}

    def _group_texts(self, examples: dict[str, list]) -> dict[str, list]:
        """Concatenate and group texts into blocks for next-token prediction."""
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // self.block_size) * self.block_size
        result = {
            k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def load_and_process(self) -> tuple[Dataset, Dataset]:
        """Main method to download, process, and split the dataset."""
        log.info(f"Loading pre-training dataset: {self.dataset_name}")
        raw_datasets = load_dataset(self.dataset_name, self.dataset_config, trust_remote_code=True)

        if isinstance(raw_datasets, Dataset):
            raw_datasets = DatasetDict({"train": raw_datasets})
        if "train" not in raw_datasets:
            first_key = next(iter(raw_datasets.keys()))
            log.warning(f"No 'train' split found. Using '{first_key}' as the training split.")
            raw_datasets["train"] = raw_datasets.pop(first_key)

        num_proc = os.cpu_count() or 1
        log.info(f"Using {num_proc} cores for data processing.")

        # Step 1: Prepare the raw text
        prepared_datasets = raw_datasets.map(
            self._prepare_text, batched=False, num_proc=num_proc
        )
        # Filter out empty or very short texts
        prepared_datasets = prepared_datasets.filter(
            lambda x: x.get("text") and len(x["text"]) > 10, num_proc=num_proc
        )

        # Step 2: Tokenize the text
        tokenized_datasets = prepared_datasets.map(
            lambda e: self.tokenizer(e["text"]),
            batched=True,
            remove_columns=prepared_datasets["train"].column_names,
            num_proc=num_proc,
        )

        # Step 3: Group into blocks
        lm_datasets = tokenized_datasets.map(
            self._group_texts, batched=True, num_proc=num_proc
        )
        full_dataset = lm_datasets["train"]

        if self.train_subset_ratio and 0.0 < self.train_subset_ratio < 1.0:
            num_samples = int(len(full_dataset) * self.train_subset_ratio)
            full_dataset = full_dataset.select(range(num_samples))
            log.info(f"Subsetting '{self.dataset_name}' to {num_samples} samples for pre-training.")

        if self.validation_split_percentage > 0:
            val_size = int(len(full_dataset) * (self.validation_split_percentage / 100))
            train_size = len(full_dataset) - val_size
            if train_size <= 0 or val_size <= 0 and len(full_dataset) > 1:
                val_size = max(1, val_size)
                train_size = len(full_dataset) - val_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        else:
            train_dataset = full_dataset
            val_dataset = [] # No validation set

        log.info(
            f"Finished processing '{self.dataset_name}': {len(train_dataset):,} train, {len(val_dataset):,} val samples."
        )

        return train_dataset, val_dataset

```

# Directory: `src/models`

## File: `src/models/__init__.py`

```python


```

# Directory: `src/models/blocks`

## File: `src/models/blocks/mod_router.py`

```python
import torch
import torch.nn as nn

class MoDTokenRouter(nn.Module):
    """
    A simple router that assigns a scalar weight to each token based on a
    linear projection. Used for Mixture-of-Depths routing.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        # Linear projection to scalar weight per token
        self.gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): The input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            torch.Tensor: A tensor of router weights for each token. Shape: (batch, seq_len).
        """
        return self.gate(hidden_states).squeeze(-1)
```

## File: `src/models/blocks/prior_ffn.py`

```python
import torch
import torch.nn as nn
import math
import logging

log = logging.getLogger(__name__)

class PriorFeedForward(nn.Module):
    """
    A small feed-forward network for prior predictions in the VPR architecture.
    It takes an `intermediate_size_factor` to scale the hidden dimension
    for its internal intermediate size, allowing for more flexible bottlenecking.
    """

    def __init__(self, config, intermediate_size_factor: float = 2.0):
        super().__init__()
        hidden_size = config.hidden_size
        
        # Calculate the raw size
        raw_intermediate_size = hidden_size * intermediate_size_factor

        # Round up to the nearest integer
        rounded_up_size = math.ceil(raw_intermediate_size)
        
        # Ensure the size is an even number by adding 1 if it's odd
        even_size = rounded_up_size + (rounded_up_size % 2)

        # Enforce a minimum size of 2 to ensure it's a valid, non-zero even number
        intermediate_size = max(2, even_size)

        log.info(
            f"Initialized PriorFeedForward with intermediate_size={intermediate_size} "
            f"(factor={intermediate_size_factor}, raw_size={raw_intermediate_size:.2f})"
        )

        # two projection layers and one gating projection (SwiGLU-like)
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

        # activation and dropout
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout", 0.0))

        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights using a normal distribution and biases to zeros.
        """
        for p in self.parameters():
            if p.ndim > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PriorFeedForward network.
        """
        # SwiGLU-like gating mechanism
        gate = self.act(self.w1(x)) * self.w3(x)
        out = self.w2(gate)
        return self.dropout(out)
```

## File: `src/models/blocks/qwen_block.py`

```python
import torch
import torch.nn as nn
import logging
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)

log = logging.getLogger(__name__)

class Qwen2Block(nn.Module):
    """
    A standalone, reusable Qwen2 transformer block.
    This module encapsulates the standard self-attention and MLP layers,
    including residual connections and layer normalization.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = Qwen2MLP(config)
        
        if not hasattr(self.self_attn, "rotary_emb") or self.self_attn.rotary_emb is None:
            log.warning(
                f"Layer {layer_idx}: Qwen2Attention unexpectedly missing rotary_emb. "
                "Initializing it manually as a fallback."
            )
            # Initialize rotary embedding manually
            self.self_attn.rotary_emb = Qwen2RotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        
        residual = hidden_states
        hidden_states_norm = self.input_layernorm(hidden_states)

        kv_seq_len = hidden_states_norm.shape[1]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        # Generate position embeddings
        cos, sin = self.self_attn.rotary_emb(hidden_states_norm, position_ids=position_ids)
        position_embeddings = (cos, sin)
        
        attn_outputs = self.self_attn(
            hidden_states_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        attn_output = attn_outputs[0]
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states_norm = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states_norm)
        hidden_states = residual + mlp_output

        outputs = (hidden_states,) + attn_outputs[1:]
        return outputs
```

## File: `src/models/blocks/vpr_router.py`

```python
import logging

import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger(__name__)


class VPRRouter(nn.Module):
    """
    Implements the Variational Predictive Routing (VPR) logic to make
    per-token or per-batch routing decisions within a Transformer layer.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.token_wise_gating = getattr(config, "token_wise_gating", True)
        self.moving_average_window_size = getattr(
            config, "moving_average_window_size", 100
        )
        # Beta CE parameter
        if getattr(config, "learn_beta_ce", False):
            self.beta_ce = nn.Parameter(torch.tensor(config.beta_ce_init, dtype=torch.float32))
        else:
            self.register_buffer('beta_ce', torch.tensor(config.beta_ce_init, dtype=torch.float32))

        # Beta CU parameter
        if getattr(config, "learn_beta_cu", False):
            self.beta_cu = nn.Parameter(torch.tensor(config.beta_cu_init, dtype=torch.float32))
        else:
            self.register_buffer('beta_cu', torch.tensor(config.beta_cu_init, dtype=torch.float32))

        # CU detection multiplier
        if getattr(config, "learn_cu_multiplier", False):
            self.cu_detection_multiplier = nn.Parameter(torch.tensor(config.cu_detection_multiplier_init, dtype=torch.float32))
        else:
            self.register_buffer('cu_detection_multiplier', torch.tensor(config.cu_detection_multiplier_init, dtype=torch.float32))

        # CE criterion offset
        if getattr(config, "learn_ce_offset", False):
            self.ce_criterion_offset = nn.Parameter(torch.tensor(config.ce_criterion_offset_init, dtype=torch.float32))
        else:
            self.register_buffer('ce_criterion_offset', torch.tensor(config.ce_criterion_offset_init, dtype=torch.float32))
            
        log.info(f"VPRRouter Layer {self.layer_idx} Parameter Trainability:")
        log.info(f"  - learn_beta_ce: {getattr(config, 'learn_beta_ce', False)}")
        log.info(f"  - learn_beta_cu: {getattr(config, 'learn_beta_cu', False)}")
        log.info(f"  - learn_cu_multiplier: {getattr(config, 'learn_cu_multiplier', False)}")
        log.info(f"  - learn_ce_offset: {getattr(config, 'learn_ce_offset', False)}")
    
    @property
    def current_beta_ce(self):
        return self.beta_ce.item()

    @property
    def current_beta_cu(self):
        return self.beta_cu.item()

    @property
    def current_cu_detection_multiplier(self):
        return self.cu_detection_multiplier.item()

    @property
    def current_ce_criterion_offset(self):
        return self.ce_criterion_offset.item()

    def _calculate_moving_average(self, d_st_values: torch.Tensor) -> torch.Tensor:
        """
        Calculates the causal moving average for d_st values.
        """
        if self.moving_average_window_size <= 0:
            return d_st_values.mean(dim=-1, keepdim=True).expand_as(d_st_values)

        padded_d_st = F.pad(
            d_st_values, (self.moving_average_window_size - 1, 0), mode="replicate"
        )
        windows = padded_d_st.unfold(
            dimension=-1, size=self.moving_average_window_size, step=1
        )
        return windows.mean(dim=-1)

    def forward(
        self,
        original_input_to_block: torch.Tensor,
        posterior_full_path_output: torch.Tensor,
        prior_hidden_states: torch.Tensor,
        capacity_gamma: float,
        is_training: bool = True,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, float, float, float, float,
    ]:
        d_st_tok = F.mse_loss(
            posterior_full_path_output,
            original_input_to_block,
            reduction="none",
        ).mean(-1)
        d_ch_tok = F.mse_loss(
            posterior_full_path_output,
            prior_hidden_states,
            reduction="none",
        ).mean(-1)

        ce_criterion_offset_val = self.ce_criterion_offset

        if self.token_wise_gating:
            CE_val = d_st_tok - (d_ch_tok - torch.log(ce_criterion_offset_val + 1e-10))
            ma_d_st_tok = self._calculate_moving_average(d_st_tok.detach())
            CU_val = d_st_tok - (self.cu_detection_multiplier * ma_d_st_tok)
        else:
            mean_d_st = d_st_tok.mean(dim=-1, keepdim=True)
            mean_d_ch = d_ch_tok.mean(dim=-1, keepdim=True)
            CE_val = mean_d_st - (mean_d_ch - torch.log(ce_criterion_offset_val + 1e-10))
            CU_val = mean_d_st - (self.cu_detection_multiplier * mean_d_st.detach())

        S_CE = torch.sigmoid(F.softplus(self.beta_ce) * CE_val)
        S_CU = torch.sigmoid(F.softplus(self.beta_cu) * CU_val)

        combined_gating_signal_continuous = S_CE + S_CU - (S_CE * S_CU)

        def get_stats(tensor):
            return {
                "mean": tensor.mean(),
                "std": tensor.std(),
                "min": tensor.min(),
                "max": tensor.max(),
            }

        s_ce_stats = get_stats(S_CE)
        s_cu_stats = get_stats(S_CU)
        g_cont_stats = get_stats(combined_gating_signal_continuous)

        if capacity_gamma >= 1.0:
            threshold = -torch.finfo(combined_gating_signal_continuous.dtype).max
        else:
            if self.token_wise_gating:
                flat_g_signal = combined_gating_signal_continuous.flatten()
                threshold = torch.quantile(flat_g_signal.float(), (1.0 - capacity_gamma))
            else:
                threshold = torch.quantile(
                    combined_gating_signal_continuous.float, (1.0 - capacity_gamma), dim=0
                )

        gate_vec_binary = (combined_gating_signal_continuous >= threshold).float()

        avg_ce_proportion = S_CE.mean()
        avg_cu_proportion = S_CU.mean()

        return (
            gate_vec_binary,
            s_ce_stats,
            s_cu_stats,
            g_cont_stats,
            d_st_tok,
            d_ch_tok
            combined_gating_signal_continuous,
            self.current_beta_ce,
            self.current_beta_cu,
            self.current_cu_detection_multiplier,
            self.current_ce_criterion_offset,
        )
```

# Directory: `src/models/layers`

## File: `src/models/layers/decision_layer.py`

```python
import logging

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from ..blocks.prior_ffn import PriorFeedForward
from ..blocks.qwen_block import Qwen2Block
from ..qwen.modeling_outputs import DecisionLayerOutput

log = logging.getLogger(__name__)


class DecisionLayer(nn.Module):
    """
    Implements the 'Decision Sub-Layer' for the VPR architecture.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.block = Qwen2Block(config, layer_idx=layer_idx)

        prior_ffn_factor = getattr(config, "prior_ffn_intermediate_size_factor", 2.0)
        self.prior_ffn = PriorFeedForward(
            config, intermediate_size_factor=prior_ffn_factor
        )
        self.prior_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> DecisionLayerOutput:
        
        vpr_signal_original_input = hidden_states

        # Block handles rotary embedding creation internally
        block_outputs = self.block(hidden_states, **kwargs)
        posterior_full_path_output = block_outputs[0]
        present_key_value = block_outputs[1] if len(block_outputs) > 1 else None
        attn_weights = block_outputs[2] if len(block_outputs) > 2 else None

        # Prior network forward pass
        prior_input_ln = self.prior_layernorm(vpr_signal_original_input)
        prior_ffn_output = self.prior_ffn(prior_input_ln)
        vpr_signal_prior_hidden_states = vpr_signal_original_input + prior_ffn_output

        prior_loss = F.mse_loss(
            vpr_signal_prior_hidden_states, posterior_full_path_output.detach()
        )

        return DecisionLayerOutput(
            hidden_states=posterior_full_path_output,
            vpr_signal_original_input=vpr_signal_original_input,
            vpr_signal_posterior_output=posterior_full_path_output,
            vpr_signal_prior_hidden_states=vpr_signal_prior_hidden_states,
            present_key_value=present_key_value,
            attention_weights=attn_weights,
            prior_loss=prior_loss,
        )
```

## File: `src/models/layers/dynamic_layer.py`

```python
import torch
import torch.nn as nn
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from ..blocks.qwen_block import Qwen2Block
from ..blocks.vpr_router import VPRRouter
from ..qwen.modeling_outputs import DecisionLayerOutput, DynamicLayerOutput


class DynamicLayer(nn.Module):
    """
    Implements the 'Dynamic Sub-Layer' for the VPR architecture using a
    numerically stable, fully batched approach.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.block = Qwen2Block(config, layer_idx=layer_idx)
        self.vpr_router = VPRRouter(config, layer_idx)
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        decision_output: DecisionLayerOutput,
        attention_mask: torch.Tensor | None = None,  # 4D causal mask
        position_ids: torch.LongTensor | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> DynamicLayerOutput:
        
        # VPR routing (first arg positional for PEFT)
        (
            gate_vec_binary, s_ce_stats, s_cu_stats, g_cont_stats,
            _, _, combined_gating_signal, beta_ce, beta_cu,
            cu_multiplier, ce_offset,
        ) = self.vpr_router(
            decision_output.vpr_signal_original_input,  # Positional argument
            posterior_full_path_output=decision_output.vpr_signal_posterior_output,
            prior_hidden_states=decision_output.vpr_signal_prior_hidden_states,
            capacity_gamma=self.config.capacity_gamma,
            is_training=self.training,
        )
        
        # Gather selected tokens
        batch_indices, token_indices = gate_vec_binary.nonzero(as_tuple=True)

        if batch_indices.numel() == 0:
            return DynamicLayerOutput(
                hidden_states=hidden_states, present_key_value=None, attention_weights=None,
                s_ce_stats=s_ce_stats, s_cu_stats=s_cu_stats, g_cont_stats=g_cont_stats,
                combined_gating_signal=combined_gating_signal, gate_vector=gate_vec_binary,
                prior_loss=decision_output.prior_loss, router_beta_ce=beta_ce, router_beta_cu=beta_cu,
                router_cu_detection_multiplier=cu_multiplier, router_ce_criterion_offset=ce_offset,
            )

        selected_tokens = hidden_states[batch_indices, token_indices]
        continuous_signal_selected = combined_gating_signal[batch_indices, token_indices]

        # Process selected tokens
        num_selected_tokens = selected_tokens.shape[0]
        selected_tokens_batched = selected_tokens.unsqueeze(0)
        
        processing_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=None, input_shape=(1, num_selected_tokens),
            inputs_embeds=selected_tokens_batched, past_key_values_length=0
        )
        
        selected_pos_ids = position_ids[batch_indices, token_indices].unsqueeze(0) if position_ids is not None else None

        block_outputs = self.block(
            hidden_states=selected_tokens_batched, attention_mask=processing_attention_mask,
            position_ids=selected_pos_ids, use_cache=use_cache, **kwargs,
        )
        processed_tokens = block_outputs[0].squeeze(0)

        # Scatter results and apply straight-through estimator
        final_hidden_states = hidden_states.clone()
        delta_output = processed_tokens - selected_tokens
        scaled_delta = delta_output * continuous_signal_selected.unsqueeze(-1)
        updated_selected_states = selected_tokens + scaled_delta

        # Match destination dtype before scattering
        final_hidden_states[batch_indices, token_indices] = updated_selected_states.to(final_hidden_states.dtype)

        return DynamicLayerOutput(
            hidden_states=final_hidden_states,
            present_key_value=block_outputs[1] if use_cache and len(block_outputs) > 1 else None,
            attention_weights=None, 
            s_ce_stats=s_ce_stats, s_cu_stats=s_cu_stats, g_cont_stats=g_cont_stats,
            combined_gating_signal=combined_gating_signal, gate_vector=gate_vec_binary,
            prior_loss=decision_output.prior_loss, router_beta_ce=beta_ce, router_beta_cu=beta_cu,
            router_cu_detection_multiplier=cu_multiplier, router_ce_criterion_offset=ce_offset,
        )
```

## File: `src/models/layers/mod_layer.py`

```python
import torch
import torch.nn as nn
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from ..blocks.mod_router import MoDTokenRouter
from ..blocks.qwen_block import Qwen2Block


class MoDLayer(nn.Module):
    """
    Implements a Mixture-of-Depths (MoD) layer using a fully-batched,
    numerically stable gather-process-scatter approach. This version is
    significantly more performant than a batch-iterative approach on parallel
    hardware.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.router = MoDTokenRouter(config.hidden_size)
        self.block = Qwen2Block(config, layer_idx)
        self.capacity_gamma = config.capacity_gamma

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """
        Forward pass for the MoD layer.

        Note: This implementation does not support `use_cache=True`. Managing a
        sparse KV cache in a fully batched context is a complex problem that
        often requires specialized kernels. For training and evaluation, disabling
        the cache is a standard and safe approach.
        """
        if use_cache:
            raise NotImplementedError(
                "The fully-batched MoDLayer does not support use_cache=True."
            )

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Get router weights for token importance scoring
        router_weights = self.router(hidden_states)

        # Calculate capacity per sequence
        k = max(1, int(self.capacity_gamma * seq_len))

        # Select top-k tokens per sequence
        top_k_weights, _ = torch.topk(router_weights, k, dim=1, sorted=False)
        
        threshold = top_k_weights[:, -1].unsqueeze(1)
        is_selected = (router_weights >= threshold).to(torch.bool)

        # Gather selected token indices
        batch_indices, token_indices = is_selected.nonzero(as_tuple=True)

        if batch_indices.numel() == 0:
            return (hidden_states, None, None)

        # Gather selected tokens for processing
        selected_tokens = hidden_states[batch_indices, token_indices]

        # Process tokens as single batch
        num_selected_tokens = selected_tokens.shape[0]
        selected_tokens_batched = selected_tokens.unsqueeze(0)

        # Create attention mask for processing
        processing_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=None,
            input_shape=(1, num_selected_tokens),
            inputs_embeds=selected_tokens_batched,
            past_key_values_length=0,
        )

        # Gather position IDs for selected tokens
        selected_pos_ids = position_ids[batch_indices, token_indices].unsqueeze(0) if position_ids is not None else None

        block_outputs = self.block(
            hidden_states=selected_tokens_batched,
            attention_mask=processing_attention_mask,
            position_ids=selected_pos_ids,
            use_cache=False,
            **kwargs,
        )
        processed_tokens = block_outputs[0].squeeze(0)

        # Scatter processed tokens back
        final_hidden_states = hidden_states.clone()

        # Scale delta by router weights for numerical stability
        delta = processed_tokens - selected_tokens
        
        selected_router_weights = router_weights[batch_indices, token_indices]
        
        scaled_delta = delta * selected_router_weights.unsqueeze(-1).to(delta.dtype)
        
        updated_selected_tokens = selected_tokens + scaled_delta

        final_hidden_states[batch_indices, token_indices] = updated_selected_tokens

        return (final_hidden_states, None, None)

```

# Directory: `src/models/qwen`

## File: `src/models/qwen/causal_lm.py`

```python
import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM, Qwen2Model
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from .config import DynamicQwenConfig
from .modeling_outputs import DynamicCausalLMOutput, VPRCausalLMOutput
from ..layers.decision_layer import DecisionLayer
from ..layers.dynamic_layer import DynamicLayer
from ..layers.mod_layer import MoDLayer
from ..blocks.qwen_block import Qwen2Block
from ..utils.patching import populate_weights_from_source_layers

logger = logging.getLogger(__name__)


class DynamicQwenForCausalLM(Qwen2ForCausalLM):
    config_class = DynamicQwenConfig

    def __init__(self, config: DynamicQwenConfig):
        # Build model with dynamic layers from start
        super(Qwen2ForCausalLM, self).__init__(config)

        self.model = Qwen2Model(config)  # Creates embed_tokens and norm
        
        # Create dynamic layer structure
        dynamic_layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.dynamic_architecture == "vpr":
                if i % 2 == 0:
                    dynamic_layers.append(DecisionLayer(config, layer_idx=i))
                else:
                    dynamic_layers.append(DynamicLayer(config, layer_idx=i))
            elif config.dynamic_architecture == "mod":
                if (i + 1) % 2 == 0:
                    dynamic_layers.append(MoDLayer(config, layer_idx=i))
                else:
                    dynamic_layers.append(Qwen2Block(config, layer_idx=i))
            else:
                raise ValueError(f"Unknown dynamic_architecture: '{config.dynamic_architecture}'")
        self.model.layers = dynamic_layers

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._freeze_main_transformer_blocks = getattr(config, "freeze_main_transformer_blocks", False)

        # Initialize weights
        self.post_init()
        # Apply freezing after weight initialization
        self._apply_main_block_freezing()

    def _apply_main_block_freezing(self):
        for layer in self.model.layers:
            block_to_freeze = None
            if hasattr(layer, 'block'):
                block_to_freeze = layer.block
            elif isinstance(layer, Qwen2Block):
                block_to_freeze = layer

            if block_to_freeze:
                for p in block_to_freeze.parameters():
                    p.requires_grad = not self._freeze_main_transformer_blocks


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        current_iter: int = 0,
        **kwargs,
    ) -> Union[Tuple, DynamicCausalLMOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Disable KV cache for VPR (incompatible with gather-scatter)
        if self.config.dynamic_architecture == "vpr" and past_key_values is not None:
            use_cache = False

        if self.config.dynamic_architecture == "mod":
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        batch_size, seq_length, _ = hidden_states.shape
        
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            ).unsqueeze(0)
        
        # Match batch size
        if position_ids.shape[0] != batch_size:
            position_ids = position_ids.expand(batch_size, -1)

        causal_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        all_dynamic_layer_outputs = []
        next_past_key_values = [] if use_cache else None

        if self.config.dynamic_architecture == "vpr":
            for i in range(0, len(self.model.layers), 2):
                decision_layer = self.model.layers[i]
                dynamic_layer = self.model.layers[i+1]
                
                # Past KV for VPR
                past_kv_decision = past_key_values[i] if past_key_values is not None else None
                past_kv_dynamic = past_key_values[i+1] if past_key_values is not None else None

                common_args = {
                    "attention_mask": causal_mask,
                    "position_ids": position_ids,
                    "use_cache": use_cache,
                    "output_attentions": output_attentions,
                    **kwargs,
                }

                decision_output = decision_layer(
                    hidden_states, past_key_value=past_kv_decision, **common_args
                )
                dynamic_output = dynamic_layer(
                    decision_output.hidden_states,
                    decision_output=decision_output,
                    past_key_value=past_kv_dynamic,
                    **common_args,
                )

                hidden_states = dynamic_output.hidden_states
                all_dynamic_layer_outputs.append(dynamic_output)
                if use_cache:
                    next_past_key_values.extend([decision_output.present_key_value, dynamic_output.present_key_value])

        elif self.config.dynamic_architecture == "mod":
            for i, layer in enumerate(self.model.layers):
                # Past KV for MoD
                past_kv = past_key_values[i] if past_key_values is not None else None
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs,
                )
                hidden_states = layer_outputs[0]
                if use_cache:
                    next_past_key_values.append(layer_outputs[1])
        
        
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if self.config.dynamic_architecture == "vpr" and all_dynamic_layer_outputs:
            def aggregate_stats(outputs_list, key_name):
                # Mean stats across layers
                stats = [o.__getattribute__(key_name) for o in outputs_list]
                return {
                    'mean': torch.stack([s['mean'] for s in stats]).mean(),
                    'std': torch.stack([s['std'] for s in stats]).mean(),
                    'min': torch.stack([s['min'] for s in stats]).mean(),
                    'max': torch.stack([s['max'] for s in stats]).mean(),
                }
            s_ce_stats_agg = aggregate_stats(all_dynamic_layer_outputs, 's_ce_stats')
            s_cu_stats_agg = aggregate_stats(all_dynamic_layer_outputs, 's_cu_stats')
            g_cont_stats_agg = aggregate_stats(all_dynamic_layer_outputs, 'g_cont_stats')
            def aggregate_router_param_stats(outputs_list, param_name):
                # Router parameter stats
                values = torch.tensor([o.__getattribute__(param_name) for o in outputs_list], device=outputs_list[0].hidden_states.device)
                return {
                    'mean': values.mean(),
                    'std': values.std()
                }
            beta_ce_stats_agg = aggregate_router_param_stats(all_dynamic_layer_outputs, 'router_beta_ce')
            beta_cu_stats_agg = aggregate_router_param_stats(all_dynamic_layer_outputs, 'router_beta_cu')
            cu_multiplier_stats_agg = aggregate_router_param_stats(all_dynamic_layer_outputs, 'router_cu_detection_multiplier')
            ce_offset_stats_agg = aggregate_router_param_stats(all_dynamic_layer_outputs, 'router_ce_criterion_offset')

        if not return_dict:
            return (logits,)

        if self.config.dynamic_architecture == "vpr":
            vpr_metrics_dict = {
                "prior_loss": torch.stack([o.prior_loss for o in all_dynamic_layer_outputs]).mean(),
                "gate_vectors_per_layer": [o.gate_vector for o in all_dynamic_layer_outputs],
                "s_ce_stats": s_ce_stats_agg,
                "s_cu_stats": s_cu_stats_agg,
                "g_cont_stats": g_cont_stats_agg,
                "router_beta_ce_stats": beta_ce_stats_agg,
                "router_beta_cu_stats": beta_cu_stats_agg,
                "router_cu_multiplier_stats": cu_multiplier_stats_agg,
                "router_ce_offset_stats": ce_offset_stats_agg,
            }

            return VPRCausalLMOutput(
                logits=logits,
                past_key_values=tuple(next_past_key_values) if use_cache else None,
                loss=None,  # Calculated in training script
                vpr_metrics=vpr_metrics_dict,
            )
        else:
            return DynamicCausalLMOutput(
                logits=logits,
                past_key_values=tuple(next_past_key_values) if use_cache else None,
            )


    @classmethod
    def from_vanilla_checkpoint(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Factory method to CONVERT a vanilla HF checkpoint into a dynamic one.
        This should only be called once at the beginning of training.
        """
        logger.info(f"Converting vanilla checkpoint '{pretrained_model_name_or_path}' to dynamic architecture.")
        model_cfg = kwargs.pop("model_cfg", {})
        config = DynamicQwenConfig.from_pretrained(pretrained_model_name_or_path, **model_cfg)
        
        # Create custom model with correct layer structure
        custom_model = cls(config)

        # Load vanilla model weights
        kwargs.pop('config', None)
        vanilla_model = Qwen2ForCausalLM.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        
        # Transfer weights to custom model
        custom_model.model.embed_tokens.load_state_dict(vanilla_model.model.embed_tokens.state_dict())
        custom_model.model.norm.load_state_dict(vanilla_model.model.norm.state_dict())
        custom_model.lm_head.load_state_dict(vanilla_model.lm_head.state_dict())
        
        # Populate transformer layer weights
        populate_weights_from_source_layers(custom_model, vanilla_model.model.layers)
        
        del vanilla_model
        return custom_model
```

## File: `src/models/qwen/config.py`

```python
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class DynamicQwenConfig(Qwen2Config):
    """
    Extends Qwen2Config with parameters for dynamic computation, supporting
    both VPR (Variational Predictive Routing) and MoD (Mixture-of-Depths)
    architectures.
    """

    model_type = "dynamic_qwen"

    def __init__(self, **kwargs):
        # Architecture settings
        self.dynamic_architecture = kwargs.pop("dynamic_architecture", "vpr")
        self.use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)
        # Dynamic compute parameters
        self.capacity_gamma = kwargs.pop("capacity_gamma", 1.0)

        # VPR-specific parameters
        self.prior_loss_schedule = kwargs.pop("prior_loss_schedule", None)

        self.learn_beta_ce = kwargs.pop("learn_beta_ce", False)
        self.learn_beta_cu = kwargs.pop("learn_beta_cu", False)
        self.learn_cu_multiplier = kwargs.pop("learn_cu_multiplier", False)
        self.learn_ce_offset = kwargs.pop("learn_ce_offset", False)
        
        self.beta_ce_init = kwargs.pop("beta_ce_init", 1.0)
        self.beta_cu_init = kwargs.pop("beta_cu_init", 1.0)
        self.cu_detection_multiplier_init = kwargs.pop("cu_detection_multiplier_init", 1.0)
        self.ce_criterion_offset_init = kwargs.pop("ce_criterion_offset_init", 0.0)
        
        self.token_wise_gating = kwargs.pop("token_wise_gating", True)
        self.moving_average_window_size = kwargs.pop("moving_average_window_size", 100)
        self.prior_ffn_intermediate_size_factor = kwargs.pop(
            "prior_ffn_intermediate_size_factor", 2.0
        )

        # Training control
        self.freeze_main_transformer_blocks = kwargs.pop(
            "freeze_main_transformer_blocks", False
        )

        super().__init__(**kwargs)
```

## File: `src/models/qwen/modeling_outputs.py`

```python
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast


@dataclass
class DecisionLayerOutput:
    """
    Structured output for a VPR Decision Layer.
    """
    hidden_states: torch.Tensor
    vpr_signal_original_input: torch.Tensor
    vpr_signal_posterior_output: torch.Tensor
    vpr_signal_prior_hidden_states: torch.Tensor
    present_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]
    attention_weights: Optional[torch.Tensor]
    prior_loss: torch.Tensor


@dataclass
class DynamicLayerOutput:
    """
    Structured output for a VPR Dynamic Layer.
    """
    hidden_states: torch.Tensor
    present_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]
    attention_weights: Optional[torch.Tensor]
    s_ce_stats: dict
    s_cu_stats: dict
    g_cont_stats: dict
    combined_gating_signal: torch.Tensor
    gate_vector: torch.Tensor
    prior_loss: torch.Tensor
    router_beta_ce: float
    router_beta_cu: float
    router_cu_detection_multiplier: float
    router_ce_criterion_offset: float


@dataclass
class VPRCausalLMOutput(CausalLMOutputWithPast):
    """
    Custom output for the VPR architecture that inherits from the standard
    CausalLMOutputWithPast to be compatible with PEFT wrappers.
    All custom VPR metrics are bundled into a single dictionary.
    """
    vpr_metrics: Optional[Dict[str, torch.Tensor]] = None

@dataclass
class DynamicCausalLMOutput(ModelOutput):
    """
    A general output class, now primarily for the MoD architecture.
    """
    logits: torch.Tensor
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
```

## File: `src/models/qwen/tokenizer.py`

```python
from transformers import Qwen2TokenizerFast

class DynamicQwenTokenizer(Qwen2TokenizerFast):
    """
    A thin subclass of the Qwen fast tokenizer.
    Use this if you need to inject any special fixes (e.g. pad_token_id)
    or methods (e.g. custom chat templates). For now, it inherits unchanged.
    """

    pass

```

# Directory: `src/models/utils`

## File: `src/models/utils/patching.py`

```python
import logging
import torch.nn as nn

from ..layers.decision_layer import DecisionLayer
from ..layers.dynamic_layer import DynamicLayer
from ..layers.mod_layer import MoDLayer
from ..blocks.qwen_block import Qwen2Block

log = logging.getLogger(__name__)

def populate_weights_from_source_layers(custom_model, source_hf_layers):
    """
    Populates the weights of a custom dynamic model from a list of source
    transformer layers. Assumes the custom model's layer structure is already built.
    """
    log.info("Populating weights from source layers into custom model...")
    
    for i, source_layer in enumerate(source_hf_layers):
        target_layer = custom_model.model.layers[i]
        source_state_dict = source_layer.state_dict()
        
        # Load weights into the '.block' attribute of custom layers
        if hasattr(target_layer, 'block') and isinstance(target_layer.block, Qwen2Block):
            target_layer.block.load_state_dict(source_state_dict, strict=True)
        # Load weights directly if the target is a standard block
        elif isinstance(target_layer, Qwen2Block):
            target_layer.load_state_dict(source_state_dict, strict=True)
        else:
            log.warning(f"Could not load weights for layer {i} of type {type(target_layer)}.")
            
    log.info("Weight population complete.")
    return custom_model
```

## File: `src/models/utils/training.py`

```python
import torch
import torch.nn.functional as F

def set_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_metrics(model, batch, global_step):
    """
    A helper function to calculate all relevant losses and metrics.
    """
    model_output = model(
        **batch,
        current_iter=global_step,
        return_dict=True
    )
    
    shift_logits = model_output.logits[..., :-1, :].contiguous()
    shift_labels = batch["labels"][..., 1:].contiguous()
    
    lm_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    total_loss = lm_loss

    perplexity = torch.exp(lm_loss)

    metrics = {
        "total_loss": total_loss,
        "lm_loss": lm_loss,
        "perplexity": perplexity,
    }

    # Dynamic prior loss weighting
    prior_loss = None
    current_prior_loss_weight = 0.0

    if hasattr(model_output, 'vpr_metrics') and model_output.vpr_metrics is not None:
        vpr_metrics = model_output.vpr_metrics
        prior_loss = vpr_metrics.get("prior_loss")

        if prior_loss is not None:
            # Get model config (handle accelerator wrapper)
            config = model.module.config if hasattr(model, 'module') else model.config
            
            # Calculate weight based on schedule
            schedule_cfg = config.prior_loss_schedule
            initial_w = schedule_cfg['initial_weight']
            final_w = schedule_cfg['final_weight']
            decay_steps = schedule_cfg['decay_steps']

            if global_step < decay_steps:
                progress = global_step / decay_steps
                current_prior_loss_weight = initial_w - progress * (initial_w - final_w)
            else:
                current_prior_loss_weight = final_w
            
            total_loss += prior_loss * current_prior_loss_weight
        
        metrics.update(vpr_metrics)

    # Include prior loss in total
    metrics["total_loss"] = total_loss

    if hasattr(model_output, 'vpr_metrics') and model_output.vpr_metrics.get("gate_vectors_per_layer"):
        gate_vectors = model_output.vpr_metrics["gate_vectors_per_layer"]
        metrics["overall_gate_activation_mean"] = torch.stack([gv.mean() for gv in gate_vectors]).mean()
        metrics["per_layer_gate_stats"] = [
            {"mean": gv.mean(), "std": gv.std() if gv.numel() > 1 else torch.tensor(0.0)}
            for gv in gate_vectors
        ]
    else:
        metrics["overall_gate_activation_mean"] = torch.tensor(0.0)
        metrics["per_layer_gate_stats"] = []

    vpr_metrics = [
        "avg_ce_proportion", "avg_cu_proportion", "avg_beta_ce", "avg_beta_cu",
        "avg_cu_detection_multiplier", "avg_ce_criterion_offset", "combined_gating_signal_mean"
    ]
    for key in vpr_metrics:
        if hasattr(model_output, key):
            metrics[key] = getattr(model_output, key)

    metrics["current_prior_loss_weight"] = current_prior_loss_weight

    return metrics

```

# Directory: ``

## File: `train.py`

```python
import logging
import os
import math
import json
from tqdm.auto import tqdm

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import wandb

from accelerate import Accelerator
from transformers import get_scheduler, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

from src.data.gate_logging import GateLogger
from src.models.qwen.causal_lm import DynamicQwenForCausalLM
from src.models.utils.training import set_seed, calculate_metrics

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg: DictConfig) -> None:
    log.info(f"--- Config ---\n{OmegaConf.to_yaml(cfg)}")
    set_seed(cfg.run.seed)
    accelerator = Accelerator(
        mixed_precision=cfg.run.precision,
        gradient_accumulation_steps=cfg.training.accumulate_grad_batches,
    )
    log.info(f"Using device: {accelerator.device}")
    datamodule = hydra.utils.instantiate(cfg.data, _convert_="partial")
    datamodule.setup()
    data_collator = DataCollatorForLanguageModeling(tokenizer=datamodule.tokenizer, mlm=False)
    train_dataloader = DataLoader(
        datamodule.train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=cfg.data.batch_size,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        datamodule.val_dataset,
        collate_fn=data_collator,
        batch_size=cfg.data.batch_size,
        num_workers=4,
    )

    log.info(f"Instantiating Model <{cfg.model.pretrained_model_name_or_path}>")
    model_load_kwargs = {
        "model_cfg": OmegaConf.to_container(cfg.model.model_cfg, resolve=True)
    }
    if getattr(cfg.model, "use_flash_attention_2", False):
        log.info("Flash Attention 2 is enabled in the config. Applying to model loading.")
        model_load_kwargs["attn_implementation"] = "flash_attention_2"
        model_load_kwargs["torch_dtype"] = torch.bfloat16
    model = DynamicQwenForCausalLM.from_vanilla_checkpoint(
        cfg.model.pretrained_model_name_or_path,
        **model_load_kwargs
    )
    if cfg.peft.enabled:
        log.info("Applying PEFT (LoRA) configuration to the model...")
        peft_config_dict = OmegaConf.to_container(cfg.peft.config, resolve=True)
        peft_config_dict.pop('_target_', None)
        peft_config = LoraConfig(**peft_config_dict)
        model = get_peft_model(model, peft_config)
        log.info("Trainable parameters after applying LoRA:")
        model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    tokenizer = datamodule.tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    
    log.info("Setting up optimizer with distinct parameter groups...")
    base_model_params, prior_params, vpr_router_params = [], [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if "vpr_router" in n:
                vpr_router_params.append(p)
            elif "prior_ffn" in n:
                prior_params.append(p)
            else:
                base_model_params.append(p)
    param_groups = [
        {"params": base_model_params, "lr": cfg.training.optimizer.base_model_lr},
        {"params": prior_params, "lr": cfg.training.optimizer.prior_lr},
        {"params": vpr_router_params, "lr": cfg.training.optimizer.vpr_router_lr},
    ]
    log.info(f"  - Base Model parameters: {sum(p.numel() for p in base_model_params):,}")
    log.info(f"  - Dynamic Component (Prior FFN) parameters: {sum(p.numel() for p in prior_params):,}")
    log.info(f"  - VPR Router parameters: {sum(p.numel() for p in vpr_router_params):,}")
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.training.optimizer.weight_decay,
    )
    gate_logger = GateLogger(model.config.num_hidden_layers)

    # Calculate epochs and training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.accumulate_grad_batches)
    if cfg.training.max_steps > 0:
        num_training_steps = cfg.training.max_steps
        num_epochs = math.ceil(num_training_steps / num_update_steps_per_epoch)
        log.info(f"Training for a fixed {num_training_steps} steps (approx. {num_epochs} epochs).")
    else:
        num_training_steps = num_update_steps_per_epoch * cfg.training.num_epochs
        num_epochs = cfg.training.num_epochs
        log.info(f"Training for {num_epochs} epochs ({num_training_steps} steps).")
    
    num_warmup_steps = int(num_training_steps * cfg.training.optimizer.warmup_ratio)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    if accelerator.is_main_process and cfg.logging.wandb.enabled:
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=cfg.run.name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    log.info("--- Starting Training ---")
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_main_process)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if progress_bar.n >= num_training_steps:
                break
            
            with accelerator.accumulate(model):
                metrics = calculate_metrics(model, batch, progress_bar.n)
                total_loss = metrics["total_loss"]
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    if cfg.training.use_gradient_clipping:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                if metrics.get("per_layer_gate_stats"):
                    gate_logger.update_rolling_history(metrics["per_layer_gate_stats"])
                    gate_logger.log_rolling_history(progress_bar.n, log_interval=100)
                
                if accelerator.is_main_process:
                    log_metrics = {
                        "train/loss": metrics["total_loss"].item(),
                        "train/lm_loss": metrics["lm_loss"].item(),
                        "train/perplexity": metrics["perplexity"].item(),
                        "lr": lr_scheduler.get_last_lr()[0]
                    }
                    if "s_ce_stats" in metrics:  # VPR metrics
                        if metrics.get("prior_loss") is not None:
                            log_metrics["train/prior_loss"] = metrics["prior_loss"].item()
                            log_metrics["train/prior_loss_weight"] = metrics["current_prior_loss_weight"]
                        def log_signal_stats(signal_name, stats_dict):
                            for key, value in stats_dict.items():
                                log_metrics[f"train_vpr_signals/{signal_name}_{key}"] = value.item()
                        log_signal_stats("S_CE", metrics["s_ce_stats"])
                        log_signal_stats("S_CU", metrics["s_cu_stats"])
                        log_signal_stats("G_cont", metrics["g_cont_stats"])
                        def log_router_param_stats(param_name, stats_dict):
                            if stats_dict:
                                log_metrics[f"train_vpr_router/{param_name}_mean"] = stats_dict["mean"].item()
                                log_metrics[f"train_vpr_router/{param_name}_std"] = stats_dict["std"].item()
                        log_router_param_stats("beta_ce", metrics.get("router_beta_ce_stats"))
                        log_router_param_stats("beta_cu", metrics.get("router_beta_cu_stats"))
                        log_router_param_stats("cu_multiplier", metrics.get("router_cu_multiplier_stats"))
                        log_router_param_stats("ce_offset", metrics.get("router_ce_offset_stats"))
                    if cfg.logging.wandb.enabled:
                        wandb.log(log_metrics, step=progress_bar.n)

                if (progress_bar.n) % cfg.training.eval_interval == 0 and progress_bar.n > 0:
                    model.eval()
                    val_losses = []
                    for val_batch in val_dataloader:
                        with torch.no_grad():
                            val_metrics_dict = calculate_metrics(model, val_batch, progress_bar.n)
                        val_losses.append(accelerator.gather(val_metrics_dict["total_loss"]))
                    val_loss = torch.stack(val_losses).mean().item()
                    
                    if accelerator.is_main_process:
                        log.info(f"Epoch {epoch}, Step {progress_bar.n}: Validation Loss = {val_loss:.4f}")
                        if cfg.logging.wandb.enabled:
                            wandb.log({"val/loss": val_loss}, step=progress_bar.n)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            log.info(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                            unwrapped_model = accelerator.unwrap_model(model)
                            save_path = os.path.join(cfg.run.output_dir, "best_model")
                            unwrapped_model.save_pretrained(save_path, safe_serialization=True)
                            tokenizer.save_pretrained(save_path)
                            
                            if cfg.logging.wandb.enabled:
                                wandb_info = {
                                    "run_id": wandb.run.id, "project": wandb.run.project,
                                    "entity": wandb.run.entity, "run_name": wandb.run.name,
                                }
                                with open(os.path.join(save_path, "wandb_info.json"), "w") as f:
                                    json.dump(wandb_info, f, indent=2)
                                log.info(f"Saved wandb run info to {save_path}")
                    model.train()
        
        if progress_bar.n >= num_training_steps:
            log.info(f"Reached max_steps ({num_training_steps}). Stopping training.")
            break

    if accelerator.is_main_process:
        log.info("--- Saving final model checkpoint ---")
        unwrapped_model = accelerator.unwrap_model(model)
        final_save_path = os.path.join(cfg.run.output_dir, "final_model")
        unwrapped_model.save_pretrained(final_save_path, safe_serialization=True)
        tokenizer.save_pretrained(final_save_path)
        
        if cfg.logging.wandb.enabled:
            wandb_info = {
                "run_id": wandb.run.id, "project": wandb.run.project,
                "entity": wandb.run.entity, "run_name": wandb.run.name,
            }
            with open(os.path.join(final_save_path, "wandb_info.json"), "w") as f:
                json.dump(wandb_info, f, indent=2)
            log.info(f"Saved wandb run info to {final_save_path}")
            
        log.info(f"Final model saved to {final_save_path}")
        if cfg.logging.wandb.enabled:
            wandb.finish()

    log.info("--- Training Finished ---")

if __name__ == "__main__":
    main()
```
