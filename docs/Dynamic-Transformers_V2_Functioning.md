# Directory: `src/data`

## File: `src/data/__init__.py`

```python
from .mixed_dataset import MixedDataset
from .huggingface_dataset import HuggingFaceDataset
from .pretraining_dataset import PretrainingDataset
from .base_dataset import BaseDatasetHandler

__all__ = [
    "MixedDataset",
    "HuggingFaceDataset",
    "PretrainingDataset",
    "BaseDatasetHandler",
]

```

## File: `src/data/base_dataset.py`

```python
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict

from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import random_split
from transformers import PreTrainedTokenizerBase

log = logging.getLogger(__name__)

class BaseDatasetHandler(ABC):
    """
    An abstract base class for handling Hugging Face datasets.

    It encapsulates the shared logic for loading, processing, and splitting,
    while delegating the specific text formatting to subclasses.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_name: str,
        text_column: str,
        block_size: int,
        dataset_config: str = None,
        validation_split_percentage: int = 5,
        train_subset_ratio: float = None,
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.block_size = block_size
        self.dataset_config = dataset_config
        self.validation_split_percentage = validation_split_percentage
        self.train_subset_ratio = train_subset_ratio

    @abstractmethod
    def _process_text_column(self, examples: Dict) -> Dict[str, str]:
        """
        Processes the raw text column into a standardized 'text' field.
        Subclasses must implement this to handle their specific data format
        (e.g., plain text vs. chat/instruction format).
        """
        pass

    def _group_texts(self, examples: Dict) -> Dict:
        """Concatenates and groups texts into fixed-size blocks."""
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
        """The main pipeline for loading and preparing the dataset."""
        log.info(f"Loading and processing dataset: {self.dataset_name}")
        raw_datasets = load_dataset(self.dataset_name, self.dataset_config, trust_remote_code=True)

        if isinstance(raw_datasets, Dataset):
            raw_datasets = DatasetDict({"train": raw_datasets})
        if "train" not in raw_datasets:
            first_key = next(iter(raw_datasets.keys()))
            log.warning(f"No 'train' split found. Using '{first_key}' as the training split.")
            raw_datasets["train"] = raw_datasets.pop(first_key)

        num_proc = os.cpu_count() or 1

        # 1. Format the text using the subclass-specific implementation
        formatted_datasets = raw_datasets.map(self._process_text_column, batched=False, num_proc=num_proc)
        
        # 2. Filter out short/empty examples
        filtered_datasets = formatted_datasets.filter(lambda x: x.get("text") and len(x["text"]) > 10, num_proc=num_proc)
        
        # 3. Tokenize
        tokenized_datasets = filtered_datasets.map(
            lambda e: self.tokenizer(e["text"]),
            batched=True,
            remove_columns=filtered_datasets["train"].column_names,
            num_proc=num_proc
        )
        
        # 4. Group into blocks
        lm_datasets = tokenized_datasets.map(self._group_texts, batched=True, num_proc=num_proc)
        full_dataset = lm_datasets["train"]

        # 5. Subset the training data if requested
        if self.train_subset_ratio and 0.0 < self.train_subset_ratio < 1.0:
            num_samples = int(len(full_dataset) * self.train_subset_ratio)
            full_dataset = full_dataset.select(range(num_samples))
            log.info(f"Subsetting '{self.dataset_name}' to {num_samples} samples.")

        # 6. Split into training and validation sets
        if self.validation_split_percentage > 0 and len(full_dataset) > 1:
            val_size = int(len(full_dataset) * (self.validation_split_percentage / 100))
            val_size = max(1, val_size) # Ensure at least one validation sample
            train_size = len(full_dataset) - val_size
            if train_size <= 0:
                raise ValueError("Dataset is too small to create a non-empty training split.")
            
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        else:
            train_dataset = full_dataset
            val_dataset = []

        log.info(f"Finished processing '{self.dataset_name}': {len(train_dataset):,} train, {len(val_dataset):,} val samples.")
        return train_dataset, val_dataset

```

## File: `src/data/huggingface_dataset.py`

```python
import json
import re
from typing import Any, Dict
from .base_dataset import BaseDatasetHandler

def _dict_list_to_chat(tokenizer, conv: list[dict[str, Any]]) -> dict[str, str]:
    """Helper function to convert a list of dicts to a chat string."""
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

class HuggingFaceDataset(BaseDatasetHandler):
    """
    Handles chat and instruction-formatted datasets (SFT).
    Inherits all processing logic from BaseDatasetHandler.
    """
    def _process_text_column(self, examples: Dict[str, Any]) -> Dict[str, str]:
        """
        Normalizes various chat and instruction formats into a single 'text' field.
        This is the specialized logic for this handler.
        """
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
            if raw.strip().startswith(("{ ", "[")):
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
        Loads and concatenates all specified datasets using a handler mapping.
        """
        log.info("Setting up mixed dataset...")
        
        handler_map = {
            "sft": HuggingFaceDataset,
            "pretrain": PretrainingDataset
        }
        
        all_train_datasets, all_val_datasets = [], []

        for cfg in self.dataset_configs:
            dataset_type = cfg.get("type", "sft")
            handler_class = handler_map.get(dataset_type)

            if not handler_class:
                raise ValueError(f"Unknown dataset type '{dataset_type}' in config. "
                                 f"Available types are: {list(handler_map.keys())}")

            log.info(f"Processing dataset '{cfg.dataset_name}' with handler: '{dataset_type}'")

            handler = handler_class(
                tokenizer=self.tokenizer,
                dataset_name=cfg.dataset_name,
                text_column=cfg.text_column,
                block_size=self.block_size,
                dataset_config=cfg.get("dataset_config"),
                validation_split_percentage=self.validation_split_percentage,
                train_subset_ratio=cfg.get("train_subset_ratio"),
            )
            
            train_data, val_data = handler.load_and_process()

            if train_data: all_train_datasets.append(train_data)
            if val_data: all_val_datasets.append(val_data)

        self.train_dataset = ConcatDataset(all_train_datasets) if all_train_datasets else []
        self.val_dataset = ConcatDataset(all_val_datasets) if all_val_datasets else []

        log.info(f"Total mixed training samples: {len(self.train_dataset):,}")
        log.info(f"Total mixed validation samples: {len(self.val_dataset):,}")

```

## File: `src/data/pretraining_dataset.py`

```python
from typing import Any, Dict
from .base_dataset import BaseDatasetHandler

class PretrainingDataset(BaseDatasetHandler):
    """
    Handles raw text datasets for continued pre-training.
    Inherits all processing logic from BaseDatasetHandler.
    """
    def _process_text_column(self, examples: Dict[str, Any]) -> Dict[str, str]:
        """Extracts and cleans text from the specified column without chat formatting."""
        raw_text = examples.get(self.text_column)
        return {"text": str(raw_text).strip() if raw_text is not None else ""}
```

# Directory: `src/models`

## File: `src/models/__init__.py`

```python
from .base.causal_lm import BaseForCausalLM
from .standard.model import StandardTransformerForCausalLM
from .mod.causal_lm import MoDForCausalLM
from .sdt.causal_lm import SDTForCausalLM # Corrected from std
from .stt.causal_lm import STTForCausalLM

__all__ = [
    "BaseForCausalLM",
    "StandardTransformerForCausalLM",
    "MoDForCausalLM",
    "SDTForCausalLM",
    "STTForCausalLM",
]

```

# Directory: `src/models/base`

## File: `src/models/base/__init__.py`

```python
from .block import DynamicBlock
from .causal_lm import BaseForCausalLM
from .priors import BasePriorNetwork
from .routers import BaseRouter, CausalRouter, BaseSurpriseRouter

__all__ = [
    "DynamicBlock",
    "BaseForCausalLM",
    "BasePriorNetwork",
    "BaseRouter",
    "CausalRouter",
    "BaseSurpriseRouter",
]

```

## File: `src/models/base/block.py`

```python
import torch
import torch.nn as nn
from typing import Tuple
from .custom_decoder_layer import CustomQwen2DecoderLayer # Import custom decoder layer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

class DynamicBlock(nn.Module):
    """
    A versatile Transformer block that serves two purposes:
    1. As a standard, dense Qwen2DecoderLayer.
    2. As a processing engine for a selected subset of tokens in dynamic models.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer = CustomQwen2DecoderLayer(config, layer_idx) # Use CustomQwen2DecoderLayer

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        """Standard forward pass for use as a dense layer."""
        return self.layer(hidden_states, **kwargs)

    def process_selected(
        self,
        hidden_states: torch.Tensor,
        batch_indices: torch.Tensor,
        token_indices: torch.Tensor,
        gating_scores: torch.Tensor,
        use_soft_gating: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Tuple, Tuple]:
        """
        Gathers, processes, and scatters a selected subset of tokens.
        This is the core utility for all dynamic computation layers.
        """
        if batch_indices.numel() == 0:
            return hidden_states, None, None

        # 1. Gather
        selected_tokens = hidden_states[batch_indices, token_indices]
        num_selected = selected_tokens.shape[0]
        selected_tokens_batched = selected_tokens.unsqueeze(0)

        # 2. Prepare inputs for the block
        position_ids = kwargs.get('position_ids')
        position_embeddings = kwargs.get('position_embeddings')
        
        selected_attn_mask = _prepare_4d_causal_attention_mask(None, (1, num_selected), selected_tokens_batched, 0)
        selected_pos_ids = position_ids[batch_indices, token_indices].unsqueeze(0) if position_ids is not None else None
        
        selected_pos_emb = None
        if position_embeddings is not None:
            cos, sin = position_embeddings
            selected_pos_emb = (cos[batch_indices, token_indices].unsqueeze(0), sin[batch_indices, token_indices].unsqueeze(0))

        # 3. Process
        block_outputs = self.layer(
            hidden_states=selected_tokens_batched,
            attention_mask=selected_attn_mask,
            position_ids=selected_pos_ids,
            position_embeddings=selected_pos_emb,
            use_cache=kwargs.get('use_cache', False)
        )
        processed_tokens = block_outputs[0].squeeze(0)

        # 4. Scatter
        final_hidden_states = hidden_states.clone()
        
        if use_soft_gating:
            # Aligns with the OLD code's logic for MoD and SDT
            delta = processed_tokens - selected_tokens
            scaled_delta = delta * gating_scores.unsqueeze(-1).to(delta.dtype)
            updated_tokens = selected_tokens + scaled_delta
            final_hidden_states[batch_indices, token_indices] = updated_tokens
        else:
            # Hard update for inference
            final_hidden_states[batch_indices, token_indices] = processed_tokens

        present_key_value = block_outputs[1] if kwargs.get('use_cache', False) and len(block_outputs) > 1 else None
        attention_weights = block_outputs[2] if len(block_outputs) > 2 else None

        return final_hidden_states, present_key_value, attention_weights
```

## File: `src/models/base/causal_lm.py`

```python
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from omegaconf import DictConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2RotaryEmbedding, Qwen2DecoderLayer, Qwen2Config
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from .block import DynamicBlock
import logging
import copy

log = logging.getLogger(__name__)

class BaseForCausalLM(nn.Module):
    """
    Contains shared components and the unified forward pass logic for all Causal LM variants.
    """
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.model_params = kwargs
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rotary_config = Qwen2Config(
            hidden_size=config.head_dim, # Use config.head_dim directly
            num_attention_heads=1, # Set num_attention_heads to 1 to make it work with the new hidden_size
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            sliding_window=config.sliding_window,
        )
        self.rotary_emb = Qwen2RotaryEmbedding(rotary_config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _setup_layers(self):
        raise NotImplementedError("Subclasses must implement `_setup_layers`")

    def _forward_layers(self, hidden_states: torch.Tensor, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement `_forward_layers`")

    def forward(
        self, input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        B, T, D = hidden_states.shape

        if position_ids is None:
            position_ids = torch.arange(T, device=hidden_states.device).unsqueeze(0).expand(B, -1)
        
        causal_mask = _prepare_4d_causal_attention_mask(attention_mask, (B, T), hidden_states, 0)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        log.debug(f"DEBUG: cos shape: {cos.shape}, sin shape: {sin.shape}") # Added debug print
        
        layer_kwargs = {**kwargs, "attention_mask": causal_mask, "position_ids": position_ids, "position_embeddings": (cos, sin)}
        
        layer_outputs = self._forward_layers(hidden_states, **layer_kwargs)
        
        final_hidden_states = self.norm(layer_outputs["hidden_states"])
        logits = self.lm_head(final_hidden_states)
        
        lm_loss = self.compute_loss(logits, labels)
        
        total_loss = lm_loss
        if "aux_loss" in layer_outputs and layer_outputs["aux_loss"] is not None and self.training:
            total_loss += layer_outputs["aux_loss"]
        
        final_outputs = {"logits": logits, "loss": total_loss, "lm_loss": lm_loss}
        final_outputs.update(layer_outputs)
        
        return final_outputs

    def compute_loss(self, logits, labels):
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            return loss_fct(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
        return None

    def gradient_checkpointing_enable(self):
        """
        Enables gradient checkpointing for all `DynamicBlock` and `Qwen2DecoderLayer`
        instances within the model's layers.
        """
        for layer in self.layers:
            if isinstance(layer, DynamicBlock):
                if hasattr(layer.layer, 'gradient_checkpointing_enable'):
                    layer.layer.gradient_checkpointing_enable()
            elif isinstance(layer, nn.ModuleDict):
                # Handle cases where DynamicBlock is nested in ModuleDict (e.g., MoD, SDT, STT)
                for sub_module in layer.values():
                    if isinstance(sub_module, DynamicBlock):
                        if hasattr(sub_module.layer, 'gradient_checkpointing_enable'):
                            sub_module.layer.gradient_checkpointing_enable()
                    elif hasattr(sub_module, 'block') and isinstance(sub_module.block, DynamicBlock):
                        if hasattr(sub_module.block.layer, 'gradient_checkpointing_enable'):
                            sub_module.block.layer.gradient_checkpointing_enable()
                    elif hasattr(sub_module, 'decision') and hasattr(sub_module.decision, 'block'): # For SDT
                        if hasattr(sub_module.decision.block.layer, 'gradient_checkpointing_enable'):
                            sub_module.decision.block.layer.gradient_checkpointing_enable()
                    elif isinstance(sub_module, Qwen2DecoderLayer): # For standard layers in ModuleDict
                        if hasattr(sub_module, 'gradient_checkpointing_enable'):
                            sub_module.gradient_checkpointing_enable()
                    elif hasattr(sub_module, 'block') and isinstance(sub_module.block, Qwen2DecoderLayer): # For STTLayer
                        if hasattr(sub_module.block, 'gradient_checkpointing_enable'):
                            sub_module.block.gradient_checkpointing_enable()
            elif isinstance(layer, Qwen2DecoderLayer): # For standard layers directly in self.layers
                if hasattr(layer, 'gradient_checkpointing_enable'):
                    layer.gradient_checkpointing_enable()
        # Also enable for the base model if it has the method (e.g., if it's a Qwen2ForCausalLM)
        if hasattr(super(), 'gradient_checkpointing_enable'):
            super().gradient_checkpointing_enable()

    def enable_input_require_grads(self):
        """
        Enables `requires_grad` for the input embeddings, which is necessary
        when using gradient checkpointing.
        """
        if hasattr(self, 'embed_tokens') and hasattr(self.embed_tokens, 'weight'):
            self.embed_tokens.weight.requires_grad_(True)
        # If the model has a get_input_embeddings method (like PreTrainedModel), use it
        elif hasattr(self, 'get_input_embeddings'):
            input_embeddings = self.get_input_embeddings()
            if input_embeddings is not None and hasattr(input_embeddings, 'weight'):
                input_embeddings.weight.requires_grad_(True)

    def copy_weights_from_pretrained(self, pretrained_model):
        """Copies weights from a pretrained Qwen2 model to this model."""
        self.embed_tokens.load_state_dict(pretrained_model.model.embed_tokens.state_dict())
        self.norm.load_state_dict(pretrained_model.model.norm.state_dict())
        self.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())

        for i, layer in enumerate(self.layers):
            if i >= len(pretrained_model.model.layers): break
            pretrained_layer = pretrained_model.model.layers[i]
            
            if isinstance(layer, DynamicBlock):
                layer.layer.load_state_dict(pretrained_layer.state_dict())
            elif isinstance(layer, nn.ModuleDict):
                if 'block' in layer:
                    layer.block.layer.load_state_dict(pretrained_layer.state_dict())
                if 'decision' in layer and hasattr(layer.decision, 'block'):
                    layer.decision.block.load_state_dict(pretrained_layer.state_dict())
                if 'dynamic_block' in layer and i + 1 < len(pretrained_model.model.layers):
                    pretrained_dynamic_layer = pretrained_model.model.layers[i + 1]
                    layer.dynamic_block.layer.load_state_dict(pretrained_dynamic_layer.state_dict())

    def get_trainable_parameters(self) -> List[Dict[str, Any]]:
        return [{'name': 'base_model', 'params': list(p for p in self.parameters() if p.requires_grad)}]

    def _create_param_groups(self, component_map: Dict[str, str]) -> List[Dict[str, Any]]:
        param_groups = {}
        base_model_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            assigned = False
            for group_name, keyword in component_map.items():
                if keyword in name:
                    param_groups.setdefault(group_name, []).append(param)
                    assigned = True
                    break
            if not assigned:
                base_model_params.append(param)
        
        groups_list = [{'name': 'base_model', 'params': base_model_params}]
        groups_list.extend([{'name': name, 'params': params} for name, params in param_groups.items()])
        return [g for g in groups_list if g['params']]

```

## File: `src/models/base/custom_attention.py`

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.cache_utils import Cache
from transformers.integrations import use_kernel_forward_from_hub
import torch.nn.functional as F

# --- Helper functions from modeling_qwen2.py ---
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

# --- Custom Qwen2Attention ---
class CustomQwen2Attention(nn.Module):
    """Custom Multi-headed attention for Qwen2, bypassing problematic apply_rotary_pos_emb."""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        
        # Manual application of rotary embeddings, bypassing apply_rotary_pos_emb
        # Ensure cos and sin are correctly unsqueezed for broadcasting
        # The unsqueeze_dim is typically 1 for (batch, heads, seq_len, head_dim)
        # and cos/sin are (batch, seq_len, head_dim)
        cos = cos.unsqueeze(1) # unsqueeze_dim=1
        sin = sin.unsqueeze(1) # unsqueeze_dim=1
        query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (rotate_half(key_states) * sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states, present_key_value = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            present_key_value = None

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            # sliding_window=self.sliding_window, # This is not used in eager_attention_forward directly
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, present_key_value, attn_weights

```

## File: `src/models/base/custom_decoder_layer.py`

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.cache_utils import Cache
from transformers.integrations import use_kernel_forward_from_hub
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm # Import original MLP and RMSNorm

from .custom_attention import CustomQwen2Attention # Import our custom attention

class CustomQwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = CustomQwen2Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Assuming config.layer_types is available and relevant for attention_type
        self.attention_type = config.layer_types[layer_idx] if hasattr(config, 'layer_types') and len(config.layer_types) > layer_idx else "full_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None, # Passed from BaseForCausalLM
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, present_key_value, attentions = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings, # Pass position_embeddings
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        # Return hidden_states, present_key_value, attentions
        return hidden_states, present_key_value, attentions

```

## File: `src/models/base/priors.py`

```python
import copy
import torch.nn as nn
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from omegaconf import DictConfig

class BasePriorNetwork(nn.Module):
    """
    Abstracts the creation of a lightweight feed-forward network.
    """
    def __init__(self, config: Qwen2Config, model_cfg: DictConfig):
        super().__init__()
        mlp_config = copy.deepcopy(config)
        factor = getattr(config, "prior_ffn_intermediate_size_factor", 0.25)
        raw_size = config.hidden_size * factor
        rounded_size = int(raw_size + 0.999)
        intermediate_size = max(2, rounded_size + (rounded_size % 2))
        mlp_config.intermediate_size = intermediate_size
        self.mlp = Qwen2MLP(mlp_config)

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward pass.")or("Subclasses must implement the forward pass.")
```

## File: `src/models/base/routers.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import copy
from omegaconf import DictConfig, OmegaConf
import logging

log = logging.getLogger(__name__)

class BaseRouter(nn.Module, ABC):
    """Abstract base class for all routing modules."""
    def __init__(self, config, capacity_attr: str, model_cfg: DictConfig): # Accept model_cfg
        super().__init__()
        self.config = config
        self.model_cfg = model_cfg
        print(f"DEBUG: BaseRouter model_cfg keys: {self.model_cfg.keys()}")
        print(f"DEBUG: BaseRouter capacity_attr: {capacity_attr}")
        self.capacity = self.model_cfg[str(capacity_attr)] # Access from model_cfg

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        raise NotImplementedError

    def select_tokens(self, scores: torch.Tensor, hidden_states: torch.Tensor):
        log.debug(f"BaseRouter.select_tokens: hidden_states shape: {hidden_states.shape}")
        B, T, D = hidden_states.shape
        k = max(1, int(T * self.capacity))
        topk_vals, topk_idx = scores.topk(k, dim=-1)
        batch_idx = torch.arange(B, device=scores.device).unsqueeze(1).expand(-1, k)
        return hidden_states[batch_idx, topk_idx].reshape(-1, D), \
               batch_idx.reshape(-1), topk_idx.reshape(-1), topk_vals.reshape(-1)

class CausalRouter(BaseRouter):
    """Unified CausalRouter for MoD, SDT, and STT inference."""
    def __init__(self, config, layer_idx: int, capacity_attr: str, model_cfg: DictConfig): # Add model_cfg
        super().__init__(config, capacity_attr, model_cfg) # Pass model_cfg
        self.router = nn.Linear(2 * config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        B, T, D = hidden_states.shape
        prev = torch.cat([torch.zeros(B, 1, D, device=hidden_states.device, dtype=hidden_states.dtype), hidden_states[:, :-1, :]], dim=1)
        logits = self.router(torch.cat([hidden_states, prev], dim=-1)).squeeze(-1)
        return logits, None, {}

class BaseSurpriseRouter(BaseRouter, ABC):
    """Abstracts the common surprise-based routing logic for SDT and STT."""
    def __init__(self, config, capacity_attr: str, model_cfg: DictConfig): # Add model_cfg
        super().__init__(config, capacity_attr, model_cfg) # Pass model_cfg
        self.raw_o_ce = nn.Parameter(torch.tensor(float(self.model_cfg['o_ce_init'])))
        self.raw_m_cu = nn.Parameter(torch.tensor(float(self.model_cfg['m_cu_init'])))
        self.ma_window = int(self.model_cfg['ma_window'])
    
    def _moving_average(self, d_st: torch.Tensor) -> torch.Tensor:
        B, T = d_st.shape
        W = min(self.ma_window, T)
        if W <= 1: return d_st
        padded = F.pad(d_st.unsqueeze(1), (W - 1, 0), 'replicate')
        return F.avg_pool1d(padded, kernel_size=W, stride=1).squeeze(1)

    def _get_vpr_signals(self, D_st, D_ch, beta_ce, beta_cu):
        o_ce_pos = F.softplus(self.raw_o_ce)
        m_cu_pos = F.softplus(self.raw_m_cu)
        
        CE = D_st - (D_ch - torch.log(o_ce_pos + 1e-10))
        CU = D_st - (m_cu_pos * self._moving_average(D_st.detach()))
        
        S_CE = torch.sigmoid(torch.tensor(beta_ce, device=CE.device) * CE)
        S_CU = torch.sigmoid(torch.tensor(beta_cu, device=CU.device) * CU)
        
        g_cont = S_CE + S_CU - (S_CE * S_CU)
        return g_cont, {"S_CE_mean": S_CE.mean().item(), "S_CU_mean": S_CU.mean().item(), "o_ce_pos": o_ce_pos.item(), "m_cu_pos": m_cu_pos.item()}
```

# Directory: `src/models/mod`

## File: `src/models/mod/__init__.py`

```python
from .model import MoDRouter
from .causal_lm import MoDForCausalLM

__all__ = [
    "MoDRouter",
    "MoDForCausalLM",
]
```

## File: `src/models/mod/causal_lm.py`

```python
import torch.nn as nn
from ..base.causal_lm import BaseForCausalLM
from ..base.block import DynamicBlock
from ..base.routers import CausalRouter
from .model import MoDRouter
from omegaconf import DictConfig

class MoDForCausalLM(BaseForCausalLM):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._setup_layers()
    
    def _setup_layers(self):
        for i in range(self.config.num_hidden_layers):
            block = DynamicBlock(self.config, i)
            if i % 2 == 1: # Dynamic MoD layer
                self.layers.append(nn.ModuleDict({
                    'block': block,
                    'router': MoDRouter(self.config, i, self.model_params), # Pass self.model_params
                    'causal_router': CausalRouter(self.config, i, 'mod_capacity', self.model_params), # Pass self.model_params
                }))
            else: # Standard layer
                self.layers.append(block)

    def _forward_layers(self, hidden_states, **kwargs):
        total_aux_loss = 0
        for layer in self.layers:
            if isinstance(layer, nn.ModuleDict):
                router = layer['router'] if self.training else layer['causal_router']
                scores, aux_loss, _ = router(hidden_states)
                if aux_loss is not None:
                    total_aux_loss += aux_loss
                
                _, batch_idx, token_idx, gating_scores = router.select_tokens(scores, hidden_states)
                hidden_states, _, _ = layer['block'].process_selected(
                    hidden_states, batch_idx, token_idx, gating_scores, use_soft_gating=self.training, **kwargs
                )
            else:
                hidden_states = layer(hidden_states, **kwargs)[0]
            
        return {"hidden_states": hidden_states, "aux_loss": total_aux_loss}
        
        def get_trainable_parameters(self):
            return self._create_param_groups({'router': 'router', 'causal_router': 'causal_router'})
```

## File: `src/models/mod/model.py`

```python
import torch
import torch.nn as nn
from typing import Tuple, Optional
from omegaconf import DictConfig
from ..base.routers import BaseRouter

class MoDRouter(BaseRouter):
    """A simple linear router for MoD that includes a load-balancing loss."""
    def __init__(self, config, layer_idx: int, model_cfg: DictConfig): # Accept model_cfg
        super().__init__(config, capacity_attr='mod_capacity', model_cfg=model_cfg) # Pass model_cfg
        self.router = nn.Linear(config.hidden_size, 1, bias=False)
        self.aux_loss_weight = model_cfg.get('mod_aux_loss_weight', 0.01) # Get from model_cfg

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        router_logits = self.router(hidden_states).squeeze(-1)
        aux_loss = None
        if self.training:
            target_load = self.capacity
            actual_load = torch.sigmoid(router_logits).mean()
            aux_loss = self.aux_loss_weight * ((actual_load - target_load) ** 2)
        return router_logits, aux_loss, {}

```

# Directory: `src/models/sdt`

## File: `src/models/sdt/__init__.py`

```python
from .model import SDTPriorNetwork, SDTDecisionLayer, SDTRouter
from .causal_lm import SDTForCausalLM

__all__ = [
    "SDTPriorNetwork",
    "SDTDecisionLayer",
    "SDTRouter",
    "SDTForCausalLM",
]

```

## File: `src/models/sdt/causal_lm.py`

```python
import torch.nn as nn
from ..base.causal_lm import BaseForCausalLM
from ..base.block import DynamicBlock
from .model import SDTRouter, SDTDecisionLayer
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

class SDTForCausalLM(BaseForCausalLM):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._setup_layers()
    
    def _setup_layers(self):
        for i in range(0, self.config.num_hidden_layers, 2):
            self.layers.append(nn.ModuleDict({
                'decision': SDTDecisionLayer(self.config, i, self.model_params),
                'dynamic_block': DynamicBlock(self.config, i + 1),
                'router': SDTRouter(self.config, i + 1, self.model_params)
            }))

    def _forward_layers(self, hidden_states, **kwargs):
        total_aux_loss = 0
        for layer_pair in self.layers:
            decision_output = layer_pair['decision'](hidden_states, **kwargs)
            hidden_states = decision_output['posterior']
            if self.training and decision_output['prior_loss'] is not None:
                # The weight is applied in the main forward pass of the base class
                total_aux_loss += self.model_params['prior_loss_weight'] * decision_output['prior_loss'] # Use self.model_params
            
            router = layer_pair['router']
            scores, _, stats = router(**decision_output, **kwargs) # Capture stats
            log.debug(f"SDTForCausalLM: scores shape: {scores.shape}, hidden_states shape: {hidden_states.shape}")
            _, batch_idx, token_idx, gating_scores = router.select_tokens(scores, hidden_states)
            hidden_states, _, _ = layer_pair['dynamic_block'].process_selected(
                hidden_states, batch_idx, token_idx, gating_scores, use_soft_gating=self.training, **kwargs
            )
        
        return {"hidden_states": hidden_states, "aux_loss": total_aux_loss, "router_stats": stats} # Return stats

    def get_trainable_parameters(self):
        return self._create_param_groups({'router': 'router', 'prior': 'decision.prior_network'})

```

## File: `src/models/sdt/model.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
from ..base.priors import BasePriorNetwork
from ..base.routers import BaseSurpriseRouter
from ..base.block import DynamicBlock
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

class SDTPriorNetwork(BasePriorNetwork):
    """Implements the SDT prior: x + MLP(RMSNorm(x))."""
    def __init__(self, config, model_cfg: DictConfig):
        super().__init__(config, model_cfg)
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))

class SDTDecisionLayer(nn.Module):
    """Computes the original, posterior, and prior states needed for SDT routing."""
    def __init__(self, config, layer_idx: int, model_cfg: DictConfig):
        super().__init__()
        log.debug(f"SDTDecisionLayer.__init__: config.use_cache={config.use_cache}, config.attn_implementation={config.attn_implementation}")
        self.block = DynamicBlock(config, layer_idx)
        self.prior_network = SDTPriorNetwork(config, model_cfg)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        original = hidden_states
        log.debug(f"SDTDecisionLayer.forward: hidden_states shape before block: {hidden_states.shape}")

        # Flatten hidden_states to (B*T, D) before passing to Qwen2DecoderLayer
        outputs = self.block(hidden_states, **kwargs)
        posterior = outputs[0] # Extract hidden_states from the tuple
        log.debug(f"SDTDecisionLayer.forward: outputs shape (posterior): {posterior.shape}")
        prior = self.prior_network(original)
        prior_loss = F.mse_loss(prior, posterior.detach())
        return {'original': original, 'posterior': posterior, 'prior': prior, 'prior_loss': prior_loss}

class SDTRouter(BaseSurpriseRouter):
    """Implements the SDT surprise calculation by inheriting VPR logic."""
    def __init__(self, config, layer_idx: int, model_cfg: DictConfig):
        super().__init__(config, capacity_attr='sdt_capacity', model_cfg=model_cfg)

    def forward(self, original: torch.Tensor, posterior: torch.Tensor, prior: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        d = float(original.shape[-1])
        D_st = torch.sum((posterior - original).pow(2), dim=-1) / d
        D_ch = torch.sum((posterior - prior).pow(2), dim=-1) / d
        
        beta_ce = kwargs.get('beta_ce', self.model_cfg['beta_ce_init']) # Get from kwargs or model_cfg
        beta_cu = kwargs.get('beta_cu', self.model_cfg['beta_cu_init']) # Get from kwargs or model_cfg

        # Use betas passed from the training loop
        g_cont, stats = self._get_vpr_signals(D_st, D_ch, beta_ce, beta_cu)
        return g_cont, None, stats

```

# Directory: `src/models/standard`

## File: `src/models/standard/__init__.py`

```python
from .model import StandardTransformerForCausalLM

__all__ = ["StandardTransformerForCausalLM"]
```

## File: `src/models/standard/model.py`

```python
import torch.nn as nn
from transformers import Qwen2ForCausalLM, Qwen2Config


class StandardTransformerForCausalLM(Qwen2ForCausalLM):
    """Standard transformer model using Qwen2 architecture.

    This is essentially a wrapper around Qwen2ForCausalLM that ensures
    we use the exact same architecture (RMSNorm, SwiGLU MLP, RoPE, etc.)
    """

    def __init__(self, config):
        super().__init__(config)
        self._init_weights_if_needed(config)

    def _init_weights_if_needed(self, config):
        """Initialize weights using Qwen's approach if training from scratch."""
        if hasattr(config, 'init_from_scratch') and config.init_from_scratch:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights matching Qwen's initialization strategy."""
        std = getattr(self.config, 'initializer_range')

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif hasattr(module, 'weight') and hasattr(module.weight, 'data'):
            # For RMSNorm layers
            module.weight.data.fill_(1.0)

    @classmethod
    def from_pretrained_or_random(cls, model_name_or_config, from_scratch=False, **kwargs):
        """Load pretrained or initialize from scratch."""
        if from_scratch:
            if isinstance(model_name_or_config, str):
                # Load config from pretrained but initialize weights randomly
                config = Qwen2Config.from_pretrained(model_name_or_config)
            else:
                config = model_name_or_config

            config.init_from_scratch = True

            # Override any size parameters if provided
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            model = cls(config)
            return model
        else:
            # Load pretrained weights
            return cls.from_pretrained(model_name_or_config, **kwargs)

    def get_trainable_parameters(self):
        """Get parameter groups for training."""
        # All parameters are trainable in standard transformer
        return [
            {
                'params': self.parameters(),
                'lr_scale': 1.0,
                'name': 'all'
            }
        ]

```

# Directory: `src/models/stt`

## File: `src/models/stt/__init__.py`

```python
from .model import STTTransitionNetwork, STTPredictiveRouter
from .causal_lm import STTForCausalLM

__all__ = [
    "STTTransitionNetwork",
    "STTPredictiveRouter",
    "STTForCausalLM",
]
```

## File: `src/models/stt/causal_lm.py`

```python
import torch.nn as nn
from ..base.causal_lm import BaseForCausalLM
from ..base.block import DynamicBlock
from .model import STTLayer
from omegaconf import DictConfig

class STTForCausalLM(BaseForCausalLM):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._setup_layers()
    
    def _setup_layers(self):
        for i in range(self.config.num_hidden_layers):
                        if i % 2 == 1: # Dynamic STT layer
                            self.layers.append(STTLayer(self.config, i, self.model_params)) # Pass self.model_params
                        else: # Standard layer
                            self.layers.append(DynamicBlock(self.config, i))
    def _forward_layers(self, hidden_states, **kwargs):
        total_aux_loss = 0
        all_router_stats = {} # Initialize all_router_stats here
        for layer in self.layers:
            if isinstance(layer, STTLayer):
                hidden_states, aux_loss, router_stats = layer(hidden_states, **kwargs)
                if aux_loss is not None:
                    total_aux_loss += aux_loss
                all_router_stats.update(router_stats)
            else: # Standard DynamicBlock
                hidden_states = layer(hidden_states, **kwargs)[0]
            
        return {"hidden_states": hidden_states, "aux_loss": total_aux_loss, "router_stats": all_router_stats}
        
        def get_trainable_parameters(self):
            return self._create_param_groups({
                'transition_network': 'transition_network',
                'predictive_router': 'predictive_router',
                'causal_router': 'causal_router'
            })

```

## File: `src/models/stt/model.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from ..base.priors import BasePriorNetwork
from ..base.routers import BaseSurpriseRouter, CausalRouter
from ..base.block import DynamicBlock
from omegaconf import DictConfig

class STTTransitionNetwork(BasePriorNetwork):
    """Implements the STT transition network with pre-normalization: MLP(RMSNorm(x))."""
    def __init__(self, config, model_cfg: DictConfig): # Accept model_cfg
        super().__init__(config, model_cfg) # Pass model_cfg
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(x))

class STTPredictiveRouter(BaseSurpriseRouter):
    """Implements the STT surprise calculation and returns binary targets."""
    def __init__(self, config, layer_idx: int, model_cfg: DictConfig): # Accept model_cfg
        super().__init__(config, capacity_attr='stt_capacity', model_cfg=model_cfg) # Pass model_cfg

    def forward(self, actual_residual: torch.Tensor, predicted_residual: torch.Tensor, beta_ce: float, beta_cu: float, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        d = float(actual_residual.shape[-1])
        D_st = torch.sum(actual_residual.pow(2), dim=-1) / d
        D_ch = torch.sum((actual_residual - predicted_residual).pow(2), dim=-1) / d
        
        g_cont, stats = self._get_vpr_signals(D_st, D_ch, beta_ce, beta_cu)
        
        B, T = g_cont.shape
        k = max(1, int(T * self.capacity))
        _, topk_idx = g_cont.topk(k, dim=-1)
        binary_targets = torch.zeros_like(g_cont)
        binary_targets.scatter_(1, topk_idx, 1.0)

        return g_cont, binary_targets, stats

class STTLayer(nn.Module):
    """A self-contained STT Layer implementing the full teacher-student paradigm."""
    def __init__(self, config, layer_idx: int, model_cfg: DictConfig):
        super().__init__()
        self.block = DynamicBlock(config, layer_idx)
        self.transition_network = STTTransitionNetwork(config, model_cfg) # Pass model_cfg
        self.predictive_router = STTPredictiveRouter(config, layer_idx, model_cfg) # Pass model_cfg
        self.causal_router = CausalRouter(config, layer_idx, 'stt_capacity', model_cfg) # Pass model_cfg
        self.config = config
        self.model_params = model_cfg # Store model_cfg

    def forward(self, hidden_states, **kwargs):
        original_hidden = hidden_states
        processed_hidden = self.block(hidden_states, **kwargs)[0]
        
        aux_loss = None
        router_stats = {} # Initialize router_stats
        if self.training:
            actual_residual = processed_hidden - original_hidden
            
            prev_final_states = torch.cat([
                torch.zeros_like(processed_hidden[:, :1, :]),
                processed_hidden[:, :-1, :]
            ], dim=1)
            predicted_residual = self.transition_network(prev_final_states)
            
            tpn_loss = F.mse_loss(predicted_residual, actual_residual.detach())
            
            g_cont, binary_targets, pred_router_stats = self.predictive_router( # Capture stats
                actual_residual, predicted_residual, **kwargs
            )
            router_stats.update(pred_router_stats) # Update router_stats
            
            causal_logits, _, causal_router_stats = self.causal_router(original_hidden) # Capture stats
            router_stats.update(causal_router_stats) # Update router_stats
            causal_loss = F.binary_cross_entropy_with_logits(causal_logits, binary_targets.detach())
            
            aux_loss = (self.model_params['tpn_loss_weight'] * tpn_loss) + \
                       (self.model_params['causal_loss_weight'] * causal_loss)
            
            # During training, no tokens are skipped to ensure stable gradient flow
            final_hidden_states = processed_hidden
        
        else: # Inference logic
            causal_logits, _, causal_router_stats = self.causal_router(original_hidden) # Capture stats
            router_stats.update(causal_router_stats) # Update router_stats
            k = max(1, int(hidden_states.shape[1] * self.causal_router.capacity))
            _, topk_indices = causal_logits.topk(k, dim=-1)
            
            mask = torch.zeros_like(causal_logits, dtype=torch.bool).scatter_(1, topk_indices, True)
            
            # Mix original and processed states based on the routing decision
            final_hidden_states = torch.where(mask.unsqueeze(-1), processed_hidden, original_hidden)

        return final_hidden_states, aux_loss, router_stats # Return stats

```

# Directory: `src/training`

## File: `src/training/eval_utils.py`

```python
import torch
from typing import List

class LMEvalAdaptor:
    """
    A wrapper class to make a Hugging Face-style model compatible with the lm-eval harness.
    """
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._batch_size = 1 # Default, can be overridden by lm-eval
    
    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # Assumes model has a config with max_position_embeddings
        return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256 # A reasonable default for generation

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(string, **kwargs)

    def tok_decode(self, tokens: List[int], **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)

    def _model_call(self, inps: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        The main model call function for lm-eval.
        :param inps: a torch.Tensor of shape [batch, sequence_length]
        :return: a torch.Tensor of shape [batch, sequence_length, vocab_size]
        """
        with torch.no_grad():
            return self.model(inps, **kwargs).get("logits")

    def _model_generate(self, context, max_length, eos_token_id):
        # lm-eval uses this for generation tasks
        return self.model.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False, # Use greedy decoding for determinism
        )

```

## File: `src/training/utils.py`

```python
import os
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from omegaconf import DictConfig, OmegaConf, OmegaConf, OmegaConf

from transformers import (
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForCausalLM,
    get_scheduler
)
from huggingface_hub import HfApi

from ..models.mod.causal_lm import MoDForCausalLM
from ..models.sdt.causal_lm import SDTForCausalLM
from ..models.stt.causal_lm import STTForCausalLM

log = logging.getLogger(__name__)


def create_model_config(
    model_size: str,
    from_scratch: bool,
    cfg: DictConfig,
) -> Qwen2Config:
    """Create model configuration.

    Args:
        model_size: '0.5B', '1.5B', or '3B'
        from_scratch: Whether to train from scratch
        cfg: The full training config

    Returns:
        Qwen2Config with appropriate settings
    """
    pretrained_name = f"Qwen/Qwen2.5-{model_size}"

    if from_scratch:
        # Create config from scratch with Qwen2.5 specifications
        if model_size not in cfg.model.scratch_config:
            raise ValueError(f"Unknown model size for scratch training: {model_size}")

        config = Qwen2Config(
            vocab_size=cfg.model.scratch_config.vocab_size,
            max_position_embeddings=cfg.model.scratch_config.max_position_embeddings,
            rope_theta=cfg.model.scratch_config.rope_theta,
            sliding_window=cfg.model.scratch_config.sliding_window,
            **cfg.model.scratch_config[model_size]
        )
        config.init_from_scratch = True
    else:
        # Load full config from pretrained
        config = Qwen2Config.from_pretrained(pretrained_name)
        log.info(f"create_model_config: intermediate_size after from_pretrained: {config.intermediate_size}")

    # Determine attention implementation based on system config
    if cfg.system.get('use_flash_attention', False):
        config.attn_implementation = cfg.model.get('attn_implementation', 'flash_attention_2')
    else:
        config.attn_implementation = 'sdpa' # Fallback to PyTorch's native SDPA

    # Ensure _attn_implementation is consistent with the public one
    config._attn_implementation = config.attn_implementation

    # Add model-type specific configurations from the provided config
    # Copy all relevant model parameters from cfg.model to the Qwen2Config object
    for key, value in cfg.model.items():
        if key not in ['scratch_config', 'size', 'type', 'pretrained_model_name_or_path', 'use_flash_attention_2', 'attn_implementation', 'use_cache', 'tie_word_embeddings', 'intermediate_size', 'params', 'beta_schedule']: # Added 'params' and 'beta_schedule' to exclusion
            setattr(config, key, value)
    log.info(f"create_model_config: intermediate_size after cfg.model.items() loop: {config.intermediate_size}")


    # Explicitly set head_dim for consistency
    config.head_dim = config.hidden_size // config.num_attention_heads

    # Ensure num_key_value_heads matches num_attention_heads for rotary embeddings
    config.num_key_value_heads = config.num_attention_heads

    # Common settings for all models
    config.use_cache = cfg.model.get('use_cache', True)
    config.tie_word_embeddings = cfg.model.get('tie_word_embeddings', True)

    # Platform-specific settings (use_flash_attention is now handled above for attn_implementation)
    config.torch_dtype = cfg.system.get('torch_dtype', 'float32')

    return config


def create_model(
    model_type: str,
    model_size: str,
    from_scratch: bool,
    cfg: DictConfig,
) -> torch.nn.Module:
    """Create and initialize model.

    Args:
        model_type: 'standard', 'mod', 'sdt', or 'stt' # Updated model types
        model_size: '0.5B', '1.5B', or '3B'
        from_scratch: Whether to train from scratch
        cfg: The full training config

    Returns:
        Initialized model
    """
    config = create_model_config(model_size, from_scratch, cfg)
    pretrained_name = f"Qwen/Qwen2.5-{model_size}"
    torch_dtype = getattr(torch, cfg.system.get('torch_dtype', 'float32'))

    if model_type == "standard":
        from ..models.standard.model import StandardTransformerForCausalLM
        return StandardTransformerForCausalLM.from_pretrained(pretrained_name, torch_dtype=torch_dtype)
    
    model_class_map = {
        "mod": MoDForCausalLM,
        "sdt": SDTForCausalLM,
        "stt": STTForCausalLM,
    }
    
    if model_type in model_class_map:
        model_kwargs = {}
        if hasattr(cfg.model, 'params') and isinstance(cfg.model.params, DictConfig):
            for param_name in [
                'sdt_capacity', 'prior_loss_weight', 'causal_loss_weight',
                'beta_ce_init', 'beta_cu_init', 'cu_detection_multiplier_init',
                'ce_criterion_offset_init', 'prior_ffn_intermediate_size_factor',
                'mod_capacity', 'mod_aux_loss_weight', 'tdtf_capacity',
                'tpn_loss_weight', 'ma_window', 'o_ce_init', 'm_cu_init', 'stt_capacity'
            ]:
                if hasattr(cfg.model.params, param_name):
                    model_kwargs[param_name] = getattr(cfg.model.params, param_name)
        
        print(f"DEBUG: model_kwargs before model creation: {model_kwargs}")
        model = model_class_map[model_type](config, **model_kwargs)
        if not from_scratch:
            log.info(f"Initializing {model_type.upper()} from pretrained {pretrained_name}")
            base_model = Qwen2ForCausalLM.from_pretrained(pretrained_name, torch_dtype=torch_dtype)
            model.copy_weights_from_pretrained(base_model)
            del base_model
        return model
    
    raise ValueError(f"Unknown model type: {model_type}")


def setup_optimizer_and_scheduler(model: torch.nn.Module, cfg: DictConfig, num_training_steps: int):
    """Setup optimizers and schedulers based on parameter groups from the model."""
    param_groups = model.get_trainable_parameters()
    optimizers, schedulers = {}, {}
    optimizer_cfg = cfg.training.optimizer
    common_kwargs = {"betas": (optimizer_cfg.adam_beta1, optimizer_cfg.adam_beta2), "eps": optimizer_cfg.adam_epsilon, "weight_decay": optimizer_cfg.weight_decay}

    for group in param_groups:
        name = group['name']
        params = group['params']
        
        # Read LR from the new 'lrs' map, with a fallback to the default lr
        lr = optimizer_cfg.lrs.get(name, optimizer_cfg.lr)
        log.info(f"Creating optimizer for group '{name}' with {sum(p.numel() for p in params)} params and LR {lr:.2e}")

        opt = torch.optim.AdamW([{'params': params}], lr=lr, **common_kwargs)
        optimizers[name] = opt
        schedulers[name] = get_scheduler(
            optimizer_cfg.scheduler, optimizer=opt,
            num_warmup_steps=int(num_training_steps * optimizer_cfg.warmup_ratio),
            num_training_steps=num_training_steps
        )
    return optimizers, schedulers


def save_checkpoint(
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    schedulers: Dict[str, Any],
    epoch: int,
    step: int,
    best_loss: float,
    save_path: Path
) -> None:
    """Save training checkpoint."""
    save_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = save_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    # Save training state
    state_path = save_path / "training_state.pt"
    optimizer_states = {name: opt.state_dict() for name, opt in optimizers.items() if opt is not None}
    scheduler_states = {name: sch.state_dict() for name, sch in schedulers.items() if sch is not None}

    torch.save({
        'optimizer_states': optimizer_states,
        'scheduler_states': scheduler_states,
        'epoch': epoch,
        'step': step,
        'best_loss': best_loss,
    }, state_path)

    log.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    schedulers: Dict[str, Any],
    load_path: Path
) -> Dict[str, Any]:
    """Load training checkpoint."""
    # Load model
    model_path = load_path / "model.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        log.info(f"Model loaded from {model_path}")

    # Load training state
    state_path = load_path / "training_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu')
        
        # Load optimizer states
        optimizer_states = state.get('optimizer_states', {})
        for name, opt in optimizers.items():
            if opt is not None and name in optimizer_states:
                opt.load_state_dict(optimizer_states[name])

        # Load scheduler states
        scheduler_states = state.get('scheduler_states', {})
        for name, sch in schedulers.items():
            if sch is not None and name in scheduler_states:
                sch.load_state_dict(scheduler_states[name])

        log.info(f"Training state loaded from {state_path}")
        return state

    return {'epoch': 0, 'step': 0, 'best_loss': float('inf')}


def push_to_hub(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    hub_model_id: str,
    commit_message: str = "Add new model",
    private: bool = False,
) -> None:
    """Push model and tokenizer to Hugging Face Hub."""
    log.info(f"Pushing model to Hugging Face Hub: {hub_model_id}")
    api = HfApi()

    # Save model and tokenizer locally first
    model_dir = Path("temp_hub_upload")
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Push to hub
    api.upload_folder(
        folder_path=model_dir,
        repo_id=hub_model_id,
        commit_message=commit_message,
        private=private,
    )
    log.info("Model successfully pushed to Hub!")

    # Clean up local files
    import shutil
    shutil.rmtree(model_dir)

def evaluate_perplexity(model, dataloader, accelerator):
    """Calculates validation loss and perplexity."""
    model.eval()
    losses = []
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        
        loss_key = "lm_loss" if "lm_loss" in outputs else "loss"
        loss = outputs[loss_key]
        losses.append(accelerator.gather(loss.repeat(batch["input_ids"].shape[0])))

    avg_loss = torch.mean(torch.cat(losses))
    perplexity = torch.exp(avg_loss)
    
    model.train() # Reset model to training mode
    return avg_loss.item(), perplexity.item()
    return avg_loss.item(), perplexity.item()
```

# Directory: ``

## File: `train.py`

```python
#!/usr/bin/env python3
"""Unified training script for all model architectures."""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb
from peft import LoraConfig, get_peft_model

from src.data.mixed_dataset import MixedDataset
from src.training.utils import (
    create_model,
    save_checkpoint,
    setup_optimizer_and_scheduler,
    evaluate_perplexity,
)

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Resolved logging level from config: {cfg.logging.level}")
    logging.basicConfig(level=cfg.logging.level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Explicitly set the level for the root logger
    logging.getLogger().setLevel(cfg.logging.level)

    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Setup accelerator
    accelerator = Accelerator(
        mixed_precision=cfg.system.precision,
        gradient_accumulation_steps=cfg.training.accumulate_grad_batches,
    )

    # Initialize Weights & Biases
    if cfg.logging.wandb.enabled and accelerator.is_main_process:
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=cfg.run.name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Load tokenizer
    tokenizer_path = cfg.data.tokenizer_name
    log.info(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create data module
    log.info("Setting up data module...")
    datamodule = MixedDataset(
        dataset_configs=cfg.data.dataset_configs,
        tokenizer_name=cfg.data.tokenizer_name,
        block_size=cfg.data.block_size,
        batch_size=cfg.data.batch_size,
        validation_split_percentage=cfg.data.validation_split_percentage,
    )
    datamodule.setup()

    # Create data loaders
    log.info("Creating data loaders...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = torch.utils.data.DataLoader(
        datamodule.train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.system.num_workers,
        pin_memory=cfg.system.pin_memory,
        drop_last=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        datamodule.val_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.system.num_workers,
        pin_memory=cfg.system.pin_memory,
        drop_last=False,
    )

    # Create model
    log.info(f"Creating {cfg.model.type} model ({cfg.model.size}, from_scratch={cfg.training.mode == 'scratch'})")
    model = create_model(
        cfg.model.type,
        cfg.model.size,
        cfg.training.mode == 'scratch',
        cfg
    )

    # Apply LoRA if enabled
    if cfg.peft.enabled:
        log.info("Applying LoRA to the model.")
        peft_config = LoraConfig(**cfg.peft.config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {total_params/1e6:.1f}M params ({trainable_params/1e6:.1f}M trainable)")

    # Enable gradient checkpointing if configured
    if cfg.training.gradient_checkpointing:
        log.info("Enabling gradient checkpointing on the model.")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Setup training
    steps_per_epoch = len(train_loader) // cfg.training.accumulate_grad_batches
    num_training_steps = steps_per_epoch * cfg.training.num_epochs

    def compute_beta_values(step: int, total_steps: int) -> Tuple[float, float]:
        """Computes scheduled beta values for STT/SDT models."""
        if cfg.model.type not in ['sdt', 'stt']:
            return 0.0, 0.0 # Return dummy values if not applicable
        
        # This check now correctly belongs here
        if "beta_schedule" not in cfg.model:
            raise ValueError(f"{cfg.model.type.upper()} model requires a 'beta_schedule' block in the config.")
        
        sched_cfg = cfg.model.beta_schedule
        warmup = int(sched_cfg.warmup_steps)
        stype = sched_cfg.type

        if step <= warmup:
            r = 0.0
        else:
            denom = max(1, total_steps - warmup)
            r = min(1.0, (step - warmup) / denom)

        def slinear(s0, s1): return s0 + r * (s1 - s0)
        def scos(s0, s1):
            import math
            return s0 + 0.5 * (1.0 - math.cos(math.pi * r)) * (s1 - s0)

        f = slinear if stype == "linear" else scos
        beta_ce = f(float(sched_cfg.beta_ce_start), float(sched_cfg.beta_ce_end))
        beta_cu = f(float(sched_cfg.beta_cu_start), float(sched_cfg.beta_cu_end))
        return beta_ce, beta_cu

    optimizers_dict, schedulers_dict = setup_optimizer_and_scheduler(model, cfg, num_training_steps)

    optimizers_to_prepare = list(optimizers_dict.values())
    schedulers_to_prepare = list(schedulers_dict.values())

    prepared_items = accelerator.prepare(
        model,
        *optimizers_to_prepare,
        train_loader, eval_loader,
        *schedulers_to_prepare
    )

    model = prepared_items[0]
    current_idx = 1
    
    for name in optimizers_dict.keys():
        optimizers_dict[name] = prepared_items[current_idx]
        current_idx += 1
    
    train_loader = prepared_items[current_idx]
    current_idx += 1
    eval_loader = prepared_items[current_idx]
    current_idx += 1

    for name in schedulers_dict.keys():
        schedulers_dict[name] = prepared_items[current_idx]
        current_idx += 1

    # Training loop
    log.info("Starting training...")
    global_step = 0
    best_eval_loss = float('inf')

    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_main_process)

    for epoch in range(cfg.training.num_epochs):
        model.train()
        
        for opt in optimizers_dict.values():
            opt.zero_grad()

        for step, batch in enumerate(train_loader):
            if global_step >= num_training_steps:
                break
            
            with accelerator.accumulate(model):
                beta_ce, beta_cu = compute_beta_values(global_step, num_training_steps)

                forward_kwargs = {}
                if cfg.model.type in ["sdt", "stt"]:
                    forward_kwargs["beta_ce"] = beta_ce
                    forward_kwargs["beta_cu"] = beta_cu

                outputs = model(**batch, **forward_kwargs)
                loss = outputs['loss']
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if cfg.training.use_gradient_clipping:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val)

                    for opt in optimizers_dict.values():
                        opt.step()
                    for sch in schedulers_dict.values():
                        sch.step()
                    for opt in optimizers_dict.values():
                        opt.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    log_metrics = {
                        "train/loss": loss.item(),
                        "train/lm_loss": outputs.get('lm_loss', torch.tensor(0.0)).item(),
                    }
                    for key, value in outputs.items():
                        if "loss" in key and key != "loss":
                            log_metrics[f"train/{key}"] = value.item()
                        elif "router_stats" in key and isinstance(value, dict):
                            for stat_key, stat_value in value.items():
                                if isinstance(stat_value, (float, int)):
                                    log_metrics[f"train/router_stats/{stat_key}"] = stat_value
                                elif isinstance(stat_value, list) and len(stat_value) > 0:
                                    log_metrics[f"train/router_stats/{stat_key}_avg"] = sum(stat_value) / len(stat_value)

                    if cfg.model.type == "stt" or cfg.model.type == "sdt":
                        log_metrics["train/beta_ce"] = beta_ce
                        log_metrics["train/beta_cu"] = beta_cu
                        # Log o_ce_pos and m_cu_pos from router_stats if present
                        if "router_stats" in outputs and "o_ce_pos" in outputs["router_stats"]:
                            log_metrics["train/router_stats/o_ce_pos"] = outputs["router_stats"]["o_ce_pos"]
                            log_metrics["train/router_stats/m_cu_pos"] = outputs["router_stats"]["m_cu_pos"]

                    if cfg.logging.wandb.enabled and wandb.run is not None:
                        wandb.log(log_metrics, step=global_step)
                    accelerator.print(
                        f"Epoch {epoch}, Step {global_step+1}: "
                        f"Loss = {loss.item():.4f}, "
                        f"LM Loss = {outputs.get('lm_loss', torch.tensor(0.0)).item():.4f}"
                    )

            # Evaluation and checkpointing
            if (global_step) % cfg.training.eval_interval == 0:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)

                val_loss, val_perplexity = evaluate_perplexity(unwrapped_model, eval_loader, accelerator)

                if accelerator.is_main_process:
                    if cfg.logging.wandb.enabled and wandb.run is not None:
                        wandb.log({
                            "val/loss": val_loss,
                            "val/perplexity": val_perplexity,
                        }, step=global_step)
                    accelerator.print(f"Validation Loss: {val_loss:.4f}, Validation Perplexity: {val_perplexity:.2f}")

                    if val_loss < best_eval_loss:
                        best_eval_loss = val_loss
                        save_checkpoint(
                            unwrapped_model,
                            optimizers_dict,
                            schedulers_dict,
                            epoch,
                            global_step,
                            best_eval_loss,
                            Path(cfg.run.output_dir) / "best_model"
                        )
                accelerator.wait_for_everyone()

                if cfg.training.max_steps > 0 and global_step >= cfg.training.max_steps:
                    log.info(f"Reached max steps ({cfg.training.max_steps})")
                    break

        if cfg.training.max_steps > 0 and global_step >= cfg.training.max_steps:
            break

    # Save final model
    save_path = Path(cfg.run.output_dir) / "final_model"
    save_checkpoint(accelerator.unwrap_model(model), 
                    optimizers_dict,
                    schedulers_dict,
                    epoch, global_step, best_eval_loss, save_path)

    # Push to Hugging Face Hub if enabled
    if cfg.push_to_hub.enabled and accelerator.is_main_process:
        from src.training.utils import push_to_hub
        push_to_hub(
            model=accelerator.unwrap_model(model),
            tokenizer=tokenizer,
            hub_model_id=cfg.push_to_hub.repo_id,
            commit_message=cfg.push_to_hub.commit_message,
            private=cfg.push_to_hub.private,
        )

    # End of training
    if accelerator.is_main_process:
        log.info("Training complete!")
        if cfg.logging.wandb.enabled:
            wandb.finish()


if __name__ == "__main__":
    main()

```
