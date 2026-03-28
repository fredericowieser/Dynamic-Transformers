import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from huggingface_hub import HfApi
from omegaconf import DictConfig, OmegaConf
from transformers import (AutoTokenizer, Qwen2Config, Qwen2ForCausalLM,
                          get_scheduler)

from ..models.configs import MoDConfig, SDTConfig, StandardConfig, STTConfig
from ..models.mod.model import MoDForCausalLM
from ..models.sdt.model import SDTForCausalLM
from ..models.stt.model import STTForCausalLM

log = logging.getLogger(__name__)


def create_model(model_type: str, cfg: DictConfig) -> torch.nn.Module:
    """Create and initialize model based on the unified config."""
    model_cfg = cfg.model
    from_scratch = model_cfg.from_scratch
    torch_dtype = getattr(torch, cfg.system.get("torch_dtype", "float32"))

    config_class_map = {
        "standard": StandardConfig,
        "mod": MoDConfig,
        "sdt": SDTConfig,
        "stt": STTConfig,
    }
    config_class = config_class_map.get(model_type, Qwen2Config)

    # Create base config
    if from_scratch:
        if model_cfg.size not in model_cfg.scratch_config:
            raise ValueError(f"Unknown model size for scratch: {model_cfg.size}")
        config = config_class(
            **model_cfg.scratch_config[model_cfg.size],
            vocab_size=model_cfg.scratch_config.vocab_size,
            max_position_embeddings=model_cfg.scratch_config.max_position_embeddings,
            rope_theta=model_cfg.scratch_config.rope_theta,
            sliding_window=model_cfg.scratch_config.sliding_window,
        )
    else:
        config = config_class.from_pretrained(model_cfg.pretrained_model_name_or_path)

    # Dynamically update config with all parameters from the 'model' section    # We convert the OmegaConf object to a standard python dict to prevent
    # serialization errors when saving the config.
    model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)
    for key, value in model_cfg_dict.items():
        setattr(config, key, value)

    config.model_type = model_type

    # Handle special cases like attention implementation
    if cfg.system.get("use_flash_attention", False):
        config.attn_implementation = model_cfg.get("attn_implementation", "flash_attention_2")
    else:
        config.attn_implementation = "eager"
    config._attn_implementation = config.attn_implementation

    # Instantiate the correct model
    log.debug(f"Final config attributes before model instantiation: {config.__dict__}")
    model_class_map = {
        "standard": "src.models.standard.model.StandardTransformerForCausalLM",
        "mod": "src.models.mod.model.MoDForCausalLM",
        "sdt": "src.models.sdt.model.SDTForCausalLM",
        "stt": "src.models.stt.model.STTForCausalLM",
    }
    if model_type not in model_class_map:
        raise ValueError(f"Unknown model type: {model_type}")

    def get_class(class_path: str):
        from importlib import import_module

        module_path, class_name = class_path.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, class_name)

    model_class = get_class(model_class_map[model_type])

    # Pass the entire model config dict as kwargs
    model_kwargs = OmegaConf.to_container(model_cfg, resolve=True)
    log.debug(f"Instantiating {model_class.__name__} with kwargs: {model_kwargs}")

    if from_scratch:
        model = model_class(config, **model_kwargs)
    else:
        # For pretrained, we still need to load the base weights
        model = model_class(config, **model_kwargs)
        log.info(
            f"Initializing {model_type.upper()} from pretrained {model_cfg.pretrained_model_name_or_path}"
        )
        base_model = Qwen2ForCausalLM.from_pretrained(
            model_cfg.pretrained_model_name_or_path, torch_dtype=torch_dtype
        )

        # This method needs to exist on the model class
        if hasattr(model, "copy_weights_from_pretrained"):
            model.copy_weights_from_pretrained(base_model)
        else:
            log.warning(
                f"Model {model_type} does not have 'copy_weights_from_pretrained' method. Weight transfer might be incomplete."
            )
            # Fallback to simple load, might fail if architectures differ
            model.load_state_dict(base_model.state_dict(), strict=False)

        del base_model

    return model


def calculate_vram_optimized_batch_size(
    cfg: DictConfig, model: torch.nn.Module, accelerator
) -> Tuple[int, int]:
    """
    Automatically computes the optimal per-device batch size and gradient accumulation steps
    to achieve a target total batch size of 64, while fitting within available VRAM.
    """
    import math

    target_total_batch_size = 64
    num_processes = accelerator.num_processes
    seq_len = cfg.data.block_size

    # Extract model dimensions
    vocab_size = getattr(model.config, "vocab_size", 151936)
    if hasattr(model.config, "hidden_size"):
        hidden_size = model.config.hidden_size
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
    else:
        size_cfg = cfg.model.scratch_config.get(cfg.model.size, {})
        hidden_size = size_cfg.get("hidden_size", 896)
        num_layers = size_cfg.get("num_hidden_layers", 24)
        num_heads = size_cfg.get("num_attention_heads", 14)

    # Count parameters
    params = sum(p.numel() for p in model.parameters())

    # 1. Estimate Fixed Overhead (Weights + Gradients + Optimizer States)
    # AdamW with fp32 master weights: 16 bytes per param
    # Using a 1.3x buffer for distributed overhead and NCCL buffers
    fixed_overhead_gb = (params * 16 * 1.3) / (1024**3)

    # 2. Estimate Activation Memory per Sample
    # Eager attention matrix: [H, T, T]
    attn_matrix_mem_gb = (num_heads * seq_len * seq_len * 2) / (1024**3)
    
    # LOGITS MEMORY: [T, V]
    # This is often the largest activation for large vocab models!
    # [1024, 151936] * 2 bytes (bf16) = 300MB per sample
    # During CrossEntropy, this can be converted to fp32 = 600MB per sample
    logits_mem_per_sample_gb = (seq_len * vocab_size * 4) / (1024**3)
    
    is_checkpointing = getattr(cfg.training, "gradient_checkpointing", True)
    is_eager = getattr(model.config, "attn_implementation", "eager") == "eager"
    eager_overhead_factor = 4.0 if is_eager else 1.2
    
    if is_checkpointing:
        # Boundary: B * T * D * L * 2 bytes
        boundary_mem_per_sample = (seq_len * hidden_size * num_layers * 2) / (1024**3)
        # Peak: One layer recomputation (Factor 8) + Attention intermediates
        peak_layer_mem = (seq_len * hidden_size * 8 * 2) / (1024**3) + (attn_matrix_mem_gb * eager_overhead_factor)
        mem_per_sample_gb = boundary_mem_per_sample + peak_layer_mem + logits_mem_per_sample_gb
    else:
        # No checkpointing: All layers + all attention
        mem_per_sample_gb = (seq_len * hidden_size * num_layers * 14 * 2 * eager_overhead_factor) / (1024**3) + \
                           (attn_matrix_mem_gb * num_layers * eager_overhead_factor) + \
                           logits_mem_per_sample_gb

    # 3. Get Available VRAM
    if torch.cuda.is_available():
        device_id = accelerator.local_process_index
        total_mem_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        # Safety margin: 35% for kernels, fragmentation, dataloader, and workspace
        safe_limit_gb = total_mem_gb * 0.65
        available_vram_gb = safe_limit_gb - fixed_overhead_gb
    else:
        return cfg.data.batch_size, cfg.training.accumulate_grad_batches

    if available_vram_gb <= 0:
        log.warning(
            f"Model overhead ({fixed_overhead_gb:.2f}GB) exceeds safe GPU limit ({safe_limit_gb:.2f}GB). "
            "Setting batch size to 1 and hoping for the best."
        )
        per_device_batch = 1
    else:
        # Max samples that can fit in available VRAM
        per_device_batch = int(available_vram_gb // mem_per_sample_gb)
        
        # Max samples needed per device to hit target without accumulation
        needed_per_device = math.ceil(target_total_batch_size / num_processes)
        
        # Final per-device batch size
        per_device_batch = max(1, min(per_device_batch, needed_per_device))

        # Prefer power of 2
        if per_device_batch > 1:
            per_device_batch = 2 ** int(math.log2(per_device_batch))

    # 4. Calculate Accumulation Steps
    total_batch_one_step = per_device_batch * num_processes
    accumulate_steps = math.ceil(target_total_batch_size / total_batch_one_step)

    if accelerator.is_main_process:
        log.info(f"--- VRAM Auto-Configurator (Hyper-Conservative) ---")
        log.info(f"GPU Total: {total_mem_gb:.2f}GB | Safe Limit: {safe_limit_gb:.2f}GB")
        log.info(f"Model Overhead: {fixed_overhead_gb:.2f}GB | Attn Implementation: {getattr(model.config, 'attn_implementation', 'unknown')}")
        log.info(f"Estimated Mem/Sample: {mem_per_sample_gb*1024:.2f}MB")
        log.info(f"Selected Per-Device Batch: {per_device_batch}")
        log.info(f"Selected Accumulation Steps: {accumulate_steps}")
        log.info(f"Resulting Total Batch Size: {per_device_batch * num_processes * accumulate_steps}")
        log.info(f"--------------------------------------------------")

    return per_device_batch, accumulate_steps


def setup_optimizer_and_scheduler(
    model: torch.nn.Module, cfg: DictConfig, num_training_steps: int, accelerator
):
    """Setup optimizers and schedulers based on parameter groups from the model."""
    # Simplify to a single optimizer group for all parameters
    optimizer_cfg = cfg.training.optimizer
    common_kwargs = {
        "betas": (optimizer_cfg.adam_beta1, optimizer_cfg.adam_beta2),
        "eps": optimizer_cfg.adam_epsilon,
        "weight_decay": optimizer_cfg.weight_decay,
    }

    lr = optimizer_cfg.lr
    log.info(f"Creating single optimizer for all model parameters with LR {lr:.2e}")

    # Create a single optimizer for the entire model
    # We filter out params that don't require grad just in case
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, **common_kwargs)

    optimizers = {"model": opt}
    schedulers = {
        "model": get_scheduler(
            optimizer_cfg.scheduler,
            optimizer=opt,
            num_warmup_steps=int(num_training_steps * optimizer_cfg.warmup_ratio),
            num_training_steps=num_training_steps,
        )
    }

    return optimizers, schedulers


def save_checkpoint(
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    schedulers: Dict[str, Any],
    epoch: int,
    step: int,
    best_loss: float,
    save_path: Path,
) -> None:
    """Save training checkpoint."""
    save_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = save_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    # Save model config (permanent fix)
    model.config.save_pretrained(save_path)

    # Save training state
    state_path = save_path / "training_state.pt"
    optimizer_states = {
        name: opt.state_dict() for name, opt in optimizers.items() if opt is not None
    }
    scheduler_states = {
        name: sch.state_dict() for name, sch in schedulers.items() if sch is not None
    }

    torch.save(
        {
            "optimizer_states": optimizer_states,
            "scheduler_states": scheduler_states,
            "epoch": epoch,
            "step": step,
            "best_loss": best_loss,
        },
        state_path,
    )

    log.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    schedulers: Dict[str, Any],
    load_path: Path,
) -> Dict[str, Any]:
    """Load training checkpoint."""
    # Load model
    model_path = load_path / "model.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        log.info(f"Model loaded from {model_path}")

    # Load training state
    state_path = load_path / "training_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu")

        # Load optimizer states
        optimizer_states = state.get("optimizer_states", {})
        for name, opt in optimizers.items():
            if opt is not None and name in optimizer_states:
                opt.load_state_dict(optimizer_states[name])

        # Load scheduler states
        scheduler_states = state.get("scheduler_states", {})
        for name, sch in schedulers.items():
            if sch is not None and name in scheduler_states:
                sch.load_state_dict(scheduler_states[name])

        log.info(f"Training state loaded from {state_path}")
        return state

    return {"epoch": 0, "step": 0, "best_loss": float("inf")}


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


def calculate_metrics(model, batch, global_step=0, max_steps=1):
    """Performs a forward pass and returns a dictionary of metrics."""
    outputs = model(
        input_ids=batch.get("input_ids"),
        attention_mask=batch.get("attention_mask"),
        labels=batch.get("labels"),
        global_step=global_step,
        max_steps=max_steps,
    )

    metrics = {}

    loss = outputs.get("loss")
    lm_loss = outputs.get("lm_loss")

    if loss is None:
        loss = lm_loss

    if lm_loss is None:
        lm_loss = loss

    metrics["loss"] = loss
    metrics["lm_loss"] = lm_loss

    # Add other metrics from outputs
    for key, value in outputs.items():
        if key not in ["loss", "lm_loss", "logits"]:
            metrics[key] = value

    return metrics


def evaluate_perplexity(model, dataloader, accelerator):
    """Calculates validation loss and perplexity, and collects auxiliary metrics."""
    model.eval()
    losses = []
    all_unscaled_losses = {}
    all_router_stats = {}

    for batch in dataloader:
        with torch.no_grad():
            metrics = calculate_metrics(model, batch)

        loss = metrics.get("lm_loss")
        if loss is not None:
            losses.append(accelerator.gather(loss.repeat(batch["input_ids"].shape[0])))

        # Collect unscaled losses
        if "unscaled_losses" in metrics:
            for k, v in metrics["unscaled_losses"].items():
                if k not in all_unscaled_losses:
                    all_unscaled_losses[k] = []
                all_unscaled_losses[k].append(v)

        # Collect router stats
        if "router_stats" in metrics:
            for k, v in metrics["router_stats"].items():
                if k not in all_router_stats:
                    all_router_stats[k] = []
                all_router_stats[k].append(v)

    if not losses:
        return 0.0, 1.0, {}, {}  # Return empty dicts if no losses

    avg_loss = torch.mean(torch.cat(losses))
    perplexity = torch.exp(avg_loss)

    # Aggregate unscaled losses
    aggregated_unscaled_losses = {
        k: (
            torch.mean(torch.stack(v)).item()
            if v and isinstance(v[0], torch.Tensor)
            else (sum(v) / len(v) if v else 0.0)
        )
        for k, v in all_unscaled_losses.items()
    }

    # Aggregate router stats (mean for now, can be more sophisticated if needed)
    aggregated_router_stats = {
        k: (
            torch.mean(torch.stack(v)).item()
            if v and isinstance(v[0], torch.Tensor)
            else (sum(v) / len(v) if v else 0.0)
        )
        for k, v in all_router_stats.items()
    }

    model.train()  # Reset model to training mode
    return avg_loss.item(), perplexity.item(), aggregated_unscaled_losses, aggregated_router_stats


def save_wandb_info(wandb_run, save_path: Path):
    """Saves essential wandb run info to a file."""
    if wandb_run is None:
        return
    info = {
        "project": wandb_run.project,
        "entity": wandb_run.entity,
        "run_id": wandb_run.id,
        "run_name": wandb_run.name,
    }
    info_path = save_path / "wandb_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    log.info(f"Wandb info saved to {info_path}")
