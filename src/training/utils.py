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
    Trial-based VRAM auto-configurator: runs a small forward+backward pass on GPU
    to measure actual memory per sample, then picks the largest safe batch size.
    """
    import math
    from contextlib import nullcontext

    target_total_batch_size = 64
    num_processes = accelerator.num_processes

    if not torch.cuda.is_available():
        return cfg.data.batch_size, cfg.training.accumulate_grad_batches

    device = torch.device(f"cuda:{accelerator.local_process_index}")
    seq_len = cfg.data.block_size
    vocab_size = getattr(model.config, "vocab_size", 151936)

    # Determine autocast dtype to match training conditions
    precision = cfg.system.get("precision", "bf16")
    if precision == "bf16":
        autocast_dtype = torch.bfloat16
    elif precision == "fp16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    # Fixed overhead NOT captured by the trial (optimizer + DDP created later)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    # AdamW: fp32 master weights + momentum + variance = 12 bytes per trainable param
    # DDP: gradient allreduce buckets ≈ param_size
    optimizer_bytes = trainable_params * 12
    ddp_bytes = all_params * 2
    fixed_overhead = optimizer_bytes + ddp_bytes

    # --- Run trial on GPU ---
    original_device = next(model.parameters()).device
    was_training = model.training

    try:
        model.to(device)
    except torch.cuda.OutOfMemoryError:
        log.warning("Model doesn't fit on GPU. Using batch_size=1.")
        return 1, max(1, math.ceil(target_total_batch_size / num_processes))

    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    model_mem = torch.cuda.memory_allocated(device)

    trial_batch = 2
    dummy_input = torch.randint(0, vocab_size, (trial_batch, seq_len), device=device)
    dummy_labels = torch.randint(0, vocab_size, (trial_batch, seq_len), device=device)

    trial_success = False
    per_sample_bytes = 0
    peak_mem = 0
    param_grad_mem = 0
    trial_outputs = None
    trial_loss = None

    try:
        amp_ctx = torch.cuda.amp.autocast(dtype=autocast_dtype) if autocast_dtype else nullcontext()
        with amp_ctx:
            trial_outputs = model(input_ids=dummy_input, labels=dummy_labels)
            trial_loss = trial_outputs["loss"] if isinstance(trial_outputs, dict) else trial_outputs.loss
            trial_loss.backward()

        torch.cuda.synchronize(device)
        peak_mem = torch.cuda.max_memory_allocated(device)

        param_grad_mem = sum(
            p.grad.numel() * p.grad.element_size()
            for p in model.parameters() if p.grad is not None
        )
        # Activation memory = peak - model weights - parameter gradients
        activation_mem = peak_mem - model_mem - param_grad_mem
        per_sample_bytes = max(activation_mem, 0) / trial_batch
        trial_success = True

    except torch.cuda.OutOfMemoryError:
        log.warning("Trial forward/backward OOMed with batch=2. Defaulting to batch_size=1.")

    # Clean up
    model.zero_grad(set_to_none=True)
    del dummy_input, dummy_labels, trial_outputs, trial_loss
    torch.cuda.empty_cache()
    model.to(original_device)
    if not was_training:
        model.eval()
    torch.cuda.empty_cache()

    if not trial_success or per_sample_bytes <= 0:
        per_device_batch = 1
        accumulate_steps = max(1, math.ceil(target_total_batch_size / (per_device_batch * num_processes)))
        if accelerator.is_main_process:
            log.info("--- VRAM Auto-Configurator (Fallback) ---")
            log.info(f"Per-Device Batch: {per_device_batch}, Accumulation: {accumulate_steps}")
        return per_device_batch, accumulate_steps

    # Calculate optimal batch size
    total_gpu_mem = torch.cuda.get_device_properties(device).total_memory
    usable_mem = total_gpu_mem * 0.90  # 10% safety for fragmentation/CUDA workspace

    # Available = usable - model weights - param gradients - optimizer states - DDP buffers
    available = usable_mem - model_mem - param_grad_mem - fixed_overhead

    if available <= 0:
        per_device_batch = 1
    else:
        per_device_batch = int(available / per_sample_bytes)
        needed = math.ceil(target_total_batch_size / num_processes)
        per_device_batch = max(1, min(per_device_batch, needed))
        if per_device_batch > 1:
            per_device_batch = 2 ** int(math.log2(per_device_batch))

    total_batch = per_device_batch * num_processes
    accumulate_steps = max(1, math.ceil(target_total_batch_size / total_batch))

    if accelerator.is_main_process:
        log.info("--- VRAM Auto-Configurator (Trial-Based) ---")
        log.info(f"GPU Total: {total_gpu_mem/(1024**3):.2f}GB | Usable (90%%): {usable_mem/(1024**3):.2f}GB")
        log.info(f"Model Weights: {model_mem/(1024**3):.2f}GB | Param Grads: {param_grad_mem/(1024**3):.2f}GB")
        log.info(f"Fixed Overhead (optim+DDP): {fixed_overhead/(1024**3):.2f}GB")
        log.info(f"Trial Peak (batch={trial_batch}): {peak_mem/(1024**3):.2f}GB")
        log.info(f"Measured Mem/Sample: {per_sample_bytes/(1024**2):.2f}MB")
        log.info(f"Available for Activations: {available/(1024**3):.2f}GB")
        log.info(f"Selected Per-Device Batch: {per_device_batch}")
        log.info(f"Selected Accumulation Steps: {accumulate_steps}")
        log.info(f"Resulting Total Batch Size: {per_device_batch * num_processes * accumulate_steps}")
        log.info("---------------------------------------------")

    return per_device_batch, accumulate_steps


def setup_optimizer_and_scheduler(
    model: torch.nn.Module, cfg: DictConfig, num_training_steps: int, accelerator
):
    """Setup optimizers and schedulers with parameter grouping for per-component LRs."""
    optimizer_cfg = cfg.training.optimizer
    lrs = optimizer_cfg.get("lrs", {})
    default_lr = optimizer_cfg.lr
    
    # Define common optimizer arguments
    common_kwargs = {
        "betas": (optimizer_cfg.adam_beta1, optimizer_cfg.adam_beta2),
        "eps": optimizer_cfg.adam_epsilon,
        "weight_decay": optimizer_cfg.weight_decay,
    }

    # Parameter Grouping Logic
    param_groups = []
    
    # 1. Component-specific groups based on name matching
    # Map from config key to internal module name patterns
    component_mapping = {
        "mod_causal_router": ["causal_router"],
        "mod_router": ["router"],
        "sdt_causal_router": ["causal_router"],
        "sdt_prior": ["prior"],
        "sdt_router": ["router"],
        "stt_causal_router": ["causal_router"],
        "stt_predictive_router": ["predictive_router"],
        "stt_transition_network": ["transition_network"],
    }
    
    # Track which parameters have been assigned to a group
    assigned_params = set()
    
    # First, identify parameters for specific components if their LR is defined
    model_type = getattr(cfg.model, "type", "standard")
    for config_key, lr_val in lrs.items():
        if config_key == "base_model" or lr_val == default_lr:
            continue
            
        # Only process keys relevant to the current model type
        if not config_key.startswith(model_type):
            continue
            
        patterns = component_mapping.get(config_key, [])
        component_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param in assigned_params:
                continue
                
            # Check if any pattern matches the parameter name
            if any(f".{pattern}." in f".{name}." for pattern in patterns):
                component_params.append(param)
                assigned_params.add(param)
        
        if component_params:
            log.info(f"Group '{config_key}': {len(component_params)} params, LR {lr_val:.2e}")
            param_groups.append({"params": component_params, "lr": lr_val})

    # 2. Base model group (everything else)
    base_lr = lrs.get("base_model", default_lr)
    base_params = [
        p for p in model.parameters() 
        if p.requires_grad and p not in assigned_params
    ]
    
    if base_params:
        log.info(f"Group 'base_model': {len(base_params)} params, LR {base_lr:.2e}")
        param_groups.append({"params": base_params, "lr": base_lr})

    # Create the optimizer with grouped parameters
    if not param_groups:
        # Fallback to single group if no params found (shouldn't happen)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], 
            lr=default_lr, 
            **common_kwargs
        )
    else:
        opt = torch.optim.AdamW(param_groups, **common_kwargs)

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
