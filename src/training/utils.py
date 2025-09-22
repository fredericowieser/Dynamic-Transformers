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

from ..models.mod.model import MoDForCausalLM
from ..models.sdt.model import SDTForCausalLM
from ..models.stt.model import STTForCausalLM

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
        config.attn_implementation = 'eager' # Fallback to eager

    # Ensure _attn_implementation is consistent with the public one
    config._attn_implementation = config.attn_implementation

    # Add model-type specific configurations from the provided config
    # Copy all relevant model parameters from cfg.model to the Qwen2Config object
    for key, value in cfg.model.items():
        if key not in ['scratch_config', 'size', 'pretrained_model_name_or_path', 'use_flash_attention_2', 'attn_implementation', 'use_cache', 'tie_word_embeddings', 'intermediate_size', 'params']: # Added 'params' and 'beta_schedule' to exclusion
            setattr(config, key, value)
    log.info(f"create_model_config: intermediate_size after cfg.model.items() loop: {config.intermediate_size}")
    log.info(f"create_model_config: config.hidden_size: {config.hidden_size}")
    log.info(f"create_model_config: config.num_attention_heads: {config.num_attention_heads}")


    # Explicitly set head_dim for consistency
    config.head_dim = config.hidden_size // config.num_attention_heads
    log.info(f"create_model_config: config.head_dim: {config.head_dim}")



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
    """Create and initialize model."""
    config = create_model_config(model_size, from_scratch, cfg)
    pretrained_name = f"Qwen/Qwen2.5-{model_size}"
    torch_dtype = getattr(torch, cfg.system.get('torch_dtype', 'float32'))

    if model_type == "standard":
        from ..models.standard.model import StandardTransformerForCausalLM
        if from_scratch:
            return StandardTransformerForCausalLM(config)
        else:
            return StandardTransformerForCausalLM.from_pretrained(pretrained_name, torch_dtype=torch_dtype)
    
    model_class_map = {
        "mod": MoDForCausalLM,
        "sdt": SDTForCausalLM,
        "stt": STTForCausalLM,
    }
    
    if model_type in model_class_map:
        # Add all model-specific hyperparameters from the 'params' block
        # to the main config object for easy access.
        if hasattr(cfg.model, 'params'):
            for key, value in cfg.model.params.items():
                setattr(config, key, value)

        model = model_class_map[model_type](config, model_cfg=cfg.model)
        if not from_scratch:
            log.info(f"Initializing {model_type.upper()} from pretrained {pretrained_name}")
            base_model = Qwen2ForCausalLM.from_pretrained(pretrained_name, torch_dtype=torch_dtype)
            model.copy_weights_from_pretrained(base_model)
            del base_model
        return model
    
    raise ValueError(f"Unknown model type: {model_type}")


def setup_optimizer_and_scheduler(model: torch.nn.Module, cfg: DictConfig, num_training_steps: int, accelerator):
    """Setup optimizers and schedulers based on parameter groups from the model."""
    unwrapped_model = accelerator.unwrap_model(model)
    param_groups = unwrapped_model.get_trainable_parameters()
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