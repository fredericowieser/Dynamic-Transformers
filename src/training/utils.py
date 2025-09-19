import os
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from omegaconf import DictConfig

from transformers import (
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForCausalLM,
    get_scheduler
)
from huggingface_hub import HfApi

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
        if key not in ['scratch_config', 'params', 'size', 'type', 'pretrained_model_name_or_path', 'use_flash_attention_2', 'attn_implementation', 'use_cache', 'tie_word_embeddings', 'intermediate_size']:
            setattr(config, key, value)
    log.info(f"create_model_config: intermediate_size after cfg.model.items() loop: {config.intermediate_size}")
    if 'model' in cfg and 'params' in cfg.model: # Keep existing params copy logic
        for key, value in cfg.model.params.items():
            setattr(config, key, value)
    log.info(f"create_model_config: intermediate_size after cfg.model.params loop: {config.intermediate_size}")

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
        model_type: 'standard', 'mod', 'dtf', or 'tdtf'
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
        return StandardTransformerForCausalLM.from_pretrained_or_random(
            pretrained_name if not from_scratch else config,
            from_scratch=from_scratch,
            config=config if from_scratch else None,
            torch_dtype=torch_dtype,
            ignore_mismatched_sizes=True
        )

    elif model_type == "mod":
        from ..models.mod.model import MoDForCausalLM
        if from_scratch:
            log.info(f"Initializing MoD model from scratch with {model_size} config")
            model = MoDForCausalLM(config)
        else:
            log.info(f"Initializing MoD model from pretrained {pretrained_name}")
            base_model = Qwen2ForCausalLM.from_pretrained(
                pretrained_name,
                torch_dtype=torch_dtype
            )
            model = MoDForCausalLM(config)
            model.copy_weights_from_pretrained(base_model)
            del base_model  # Free memory
        return model

    elif model_type == "dtf":
        from ..models.dtf.model import DTFForCausalLM
        if from_scratch:
            log.info(f"Initializing DTF model from scratch with {model_size} config")
            model = DTFForCausalLM(config)
        else:
            log.info(f"Initializing DTF model from pretrained {pretrained_name}")
            log.info(f"Config intermediate_size before DTFForCausalLM init: {config.intermediate_size}")
            base_model = Qwen2ForCausalLM.from_pretrained(
                pretrained_name,
                torch_dtype=torch_dtype
            )
            model = DTFForCausalLM(config)
            model.copy_weights_from_pretrained(base_model)
            del base_model  # Free memory
        return model

    elif model_type == "tdtf":
        from ..models.tdtf.model import TDTFForCausalLM
        if from_scratch:
            log.info(f"Initializing TDTF model from scratch with {model_size} config")
            model = TDTFForCausalLM(config)
        else:
            log.info(f"Initializing TDTF model from pretrained {pretrained_name}")
            base_model = Qwen2ForCausalLM.from_pretrained(
                pretrained_name,
                torch_dtype=torch_dtype
            )
            model = TDTFForCausalLM(config)
            model.copy_weights_from_pretrained(base_model)
            del base_model  # Free memory
        return model

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def setup_optimizer_and_scheduler(
    model: torch.nn.Module,
    cfg: DictConfig,
    num_training_steps: int,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Setup optimizer with parameter groups and scheduler."""

    # Get parameter groups from model
    if hasattr(model, 'get_trainable_parameters'):
        param_groups = model.get_trainable_parameters()
    else:
        param_groups = [{'params': model.parameters(), 'lr_scale': 1.0, 'name': 'all'}]

    # Apply learning rate scales
    base_lr = cfg.training.optimizer.lr
    optimizer_groups = []

    for group in param_groups:
        optimizer_groups.append({
            'params': group['params'],
            'lr': base_lr * group['lr_scale'],
            'weight_decay': cfg.training.optimizer.weight_decay,
        })

        log.info(f"  {group['name']}: {sum(p.numel() for p in group['params'])} params, "
                f"lr={base_lr * group['lr_scale']:.2e}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(cfg.training.optimizer.adam_beta1, cfg.training.optimizer.adam_beta2),
        eps=cfg.training.optimizer.adam_epsilon,
    )

    # Create scheduler
    num_warmup_steps = int(num_training_steps * cfg.training.optimizer.warmup_ratio)
    scheduler = get_scheduler(
        cfg.training.optimizer.scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
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
    torch.save({
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'step': step,
        'best_loss': best_loss,
    }, state_path)

    log.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
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
    if state_path.exists() and optimizer is not None:
        state = torch.load(state_path, map_location='cpu')
        optimizer.load_state_dict(state['optimizer'])
        if scheduler and state.get('scheduler'):
            scheduler.load_state_dict(state['scheduler'])
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