"""Training utilities for dynamic transformer models."""

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

log = logging.getLogger(__name__)


class PlatformOptimizer:
    """Platform-specific optimization settings."""

    @staticmethod
    def detect_platform() -> str:
        """Detect available compute platform."""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    @staticmethod
    def get_optimal_settings(platform: str, config: DictConfig) -> Dict[str, Any]:
        """Get optimal settings for the platform."""
        # Handle nested config from presets
        cfg = config.presets if 'presets' in config else config

        settings = {
            'device': torch.device(platform),
            'dtype': torch.float32,
            'use_amp': False,
            'use_flash_attention': False,
            'compile_model': False,
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
        }

        if platform == 'cuda':
            log.info("✅ Using CUDA GPU")
            settings.update({
                'dtype': torch.bfloat16 if cfg.training.get('use_bf16', True) else torch.float16,
                'use_amp': True,
                'use_flash_attention': cfg.model.get('use_flash_attention', True),
                'compile_model': cfg.system.get('compile', False),
                'num_workers': cfg.system.get('num_workers', 4),
                'pin_memory': True,
                'persistent_workers': True,
            })

        elif platform == 'mps':
            log.info("✅ Using Mac GPU (Metal Performance Shaders)")
            # MPS doesn't support BF16 or AMP well yet
            settings.update({
                'dtype': torch.float32,
                'use_amp': False,
                'use_flash_attention': False,
                'compile_model': False,
                'num_workers': 0,  # MPS has issues with multiprocessing
                'pin_memory': False,
                'persistent_workers': False,
            })

        else:
            log.info("⚠️ Using CPU (this will be slow)")
            settings.update({
                'dtype': torch.float32,
                'use_amp': False,
                'num_workers': min(4, os.cpu_count() or 1),
            })

        return settings


def create_model_config(
    model_type: str,
    model_size: str,
    from_scratch: bool,
    platform_settings: Dict[str, Any]
) -> Qwen2Config:
    """Create model configuration.

    Args:
        model_type: 'standard', 'mod', or 'dtf'
        model_size: '0.5B', '1.5B', or '3B'
        from_scratch: Whether to train from scratch
        platform_settings: Platform-specific settings

    Returns:
        Qwen2Config with appropriate settings
    """
    pretrained_name = f"Qwen/Qwen2.5-{model_size}"

    if from_scratch:
        # Create config from scratch with Qwen2.5 specifications
        size_configs = {
            "0.5B": {
                "hidden_size": 896,
                "intermediate_size": 4864,
                "num_hidden_layers": 24,
                "num_attention_heads": 14,
                "num_key_value_heads": 2,
            },
            "1.5B": {
                "hidden_size": 1536,
                "intermediate_size": 8960,
                "num_hidden_layers": 28,
                "num_attention_heads": 12,
                "num_key_value_heads": 2,
            },
            "3B": {
                "hidden_size": 2048,
                "intermediate_size": 11008,
                "num_hidden_layers": 36,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
            },
        }

        if model_size not in size_configs:
            raise ValueError(f"Unknown model size: {model_size}")

        config = Qwen2Config(
            vocab_size=151936,
            max_position_embeddings=32768,
            rope_theta=1000000.0,
            sliding_window=131072,
            **size_configs[model_size]
        )
        config.init_from_scratch = True
    else:
        # Load full config from pretrained
        config = Qwen2Config.from_pretrained(pretrained_name)

    # Ensure _attn_implementation is set
    config._attn_implementation = 'eager'

    # Add model-type specific configurations
    if model_type == "dtf":
        config.prior_ffn_intermediate_size_factor = 0.25
        config.prior_loss_weight = 0.05
        config.capacity_gamma = 0.5
        config.beta_ce_init = -0.5
        config.beta_cu_init = -0.8
        config.cu_detection_multiplier_init = 1.2
        config.ce_criterion_offset_init = 1.0

    elif model_type == "mod":
        config.mod_capacity = 0.125
        config.mod_aux_loss_weight = 0.01
        config.mod_total_aux_loss_weight = 0.01

    # Common settings for all models
    config.initializer_range = 0.02
    config.rms_norm_eps = 1e-6
    config.use_cache = True
    config.tie_word_embeddings = True

    # Platform-specific settings
    config.use_flash_attention = platform_settings['use_flash_attention']
    config.torch_dtype = str(platform_settings['dtype']).split('.')[-1]

    return config


def create_model(
    model_type: str,
    model_size: str,
    from_scratch: bool,
    platform_settings: Dict[str, Any]
) -> torch.nn.Module:
    """Create and initialize model.

    Args:
        model_type: 'standard', 'mod', or 'dtf'
        model_size: '0.5B', '1.5B', or '3B'
        from_scratch: Whether to train from scratch
        platform_settings: Platform-specific settings

    Returns:
        Initialized model
    """
    config = create_model_config(model_type, model_size, from_scratch, platform_settings)
    pretrained_name = f"Qwen/Qwen2.5-{model_size}"

    if model_type == "standard":
        from ..models.standard.model import StandardTransformerForCausalLM
        return StandardTransformerForCausalLM.from_pretrained_or_random(
            pretrained_name if not from_scratch else config,
            from_scratch=from_scratch,
            config=config if from_scratch else None,
            torch_dtype=platform_settings['dtype'],
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
                torch_dtype=platform_settings['dtype']
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
            base_model = Qwen2ForCausalLM.from_pretrained(
                pretrained_name,
                torch_dtype=platform_settings['dtype']
            )
            model = DTFForCausalLM(config)
            model.copy_weights_from_pretrained(base_model)
            del base_model  # Free memory
        return model

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def setup_optimizer_and_scheduler(
    model: torch.nn.Module,
    cfg: DictConfig,
    num_training_steps: int,
    platform_settings: Dict[str, Any]
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