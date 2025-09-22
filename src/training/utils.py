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


def create_model(model_type: str, cfg: DictConfig) -> torch.nn.Module:
    """Create and initialize model based on the unified config."""
    model_cfg = cfg.model
    from_scratch = model_cfg.from_scratch
    torch_dtype = getattr(torch, cfg.system.get('torch_dtype', 'float32'))

    # 1. Create base Qwen2Config
    if from_scratch:
        if model_cfg.size not in model_cfg.scratch_config:
            raise ValueError(f"Unknown model size for scratch: {model_cfg.size}")
        config = Qwen2Config(
            **model_cfg.scratch_config[model_cfg.size],
            vocab_size=model_cfg.scratch_config.vocab_size,
            max_position_embeddings=model_cfg.scratch_config.max_position_embeddings,
            rope_theta=model_cfg.scratch_config.rope_theta,
            sliding_window=model_cfg.scratch_config.sliding_window,
        )
    else:
        config = Qwen2Config.from_pretrained(model_cfg.pretrained_model_name_or_path)

    # 2. Dynamically update config with all parameters from the 'model' section
    for key, value in model_cfg.items():
        setattr(config, key, value)
    
    # Unpack model-specific config into the main config
    if model_type in model_cfg:
        model_specific_cfg = model_cfg[model_type]
        for key, value in model_specific_cfg.items():
            setattr(config, key, value)
    
    # Handle special cases like attention implementation
    if cfg.system.get('use_flash_attention', False):
        config.attn_implementation = model_cfg.get('attn_implementation', 'flash_attention_2')
    else:
        config.attn_implementation = 'eager'
    config._attn_implementation = config.attn_implementation


    # 3. Instantiate the correct model
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
        module_path, class_name = class_path.rsplit('.', 1)
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
        log.info(f"Initializing {model_type.upper()} from pretrained {model_cfg.pretrained_model_name_or_path}")
        base_model = Qwen2ForCausalLM.from_pretrained(model_cfg.pretrained_model_name_or_path, torch_dtype=torch_dtype)
        
        # This method needs to exist on the model class
        if hasattr(model, 'copy_weights_from_pretrained'):
             model.copy_weights_from_pretrained(base_model)
        else:
            log.warning(f"Model {model_type} does not have 'copy_weights_from_pretrained' method. Weight transfer might be incomplete.")
            # Fallback to simple load, might fail if architectures differ
            model.load_state_dict(base_model.state_dict(), strict=False)

        del base_model
        
    return model


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