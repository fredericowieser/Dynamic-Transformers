#!/usr/bin/env python3
"""Unified training script for all model architectures."""

import logging
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data.datasets import get_dataloader
from src.training.utils import (
    PlatformOptimizer,
    create_model,
    save_checkpoint,
    setup_optimizer_and_scheduler,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def train_step(model: nn.Module, batch: dict, scaler, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Execute single training step."""
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    # Use appropriate precision
    if scaler:  # CUDA with AMP
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
    else:  # MPS or CPU
        with torch.autocast(device_type=str(device).split(':')[0], dtype=dtype, enabled=(dtype != torch.float32)):
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

    return loss


def evaluate(model: nn.Module, eval_loader, device: torch.device, dtype: torch.dtype, max_batches: int = 100) -> float:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_loader:
            if num_batches >= max_batches:
                break

            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            with torch.autocast(device_type=str(device).split(':')[0], dtype=dtype, enabled=(dtype != torch.float32)):
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    """Main training loop."""
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Platform optimization
    platform = PlatformOptimizer.detect_platform()
    platform_settings = PlatformOptimizer.get_optimal_settings(platform, cfg)

    # Load tokenizer
    model_name = f"Qwen/Qwen2.5-{cfg.model.size}"
    log.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create data loaders
    log.info("Creating data loaders...")
    train_loader = get_dataloader(cfg, tokenizer, split='train')
    eval_loader = get_dataloader(cfg, tokenizer, split='validation')

    # Create model
    log.info(f"Creating {cfg.model.type} model ({cfg.model.size}, from_scratch={cfg.training.from_scratch})")
    model = create_model(
        cfg.model.type,
        cfg.model.size,
        cfg.training.from_scratch,
        platform_settings
    )

    # Setup device and precision
    device = platform_settings['device']
    dtype = platform_settings['dtype']
    model = model.to(device)
    if dtype != torch.float32:
        model = model.to(dtype)

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {total_params/1e6:.1f}M params ({trainable_params/1e6:.1f}M trainable)")

    # Setup training
    steps_per_epoch = len(train_loader) // cfg.training.gradient_accumulation_steps
    num_training_steps = steps_per_epoch * cfg.training.num_epochs

    optimizer, scheduler = setup_optimizer_and_scheduler(
        model, cfg, num_training_steps, platform_settings
    )

    scaler = GradScaler() if platform_settings['use_amp'] else None

    # Training loop
    log.info("Starting training...")
    global_step = 0
    best_eval_loss = float('inf')

    for epoch in range(cfg.training.num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(progress):
            # Forward pass
            loss = train_step(model, batch, scaler, device, dtype)
            loss = loss / cfg.training.gradient_accumulation_steps

            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step every N accumulation steps
            if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
                # Gradient clipping
                if cfg.training.gradient_clip_val > 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val)

                # Update weights
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Track loss
                epoch_loss += loss.item() * cfg.training.gradient_accumulation_steps
                num_batches += 1

                # Update progress bar
                if global_step % cfg.logging.log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    progress.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })

                # Evaluation
                if global_step % cfg.training.eval_interval == 0:
                    eval_loss = evaluate(
                        model, eval_loader, device, dtype,
                        max_batches=cfg.training.eval_samples // cfg.data.batch_size
                    )
                    log.info(f"Step {global_step}: eval_loss={eval_loss:.4f}")

                    # Save best model
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_path = Path(cfg.system.output_dir) / "best_model"
                        save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_eval_loss, save_path)

                # Early stopping
                if cfg.training.max_steps > 0 and global_step >= cfg.training.max_steps:
                    log.info(f"Reached max steps ({cfg.training.max_steps})")
                    break

        if cfg.training.max_steps > 0 and global_step >= cfg.training.max_steps:
            break

    # Save final model
    save_path = Path(cfg.system.output_dir) / "final_model"
    save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_eval_loss, save_path)
    log.info("Training complete!")


if __name__ == "__main__":
    main()