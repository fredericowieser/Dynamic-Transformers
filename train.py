#!/usr/bin/env python3

import logging
import os
from pathlib import Path

import hydra
import torch
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, get_scheduler
from tqdm.auto import tqdm

from src.data.datasets import get_dataloader
from src.models.dtf.model import DTFForCausalLM
from src.models.mod.model import MoDForCausalLM

log = logging.getLogger(__name__)


def get_model(config: DictConfig):
    """Get model based on configuration."""
    model_classes = {
        "dtf": DTFForCausalLM,
        "mod": MoDForCausalLM,
    }

    if config.model_type not in model_classes:
        raise ValueError(f"Unknown model type: {config.model_type}")

    log.info(f"Loading {config.model_type.upper()} model from {config.model.pretrained_model_name}")

    model_class = model_classes[config.model_type]
    model_config = OmegaConf.to_container(config.model, resolve=True)

    # Load model
    model = model_class.from_pretrained(
        config.model.pretrained_model_name,
        config_dict=model_config,
        torch_dtype=torch.bfloat16 if config.model.use_flash_attention else torch.float32,
        attn_implementation="flash_attention_2" if config.model.use_flash_attention else "eager"
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    return model


def get_optimizer_groups(model, config):
    """Create optimizer parameter groups with different learning rates."""
    base_params, router_params, prior_params = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "router" in name:
            router_params.append(param)
        elif "prior" in name:
            prior_params.append(param)
        else:
            base_params.append(param)

    param_groups = [
        {"params": base_params, "lr": config.training.optimizer.lr},
        {"params": router_params, "lr": config.training.optimizer.lr * config.training.lr_multipliers.router},
        {"params": prior_params, "lr": config.training.optimizer.lr * config.training.lr_multipliers.prior},
    ]

    log.info(f"Parameter groups:")
    log.info(f"  Base: {sum(p.numel() for p in base_params):,} params")
    log.info(f"  Router: {sum(p.numel() for p in router_params):,} params")
    log.info(f"  Prior: {sum(p.numel() for p in prior_params):,} params")

    return param_groups


def train_epoch(model, dataloader, optimizer, scheduler, accelerator, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)

    for step, batch in enumerate(progress_bar):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.gradient_clip_val)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        if step % config.logging.log_interval == 0:
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=scheduler.get_last_lr()[0])

            if config.logging.wandb.enabled and accelerator.is_local_main_process:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": step
                })

    return total_loss / len(dataloader)


def evaluate(model, dataloader, accelerator):
    """Evaluate model."""
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            outputs = model(**batch)
            losses.append(accelerator.gather(outputs.loss))

    return torch.stack(losses).mean().item()


@hydra.main(version_base=None, config_path="config", config_name="base")
def main(config: DictConfig) -> None:
    """Main training loop."""

    # Set seed
    torch.manual_seed(config.system.seed)

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.system.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )

    log.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Get data loaders
    train_dataloader = get_dataloader(config, tokenizer, split="train")
    val_dataloader = get_dataloader(config, tokenizer, split="validation")

    # Load model
    model = get_model(config)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Setup optimizer
    param_groups = get_optimizer_groups(model, config)
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.training.optimizer.weight_decay
    )

    # Setup scheduler
    num_training_steps = (
        config.training.max_steps if config.training.max_steps > 0
        else len(train_dataloader) * config.training.num_epochs // config.training.gradient_accumulation_steps
    )
    num_warmup_steps = int(num_training_steps * config.training.optimizer.warmup_ratio)

    scheduler = get_scheduler(
        config.training.optimizer.scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    # Initialize wandb
    if config.logging.wandb.enabled and accelerator.is_local_main_process:
        wandb.init(
            project=config.logging.wandb.project,
            entity=config.logging.wandb.entity,
            name=f"{config.model_type}_{config.model.capacity_gamma}",
            config=OmegaConf.to_container(config, resolve=True)
        )

    # Create output directory
    output_dir = Path(config.system.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(config.training.num_epochs):
        log.info(f"Epoch {epoch + 1}/{config.training.num_epochs}")

        # Train
        train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, accelerator, config
        )

        # Evaluate
        if (epoch + 1) % config.training.eval_interval == 0:
            val_loss = evaluate(model, val_dataloader, accelerator)

            if accelerator.is_local_main_process:
                log.info(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

                if config.logging.wandb.enabled:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train/epoch_loss": train_loss,
                        "val/loss": val_loss
                    })

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(output_dir / "best_model")
                    tokenizer.save_pretrained(output_dir / "best_model")
                    log.info(f"Saved best model with val_loss={val_loss:.4f}")

        global_step += len(train_dataloader)

        # Check max steps
        if config.training.max_steps > 0 and global_step >= config.training.max_steps:
            log.info(f"Reached max_steps ({config.training.max_steps})")
            break

    # Save final model
    if accelerator.is_local_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir / "final_model")
        tokenizer.save_pretrained(output_dir / "final_model")
        log.info(f"Training complete. Models saved to {output_dir}")

        if config.logging.wandb.enabled:
            wandb.finish()


if __name__ == "__main__":
    main()