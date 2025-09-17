#!/usr/bin/env python3
"""Mac-optimized training script for Dynamic Transformer models."""

import logging
import warnings
from pathlib import Path

import hydra
import torch
import torch.backends.mps
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, get_scheduler
from tqdm.auto import tqdm

from src.data.datasets import get_dataloader
from src.models.dtf.model import DTFForCausalLM
from src.models.mod.model import MoDForCausalLM

# Suppress warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


def check_mac_gpu():
    """Check for Mac GPU (Metal Performance Shaders) availability."""
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            device = torch.device("mps")
            log.info("✅ Using Mac GPU (Metal Performance Shaders)")
            return device

    device = torch.device("cpu")
    log.warning("⚠️ Mac GPU not available, using CPU")
    return device


def get_model(config: DictConfig, device):
    """Initialize model with Mac optimizations."""
    model_classes = {
        "dtf": DTFForCausalLM,
        "mod": MoDForCausalLM,
    }

    if config.model_type not in model_classes:
        raise ValueError(f"Unknown model type: {config.model_type}")

    log.info(f"Initializing {config.model_type.upper()} model (≈50M params)")

    model_class = model_classes[config.model_type]
    model_config = OmegaConf.to_container(config.model, resolve=True)

    # Create model with small architecture
    model = model_class.from_pretrained(
        config.model.pretrained_model_name,
        config_dict=model_config,
        torch_dtype=torch.float32,  # Use float32 on Mac
        low_cpu_mem_usage=True
    )

    # Enable memory-efficient training
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Move to device
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model size: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")

    return model


def create_optimizer(model, config):
    """Create optimizer with parameter groups."""
    # Separate parameter groups
    base_params = []
    router_params = []
    prior_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "router" in name:
            router_params.append(param)
        elif "prior" in name:
            prior_params.append(param)
        else:
            base_params.append(param)

    # Create parameter groups with different learning rates
    param_groups = [
        {"params": base_params, "lr": config.training.optimizer.lr, "name": "base"},
        {"params": router_params, "lr": config.training.optimizer.lr * config.training.lr_multipliers.router, "name": "router"},
        {"params": prior_params, "lr": config.training.optimizer.lr * config.training.lr_multipliers.prior, "name": "prior"},
    ]

    # Log parameter counts
    log.info("Parameter groups:")
    for group in param_groups:
        count = sum(p.numel() for p in group["params"])
        log.info(f"  {group['name']}: {count:,} params, lr={group['lr']:.2e}")

    # Create AdamW optimizer
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(config.training.optimizer.adam_beta1, config.training.optimizer.adam_beta2),
        eps=config.training.optimizer.adam_epsilon,
        weight_decay=config.training.optimizer.weight_decay
    )

    return optimizer


def train_step(model, batch, optimizer, scheduler, config, device):
    """Single training step optimized for Mac."""
    model.train()

    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}

    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss

    # Scale loss for gradient accumulation
    loss = loss / config.training.gradient_accumulation_steps

    # Backward pass
    loss.backward()

    # Gradient clipping
    if config.training.gradient_clip_val > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_val)

    return loss.item() * config.training.gradient_accumulation_steps


def evaluate(model, dataloader, config, device):
    """Evaluation loop."""
    model.eval()
    total_loss = 0
    num_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            if i >= config.training.eval_samples // config.data.batch_size:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            total_loss += outputs.loss.item() * batch["input_ids"].size(0)
            num_samples += batch["input_ids"].size(0)

    return total_loss / num_samples if num_samples > 0 else float('inf')


@hydra.main(version_base=None, config_path="config", config_name="mac_training")
def main(config: DictConfig) -> None:
    """Main training loop for Mac."""

    # Set seed
    torch.manual_seed(config.system.seed)

    # Setup device
    device = check_mac_gpu()

    # Log configuration
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(config))

    # Initialize tokenizer
    log.info(f"Loading tokenizer: {config.model.pretrained_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create data loaders
    log.info("Creating data loaders...")
    train_loader = get_dataloader(config, tokenizer, split="train")
    val_loader = get_dataloader(config, tokenizer, split="validation")

    # Initialize model
    model = get_model(config, device)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Create scheduler
    num_training_steps = min(
        len(train_loader) * config.training.num_epochs,
        config.training.max_steps
    ) if config.training.max_steps > 0 else len(train_loader) * config.training.num_epochs

    num_warmup_steps = int(num_training_steps * config.training.optimizer.warmup_ratio)

    scheduler = get_scheduler(
        config.training.optimizer.scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Initialize wandb
    if config.logging.wandb.enabled:
        wandb.init(
            project=config.logging.wandb.project,
            entity=config.logging.wandb.entity,
            name=f"{config.model_type}_50m_mac",
            config=OmegaConf.to_container(config, resolve=True),
            tags=config.logging.wandb.tags
        )

    # Create output directory
    output_dir = Path(config.system.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training variables
    global_step = 0
    best_val_loss = float('inf')
    accumulation_steps = 0
    accumulated_loss = 0

    # Training loop
    log.info("Starting training...")
    progress_bar = tqdm(total=num_training_steps, desc="Training")

    for epoch in range(config.training.num_epochs):
        log.info(f"\n📚 Epoch {epoch + 1}/{config.training.num_epochs}")

        for batch_idx, batch in enumerate(train_loader):
            # Training step
            loss = train_step(model, batch, optimizer, scheduler, config, device)
            accumulated_loss += loss
            accumulation_steps += 1

            # Gradient accumulation
            if accumulation_steps >= config.training.gradient_accumulation_steps:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                progress_bar.update(1)

                # Logging
                if global_step % config.logging.log_interval == 0:
                    avg_loss = accumulated_loss / accumulation_steps
                    lr = scheduler.get_last_lr()[0]

                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "epoch": epoch + 1
                    })

                    if config.logging.wandb.enabled:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/epoch": epoch + 1,
                            "train/step": global_step
                        })

                accumulated_loss = 0
                accumulation_steps = 0

                # Evaluation
                if global_step % config.training.eval_interval == 0:
                    val_loss = evaluate(model, val_loader, config, device)

                    log.info(f"📊 Step {global_step}: val_loss = {val_loss:.4f}")

                    if config.logging.wandb.enabled:
                        wandb.log({
                            "val/loss": val_loss,
                            "val/perplexity": torch.exp(torch.tensor(val_loss)).item()
                        })

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                        log.info(f"💾 Saving best model (val_loss: {val_loss:.4f})")
                        model.save_pretrained(output_dir / "best_model")
                        tokenizer.save_pretrained(output_dir / "best_model")

                # Save checkpoint
                if global_step % config.training.save_interval == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

                # Check max steps
                if config.training.max_steps > 0 and global_step >= config.training.max_steps:
                    break

        if config.training.max_steps > 0 and global_step >= config.training.max_steps:
            log.info(f"Reached max_steps ({config.training.max_steps})")
            break

    progress_bar.close()

    # Save final model
    log.info("💾 Saving final model...")
    model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")

    # Final evaluation
    final_val_loss = evaluate(model, val_loader, config, device)
    log.info(f"✅ Training complete! Final val_loss: {final_val_loss:.4f}")

    if config.logging.wandb.enabled:
        wandb.log({"final/val_loss": final_val_loss})
        wandb.finish()

    log.info(f"📁 Models saved to: {output_dir}")


if __name__ == "__main__":
    main()