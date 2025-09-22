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
    calculate_metrics,
)

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="default", version_base="1.3")
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

    # Log the final configuration
    if accelerator.is_main_process:
        log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

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
    log.info(f"Creating {cfg.model.type} model ({cfg.model.size}, from_scratch={cfg.model.from_scratch})")
    model = create_model(
        cfg.model.type,
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
    # if cfg.training.gradient_checkpointing:
    #     log.info("Enabling gradient checkpointing on the model.")
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.gradient_checkpointing_enable()
    #     unwrapped_model.enable_input_require_grads()

    # Setup training
    steps_per_epoch = len(train_loader) // cfg.training.accumulate_grad_batches
    num_training_steps = steps_per_epoch * cfg.training.num_epochs

    optimizers_dict, schedulers_dict = setup_optimizer_and_scheduler(model, cfg, num_training_steps, accelerator)

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
                metrics = calculate_metrics(model, batch, global_step, num_training_steps)
                loss = metrics['loss']
                
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
                        "train/lm_loss": metrics.get('lm_loss', torch.tensor(0.0)).item(),
                    }
                    for key, value in metrics.items():
                        if "loss" in key and key != "loss":
                            log_metrics[f"train/{key}"] = value.item()
                        elif "router_stats" in key and isinstance(value, dict):
                            for stat_key, stat_value in value.items():
                                if isinstance(stat_value, (float, int)):
                                    log_metrics[f"train/router_stats/{stat_key}"] = stat_value
                                elif isinstance(stat_value, list) and len(stat_value) > 0:
                                    log_metrics[f"train/router_stats/{stat_key}_avg"] = sum(stat_value) / len(stat_value)

                    if cfg.model.type in ["sdt", "stt"]:
                        beta_ce = metrics.get('beta_ce', 0.0)
                        beta_cu = metrics.get('beta_cu', 0.0)
                        log_metrics["train/beta_ce"] = beta_ce
                        log_metrics["train/beta_cu"] = beta_cu
                        if "router_stats" in metrics and "o_ce_pos" in metrics["router_stats"]:
                            log_metrics["train/router_stats/o_ce_pos"] = metrics["router_stats"]["o_ce_pos"]
                            log_metrics["train/router_stats/m_cu_pos"] = metrics["router_stats"]["m_cu_pos"]

                    if cfg.logging.wandb.enabled and wandb.run is not None:
                        wandb.log(log_metrics, step=global_step)
                    accelerator.print(
                        f"Epoch {epoch}, Step {global_step}: "
                        f"Loss = {loss.item():.4f}, "
                        f"LM Loss = {metrics.get('lm_loss', torch.tensor(0.0)).item():.4f}"
                    )

                # Evaluation and checkpointing
                if global_step > 0 and global_step % cfg.training.eval_interval == 0:
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
