import logging
import os
import math
import json
from tqdm.auto import tqdm

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import wandb

from accelerate import Accelerator
from transformers import get_scheduler, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

from src.data.gate_logging import GateLogger
from src.models.qwen.causal_lm import DynamicQwenForCausalLM
from src.models.utils.training import set_seed, calculate_metrics

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg: DictConfig) -> None:
    log.info(f"--- Config ---\n{OmegaConf.to_yaml(cfg)}")
    set_seed(cfg.run.seed)
    accelerator = Accelerator(
        mixed_precision=cfg.run.precision,
        gradient_accumulation_steps=cfg.training.accumulate_grad_batches,
    )
    log.info(f"Using device: {accelerator.device}")
    datamodule = hydra.utils.instantiate(cfg.data, _convert_="partial")
    datamodule.setup()
    data_collator = DataCollatorForLanguageModeling(tokenizer=datamodule.tokenizer, mlm=False)
    train_dataloader = DataLoader(
        datamodule.train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=cfg.data.batch_size,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        datamodule.val_dataset,
        collate_fn=data_collator,
        batch_size=cfg.data.batch_size,
        num_workers=4,
    )

    log.info(f"Instantiating Model <{cfg.model.pretrained_model_name_or_path}>")
    model_load_kwargs = {
        "model_cfg": OmegaConf.to_container(cfg.model.model_cfg, resolve=True)
    }
    if getattr(cfg.model, "use_flash_attention_2", False):
        log.info("Flash Attention 2 is enabled in the config. Applying to model loading.")
        model_load_kwargs["attn_implementation"] = "flash_attention_2"
        model_load_kwargs["torch_dtype"] = torch.bfloat16
    model = DynamicQwenForCausalLM.from_vanilla_checkpoint(
        cfg.model.pretrained_model_name_or_path,
        **model_load_kwargs
    )
    if cfg.peft.enabled:
        log.info("Applying PEFT (LoRA) configuration to the model...")
        peft_config = hydra.utils.instantiate(cfg.peft.config)
        model = get_peft_model(model, peft_config)
        log.info("Trainable parameters after applying LoRA:")
        model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    tokenizer = datamodule.tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    
    log.info("Setting up optimizer with distinct parameter groups...")
    base_model_params, prior_params, vpr_router_params = [], [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if "vpr_router" in n:
                vpr_router_params.append(p)
            elif "prior_ffn" in n:
                prior_params.append(p)
            else:
                base_model_params.append(p)
    param_groups = [
        {"params": base_model_params, "lr": cfg.training.optimizer.base_model_lr},
        {"params": prior_params, "lr": cfg.training.optimizer.prior_lr},
        {"params": vpr_router_params, "lr": cfg.training.optimizer.vpr_router_lr},
    ]
    log.info(f"  - Base Model parameters: {sum(p.numel() for p in base_model_params):,}")
    log.info(f"  - Dynamic Component (Prior FFN) parameters: {sum(p.numel() for p in prior_params):,}")
    log.info(f"  - VPR Router parameters: {sum(p.numel() for p in vpr_router_params):,}")
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.training.optimizer.weight_decay,
    )
    gate_logger = GateLogger(model.config.num_hidden_layers)

    # Epochs and Steps Calculation
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.accumulate_grad_batches)
    if cfg.training.max_steps > 0:
        num_training_steps = cfg.training.max_steps
        num_epochs = math.ceil(num_training_steps / num_update_steps_per_epoch)
        log.info(f"Training for a fixed {num_training_steps} steps (approx. {num_epochs} epochs).")
    else:
        num_training_steps = num_update_steps_per_epoch * cfg.training.num_epochs
        num_epochs = cfg.training.num_epochs
        log.info(f"Training for {num_epochs} epochs ({num_training_steps} steps).")
    
    num_warmup_steps = int(num_training_steps * cfg.training.optimizer.warmup_ratio)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    if accelerator.is_main_process and cfg.logging.wandb.enabled:
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=cfg.run.name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    log.info("--- Starting Training ---")
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_main_process)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if progress_bar.n >= num_training_steps:
                break
            
            with accelerator.accumulate(model):
                metrics = calculate_metrics(model, batch, progress_bar.n)
                total_loss = metrics["total_loss"]
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    if cfg.training.use_gradient_clipping:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                if metrics.get("per_layer_gate_stats"):
                    gate_logger.update_rolling_history(metrics["per_layer_gate_stats"])
                    gate_logger.log_rolling_history(progress_bar.n, log_interval=100)
                
                if accelerator.is_main_process:
                    log_metrics = {
                        "train/loss": metrics["total_loss"].item(),
                        "train/lm_loss": metrics["lm_loss"].item(),
                        "train/perplexity": metrics["perplexity"].item(),
                        "lr": lr_scheduler.get_last_lr()[0]
                    }
                    if "s_ce_stats" in metrics: # Check for any representative VPR metric
                        if metrics.get("prior_loss") is not None:
                            log_metrics["train/prior_loss"] = metrics["prior_loss"].item()
                            log_metrics["train/prior_loss_weight"] = metrics["current_prior_loss_weight"]
                        def log_signal_stats(signal_name, stats_dict):
                            for key, value in stats_dict.items():
                                log_metrics[f"train_vpr_signals/{signal_name}_{key}"] = value.item()
                        log_signal_stats("S_CE", metrics["s_ce_stats"])
                        log_signal_stats("S_CU", metrics["s_cu_stats"])
                        log_signal_stats("G_cont", metrics["g_cont_stats"])
                        def log_router_param_stats(param_name, stats_dict):
                            if stats_dict:
                                log_metrics[f"train_vpr_router/{param_name}_mean"] = stats_dict["mean"].item()
                                log_metrics[f"train_vpr_router/{param_name}_std"] = stats_dict["std"].item()
                        log_router_param_stats("beta_ce", metrics.get("router_beta_ce_stats"))
                        log_router_param_stats("beta_cu", metrics.get("router_beta_cu_stats"))
                        log_router_param_stats("cu_multiplier", metrics.get("router_cu_multiplier_stats"))
                        log_router_param_stats("ce_offset", metrics.get("router_ce_offset_stats"))
                    if cfg.logging.wandb.enabled:
                        wandb.log(log_metrics, step=progress_bar.n)

                if (progress_bar.n) % cfg.training.eval_interval == 0 and progress_bar.n > 0:
                    model.eval()
                    val_losses = []
                    for val_batch in val_dataloader:
                        with torch.no_grad():
                            val_metrics_dict = calculate_metrics(model, val_batch, progress_bar.n)
                        val_losses.append(accelerator.gather(val_metrics_dict["total_loss"]))
                    val_loss = torch.stack(val_losses).mean().item()
                    
                    if accelerator.is_main_process:
                        log.info(f"Epoch {epoch}, Step {progress_bar.n}: Validation Loss = {val_loss:.4f}")
                        if cfg.logging.wandb.enabled:
                            wandb.log({"val/loss": val_loss}, step=progress_bar.n)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            log.info(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                            unwrapped_model = accelerator.unwrap_model(model)
                            save_path = os.path.join(cfg.run.output_dir, "best_model")
                            unwrapped_model.save_pretrained(save_path, safe_serialization=True)
                            tokenizer.save_pretrained(save_path)
                            
                            if cfg.logging.wandb.enabled:
                                wandb_info = {
                                    "run_id": wandb.run.id, "project": wandb.run.project,
                                    "entity": wandb.run.entity, "run_name": wandb.run.name,
                                }
                                with open(os.path.join(save_path, "wandb_info.json"), "w") as f:
                                    json.dump(wandb_info, f, indent=2)
                                log.info(f"Saved wandb run info to {save_path}")
                    model.train()
        
        if progress_bar.n >= num_training_steps:
            log.info(f"Reached max_steps ({num_training_steps}). Stopping training.")
            break

    if accelerator.is_main_process:
        log.info("--- Saving final model checkpoint ---")
        unwrapped_model = accelerator.unwrap_model(model)
        final_save_path = os.path.join(cfg.run.output_dir, "final_model")
        unwrapped_model.save_pretrained(final_save_path, safe_serialization=True)
        tokenizer.save_pretrained(final_save_path)
        
        if cfg.logging.wandb.enabled:
            wandb_info = {
                "run_id": wandb.run.id, "project": wandb.run.project,
                "entity": wandb.run.entity, "run_name": wandb.run.name,
            }
            with open(os.path.join(final_save_path, "wandb_info.json"), "w") as f:
                json.dump(wandb_info, f, indent=2)
            log.info(f"Saved wandb run info to {final_save_path}")
            
        log.info(f"Final model saved to {final_save_path}")
        if cfg.logging.wandb.enabled:
            wandb.finish()

    log.info("--- Training Finished ---")

if __name__ == "__main__":
    main()