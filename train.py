import logging
import os
import math
from tqdm.auto import tqdm

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import wandb

from accelerate import Accelerator
from transformers import get_scheduler, AutoTokenizer

# Assuming your project structure allows this import path
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
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    log.info(f"Instantiating Model <{cfg.model.model_cfg.model_name}>")
    model = DynamicQwenForCausalLM.from_pretrained(
        cfg.model.model_cfg.model_name,
        model_cfg=OmegaConf.to_container(cfg.model.model_cfg, resolve=True)
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    gate_logger = GateLogger(model.config.num_hidden_layers)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.training.optimizer.base_lr,
        weight_decay=cfg.training.optimizer.weight_decay,
    )

    num_training_steps = math.ceil(len(train_dataloader) / cfg.training.accumulate_grad_batches) * cfg.training.num_epochs
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
        if not cfg.logging.wandb.entity:
            raise ValueError("WandB entity not set.")
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=cfg.run.name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    log.info("--- Starting Training ---")
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_main_process)
    best_val_loss = float('inf')
    
    for epoch in range(cfg.training.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                metrics = calculate_metrics(model, batch, progress_bar.n)
                total_loss = metrics["total_loss"]
                
                accelerator.backward(total_loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.training.gradient_clip_val)
                
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
                    if metrics["prior_loss"] is not None:
                        log_metrics["train/prior_loss"] = metrics["prior_loss"].item()
                    
                    if metrics.get("avg_ce_proportion") is not None:
                         log_metrics["train_vpr/avg_ce_proportion"] = metrics["avg_ce_proportion"].item()
                         log_metrics["train_vpr/avg_cu_proportion"] = metrics["avg_cu_proportion"].item()
                         log_metrics["train_vpr/gating_signal_mean"] = metrics["combined_gating_signal_mean"].item()
                         log_metrics["train_vpr/router_beta_ce"] = metrics["avg_beta_ce"].item()
                         log_metrics["train_vpr/router_beta_cu"] = metrics["avg_beta_cu"].item()
                         log_metrics["train_vpr/router_cu_multiplier"] = metrics["avg_cu_detection_multiplier"].item()
                         log_metrics["train_vpr/router_ce_offset"] = metrics["avg_ce_criterion_offset"].item()
                    
                    if cfg.logging.wandb.enabled:
                        wandb.log(log_metrics, step=progress_bar.n)

            if (progress_bar.n + 1) % cfg.training.eval_interval == 0:
                model.eval()
                val_losses = []
                for val_batch in val_dataloader:
                    with torch.no_grad():
                        val_metrics_dict = calculate_metrics(model, val_batch, progress_bar.n)
                    val_losses.append(accelerator.gather(val_metrics_dict["total_loss"]))
                
                val_loss = torch.cat(val_losses).mean().item()
                
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

                model.train()

    if accelerator.is_main_process:
        if cfg.logging.wandb.enabled:
            wandb.finish()

    log.info("--- Training Finished ---")

if __name__ == "__main__":
    main()
