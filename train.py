import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from src.data.mixed_dataset import MixedDataset
from src.training.utils import (calculate_metrics, create_model,
                                evaluate_perplexity, save_checkpoint,
                                setup_optimizer_and_scheduler)

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Resolved logging level from config: {cfg.logging.level}")
    logging.basicConfig(level=cfg.logging.level, format="%(asctime)s - %(levelname)s - %(message)s")

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
            config=OmegaConf.to_container(cfg, resolve=True),
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
    log.info(
        f"Creating {cfg.model.type} model ({cfg.model.size}, from_scratch={cfg.model.from_scratch})"
    )
    model = create_model(cfg.model.type, cfg)

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
    num_training_steps_from_epochs = steps_per_epoch * cfg.training.num_epochs

    if cfg.training.max_steps > 0:
        num_training_steps = min(num_training_steps_from_epochs, cfg.training.max_steps)
        log.info(f"Training will run for {num_training_steps} steps (capped by max_steps).")
    else:
        num_training_steps = num_training_steps_from_epochs
        log.info(
            f"Training will run for {num_training_steps} steps ({cfg.training.num_epochs} epochs)."
        )

    optimizers_dict, schedulers_dict = setup_optimizer_and_scheduler(
        model, cfg, num_training_steps, accelerator
    )

    optimizers_to_prepare = list(optimizers_dict.values())
    schedulers_to_prepare = list(schedulers_dict.values())

    prepared_items = accelerator.prepare(
        model, *optimizers_to_prepare, train_loader, eval_loader, *schedulers_to_prepare
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
    best_eval_loss = float("inf")

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
                loss = metrics["loss"]

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if cfg.training.use_gradient_clipping:
                        accelerator.clip_grad_norm_(
                            model.parameters(), cfg.training.gradient_clip_val
                        )

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
                        "train/lm_loss": metrics.get("lm_loss", torch.tensor(0.0)).item(),
                    }
                    for key, value in metrics.items():
                        if "loss" in key and key != "loss":
                            if hasattr(value, "item"):
                                log_metrics[f"train/{key}"] = value.item()
                            else:
                                log_metrics[f"train/{key}"] = value  # Already a float
                        elif "router_stats" in key and isinstance(value, dict):
                            per_layer_stats = {}
                            other_stats = {}

                            # Separate per-layer stats from other stats
                            for stat_key, stat_value in value.items():
                                if "/layer_" in stat_key and isinstance(stat_value, (float, int)):
                                    # e.g., key is "sdt/layer_1/S_CE_mean"
                                    # metric_name is "sdt/S_CE_mean"
                                    parts = stat_key.split("/")
                                    metric_name = f"{parts[0]}/{parts[2]}"
                                    if metric_name not in per_layer_stats:
                                        per_layer_stats[metric_name] = []
                                    per_layer_stats[metric_name].append(stat_value)
                                    # Log the individual value to the "extra" section
                                    log_metrics[f"extra/router_stats/{stat_key}"] = stat_value
                                else:
                                    other_stats[stat_key] = stat_value

                            # Log the non-per-layer stats directly to train/
                            for stat_key, stat_value in other_stats.items():
                                if isinstance(stat_value, (float, int)):
                                    log_metrics[f"train/router_stats/{stat_key}"] = stat_value

                            # Calculate and log the mean of the per-layer stats to train/
                            for metric_name, values_list in per_layer_stats.items():
                                if values_list:
                                    mean_value = sum(values_list) / len(values_list)
                                    log_metrics[f"train/router_stats/{metric_name}_mean"] = (
                                        mean_value
                                    )

                    if cfg.model.type in ["sdt", "stt"]:
                        beta_ce = metrics.get("beta_ce", 0.0)
                        beta_cu = metrics.get("beta_cu", 0.0)
                        log_metrics["train/beta_ce"] = beta_ce
                        log_metrics["train/beta_cu"] = beta_cu
                        if "router_stats" in metrics and "o_ce" in metrics["router_stats"]:
                            log_metrics["train/router_stats/o_ce"] = metrics["router_stats"]["o_ce"]
                            log_metrics["train/router_stats/m_cu"] = metrics["router_stats"]["m_cu"]

                    # Log learning rates for each parameter group
                    for name, opt in optimizers_dict.items():
                        if opt.param_groups:  # Ensure there are param groups
                            log_metrics[f"lr/{name}"] = opt.param_groups[0]["lr"]

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

                    val_loss, val_perplexity, val_unscaled_losses, val_router_stats = (
                        evaluate_perplexity(unwrapped_model, eval_loader, accelerator)
                    )

                    if accelerator.is_main_process:
                        if cfg.logging.wandb.enabled and wandb.run is not None:
                            val_log_metrics = {
                                "val/loss": val_loss,
                                "val/perplexity": val_perplexity,
                            }
                            # Log unscaled losses
                            for k, v in val_unscaled_losses.items():
                                val_log_metrics[f"val/unscaled_losses/{k}"] = v

                            # Log router stats
                            for k, v in val_router_stats.items():
                                # Separate per-layer stats from other stats for validation logging
                                if "/layer_" in k:
                                    val_log_metrics[f"extra/val_router_stats/{k}"] = v
                                else:
                                    val_log_metrics[f"val/router_stats/{k}"] = v

                            wandb.log(val_log_metrics, step=global_step)
                        accelerator.print(
                            f"Validation Loss: {val_loss:.4f}, Validation Perplexity: {val_perplexity:.2f}"
                        )

                        if val_loss < best_eval_loss:
                            best_eval_loss = val_loss
                            best_model_path = Path(cfg.run.output_dir) / "best_model"
                            log.info(f"New best model found! Saving to {best_model_path}")
                            # FIX: Use save_pretrained to save the model in the standard Hugging Face format,
                            # which creates pytorch_model.bin or model.safetensors that from_pretrained can find.
                            unwrapped_model.save_pretrained(best_model_path)
                            tokenizer.save_pretrained(best_model_path)

                    accelerator.wait_for_everyone()

    # Save final model
    save_path = Path(cfg.run.output_dir) / "final_model"
    save_checkpoint(
        accelerator.unwrap_model(model),
        optimizers_dict,
        schedulers_dict,
        epoch,
        global_step,
        best_eval_loss,
        save_path,
    )

    # Run final evaluation if enabled
    if accelerator.is_main_process and cfg.run.run_final_evaluation:
        log.info("Saving final model in Hugging Face format for evaluation...")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        # Brute-force fix for incorrect model_type in saved config.json
        if accelerator.is_main_process:
            import json

            try:
                config_path = save_path / "config.json"
                with open(config_path, "r") as f:
                    config_data = json.load(f)

                correct_model_type = cfg.model.type
                if config_data.get("model_type") != correct_model_type:
                    log.warning(
                        f"Overwriting incorrect model_type in config.json. Was: {config_data.get('model_type')}, should be: {correct_model_type}"
                    )
                    config_data["model_type"] = correct_model_type
                    with open(config_path, "w") as f:
                        json.dump(config_data, f, indent=2)
            except Exception as e:
                log.error(f"Failed to manually correct model_type in config.json: {e}")

        if cfg.logging.wandb.enabled and wandb.run is not None:
            from src.training.utils import save_wandb_info

            save_wandb_info(wandb.run, save_path)

        if cfg.lm_eval.enabled:
            log.info("Starting final benchmark evaluation...")

            # Free up GPU memory before starting the evaluation subprocess
            log.info("Releasing GPU memory before evaluation...")
            del model, optimizers_dict, schedulers_dict, train_loader, eval_loader, prepared_items
            accelerator.free_memory()
            torch.cuda.empty_cache()

            import json
            import subprocess

            eval_command = [
                "python",
                "run_benchmark_eval.py",
                "--model_path",
                str(save_path),
                "--tasks",
                cfg.lm_eval.tasks,
                "--batch_size",
                str(cfg.lm_eval.batch_size),
            ]
            log.info(f"Running evaluation command: {' '.join(eval_command)}")
            try:
                result = subprocess.run(
                    eval_command,
                    check=True,
                    stdout=subprocess.PIPE,  # Capture stdout for results
                    text=True,  # Decode stdout/stderr as text
                )
                eval_results_json = result.stdout
                eval_results = json.loads(eval_results_json)

                log.info("Benchmark evaluation complete.")

                # Log results to wandb from the main process
                if cfg.logging.wandb.enabled and wandb.run is not None:
                    log.info("Uploading evaluation results to wandb...")

                    summary_metrics = {}
                    for task, res in eval_results.get("results", {}).items():
                        for metric, value in res.items():
                            if isinstance(value, (int, float)):
                                summary_metrics[f"lm_eval/final/{task}/{metric}"] = value
                    wandb.run.summary.update(summary_metrics)

                    # Save results to a file and log as an artifact
                    output_filename = "final_benchmark_results.json"
                    output_path = save_path / output_filename
                    with open(output_path, "w") as f:
                        json.dump(eval_results, f, indent=2)

                    artifact = wandb.Artifact(
                        name=f"{wandb.run.name}-evaluation", type="evaluation-results"
                    )
                    artifact.add_file(str(output_path))
                    wandb.run.log_artifact(artifact)
                    log.info("Evaluation results uploaded to wandb.")

            except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
                log.error(f"Benchmark evaluation script failed or produced invalid output: {e}")
                if isinstance(e, subprocess.CalledProcessError):
                    log.error(f"Evaluation script stderr:\n{e.stderr}")

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
