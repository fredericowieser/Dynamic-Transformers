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
from transformers import AutoTokenizer

from src.data.fineweb_dataloader import FineWebDataloader
from src.training.utils import (create_model, save_checkpoint,
                                setup_optimizer_and_scheduler)

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.logging.level, format="%(asctime)s - %(levelname)s - %(message)s")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    accelerator = Accelerator(
        mixed_precision=cfg.system.precision,
        gradient_accumulation_steps=cfg.training.accumulate_grad_batches,
    )

    if accelerator.is_main_process:
        log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        if cfg.logging.wandb.enabled:
            wandb.init(
                project=cfg.logging.wandb.project,
                entity=cfg.logging.wandb.entity,
                name=cfg.run.name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Setting up FineWebDataloaders...")
    train_loader = FineWebDataloader(
        tokenizer=tokenizer,
        B=cfg.data.batch_size,
        T=cfg.data.block_size,
        split="train",
        data_dir=cfg.data.local_dir,
        num_shards=cfg.data.num_shards_train,
        device=accelerator.device,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    eval_loader = FineWebDataloader(
        tokenizer=tokenizer,
        B=cfg.data.batch_size,
        T=cfg.data.block_size,
        split="val",
        data_dir=cfg.data.local_dir,
        num_shards=cfg.data.num_shards_train, # Reuse same shards pool
        device=accelerator.device,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )

    log.info(f"Creating {cfg.model.type} model...")
    model = create_model(cfg.model.type, cfg)

    if cfg.peft.enabled:
        log.info("Applying LoRA to the model.")
        peft_config = LoraConfig(**cfg.peft.config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {total_params/1e6:.1f}M params ({trainable_params/1e6:.1f}M trainable)")

    if cfg.training.max_steps > 0:
        num_training_steps = cfg.training.max_steps
        log.info(f"Training will run for {num_training_steps} steps.")
    else:
        log.info(f"max_steps not provided, calculating based on num_epochs: {cfg.training.num_epochs}")
        if cfg.data.name == "fineweb":
            # karpathy/fineweb-edu-100b-shuffle has ~54.85M tokens per shard.
            # With best-fit packing and 35% crop loss, we have ~35.65M tokens per shard.
            # Each sample is block_size tokens.
            SAMPLES_PER_SHARD = int((54.85e6 * 0.65) // cfg.data.block_size)
            num_samples = train_loader.num_shards * SAMPLES_PER_SHARD
            num_batches_per_epoch = num_samples // cfg.data.batch_size
            steps_per_epoch = num_batches_per_epoch // (cfg.training.accumulate_grad_batches * accelerator.num_processes)
            num_training_steps = int(steps_per_epoch * cfg.training.num_epochs)
        else:
            raise ValueError(
                f"Epoch-based training is only supported for 'fineweb' dataset with streaming. "
                f"Got '{cfg.data.name}'. Please specify training.max_steps."
            )

        if num_training_steps <= 0:
            raise ValueError(
                f"Calculated training steps are {num_training_steps}. Please increase num_epochs."
            )
        log.info(f"Training for {cfg.training.num_epochs} epochs, estimated to be {num_training_steps} steps.")

    optimizers_dict, schedulers_dict = setup_optimizer_and_scheduler(
        model, cfg, num_training_steps, accelerator
    )

    prepared_items = accelerator.prepare(
        model, *optimizers_dict.values(), train_loader, eval_loader, *schedulers_dict.values()
    )
    model = prepared_items[0]
    prepared_optimizers = prepared_items[1 : 1 + len(optimizers_dict)]
    train_loader, eval_loader = prepared_items[1 + len(optimizers_dict) : 3 + len(optimizers_dict)]
    prepared_schedulers = prepared_items[3 + len(optimizers_dict) :]

    optimizers_dict = dict(zip(optimizers_dict.keys(), prepared_optimizers))
    schedulers_dict = dict(zip(schedulers_dict.keys(), prepared_schedulers))

    log.info("Starting training...")
    global_step = 0
    best_eval_loss = float("inf")
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_main_process)

    model.train()
    for batch in train_loader:
        if global_step >= num_training_steps:
            break

        batch_x, batch_y = batch
        batch_x = batch_x.to(accelerator.device)
        batch_y = batch_y.to(accelerator.device)

        with accelerator.accumulate(model):
            outputs = model(
                input_ids=batch_x,
                labels=batch_y,
                global_step=global_step,
                max_steps=num_training_steps,
            )
            loss = outputs["loss"]
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
                log_metrics = {"train/loss": loss.item()}
                causal_info = ""
                
                if "aux_metrics" in outputs:
                    for k, v in outputs["aux_metrics"].items():
                        if "causal_router" in k:
                            log_metrics[f"train/{k}"] = v
                            if "loss" in k:
                                causal_info += f", Causal Loss: {v:.4f}"
                            elif "acc" in k:
                                causal_info += f", Causal Acc: {v:.4f}"

                if cfg.logging.wandb.enabled:
                    wandb.log(log_metrics, step=global_step)

                accelerator.print(f"Step {global_step}: Loss = {loss.item():.4f}{causal_info}")

            if global_step > 0 and global_step % cfg.training.eval_interval == 0:
                model.eval()
                eval_losses = []
                eval_batches = []
                for i, eval_batch in enumerate(eval_loader):
                    if i >= 50:
                        break
                    eval_x, eval_y = eval_batch
                    eval_x = eval_x.to(accelerator.device)
                    eval_y = eval_y.to(accelerator.device)
                    eval_batches.append((eval_x, eval_y))

                for eval_batch in eval_batches:
                    with torch.no_grad():
                        eval_x, eval_y = eval_batch
                        val_outputs = model(input_ids=eval_x, labels=eval_y)
                        # Unsqueeze the loss to make it a 1D tensor before gathering
                        loss_tensor = val_outputs["loss"].unsqueeze(0)
                        eval_losses.append(accelerator.gather(loss_tensor))

                if eval_losses:
                    val_loss = torch.mean(torch.cat(eval_losses))
                    val_perplexity = torch.exp(val_loss)

                    if accelerator.is_main_process:
                        if cfg.logging.wandb.enabled:
                            wandb.log(
                                {
                                    "val/loss": val_loss.item(),
                                    "val/perplexity": val_perplexity.item(),
                                },
                                step=global_step,
                            )
                        accelerator.print(
                            f"Validation Loss: {val_loss.item():.4f}, Perplexity: {val_perplexity.item():.2f}"
                        )

                        if val_loss < best_eval_loss:
                            best_eval_loss = val_loss
                            best_model_path = Path(cfg.run.output_dir) / "best_model"
                            log.info(f"New best model! Saving to {best_model_path}")
                            accelerator.unwrap_model(model).save_pretrained(best_model_path)
                            tokenizer.save_pretrained(best_model_path)

                model.train()

    # Final save and evaluation
    if accelerator.is_main_process:
        log.info("Training complete. Saving final model...")
        save_path = Path(cfg.run.output_dir) / "final_model"
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        # Ensure config.json has the correct model_type and architectures for AutoModel and lm-eval
        if accelerator.is_main_process:
            try:
                config_path = save_path / "config.json"
                with open(config_path, "r") as f:
                    config_data = json.load(f)

                # model_type is set by the config class, but we'll double check
                config_data["model_type"] = cfg.model.type
                
                model_class_map = {
                    "standard": "StandardTransformerForCausalLM",
                    "mod": "MoDForCausalLM",
                    "sdt": "SDTForCausalLM",
                    "stt": "STTForCausalLM",
                }
                correct_arch = model_class_map.get(cfg.model.type)
                if correct_arch:
                    config_data["architectures"] = [correct_arch]

                with open(config_path, "w") as f:
                    json.dump(config_data, f, indent=2)
                log.info(f"Final config.json verified for model_type '{cfg.model.type}' and architecture '{correct_arch}'")
            except Exception as e:
                log.error(f"Failed to verify config.json: {e}")

        if cfg.run.run_final_evaluation and cfg.lm_eval.enabled:
            log.info("Starting final benchmark evaluation...")
            # Free up memory before eval
            del model, optimizers_dict, schedulers_dict, train_loader, eval_loader
            accelerator.free_memory()
            torch.cuda.empty_cache()

            import subprocess

            eval_command = [
                "python",
                "bench.py",
                "--model_path",
                str(save_path),
                "--tasks",
                cfg.lm_eval.tasks,
                "--batch_size",
                str(cfg.lm_eval.batch_size),
            ]
            log.info(f"Running evaluation: {' '.join(eval_command)}")
            try:
                result = subprocess.run(eval_command, check=True, capture_output=True, text=True)
                log.info("Benchmark evaluation complete.")
                if cfg.logging.wandb.enabled:
                    eval_results = json.loads(result.stdout)
                    summary_metrics = {
                        f"lm_eval/final/{task}/{metric}": value
                        for task, res in eval_results.get("results", {}).items()
                        for metric, value in res.items()
                        if isinstance(value, (int, float))
                    }
                    wandb.run.summary.update(summary_metrics)
            except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
                log.error(f"Benchmark evaluation failed: {e}")
                if isinstance(e, subprocess.CalledProcessError):
                    log.error(f"Stderr: {e.stderr}")

        if cfg.logging.wandb.enabled:
            wandb.finish()


if __name__ == "__main__":
    main()
