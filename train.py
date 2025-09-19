#!/usr/bin/env python3
"""Unified training script for all model architectures."""

import logging
from pathlib import Path
from typing import List, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from lm_eval import evaluator, tasks

from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb
from peft import LoraConfig, get_peft_model

from src.data.mixed_dataset import MixedDataset # Changed import
from src.training.utils import (
    create_model,
    save_checkpoint,
    setup_optimizer_and_scheduler,
)

from typing import Dict, Any # Added Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def train_step(model: nn.Module, batch: dict, **forward_kwargs) -> Dict[str, Any]:
    """Execute single training step."""
    outputs = model(**batch, **forward_kwargs)
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
    return outputs, loss


def evaluate(model, dataloader, accelerator, cfg, tokenizer):
    model.eval()
    losses = []
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs["loss"]
        losses.append(accelerator.gather(loss.repeat(batch["input_ids"].shape[0])))

    losses = torch.cat(losses)
    val_loss = torch.mean(losses).item()

    # lm_eval integration
    if cfg.lm_eval.enabled and accelerator.is_main_process:
        log.info("Running lm_eval benchmarks...")
        # Unwrap model for lm_eval
        unwrapped_model = accelerator.unwrap_model(model)

        # Merge LoRA weights if enabled and configured
        if cfg.peft.enabled and cfg.lm_eval.merge_lora_for_eval:
            log.info("Merging LoRA weights for lm_eval...")
            unwrapped_model = unwrapped_model.merge_and_unload()

        # Prepare model for lm_eval
        class LMEvalModel(unwrapped_model.__class__):
            def __init__(self, model, tokenizer):
                super().__init__(model.config)
                self.model = model # The actual model (unwrapped, potentially merged)
                self.tokenizer = tokenizer

            def _model_call(self, inputs):
                return self.model(inputs["input_ids"].to(self.model.device)).logits

            def tok_encode(self, string: str, **kwargs) -> List[int]:
                return self.tokenizer.encode(string, **kwargs)

            def tok_decode(self, tokens: List[int], **kwargs) -> str:
                return self.tokenizer.decode(tokens, **kwargs)

            def _loglikelihood_tokens(self, requests, disable_tqdm=False):
                # This is a simplified version. lm_eval expects a specific format.
                # For full implementation, refer to lm_eval's HFLM class.
                res = []
                for context, continuation in requests:
                    # Encode context and continuation
                    context_enc = self.tok_encode(context)
                    continuation_enc = self.tok_encode(continuation)

                    # Prepare input_ids and labels
                    input_ids = torch.tensor(context_enc + continuation_enc, dtype=torch.long).unsqueeze(0).to(self.model.device)
                    labels = torch.tensor(context_enc + continuation_enc, dtype=torch.long).unsqueeze(0).to(self.model.device)
                    labels[:, :len(context_enc)] = -100 # Mask context

                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        # FIX: Access loss using dictionary key for consistency with model outputs
                        neg_log_likelihood = outputs["loss"]
                    res.append((neg_log_likelihood.item(), True)) # (loglikelihood, is_greedy)
                return res

        lm_eval_model = LMEvalModel(unwrapped_model, tokenizer)

        # Load lm_eval tasks
        task_names = cfg.lm_eval.tasks
        if "all" in task_names:
            task_names = tasks.ALL_TASKS

        results = evaluator.simple_evaluate(
            model=lm_eval_model,
            tasks=task_names,
            batch_size=cfg.lm_eval.batch_size,
            device=accelerator.device,
            no_cache=True, # Set to False for faster subsequent runs
        )

        log.info(f"lm_eval results: {results}")

        # Log lm_eval results to wandb
        if cfg.logging.wandb.enabled:
            lm_eval_log = {"lm_eval": {}}
            for task_name, task_results in results["results"].items():
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)):
                        lm_eval_log["lm_eval"][f"{task_name}/{metric}"] = value
            wandb.log(lm_eval_log, step=accelerator.num_steps)

        # Unmerge LoRA weights if they were merged
        if cfg.peft.enabled and cfg.lm_eval.merge_lora_for_eval:
            log.info("Unmerging LoRA weights...")
            # This is a placeholder. PEFT doesn't have a direct unmerge.
            # You'd typically reload the original model or save/load the adapter state.
            # For now, we'll just log a warning if the model is not in training mode.
            if not model.training:
                log.warning("Model is not in training mode after lm_eval. If you merged LoRA, you might need to reload the model or its adapters.")

    return val_loss


@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    """Main training loop."""
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Setup accelerator
    accelerator = Accelerator(
        mixed_precision=cfg.system.precision,
        gradient_accumulation_steps=cfg.training.accumulate_grad_batches,
    )

    # Initialize Weights & Biases
    if cfg.logging.wandb.enabled and accelerator.is_main_process:
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=cfg.run.name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Load tokenizer
    tokenizer_path = cfg.data.tokenizer_name # Use the tokenizer path from config
    log.info(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create data module (replaces get_dataloader)
    log.info("Setting up data module...")
    datamodule = MixedDataset(
        dataset_configs=cfg.data.dataset_configs,
        tokenizer_name=cfg.data.tokenizer_name,
        block_size=cfg.data.block_size,
        batch_size=cfg.data.batch_size, # Passed for Hydra compatibility
        validation_split_percentage=cfg.data.validation_split_percentage,
    )
    datamodule.setup() # Process and load datasets

    # Create data loaders
    log.info("Creating data loaders...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # Define data_collator here

    train_loader = torch.utils.data.DataLoader(
        datamodule.train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.system.num_workers, # Use num_workers from system config
        pin_memory=cfg.system.pin_memory,
        drop_last=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        datamodule.val_dataset,
        shuffle=False, # No need to shuffle validation data
        collate_fn=data_collator,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.system.num_workers, # Use num_workers from system config
        pin_memory=cfg.system.pin_memory,
        drop_last=False, # Don't drop last batch for evaluation
    )

    # Create model
    log.info(f"Creating {cfg.model.type} model ({cfg.model.size}, from_scratch={cfg.training.mode == 'scratch'})")
    model = create_model(
        cfg.model.type,
        cfg.model.size,
        cfg.training.mode == 'scratch',
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

    # Setup training
    steps_per_epoch = len(train_loader) // cfg.training.accumulate_grad_batches
    num_training_steps = steps_per_epoch * cfg.training.num_epochs

    if "router" not in cfg.model or "beta_schedule" not in cfg.model.router:
        raise ValueError("Missing cfg.model.router.beta_schedule")
    sched_cfg = cfg.model.router.beta_schedule
    required_sched_keys = ["type", "beta_ce_start", "beta_ce_end", "beta_cu_start", "beta_cu_end", "warmup_steps"]
    for k in required_sched_keys:
        if k not in sched_cfg:
            raise ValueError(f"Missing cfg.model.router.beta_schedule.{k}")

    def compute_beta_values(step: int, total_steps: int) -> Tuple[float, float]:
        # Warmup then schedule over [warmup_steps, total_steps]
        warmup = int(sched_cfg.warmup_steps)
        stype = sched_cfg.type  # 'linear' or 'cosine' (extend as needed)

        if step <= warmup:
            r = 0.0
        else:
            denom = max(1, total_steps - warmup)
            r = min(1.0, (step - warmup) / denom)

        def slinear(s0, s1):
            return s0 + r * (s1 - s0)

        def scos(s0, s1):
            # cosine from s0 to s1
            import math
            return s0 + 0.5 * (1.0 - math.cos(math.pi * r)) * (s1 - s0)

        f = slinear if stype == "linear" else scos
        beta_ce = f(float(sched_cfg.beta_ce_start), float(sched_cfg.beta_ce_end))
        beta_cu = f(float(sched_cfg.beta_cu_start), float(sched_cfg.beta_cu_end))
        return beta_ce, beta_cu

    optimizer, scheduler = setup_optimizer_and_scheduler(
        model, cfg, num_training_steps
    )

    # Prepare for distributed training
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    # Training loop
    log.info("Starting training...")
    global_step = 0
    best_eval_loss = float('inf')

    for epoch in range(cfg.training.num_epochs):
        model.train()
        progress_bar = tqdm(range(len(train_loader)), disable=not accelerator.is_main_process)
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            if progress_bar.n >= num_training_steps:
                break
            
            with accelerator.accumulate(model):
                # Compute scheduled β values for this step
                beta_ce, beta_cu = compute_beta_values(global_step, num_training_steps)

                # Forward with scheduled β (only for tdtf)
                forward_kwargs = {}
                if cfg.model.type == "tdtf":
                    forward_kwargs["beta_ce"] = beta_ce
                    forward_kwargs["beta_cu"] = beta_cu

                outputs, loss = train_step(model, batch, **forward_kwargs)
                accelerator.backward(loss)

                if cfg.training.use_gradient_clipping:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Extract auxiliary losses and router stats
            total_prior_loss = outputs.get("prior_loss", torch.tensor(0.0, device=loss.device)).detach().float()
            causal_loss = outputs.get("causal_loss", torch.tensor(0.0, device=loss.device)).detach().float()
            aux_loss = outputs.get("aux_loss", torch.tensor(0.0, device=loss.device)).detach().float() # For MoD
            router_stats = outputs.get("router_stats", {})

            # Calculate perplexity
            perplexity = torch.exp(loss.detach().float())

            # Prepare metrics for logging
            log_metrics = {
                "train/loss": loss.detach().float().item(),
                "train/perplexity": perplexity.item(),
            }
            if cfg.model.type == "tdtf":
                log_metrics["train/beta_ce"] = beta_ce
                log_metrics["train/beta_cu"] = beta_cu

            current_prior_loss_weight = 0.0
            if cfg.model.get("prior_loss_schedule") and total_prior_loss > 0:
                schedule_cfg = cfg.model.prior_loss_schedule
                initial_w = schedule_cfg.initial_weight
                final_w = schedule_cfg.final_weight
                decay_steps = schedule_cfg.decay_steps

                if decay_steps > 0 and global_step < decay_steps:
                    progress = global_step / decay_steps
                    current_prior_loss_weight = initial_w - progress * (initial_w - final_w)
                else:
                    current_prior_loss_weight = final_w
                
                loss = loss + total_prior_loss * current_prior_loss_weight
            
            log_metrics["train/prior_loss"] = total_prior_loss.item()
            log_metrics["train/prior_loss_weight"] = current_prior_loss_weight
            log_metrics["train/causal_loss"] = causal_loss.item()
            log_metrics["train/mod_aux_loss"] = aux_loss.item() # For MoD


            # Process and add router stats
            def process_router_stats(stats: Dict[str, Any], model_type: str) -> Dict[str, float]:
                processed = {}
                if model_type == "dtf":
                    # Log VPR signals
                    if "S_CE_mean" in stats:
                        processed["train_vpr_signals/S_CE_mean"] = stats["S_CE_mean"]
                    if "S_CU_mean" in stats:
                        processed["train_vpr_signals/S_CU_mean"] = stats["S_CU_mean"]
                    if "G_cont_mean" in stats:
                        processed["train_vpr_signals/G_cont_mean"] = stats["G_cont_mean"]
                    
                    # Log VPR router parameters
                    if "beta_ce" in stats:
                        processed["train_vpr_router/beta_ce"] = stats["beta_ce"]
                    if "beta_cu" in stats:
                        processed["train_vpr_router/beta_cu"] = stats["beta_cu"]
                    if "o_ce" in stats:
                        processed["train_vpr_router/o_ce"] = stats["o_ce"]
                    if "m_cu" in stats:
                        processed["train_vpr_router/m_cu"] = stats["m_cu"]

                elif model_type == "mod":
                    for i, layer_stats in enumerate(stats):
                        for k, v in layer_stats.items():
                            processed[f"train/router_stats/layer_{i}/{k}"] = v
                elif model_type == "tdtf":
                    for k, v_list in stats.items():
                        if isinstance(v_list, list) and len(v_list) > 0:
                            processed[f"train/router_stats/{k}_avg"] = sum(v_list) / len(v_list)
                return processed

            processed_router_stats = process_router_stats(router_stats, cfg.model.type)
            log_metrics.update(processed_router_stats)

            if accelerator.is_main_process and (global_step + 1) % cfg.logging.wandb.log_interval == 0:
                wandb.log(log_metrics, step=global_step)
                accelerator.print(
                    f"Epoch {epoch}, Step {global_step+1}: "
                    f"Loss = {loss.detach().float():.4f}, "
                    f"Perplexity = {perplexity:.2f}, "
                    f"Prior Loss = {log_metrics['train/prior_loss']:.4f}, "
                    f"Causal Loss = {log_metrics['train/causal_loss']:.4f}, "
                    f"MoD Aux Loss = {log_metrics['train/mod_aux_loss']:.4f}, "
                    f"Router Stats Logged: {bool(processed_router_stats)}"
                )

            global_step += 1
            progress_bar.update(1)

            # Evaluation and checkpointing
            if (global_step) % cfg.training.eval_interval == 0:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                eval_loss = evaluate(unwrapped_model, eval_loader, accelerator, cfg, tokenizer)
                eval_perplexity = torch.exp(torch.tensor(eval_loss))

                wandb.log({
                    "val/loss": eval_loss,
                    "val/perplexity": eval_perplexity.item(),
                }, step=global_step)
                accelerator.print(f"Validation Loss: {eval_loss:.4f}, Validation Perplexity: {eval_perplexity:.2f}")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    save_checkpoint(
                        unwrapped_model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        best_eval_loss,
                        Path(cfg.run.output_dir) / "best_model"
                    )
                accelerator.wait_for_everyone()
                model.train() # Resume training mode

                # Early stopping
                if cfg.training.max_steps > 0 and global_step >= cfg.training.max_steps:
                    log.info(f"Reached max steps ({cfg.training.max_steps})")
                    break

        if cfg.training.max_steps > 0 and global_step >= cfg.training.max_steps:
            break

    # Save final model
    save_path = Path(cfg.run.output_dir) / "final_model"
    save_checkpoint(accelerator.unwrap_model(model), optimizer, scheduler, epoch, global_step, best_eval_loss, save_path)

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
