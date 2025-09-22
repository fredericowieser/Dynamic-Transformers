import logging
import torch
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
import wandb
from lm_eval import evaluator, tasks

from src.data.mixed_dataset import MixedDataset
from src.training.utils import create_model
from src.training.eval_utils import LMEvalAdaptor
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

log = logging.getLogger(__name__)

def calculate_perplexity(model, dataloader, accelerator, limit_batches=50):
    """Calculates perplexity on a given dataloader."""
    model.eval()
    losses = []
    for step, batch in enumerate(dataloader):
        if step >= limit_batches:
            break
        with torch.no_grad():
            outputs = model(**batch)
        
        # Use 'lm_loss' if available, otherwise fall back to 'loss'
        loss_key = "lm_loss" if "lm_loss" in outputs else "loss"
        loss = outputs[loss_key]
        losses.append(accelerator.gather(loss.repeat(batch["input_ids"].shape[0])))

    losses = torch.cat(losses)
    avg_loss = torch.mean(losses)
    perplexity = torch.exp(avg_loss).item()
    log.info(f"Validation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    return perplexity

def run_lm_eval(model, tokenizer, accelerator, cfg):
    """Runs the lm-eval harness."""
    log.info("Running lm_eval benchmarks...")
    unwrapped_model = accelerator.unwrap_model(model)

    if cfg.peft.enabled and cfg.lm_eval.merge_lora_for_eval:
        log.info("Merging LoRA weights for lm-eval...")
        unwrapped_model = unwrapped_model.merge_and_unload()

    lm_eval_model = LMEvalAdaptor(unwrapped_model, tokenizer, accelerator.device)
    
    task_names = list(cfg.lm_eval.tasks)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=cfg.lm_eval.batch_size,
        device=str(accelerator.device),
        no_cache=True,
    )

    log.info(f"lm-eval results:\n{OmegaConf.to_yaml(results)}")
    return results

@hydra.main(config_path="config", config_name="eval", version_base="1.3")
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.logging.level, format='%(asctime)s - %(levelname)s - %(message)s')
    log.info(f"Configuration for evaluation:\n{OmegaConf.to_yaml(cfg)}")

    accelerator = Accelerator()

    if cfg.logging.wandb.enabled and accelerator.is_main_process:
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=f"eval-{cfg.run.name}",
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load Model
    log.info(f"Loading model checkpoint from: {cfg.checkpoint_path}")
    model = create_model(
        cfg.model.type,
        cfg.model.size,
        from_scratch=False, # Always loading a trained model for eval
        cfg=cfg
    )
    # Load the state dict from the checkpoint
    state_dict = torch.load(Path(cfg.checkpoint_path) / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)

    # Prepare Data
    datamodule = MixedDataset(**cfg.data, tokenizer_name=cfg.model.pretrained_model_name_or_path)
    datamodule.setup()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    eval_loader = torch.utils.data.DataLoader(
        datamodule.val_dataset,
        collate_fn=data_collator,
        batch_size=cfg.data.batch_size,
    )

    model, eval_loader = accelerator.prepare(model, eval_loader)

    # --- Run Evaluations ---
    
    # 1. Perplexity
    perplexity = calculate_perplexity(model, eval_loader, accelerator)
    if cfg.logging.wandb.enabled and accelerator.is_main_process:
        wandb.log({"eval/perplexity": perplexity})

    # 2. LM-Eval
    if cfg.lm_eval.enabled and accelerator.is_main_process:
        lm_eval_results = run_lm_eval(model, tokenizer, accelerator, cfg)
        if cfg.logging.wandb.enabled:
            # Log lm_eval results to wandb
            lm_eval_log = {}
            for task, res in lm_eval_results["results"].items():
                for metric, value in res.items():
                    if isinstance(value, (int, float)):
                        lm_eval_log[f"lm_eval/{task}/{metric}"] = value
            wandb.log(lm_eval_log)

    if cfg.logging.wandb.enabled and accelerator.is_main_process:
        wandb.finish()
    log.info("Evaluation complete.")