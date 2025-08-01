import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg: DictConfig) -> None:
    log.info(f"--- Config ---\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.run.seed, workers=True)

    # Instantiate DataModule
    log.info(f"Instantiating DataModule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data, _convert_="partial")

    # Instantiate Model
    log.info(f"Instantiating Model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model, training_cfg=cfg.training)

    # Instantiate Loggers
    loggers = []
    if cfg.logging.tensorboard.enabled:
        loggers.append(TensorBoardLogger("logs/tensorboard", name=cfg.run.name))
    if cfg.logging.wandb.enabled:
        if not cfg.logging.wandb.entity:
            raise ValueError("WandB entity not set.")
        loggers.append(
            WandbLogger(
                project=cfg.logging.wandb.project,
                entity=cfg.logging.wandb.entity,
                name=cfg.run.name,
            )
        )

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.run.output_dir, "checkpoints"),
        filename="best-model-{epoch}-{val/loss:.2f}",
        save_top_k=1,
        monitor="val/loss",
        mode="min",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # Instantiate Trainer
    log.info("Instantiating Trainer")
    trainer = pl.Trainer(
        default_root_dir=cfg.run.output_dir,
        accelerator=cfg.run.device,
        devices="auto",
        max_steps=cfg.training.max_iters,
        val_check_interval=cfg.training.eval_interval,
        check_val_every_n_epoch=None,
        log_every_n_steps=10,
        logger=loggers,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=cfg.run.precision,
    )

    # Train
    log.info("--- Starting Training ---")
    trainer.fit(model=model, datamodule=datamodule)
    log.info("--- Training Finished ---")

    # Test
    if cfg.run.run_final_evaluation:
        log.info("--- Starting Final Evaluation on Test Set ---")
        trainer.test(datamodule=datamodule, ckpt_path="best")
        log.info("--- Evaluation Finished ---")

    # Save final model
    log.info("--- Saving Final Model ---")
    save_path = os.path.join(cfg.run.output_dir, "final_model")

    # Access the underlying Hugging Face model
    hf_model = model.model

    # --- FIX: Convert ListConfig to list before saving ---
    # These attributes are Omegaconf ListConfig objects due to Hydra parsing.
    # They need to be converted to native Python lists for JSON serialization.
    if hasattr(hf_model.config, 'lora_target_modules_main') and OmegaConf.is_list(hf_model.config.lora_target_modules_main):
        hf_model.config.lora_target_modules_main = OmegaConf.to_container(hf_model.config.lora_target_modules_main, resolve=True)
    if hasattr(hf_model.config, 'lora_target_modules_prior_ffn') and OmegaConf.is_list(hf_model.config.lora_target_modules_prior_ffn):
        hf_model.config.lora_target_modules_prior_ffn = OmegaConf.to_container(hf_model.config.lora_target_modules_prior_ffn, resolve=True)
    
    # Save the underlying Hugging Face model with the updated config
    hf_model.save_pretrained(save_path, safe_serialization=True)
    # Save the tokenizer
    model.tokenizer.save_pretrained(save_path)
    log.info(f"Final model saved to {save_path}")


if __name__ == "__main__":
    main()
