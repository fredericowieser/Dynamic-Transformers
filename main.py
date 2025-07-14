import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training and evaluation, orchestrated by Hydra.

    Args:
        cfg: A DictConfig object composed by Hydra from YAML files and CLI overrides.
    """
    # Initial Setup
    log.info("--- Hydra Config File ---")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    log.info("------------------------------")

    # Set the global seed for reproducibility
    pl.seed_everything(cfg.run_params.seed, workers=True)

    # Instantiate DataModule
    # The DataModule handles all data loading, tokenization, and batching.
    log.info(f"Instantiating DataModule <{cfg.data_params._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data_params)

    # Instantiate Model
    # The LightningModule defines the model architecture and the training logic.
    log.info(f"Instantiating Model <{cfg.model_params._target_}>")
    model = hydra.utils.instantiate(
        cfg.model_params,
        training_params=cfg.training_params,
        lora_params=cfg.lora_params,
        _recursive_=False,  # We pass configs as objects, not instantiated classes
    )

    # Instantiate Loggers and Callbacks
    loggers = []
    if cfg.logging_params.tensorboard.enabled:
        log.info("Instantiating TensorBoard Logger")
        loggers.append(
            TensorBoardLogger(
                "logs/tensorboard", name=cfg.run_params.get("name")
            )
        )
    if cfg.logging_params.wandb.enabled:
        log.info("Instantiating WandB Logger")
        if not cfg.logging_params.wandb.entity:
            raise ValueError(
                "WandB logging is enabled, but `logging_params.wandb.entity` is not set."
            )
        loggers.append(
            WandbLogger(
                project=cfg.logging_params.wandb.project,
                entity=cfg.logging_params.wandb.entity,
                name=cfg.run_params.get("name"),
            )
        )

    # ModelCheckpoint callback to save the best model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{cfg.run_params.output_dir}/checkpoints",
        filename="best-model-{epoch}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    # Instantiate Trainer
    # The PyTorch Lightning Trainer handles the training loop, hardware, and more.
    log.info("Instantiating PyTorch Lightning Trainer")
    trainer = pl.Trainer(
        default_root_dir=cfg.run_params.output_dir,
        accelerator=cfg.run_params.device,
        devices="auto",
        max_steps=cfg.training_params.max_iters,
        val_check_interval=cfg.training_params.eval_interval,
        log_every_n_steps=10,
        logger=loggers,
        callbacks=[checkpoint_callback],
        precision=cfg.run_params.get("precision", "32-true"),
    )

    # Start Training
    log.info("--- Starting Training ---")
    trainer.fit(model=model, datamodule=datamodule)
    log.info("--- Training Finished ---")

    # Final Eval (Optional)
    if cfg.get("run_final_evaluation", True):
        log.info("--- Starting Final Evaluation on Test Set ---")
        # `trainer.test` will automatically use the best model checkpoint
        trainer.test(datamodule=datamodule, ckpt_path="best")
        log.info("--- Evaluation Finished ---")


if __name__ == "__main__":
    main()