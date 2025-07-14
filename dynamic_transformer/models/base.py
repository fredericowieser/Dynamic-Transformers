# dynamic_transformer/models/base.py
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoConfig
import torch
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)


class HuggingFaceModel(pl.LightningModule):
    """
    Wrapper for a Hugging Face Causal LM.

    This class handles loading pre-trained models, applying PEFT/LoRA modifications,
    and defining the training, validation, and test logic.
    """

    def __init__(
        self,
        base_model_id: str,
        training_params: DictConfig,
        lora_params: DictConfig,
    ):
        """
        Args:
            base_model_id: The Hugging Face model identifier (e.g., "gpt2").
            training_params: A DictConfig object with optimizer and learning rate settings.
            lora_params: A DictConfig object with LoRA settings.
        """
        super().__init__()
        # `save_hyperparameters` is crucial for Lightning to know about these configs,
        # allowing for easy checkpointing and loading.
        self.save_hyperparameters()

        # Load model configuration
        config = AutoConfig.from_pretrained(base_model_id)

        # Load the base model
        log.info(f"Loading base model: {base_model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id, config=config
        )

        # Apply LoRA if enabled in the config
        if lora_params.enabled:
            self._apply_lora(lora_params)

    def _apply_lora(self, lora_cfg: DictConfig):
        """Applies LoRA modifications to the model using PEFT."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
        except ImportError:
            raise ImportError(
                "PEFT is not installed. Please install it with `pip install peft` to use LoRA."
            )

        log.info("LoRA is enabled. Applying PEFT modifications...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            target_modules=list(lora_cfg.target_modules),
        )
        self.model = get_peft_model(self.model, peft_config)
        log.info("Trainable parameters after applying LoRA:")
        self.model.print_trainable_parameters()

    def forward(self, batch):
        """
        Args:
            batch: A dictionary containing 'input_ids', 'attention_mask', and 'labels'.

        Returns:
            The output from the underlying Hugging Face model, which includes the loss.
        """
        # The model expects keyword arguments, so we unpack the dictionary
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        """Performs a single training step."""
        outputs = self(batch)
        loss = outputs.loss
        # Log the training loss for monitoring
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step."""
        outputs = self(batch)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Performs a single test step."""
        outputs = self(batch)
        loss = outputs.loss
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        This method is called by PyTorch Lightning to get the optimizer.
        """
        # We can support multiple optimizers based on the config
        optimizer_name = self.hparams.training_params.optimizer_name.lower()
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),  # `self.parameters()` automatically includes only trainable params
                lr=self.hparams.training_params.learning_rate,
                weight_decay=self.hparams.training_params.weight_decay,
                betas=(
                    self.hparams.training_params.beta1,
                    self.hparams.training_params.beta2,
                ),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Here you could also add a learning rate scheduler if needed
        return optimizer