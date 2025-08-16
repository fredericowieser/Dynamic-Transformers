import logging
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.models.d_qwen_causal_lm import DynamicQwenForCausalLM
from src.models.d_qwen_config import DynamicQwenConfig

log = logging.getLogger(__name__)

# Define the rolling window size
ROLLING_WINDOW_SIZE = 100

class DynamicQwenTrainer(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, training_cfg: DictConfig):
        super().__init__()
        # save for logging and checkpointing
        self.save_hyperparameters("model_cfg", "training_cfg")

        # --- START OF CHANGE ---
        log.info(f"Loading and configuring model: {model_cfg.model_name}")
        # Load the base config and set dynamic parameters
        config = DynamicQwenConfig.from_pretrained(
            model_cfg.model_name,
            dynamic_k=model_cfg.dynamic_k,
            ce_bias=model_cfg.ce_bias,
            gate_warmup_iters=training_cfg.gate_warmup_iters,
        )
        # Instantiate our DynamicQwen model with the configured config
        self.model = DynamicQwenForCausalLM(config)
        # --- END OF CHANGE ---

        # tokenizer (pad_token may need fixing)
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.model.config.pad_token_id
                or self.tokenizer.eos_token_id
            )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # learning rate
        self.lr = training_cfg.optimizer.lr

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("val/loss", outputs.loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("test/loss", outputs.loss, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)