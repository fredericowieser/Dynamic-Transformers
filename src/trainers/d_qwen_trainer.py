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

        log.info(f"Loading and configuring model: {model_cfg.model_name}")
        # --- START OF CHANGE ---
        # Load the base config. dynamic_k, ce_bias, gate_warmup_iters will be None initially
        config = DynamicQwenConfig.from_pretrained(model_cfg.model_name)

        # Explicitly set dynamic parameters on the loaded config object
        # This mirrors the Llama trainer's approach to setting config attributes
        if model_cfg.dynamic_k is None:
            raise ValueError("model_cfg.dynamic_k must be provided in the Hydra config.")
        config.dynamic_k = model_cfg.dynamic_k

        if model_cfg.ce_bias is None:
            raise ValueError("model_cfg.ce_bias must be provided in the Hydra config.")
        config.ce_bias = model_cfg.ce_bias

        if training_cfg.gate_warmup_iters is None:
            raise ValueError("training_cfg.gate_warmup_iters must be provided in the Hydra config.")
        config.gate_warmup_iters = training_cfg.gate_warmup_iters

        # Instantiate our DynamicQwen model with the fully configured config
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