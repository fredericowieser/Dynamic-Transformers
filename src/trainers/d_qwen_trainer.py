import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer
from omegaconf import DictConfig
from hydra.utils import instantiate

class DynamicQwenTrainer(pl.LightningModule):
    """
    LightningModule for fine-tuning DynamicQwenForCausalLM models.
    Minimal, single-optimizer setup. Expects:
      cfg.model: _target_ = src.models.d_qwen_causal_lm.DynamicQwenForCausalLM
      cfg.model.model_cfg.model_name: pretrained name
      cfg.training.optimizer.lr: learning rate
    """
    def __init__(self, model_cfg: DictConfig, training_cfg: DictConfig):
        super().__init__()
        # save for logging and checkpointing
        self.save_hyperparameters("model_cfg", "training_cfg")

        # instantiate our DynamicQwen model via Hydra
        self.model = instantiate(model_cfg, _convert_="partial")

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