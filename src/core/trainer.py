import logging
import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from src.models.dynamic_llama import DynamicLlamaDecoderLayer

log = logging.getLogger(__name__)


class LightningModel(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, training_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg

        log.info(f"Loading pre-trained model: {self.model_cfg.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self._modify_model_architecture()
        self._setup_parameter_groups()

    def _modify_model_architecture(self):
        log.info("Replacing LlamaDecoderLayer with DynamicLlamaDecoderLayer...")
        new_layers = nn.ModuleList()
        for i, layer in enumerate(self.model.model.layers):
            custom_layer = DynamicLlamaDecoderLayer(self.model.config, i)
            custom_layer.load_state_dict(layer.state_dict(), strict=False)
            new_layers.append(custom_layer)
        self.model.model.layers = new_layers
        log.info("All Llama decoder layers have been replaced.")

    def _setup_parameter_groups(self):
        log.info("Setting up parameter groups for differential learning rates.")
        self.original_params = []
        self.new_prior_params = []
        for name, param in self.model.named_parameters():
            if "prior_ffn" in name or "prior_layernorm" in name:
                self.new_prior_params.append(param)
            else:
                param.requires_grad = True
                self.original_params.append(param)
        log.info(
            f"Found {len(self.original_params)} original parameters and "
            f"{len(self.new_prior_params)} new prior parameters."
        )

    def forward(self, **inputs):
        # The custom layer returns an extra element (prior_loss).
        # We must manually iterate through the layers to capture it.
        hidden_states = self.model.model.embed_tokens(inputs["input_ids"])
        attention_mask = inputs.get("attention_mask")
        
        prior_losses = []
        for layer in self.model.model.layers:
            layer_outputs = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0]
            prior_losses.append(layer_outputs[-1])

        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
        
        avg_prior_loss = torch.stack(prior_losses).mean()
        return logits, avg_prior_loss

    def _calculate_loss(self, batch):
        logits, prior_loss = self.forward(**batch)
        
        # Shift so that models predicting the next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()

        # Standard Language Modeling Loss
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )

        # Total loss
        total_loss = lm_loss + self.model_cfg.prior_loss_weight * prior_loss
        
        # Perplexity
        perplexity = torch.exp(lm_loss)

        return total_loss, lm_loss, prior_loss, perplexity

    def training_step(self, batch, batch_idx):
        total_loss, lm_loss, prior_loss, perplexity = self._calculate_loss(batch)
        self.log("train/loss", total_loss)
        self.log("train/lm_loss", lm_loss, prog_bar=True)
        self.log("train/prior_loss", prior_loss, prog_bar=True)
        self.log("train/perplexity", perplexity)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, lm_loss, prior_loss, perplexity = self._calculate_loss(batch)
        self.log("val/loss", total_loss)
        self.log("val/lm_loss", lm_loss, prog_bar=True)
        self.log("val/perplexity", perplexity)

    def test_step(self, batch, batch_idx):
        total_loss, lm_loss, prior_loss, perplexity = self._calculate_loss(batch)
        self.log("test/loss", total_loss)
        self.log("test/lm_loss", lm_loss)
        self.log("test/perplexity", perplexity)

    def configure_optimizers(self):
        param_groups = [
            {"params": self.original_params, "lr": self.training_cfg.optimizer.base_lr},
            {"params": self.new_prior_params, "lr": self.training_cfg.optimizer.prior_ffn_lr},
        ]
        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.training_cfg.optimizer.weight_decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.training_cfg.scheduler.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }