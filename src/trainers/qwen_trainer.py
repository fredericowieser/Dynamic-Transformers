import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, get_scheduler

from ..models.qwen.causal_lm import DynamicQwenForCausalLM
from .gate_logging import GateLogger

log = logging.getLogger(__name__)


class DynamicQwenTrainer(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, training_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg

        log.info(f"Loading and configuring model: {self.model_cfg.model_name}")
        
        # Centralized configuration is now handled by the model's from_pretrained method
        self.model = DynamicQwenForCausalLM.from_pretrained(
            self.model_cfg.model_name,
            model_cfg=OmegaConf.to_container(self.model_cfg, resolve=True)
        )
        
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.gate_logger = GateLogger(self.model.config.num_hidden_layers)

    def forward(self, **inputs):
        return self.model(
            **inputs,
            current_iter=self.global_step,
            return_dict=True,
        )

    def _calculate_loss(self, batch):
        model_output = self.forward(**batch)
        
        shift_logits = model_output.logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        total_loss = lm_loss
        
        # Add auxiliary loss for VPR architecture if it exists
        if model_output.prior_loss is not None:
            # You can add a hyperparameter to scale this loss if needed
            total_loss += model_output.prior_loss

        perplexity = torch.exp(lm_loss)

        if model_output.gate_vectors_per_layer:
            overall_gate_activation_mean = torch.stack(
                [gv.mean() for gv in model_output.gate_vectors_per_layer]
            ).mean()
            per_layer_gate_stats = [
                {"mean": gv.mean(), "std": gv.std() if gv.numel() > 1 else torch.tensor(0.0)}
                for gv in model_output.gate_vectors_per_layer
            ]
        else:
            overall_gate_activation_mean = torch.tensor(0.0)
            per_layer_gate_stats = []

        # This tuple is now populated safely from the dataclass
        return (
            total_loss,
            lm_loss,
            model_output.prior_loss,
            perplexity,
            overall_gate_activation_mean,
            model_output.avg_ce_proportion,
            model_output.avg_cu_proportion,
            per_layer_gate_stats,
            model_output.avg_beta_ce,
            model_output.avg_beta_cu,
            model_output.avg_cu_detection_multiplier,
            model_output.avg_ce_criterion_offset,
            model_output.combined_gating_signal_mean,
        )

    def _log_step_metrics(self, prefix: str, outputs: tuple, on_step: bool, on_epoch: bool):
        (
            total_loss, lm_loss, prior_loss, perplexity, gate_mean,
            avg_ce, avg_cu, per_layer_stats, beta_ce, beta_cu,
            cu_multiplier, ce_offset, gating_signal_mean
        ) = outputs

        self.log(f"{prefix}/loss", total_loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True)
        self.log(f"{prefix}/lm_loss", lm_loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True)
        self.log(f"{prefix}/perplexity", perplexity, on_step=on_step, on_epoch=on_epoch, prog_bar=True)

        if prior_loss is not None:
            self.log(f"{prefix}/prior_loss", prior_loss, on_step=on_step, on_epoch=on_epoch)
        if avg_ce is not None:
            self.log(f"{prefix}_vpr/avg_ce_proportion", avg_ce, on_step=on_step, on_epoch=on_epoch, prog_bar=True)
        if avg_cu is not None:
            self.log(f"{prefix}_vpr/avg_cu_proportion", avg_cu, on_step=on_step, on_epoch=on_epoch, prog_bar=True)
        if gating_signal_mean is not None:
            self.log(f"{prefix}_vpr/gating_signal_mean", gating_signal_mean, on_step=on_step, on_epoch=on_epoch)
        
        # Log router parameters if they exist (VPR specific)
        if beta_ce is not None:
            self.log(f"{prefix}_vpr/router_beta_ce", beta_ce, on_step=on_step, on_epoch=on_epoch)
            self.log(f"{prefix}_vpr/router_beta_cu", beta_cu, on_step=on_step, on_epoch=on_epoch)
            self.log(f"{prefix}_vpr/router_cu_multiplier", cu_multiplier, on_step=on_step, on_epoch=on_epoch)
            self.log(f"{prefix}_vpr/router_ce_offset", ce_offset, on_step=on_step, on_epoch=on_epoch)

        GateLogger.log_gate_metrics(self, prefix, gate_mean, per_layer_stats, on_step, on_epoch)

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        
        outputs = self._calculate_loss(batch)
        total_loss = outputs[0]
        
        self.manual_backward(total_loss)
        
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.training_cfg.gradient_clip_val,
            gradient_clip_algorithm="norm" # Or "value"
        )
        
        optimizer.step()
        
        scheduler = self.lr_schedulers()
        scheduler.step()

        self._log_step_metrics("train", outputs, on_step=True, on_epoch=True)
        if self.trainer.global_step > 0 and outputs[7]: # per_layer_gate_stats is not empty
             self.gate_logger.update_rolling_history(outputs[7])


    def validation_step(self, batch, batch_idx):
        outputs = self._calculate_loss(batch)
        self._log_step_metrics("val", outputs, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        outputs = self._calculate_loss(batch)
        self._log_step_metrics("test", outputs, on_step=False, on_epoch=True)
        
    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        log.info(f"Configuring optimizer for {len(trainable_params)} trainable parameters.")
        
        if not trainable_params:
            log.warning("No trainable parameters found. Optimizer will not be configured.")
            return [], []

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.training_cfg.optimizer.base_lr,
            weight_decay=self.training_cfg.optimizer.weight_decay,
        )
        
        num_training_steps = self.training_cfg.max_iters
        num_warmup_steps = int(num_training_steps * self.training_cfg.optimizer.warmup_ratio)

        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]