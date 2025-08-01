import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoTokenizer

from src.models.d_llama_causal_lm import DynamicLlamaForCausalLM
from src.models.d_llama_config import DynamicLlamaConfig
from src.trainers.gate_logging import GateLogger
from src.utils.llama_config_utils import fix_rope_scaling, fix_pad_token_id

log = logging.getLogger(__name__)

# Define the rolling window size
ROLLING_WINDOW_SIZE = 100


class DynamicLlamaTrainer(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, training_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg

        log.info(f"Loading pre-trained model: {self.model_cfg.model_name}")
        config = DynamicLlamaConfig.from_pretrained(self.model_cfg.model_name)
        log.info("Applying config fixes for rope_scaling and pad_token_id...")
        config = fix_rope_scaling(config)
        config = fix_pad_token_id(config)
        log.info("Config fixes applied.")
        required_params = {
            "dynamic_k":          self.model_cfg.dynamic_k,
            "ce_bias":            self.model_cfg.ce_bias,
            "token_wise":         self.model_cfg.token_wise,
            "gate_warmup_iters":  self.training_cfg.gate_warmup_iters,
            "prior_loss_weight":  self.model_cfg.prior_loss_weight,
            "init_prior_from_mlp": self.model_cfg.init_prior_from_mlp, # New
            # LoRA parameters from model.lora in base.yaml
            "enable_lora_main_path": self.model_cfg.lora.enable_lora_main_path, # New
            "enable_lora_prior_ffn": self.model_cfg.lora.enable_lora_prior_ffn, # New
            "lora_r": self.model_cfg.lora.r,
            "lora_alpha": self.model_cfg.lora.lora_alpha,
            "lora_dropout": self.model_cfg.lora.lora_dropout,
            "lora_bias": self.model_cfg.lora.bias,
            "lora_target_modules_main": self.model_cfg.lora.lora_target_modules_main,
            "lora_target_modules_prior_ffn": self.model_cfg.lora.lora_target_modules_prior_ffn,
        }
        for param, value in required_params.items():
            if value is None:
                raise ValueError(f"{param} must be provided in the Hydra config.")
            setattr(config, param, value)

        self.model = DynamicLlamaForCausalLM(config)
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                config.pad_token_id or self.tokenizer.eos_token_id
            )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self._setup_parameter_groups()

        # Gate logging utility
        self.gate_logger = GateLogger(self.model.config.num_hidden_layers)
        self.per_layer_gate_activation_rolling_history = (
            self.gate_logger.per_layer_gate_activation_rolling_history
        )

    def _setup_parameter_groups(self):
        log.info("Setting up parameter groups for differential learning rates and LoRA.")
        named_params = list(self.model.named_parameters())

        self.original_params = []
        self.new_prior_params = []

        enable_lora_main = getattr(self.model.config, "enable_lora_main_path", False)
        enable_lora_prior = getattr(self.model.config, "enable_lora_prior_ffn", False)

        for n, p in named_params:
            if p.requires_grad:
                is_prior_param = "prior_ffn" in n or "prior_layernorm" in n

                if is_prior_param:
                    self.new_prior_params.append(p)
                else:
                    self.original_params.append(p)
            else:
                log.debug(f"Parameter '{n}' is frozen.")

        if enable_lora_main:
            log.info(f"LoRA enabled for main decoder path. Training {len(self.original_params)} LoRA parameters.")
            if len(self.original_params) == 0:
                log.warning("No trainable parameters found for the main decoder path despite LoRA being enabled. Check target modules.")
        else:
            log.info(f"Full training for main decoder path. Training {len(self.original_params)} parameters.")

        if enable_lora_prior:
            log.info(f"LoRA enabled for prior FFN. Training {len(self.new_prior_params)} LoRA parameters.")
            if len(self.new_prior_params) == 0:
                log.warning("No trainable parameters found for the prior FFN despite LoRA being enabled. Check target modules.")
        else:
            log.info(f"Full training for prior FFN. Training {len(self.new_prior_params)} parameters.")

        total_trainable = len(self.original_params) + len(self.new_prior_params)
        model_total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Checking sum of numel for a more accurate comparison of trainable parameters
        if total_trainable != model_total_trainable:
             log.warning(f"Mismatch in trainable parameter count! _setup_parameter_groups found {total_trainable} (numel), model reports {model_total_trainable} (numel). This might indicate a miscategorization.")

    def forward(self, **inputs):
        if "input_ids" not in inputs:
            raise ValueError("input_ids must be provided.")
        inputs.update({
            "current_iter":   self.global_step,
            "return_metrics": True,
        })
        return self.model(**inputs)

    def _calculate_loss(self, batch):
        (
            logits,
            prior_loss,
            gate_vecs_per_layer,
            ce_proportions_per_layer,
            cu_proportions_per_layer,
        ) = self.forward(**batch)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        total_loss = lm_loss + self.model_cfg.prior_loss_weight * prior_loss
        perplexity = torch.exp(lm_loss)

        overall_avg_ce = (
            torch.stack(ce_proportions_per_layer).mean()
            if ce_proportions_per_layer
            else torch.tensor(0.0, device=self.device)
        )
        overall_avg_cu = (
            torch.stack(cu_proportions_per_layer).mean()
            if cu_proportions_per_layer
            else torch.tensor(0.0, device=self.device)
        )
        overall_gate_activation_mean = (
            torch.stack(gate_vecs_per_layer).mean()
            if gate_vecs_per_layer
            else torch.tensor(0.0, device=self.device)
        )

        per_layer_gate_stats = [
            {
                "mean": gv.mean(),
                "std": (
                    gv.std()
                    if gv.numel() > 1
                    else torch.tensor(0.0, device=self.device)
                ),
            }
            for gv in gate_vecs_per_layer
        ]
        return (
            total_loss,
            lm_loss,
            prior_loss,
            perplexity,
            overall_gate_activation_mean,
            per_layer_gate_stats,
            overall_avg_ce,
            overall_avg_cu,
            ce_proportions_per_layer,
            cu_proportions_per_layer,
        )

    def _log_step_metrics(
            self,
            prefix: str,
            outputs: tuple,
            on_step: bool,
            on_epoch: bool,
        ):
        (
            total_loss,
            lm_loss,
            prior_loss,
            perplexity,
            overall_gate_activation_mean,
            per_layer_gate_stats,
            overall_avg_ce,
            overall_avg_cu,
            ce_proportions_per_layer,
            cu_proportions_per_layer,
        ) = outputs

        self.log(f"{prefix}/loss", total_loss,
                 on_step=on_step, on_epoch=on_epoch, prog_bar=True)
        self.log(f"{prefix}/lm_loss", lm_loss,
                 on_step=on_step, on_epoch=on_epoch, prog_bar=True)
        self.log(f"{prefix}/prior_loss", prior_loss,
                 on_step=on_step, on_epoch=on_epoch)
        self.log(f"{prefix}/perplexity", perplexity,
                 on_step=on_step, on_epoch=on_epoch, prog_bar=True)

        # Log overall CE and CU metrics
        self.log(
            f"{prefix}_dynamic_model/overall_avg_ce_proportion",
            overall_avg_ce,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
        )
        self.log(
            f"{prefix}_dynamic_model/overall_avg_cu_proportion",
            overall_avg_cu,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
        )

        # Gate-related logging delegated out
        GateLogger.log_gate_metrics(
            self,
            prefix,
            overall_gate_activation_mean,
            per_layer_gate_stats,
            on_step,
            on_epoch,
        )

    def training_step(self, batch, batch_idx):
        opt_base, opt_prior = self.optimizers()
        opt_base.zero_grad()
        opt_prior.zero_grad()

        outputs = self._calculate_loss(batch)
        total_loss = outputs[0]
        self.manual_backward(total_loss)
        opt_base.step()
        opt_prior.step()

        self._log_step_metrics("train", outputs, on_step=True, on_epoch=True)

        # Gate rolling stats update & logging
        per_layer_gate_stats = outputs[5]
        if self.trainer.global_step > 0:
            self.gate_logger.update_rolling_history(per_layer_gate_stats)
            self.gate_logger.log_rolling_history(
                self.trainer.global_step,
                self.trainer.log_every_n_steps,
            )

    def validation_step(self, batch, batch_idx):
        outputs = self._calculate_loss(batch)
        self._log_step_metrics("val", outputs, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        outputs = self._calculate_loss(batch)
        self._log_step_metrics("test", outputs, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizers = [
            torch.optim.AdamW(
                self.original_params,
                lr=self.training_cfg.optimizer.base_lr,
                weight_decay=self.training_cfg.optimizer.weight_decay,
            ),
            torch.optim.AdamW(
                self.new_prior_params,
                lr=self.training_cfg.optimizer.prior_ffn_lr,
                weight_decay=self.training_cfg.optimizer.weight_decay,
            ),
        ]
        schedulers = []
        for opt in optimizers:
            schedulers.append({
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt,
                    mode="min",
                    factor=self.training_cfg.scheduler.factor,
                    patience=self.training_cfg.scheduler.patience,
                    min_lr=1e-5,  # Shared minimum LR
                ),
                "monitor":  "val/loss",
                "interval": "epoch",
            })
        return optimizers, schedulers
