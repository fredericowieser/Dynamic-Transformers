# src/trainers/d_qwen_trainer.py

import logging
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer
from omegaconf import DictConfig
from src.models.d_qwen_causal_lm import DynamicQwenForCausalLM
from src.models.d_qwen_config import DynamicQwenConfig
from src.trainers.gate_logging import GateLogger

log = logging.getLogger(__name__)

# Define the rolling window size
ROLLING_WINDOW_SIZE = 100


class DynamicQwenTrainer(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, training_cfg: DictConfig):
        super().__init__()
        # save for logging and checkpointing
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg

        log.info(f"Loading and configuring model: {self.model_cfg.model_name}")
        # Load the base config. dynamic_k, ce_bias, gate_warmup_iters will be None initially
        config = DynamicQwenConfig.from_pretrained(self.model_cfg.model_name)

        # Explicitly set dynamic parameters on the loaded config object
        # This mirrors the Llama trainer's approach to setting config attributes
        # --- START OF CHANGE ---
        # Define parameters that are meant to be set on the *model's config* (DynamicQwenConfig)
        required_params_for_model_config = {
            "dynamic_k": self.model_cfg.dynamic_k,
            "ce_bias": self.model_cfg.ce_bias,
            "gate_warmup_iters": self.training_cfg.gate_warmup_iters,
            # LoRA parameters are specifically handled by getattr with defaults
            "enable_lora_main_path": getattr(self.model_cfg, "lora", {}).get("enable_lora_main_path", False),
            "enable_lora_prior_ffn": getattr(self.model_cfg, "lora", {}).get("enable_lora_prior_ffn", False),
            "lora_r": getattr(self.model_cfg, "lora", {}).get("r", 8),
            "lora_alpha": getattr(self.model_cfg, "lora", {}).get("lora_alpha", 16),
            "lora_dropout": getattr(self.model_cfg, "lora", {}).get("lora_dropout", 0.05),
            "lora_bias": getattr(self.model_cfg, "lora", {}).get("bias", "none"),
            "lora_target_modules_main": getattr(self.model_cfg, "lora", {}).get("lora_target_modules_main", []),
            "lora_target_modules_prior_ffn": getattr(self.model_cfg, "lora", {}).get("lora_target_modules_prior_ffn", []),
            "init_prior_from_mlp": getattr(self.model_cfg, "init_prior_from_mlp", False),
        }

        # Set these parameters on the model's config object
        for param, value in required_params_for_model_config.items():
            # The 'None' check below ensures that if any of these explicitly listed
            # parameters are found to be None (e.g., if Hydra provides 'null' or it's implicitly None),
            # an error is raised, unless it's a LoRA flag which can be False by default.
            if value is None and param not in ["enable_lora_main_path", "enable_lora_prior_ffn", "init_prior_from_mlp"]:
                raise ValueError(f"{param} must be provided in the Hydra config for the Qwen model configuration.")
            setattr(config, param, value)

        # Separately check for 'prior_loss_weight' as it's a trainer-level parameter
        # and not an attribute of DynamicQwenConfig.
        # This ensures it's always provided by the Hydra config before it's used in _calculate_loss.
        if not hasattr(self.model_cfg, "prior_loss_weight") or self.model_cfg.prior_loss_weight is None:
            raise ValueError("model_cfg.prior_loss_weight must be provided in the Hydra config for the DynamicQwenTrainer.")
        # The value will be accessed later directly via self.model_cfg.prior_loss_weight
        # --- END OF CHANGE ---

        # Instantiate our DynamicQwen model with the fully configured config
        self.model = DynamicQwenForCausalLM(config)
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        # tokenizer (pad_token may need fixing)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.model.config.pad_token_id
                or self.tokenizer.eos_token_id
            )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # learning rate
        self._setup_parameter_groups()

        # Gate logging utility
        self.gate_logger = GateLogger(self.model.config.num_hidden_layers)
        self.per_layer_gate_activation_rolling_history = (
            self.gate_logger.per_layer_gate_activation_rolling_history
        )

    def forward(self, **inputs):
        if "input_ids" not in inputs:
            raise ValueError("input_ids must be provided.")
        inputs.update(
            {
                "current_iter": self.global_step,
                "return_metrics": True,
            }
        )
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

        self.log(
            f"{prefix}/loss",
            total_loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
        )
        self.log(
            f"{prefix}/lm_loss",
            lm_loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
        )
        self.log(f"{prefix}/prior_loss", prior_loss, on_step=on_step, on_epoch=on_epoch)
        self.log(
            f"{prefix}/perplexity",
            perplexity,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
        )

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
            schedulers.append(
                {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        opt,
                        mode="min",
                        factor=self.training_cfg.scheduler.factor,
                        patience=self.training_cfg.scheduler.patience,
                        min_lr=1e-5,  # Shared minimum LR
                    ),
                    "monitor": "val/loss",
                    "interval": "epoch",
                }
            )
        return optimizers, schedulers