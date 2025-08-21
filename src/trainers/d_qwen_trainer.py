import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, get_scheduler

from src.models.d_qwen_causal_lm import DynamicQwenForCausalLM
from src.models.d_qwen_config import DynamicQwenConfig
from src.trainers.gate_logging import GateLogger

log = logging.getLogger(__name__)


class DynamicQwenTrainer(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, training_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg

        log.info(f"Loading and configuring model: {self.model_cfg.model_name}")
        config = DynamicQwenConfig.from_pretrained(self.model_cfg.model_name)

        required_params_for_model_config = {
            "capacity_gamma": self.model_cfg.capacity_gamma,
            "beta_ce_init": self.model_cfg.beta_ce_init,
            "beta_cu_init": self.model_cfg.beta_cu_init,
            "cu_detection_multiplier_init": self.model_cfg.cu_detection_multiplier_init,
            "ce_criterion_offset_init": self.model_cfg.ce_criterion_offset_init,
            "token_wise_gating": getattr(self.model_cfg, "token_wise_gating", True),
            "moving_average_window_size": getattr(
                self.model_cfg, "moving_average_window_size", 100
            ),
            "prior_ffn_intermediate_size_factor": getattr(
                self.model_cfg, "prior_ffn_intermediate_size_factor", 2.0
            ),
            "freeze_main_transformer_blocks": getattr(
                self.model_cfg, "freeze_main_transformer_blocks", False
            ),
            "init_prior_from_mlp": False,
        }

        for param, value in required_params_for_model_config.items():
            if (
                value is None
                and not isinstance(value, bool)
                and "init" not in param
                and "factor" not in param
            ):
                raise ValueError(
                    f"{param} must be provided in the Hydra config for the Qwen model configuration."
                )
            setattr(config, param, value)

        self.model = DynamicQwenForCausalLM.from_pretrained(
            self.model_cfg.model_name, config=config
        )
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.model.config.pad_token_id or self.tokenizer.eos_token_id
            )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # --- START OF MODIFICATION ---
        # Setup a single list of all trainable parameters
        self._setup_parameter_groups()
        # --- END OF MODIFICATION ---

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
        # --- START OF MODIFICATION ---
        # Unpack all metrics returned by DynamicQwenForCausalLM.forward
        (
            logits,
            overall_prior_loss,  # Actual aggregated prior loss for monitoring
            gate_vecs_per_layer,  # List of (B, T) binary gate vectors for Dynamic layers
            overall_avg_ce_prop_from_model,  # Scalar mean of CE from DynamicQwenForCausalLM
            overall_avg_cu_prop_from_model,  # Scalar mean of CU from DynamicQwenForCausalLM
            overall_combined_gating_signal_mean,  # NEW: Overall mean of continuous signal
            ce_proportions_per_layer,  # List of scalar CE proportions per Dynamic layer (from router)
            cu_proportions_per_layer,  # List of scalar CU proportions per Dynamic layer (from router)
            overall_beta_ce,  # Scalar average of learnable beta_ce across layers
            overall_beta_cu,  # Scalar average of learnable beta_cu across layers
            overall_cu_detection_multiplier,  # Scalar average of non-learnable cu_detection_multiplier across layers
            overall_ce_criterion_offset,  # Scalar average of learnable ce_criterion_offset across layers
        ) = self.forward(**batch)
        # --- END OF MODIFICATION ---

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        total_loss = lm_loss
        perplexity = torch.exp(lm_loss)

        overall_gate_activation_mean = (
            torch.stack([gv.mean() for gv in gate_vecs_per_layer]).mean()
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
            overall_prior_loss,
            perplexity,
            overall_gate_activation_mean,
            overall_avg_ce_prop_from_model,
            overall_avg_cu_prop_from_model,
            per_layer_gate_stats,
            overall_beta_ce,
            overall_beta_cu,
            overall_cu_detection_multiplier,
            overall_ce_criterion_offset,
            overall_combined_gating_signal_mean,
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
            overall_avg_ce,
            overall_avg_cu,
            per_layer_gate_stats,
            overall_beta_ce,
            overall_beta_cu,
            overall_cu_detection_multiplier,
            overall_ce_criterion_offset,
            overall_combined_gating_signal_mean,
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
        # NEW: Log overall mean of the continuous gating signal
        self.log(
            f"{prefix}_dynamic_model/overall_combined_gating_signal_mean",
            overall_combined_gating_signal_mean,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
        )

        self.log(
            f"{prefix}_dynamic_model/router_beta_ce",
            overall_beta_ce,
            on_step=on_step,
            on_epoch=on_epoch,
        )
        self.log(
            f"{prefix}_dynamic_model/router_beta_cu",
            overall_beta_cu,
            on_step=on_step,
            on_epoch=on_epoch,
        )
        self.log(
            f"{prefix}_dynamic_model/router_cu_detection_multiplier",
            overall_cu_detection_multiplier,
            on_step=on_step,
            on_epoch=on_epoch,
        )
        self.log(
            f"{prefix}_dynamic_model/router_ce_criterion_offset",
            overall_ce_criterion_offset,
            on_step=on_step,
            on_epoch=on_epoch,
        )

        GateLogger.log_gate_metrics(
            self,
            prefix,
            overall_gate_activation_mean,
            per_layer_gate_stats,
            on_step,
            on_epoch,
        )

    def training_step(self, batch, batch_idx):
        # --- START OF MODIFICATION ---
        optimizer = self.optimizers()  # Get the single optimizer
        optimizer.zero_grad()

        outputs = self._calculate_loss(batch)
        total_loss = outputs[0]
        self.manual_backward(total_loss)

        optimizer.step()

        scheduler = self.lr_schedulers()  # Get the single scheduler
        scheduler.step()
        # --- END OF MODIFICATION ---

        self._log_step_metrics("train", outputs, on_step=True, on_epoch=True)

        if self.trainer.global_step > 0:
            self.gate_logger.update_rolling_history(outputs[7])
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

    def _setup_parameter_groups(self):
        log.info(
            "Setting up parameter groups for dynamic components and main block freezing."
        )
        # --- START OF MODIFICATION ---
        self.all_trainable_params = []

        # Iterate through all parameters that require gradients
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.all_trainable_params.append(p)
            else:
                log.debug(f"Parameter '{n}' is frozen.")

        log.info(
            f"Identified {len(self.all_trainable_params)} total trainable parameters."
        )
        total_trainable_summed = sum(p.numel() for p in self.all_trainable_params)
        model_total_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        if total_trainable_summed != model_total_trainable:
            log.warning(
                f"Mismatch in trainable parameter count! _setup_parameter_groups found {total_trainable_summed} (numel), model reports {model_total_trainable} (numel). This might indicate a miscategorization or initial freezing."
            )
        # --- END OF MODIFICATION ---

    def configure_optimizers(self):
        # --- START OF MODIFICATION ---
        if not self.all_trainable_params:
            raise ValueError(
                "No trainable parameters found for the model. Check freezing configuration and model setup."
            )

        optimizer = torch.optim.AdamW(
            self.all_trainable_params,
            lr=self.training_cfg.optimizer.base_lr,  # Use base_lr for the single optimizer
            weight_decay=self.training_cfg.optimizer.weight_decay,
        )
        log.info(
            f"Single Optimizer created with LR: {self.training_cfg.optimizer.base_lr}"
        )

        num_training_steps = self.training_cfg.max_iters
        warmup_ratio = getattr(self.training_cfg.optimizer, "warmup_ratio", 0.0)
        num_warmup_steps = int(num_training_steps * warmup_ratio)

        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return [optimizer], [
            {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            }
        ]
        # --- END OF MODIFICATION ---
