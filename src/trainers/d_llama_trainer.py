import logging
from collections import deque

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from transformers import AutoConfig, AutoTokenizer

from src.models.d_llama_causal_lm import DynamicLlamaForCausalLM
from src.models.d_llama_config import DynamicLlamaConfig

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
        required_params = {
            "dynamic_k": self.model_cfg.dynamic_k,
            "ce_bias": self.model_cfg.ce_bias,
            "token_wise": self.model_cfg.token_wise,
            "gate_warmup_iters": self.training_cfg.gate_warmup_iters,
            "prior_loss_weight": self.model_cfg.prior_loss_weight,
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

        self._setup_parameter_groups()
        self.per_layer_gate_activation_rolling_history = [
            {
                "mean": deque(maxlen=ROLLING_WINDOW_SIZE),
                "std":  deque(maxlen=ROLLING_WINDOW_SIZE),
            }
            for _ in range(self.model.config.num_hidden_layers)
        ]

    def _setup_parameter_groups(self):
        log.info("Setting up parameter groups for differential learning rates.")
        named = list(self.model.named_parameters())
        self.new_prior_params = [
            p for n, p in named
            if "prior_ffn" in n or "prior_layernorm" in n
        ]
        self.original_params = [
            p for n, p in named
            if "prior_ffn" not in n and "prior_layernorm" not in n
        ]
        for p in self.original_params:
            p.requires_grad = True

        log.info(
            f"Found {len(self.original_params)} original parameters and "
            f"{len(self.new_prior_params)} new prior parameters."
        )

    def forward(self, **inputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "input_ids" not in inputs:
            raise ValueError("input_ids must be provided.")
        
        # Inject training-specific params
        inputs["current_iter"] = self.global_step
        inputs["return_metrics"] = True
        
        return self.model(**inputs)

    def _calculate_loss(self, batch) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, torch.Tensor]],
    ]:
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
                "std":  gv.std()
                        if gv.numel() > 1
                        else torch.tensor(0.0, device=self.device),
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
        self, prefix: str, outputs: tuple, on_step: bool, on_epoch: bool
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
        self.log(
            f"{prefix}_dynamic_model/overall_gate_activation_mean",
            overall_gate_activation_mean,
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

        # Log per-layer metrics
        for i, stats in enumerate(per_layer_gate_stats):
            self.log(
                f"{prefix}_dynamic_layer/gate_mean/layer_{i}",
                stats["mean"],
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                f"{prefix}_dynamic_layer/gate_std/layer_{i}",
                stats["std"],
                on_step=on_step,
                on_epoch=on_epoch,
            )

        for i, prop in enumerate(ce_proportions_per_layer):
            self.log(
                f"{prefix}_dynamic_layer/ce_proportion/layer_{i}",
                prop,
                on_step=on_step,
                on_epoch=on_epoch,
            )

        for i, prop in enumerate(cu_proportions_per_layer):
            self.log(
                f"{prefix}_dynamic_layer/cu_proportion/layer_{i}",
                prop,
                on_step=on_step,
                on_epoch=on_epoch,
            )

    def training_step(self, batch, batch_idx):
        # Manual optimization requires you to get the optimizers
        opt_base, opt_prior = self.optimizers()

        # Zero gradients for both optimizers
        opt_base.zero_grad()
        opt_prior.zero_grad()

        # Calculate loss
        outputs = self._calculate_loss(batch)
        total_loss = outputs[0]

        # Manually perform the backward pass
        self.manual_backward(total_loss)

        # Step both optimizers
        opt_base.step()
        opt_prior.step()

        # Log metrics
        self._log_step_metrics("train", outputs, on_step=True, on_epoch=True)

        # The rest of your logging logic remains unchanged
        per_layer_gate_stats = outputs[5]
        if self.trainer.global_step > 0:
            log_interval = self.trainer.log_every_n_steps
            for i, stats in enumerate(per_layer_gate_stats):
                self.per_layer_gate_activation_rolling_history[i]["mean"].append(
                    stats["mean"].item()
                )
                self.per_layer_gate_activation_rolling_history[i]["std"].append(
                    stats["std"].item()
                )

            if self.trainer.global_step % log_interval == 0:
                log_lines = [
                    f"--- Per-Layer Gate Activations (Training, Rolling Avg over last {ROLLING_WINDOW_SIZE} steps) ---"
                ]
                for i, history in enumerate(
                    self.per_layer_gate_activation_rolling_history
                ):
                    if history["mean"]:
                        rolling_mean = sum(history["mean"]) / len(history["mean"])
                        rolling_std = (
                            torch.tensor(list(history["std"])).std().item()
                            if len(history["std"]) > 1
                            else 0.0
                        )
                        log_lines.append(
                            f"  Layer {i}: Mean = {rolling_mean:.3f}, Std = {rolling_std:.3f}"
                        )
                log.info("\n".join(log_lines))

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
