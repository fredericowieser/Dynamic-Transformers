# src/trainers/d_qwen_trainer.py

import logging
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer, get_scheduler
from omegaconf import DictConfig, OmegaConf # Import OmegaConf to handle ListConfig
from src.models.d_qwen_causal_lm import DynamicQwenForCausalLM
from src.models.d_qwen_config import DynamicQwenConfig
from src.trainers.gate_logging import GateLogger

log = logging.getLogger(__name__)


class DynamicQwenTrainer(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, training_cfg: DictConfig):
        super().__init__()
        # Save for logging and checkpointing
        self.save_hyperparameters()
        self.automatic_optimization = False # Manual optimization for multiple optimizers
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg

        log.info(f"Loading and configuring model: {self.model_cfg.model_name}")
        # Load the base config. New dynamic parameters will be None initially
        config = DynamicQwenConfig.from_pretrained(self.model_cfg.model_name)

        # Explicitly set dynamic/training parameters on the loaded config object
        # This mirrors the Llama trainer's approach to setting config attributes
        # --- START OF CHANGE: Updated config parameters ---
        required_params_for_model_config = {
            "capacity_gamma": self.model_cfg.capacity_gamma,
            "beta_ce_init": self.model_cfg.beta_ce_init,
            "beta_cu_init": self.model_cfg.beta_cu_init,
            "cu_detection_multiplier_init": self.model_cfg.cu_detection_multiplier_init,
            "ce_criterion_offset_init": self.model_cfg.ce_criterion_offset_init,
            "token_wise_gating": getattr(self.model_cfg, "token_wise_gating", True), # Default to True
            "moving_average_window_size": getattr(self.model_cfg, "moving_average_window_size", 100),
            "prior_ffn_intermediate_size_factor": getattr(self.model_cfg, "prior_ffn_intermediate_size_factor", 2.0),
            "freeze_main_transformer_blocks": getattr(self.model_cfg, "freeze_main_transformer_blocks", False),
        }

        for param, value in required_params_for_model_config.items():
            if value is None:
                # Only warn for non-boolean/non-optional parameters that are None
                if not isinstance(value, bool) and "init" not in param and "factor" not in param:
                    raise ValueError(f"{param} must be provided in the Hydra config for the Qwen model configuration.")
            setattr(config, param, value)
        # --- END OF CHANGE ---

        # Instantiate our DynamicQwen model with the fully configured config
        self.model = DynamicQwenForCausalLM(config)
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        # Tokenizer (pad_token may need fixing)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.model.config.pad_token_id
                or self.tokenizer.eos_token_id
            )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Parameter grouping for differential learning rates and freezing
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
        # --- START OF CHANGE: Updated return signature from model.forward ---
        (
            logits,
            _, # prior_loss is no longer returned
            gate_vecs_per_layer,
            overall_avg_ce, # Overall means directly from model
            overall_avg_cu, # Overall means directly from model
            ce_proportions_per_layer, # Per-layer proportions still available
            cu_proportions_per_layer, # Per-layer proportions still available
        ) = self.forward(**batch)
        # --- END OF CHANGE ---

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        total_loss = lm_loss # Prior loss removed from total loss
        perplexity = torch.exp(lm_loss)

        # Gate stats calculation
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
            None, # prior_loss is None now
            perplexity,
            overall_avg_ce, # Now directly the overall mean
            overall_avg_cu, # Now directly the overall mean
            per_layer_gate_stats,
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
        # --- START OF CHANGE: Updated outputs unpacking ---
        (
            total_loss,
            lm_loss,
            prior_loss, # Will be None
            perplexity,
            overall_avg_ce,
            overall_avg_cu,
            per_layer_gate_stats,
            ce_proportions_per_layer,
            cu_proportions_per_layer,
        ) = outputs
        # --- END OF CHANGE ---

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
        # prior_loss is no longer logged as it's removed
        self.log(
            f"{prefix}/perplexity",
            perplexity,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
        )

        # Log overall CE and CU metrics (now directly provided)
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
        # Overall gate activation mean now derived from overall_avg_ce and overall_avg_cu conceptually
        # We will use the direct per-layer gate means from per_layer_gate_stats for logging.
        # So overall_gate_activation_mean argument might be removed from GateLogger.log_gate_metrics if not needed.
        # For simplicity, pass the mean of per-layer gate means if needed, or update GateLogger.
        # Let's derive overall_gate_activation_mean here for GateLogger.
        overall_gate_activation_mean_for_logger = (
            torch.tensor([s["mean"] for s in per_layer_gate_stats]).mean()
            if per_layer_gate_stats
            else torch.tensor(0.0, device=self.device)
        )
        GateLogger.log_gate_metrics(
            self,
            prefix,
            overall_gate_activation_mean_for_logger, # Pass computed mean
            per_layer_gate_stats,
            on_step,
            on_epoch,
        )

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        # Zero grads for all optimizers
        for opt in optimizers:
            opt.zero_grad()

        outputs = self._calculate_loss(batch)
        total_loss = outputs[0]
        self.manual_backward(total_loss)

        # Step all optimizers
        for opt in optimizers:
            opt.step()

        # Update learning rate schedulers
        schedulers = self.lr_schedulers()
        if isinstance(schedulers, list):
            for sch in schedulers:
                sch.step()
        else:
            schedulers.step()

        self._log_step_metrics("train", outputs, on_step=True, on_epoch=True)

        # Gate rolling stats update & logging
        # per_layer_gate_stats is outputs[6]
        if self.trainer.global_step > 0:
            self.gate_logger.update_rolling_history(outputs[6])
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
        # Initialize lists for different parameter groups
        self.main_block_params = []
        self.prior_ffn_params = []
        self.vpr_router_params = []
        self.other_params = [] # For other model params like embeddings, lm_head, etc.

        freeze_main = self.model.freeze_main_transformer_blocks

        # Iterate through the model's named parameters
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                log.debug(f"Parameter '{n}' is frozen (requires_grad=False).")
                continue # Skip if already frozen (e.g., from base Hugging Face model or explicit .eval())

            if "decision_layer" in n:
                if "prior_ffn" in n or "prior_layernorm" in n:
                    self.prior_ffn_params.append(p)
                else:
                    # These are original Qwen2 block parameters within DecisionLayer
                    # Their requires_grad status is controlled by model._apply_main_block_freezing
                    self.main_block_params.append(p)
            elif "dynamic_layer.vpr_router" in n:
                self.vpr_router_params.append(p)
            else:
                # Parameters like embeddings, LM head, etc.
                self.other_params.append(p)

        # Log counts for verification
        log.info(f"Identified {len(self.main_block_params)} main block parameters (trainable if not frozen).")
        log.info(f"Identified {len(self.prior_ffn_params)} prior FFN parameters (always trainable).")
        log.info(f"Identified {len(self.vpr_router_params)} VPR router parameters (always trainable).")
        log.info(f"Identified {len(self.other_params)} other trainable parameters (e.g., embeddings, LM head).")

        total_trainable_summed = (
            sum(p.numel() for p in self.main_block_params)
            + sum(p.numel() for p in self.prior_ffn_params)
            + sum(p.numel() for p in self.vpr_router_params)
            + sum(p.numel() for p in self.other_params)
        )
        model_total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        if total_trainable_summed != model_total_trainable:
            log.warning(
                f"Mismatch in trainable parameter count! _setup_parameter_groups found {total_trainable_summed} (numel), model reports {model_total_trainable} (numel). This might indicate a miscategorization or initial freezing."
            )

    def configure_optimizers(self):
        # We will use two optimizers: one for main blocks (+ other params), one for dynamic components.
        # This gives flexibility for freezing or different LRs.

        optimizer_groups = []

        # 1. Optimizer for Main Transformer Blocks and other base model parameters (embeddings, LM head)
        # This group is only added if main blocks are not frozen, or if there are other_params.
        main_and_other_params = self.main_block_params + self.other_params
        if main_and_other_params: # Only create optimizer if there are parameters to optimize
            optimizer_main = torch.optim.AdamW(
                main_and_other_params,
                lr=self.training_cfg.optimizer.base_lr,
                weight_decay=self.training_cfg.optimizer.weight_decay,
            )
            optimizer_groups.append(optimizer_main)
            log.info(f"Main/Other Params Optimizer created with LR: {self.training_cfg.optimizer.base_lr}")
        else:
            log.info("No parameters for Main/Other Params Optimizer (possibly frozen or empty).")


        # 2. Optimizer for Prior FFN and VPR Router parameters (the 'dynamic' parts)
        dynamic_component_params = self.prior_ffn_params + self.vpr_router_params
        if not dynamic_component_params:
            raise ValueError("No trainable parameters found for prior FFN or VPR Router. Check your model configuration and parameter groups.")

        optimizer_dynamic = torch.optim.AdamW(
            dynamic_component_params,
            lr=self.training_cfg.optimizer.prior_ffn_lr, # Using prior_ffn_lr for all dynamic components
            weight_decay=self.training_cfg.optimizer.weight_decay,
        )
        optimizer_groups.append(optimizer_dynamic)
        log.info(f"Prior FFN/VPR Router Optimizer created with LR: {self.training_cfg.optimizer.prior_ffn_lr}")


        # --- Learning Rate Schedulers ---
        num_training_steps = self.training_cfg.max_iters
        # No more gate_warmup_iters for LR warmup directly, use a general warmup
        num_warmup_steps = int(num_training_steps * getattr(self.training_cfg.optimizer, "warmup_ratio", 0.0))

        schedulers = []
        if optimizer_main in optimizer_groups: # Check if optimizer_main was actually added
            lr_scheduler_main = get_scheduler(
                name="cosine", # Or "linear"
                optimizer=optimizer_main,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
            schedulers.append({
                "scheduler": lr_scheduler_main,
                "interval": "step",
                "frequency": 1,
            })

        lr_scheduler_dynamic = get_scheduler(
            name="cosine", # Or "linear"
            optimizer=optimizer_dynamic,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        schedulers.append({
            "scheduler": lr_scheduler_dynamic,
            "interval": "step",
            "frequency": 1,
        })

        return optimizer_groups, schedulers