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
            "ce_criterion_offset_init": self.model_cfg.ce_criterion_offset_init, # Now a learnable parameter
            "token_wise_gating": getattr(self.model_cfg, "token_wise_gating", True),
            "moving_average_window_size": getattr(self.model_cfg, "moving_average_window_size", 100),
            "prior_ffn_intermediate_size_factor": getattr(self.model_cfg, "prior_ffn_intermediate_size_factor", 2.0),
            "freeze_main_transformer_blocks": getattr(self.model_cfg, "freeze_main_transformer_blocks", False),
        }

        for param, value in required_params_for_model_config.items():
            if value is None and not isinstance(value, bool) and "init" not in param and "factor" not in param:
                raise ValueError(f"{param} must be provided in the Hydra config for the Qwen model configuration.")
            setattr(config, param, value)

        self.model = DynamicQwenForCausalLM.from_pretrained(self.model_cfg.model_name, config=config) # Pass full config to from_pretrained directly
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.model.config.pad_token_id
                or self.tokenizer.eos_token_id
            )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self._setup_parameter_groups()

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
        # Unpack all metrics returned by DynamicQwenForCausalLM.forward
        (
            logits,
            overall_prior_loss, # Actual aggregated prior loss for monitoring
            gate_vecs_per_layer, # List of (B, T) gate vectors for Dynamic layers
            overall_avg_ce_prop_from_model, # Scalar mean of CE from DynamicQwenForCausalLM
            overall_avg_cu_prop_from_model, # Scalar mean of CU from DynamicQwenForCausalLM
            ce_proportions_per_layer, # List of scalar CE proportions per Dynamic layer
            cu_proportions_per_layer, # List of scalar CU proportions per Dynamic layer
            overall_beta_ce, # Scalar
            overall_beta_cu, # Scalar
            overall_cu_detection_multiplier, # Scalar
            overall_ce_criterion_offset, # Scalar
        ) = self.forward(**batch)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        total_loss = lm_loss # Prior loss is only for monitoring, not added to total_loss
        perplexity = torch.exp(lm_loss)

        # Calculate overall gate activation mean from per-layer gate_vecs_per_layer
        overall_gate_activation_mean = (
            torch.stack([gv.mean() for gv in gate_vecs_per_layer]).mean()
            if gate_vecs_per_layer
            else torch.tensor(0.0, device=self.device)
        )

        # Calculate per-layer gate statistics (mean and std)
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
            overall_prior_loss, # Now includes the actual prior loss for logging
            perplexity,
            overall_gate_activation_mean,
            overall_avg_ce_prop_from_model, # Overall average CE proportion
            overall_avg_cu_prop_from_model, # Overall average CU proportion
            per_layer_gate_stats, # Per-layer mean/std for gate activations
            overall_beta_ce,
            overall_beta_cu,
            overall_cu_detection_multiplier,
            overall_ce_criterion_offset,
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
            prior_loss, # This is now the actual aggregated prior loss
            perplexity,
            overall_gate_activation_mean,
            overall_avg_ce, # Overall average CE proportion (from model.forward)
            overall_avg_cu, # Overall average CU proportion (from model.forward)
            per_layer_gate_stats, # Per-layer gate stats
            overall_beta_ce,
            overall_beta_cu,
            overall_cu_detection_multiplier,
            overall_ce_criterion_offset,
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
        self.log(f"{prefix}/prior_loss", prior_loss, on_step=on_step, on_epoch=on_epoch) # Log the actual prior_loss
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

        # Log new learnable parameters from VPRRouter
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
        optimizers = self.optimizers()
        for opt in optimizers:
            opt.zero_grad()

        outputs = self._calculate_loss(batch)
        total_loss = outputs[0]
        self.manual_backward(total_loss)

        for opt in optimizers:
            opt.step()

        schedulers = self.lr_schedulers()
        if isinstance(schedulers, list):
            for sch in schedulers:
                sch.step()
        else:
            schedulers.step()

        self._log_step_metrics("train", outputs, on_step=True, on_epoch=True)

        if self.trainer.global_step > 0:
            # per_layer_gate_stats are at outputs[7] based on _calculate_loss return
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
        self.main_block_params = []
        self.prior_ffn_params = []
        self.vpr_router_params = []
        self.other_params = []

        # Iterate through all parameters that require gradients
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                # This parameter is frozen (e.g., due to freeze_main_transformer_blocks or from HF base model)
                log.debug(f"Parameter '{n}' is frozen.")
                continue

            # Check for specific component names within the parameter name
            # These names come from the internal structure of DecisionQwenDecoderLayer and DynamicQwenDecoderLayer
            if "prior_ffn" in n or "prior_layernorm" in n:
                # Parameters belonging to the Prior FFN or its LayerNorm (in Decision layers)
                self.prior_ffn_params.append(p)
            elif "vpr_router" in n:
                # Parameters belonging to the VPR Router (in Dynamic layers)
                self.vpr_router_params.append(p)
            elif "model.layers" in n:
                # These are parameters of the core Qwen2 attention/MLP/LayerNorms
                # within either Decision or Dynamic layers.
                # Their requires_grad status is controlled by model._apply_main_block_freezing.
                self.main_block_params.append(p)
            else:
                # These are top-level model parameters: embed_tokens, lm_head, model.norm
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
        optimizer_groups = []

        main_and_other_params = self.main_block_params + self.other_params
        if main_and_other_params:
            optimizer_main = torch.optim.AdamW(
                main_and_other_params,
                lr=self.training_cfg.optimizer.base_lr,
                weight_decay=self.training_cfg.optimizer.weight_decay,
            )
            optimizer_groups.append(optimizer_main)
            log.info(f"Main/Other Params Optimizer created with LR: {self.training_cfg.optimizer.base_lr}")
        else:
            log.info("No parameters for Main/Other Params Optimizer (possibly frozen or empty).")

        dynamic_component_params = self.prior_ffn_params + self.vpr_router_params
        if not dynamic_component_params:
            raise ValueError("No trainable parameters found for prior FFN or VPR Router. Check your model configuration and parameter groups.")

        optimizer_dynamic = torch.optim.AdamW(
            dynamic_component_params,
            lr=self.training_cfg.optimizer.prior_ffn_lr,
            weight_decay=self.training_cfg.optimizer.weight_decay,
        )
        optimizer_groups.append(optimizer_dynamic)
        log.info(f"Prior FFN/VPR Router Optimizer created with LR: {self.training_cfg.optimizer.prior_ffn_lr}")

        num_training_steps = self.training_cfg.max_iters
        num_warmup_steps = int(num_training_steps * getattr(self.training_cfg.optimizer, "warmup_ratio", 0.0))

        schedulers = []
        if optimizer_main in optimizer_groups: # Check if optimizer_main was actually added
            lr_scheduler_main = get_scheduler(
                name="cosine",
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
            name="cosine",
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