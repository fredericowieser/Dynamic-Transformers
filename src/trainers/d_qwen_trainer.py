import logging
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer
from omegaconf import DictConfig
from hydra.utils import instantiate
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
        self.save_hyperparameters() # Changed to save all
        self.automatic_optimization = False # NEW: Enable manual optimization
        self.model_cfg = model_cfg # NEW: Store model_cfg
        self.training_cfg = training_cfg # NEW: Store training_cfg

        log.info(f"Loading and configuring model: {model_cfg.model_name}")
        # --- START OF CHANGE ---
        # Load the base config. dynamic_k, ce_bias, gate_warmup_iters will be None initially
        config = DynamicQwenConfig.from_pretrained(self.model_cfg.model_name) # Changed to use self.model_cfg

        # Explicitly set dynamic parameters on the loaded config object
        # This mirrors the Llama trainer's approach to setting config attributes
        # Ensure all required params for model config and LoRA are set
        required_params = { # NEW: Defined required_params like LlamaTrainer
            "dynamic_k": self.model_cfg.dynamic_k,
            "ce_bias": self.model_cfg.ce_bias,
            # Qwen does not have 'token_wise' config attribute, only Llama. So omit here.
            "gate_warmup_iters": self.training_cfg.gate_warmup_iters,
            "prior_loss_weight": self.model_cfg.prior_loss_weight,
            # Qwen does not have Lora parameters in its model config directly like Llama.
            # LoRA config should be passed through in a different way or handled internally by the model if enabled.
            # Assuming for now that DynamicQwenForCausalLM handles its own LoRA config application.
            # However, for consistency and future expansion, we can set it if the config supports it.
            # If your Qwen model_cfg will not include LORA specific parameters, remove these lines.
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
        for param, value in required_params.items(): # NEW: Loop to set attributes
            if value is None and param not in ["enable_lora_main_path", "enable_lora_prior_ffn", "init_prior_from_mlp"]: # Allow LoRA and init_prior to be None for default
                raise ValueError(f"{param} must be provided in the Hydra config.")
            setattr(config, param, value)

        # Instantiate our DynamicQwen model with the fully configured config
        self.model = DynamicQwenForCausalLM(config)
        self.model.gradient_checkpointing_enable() # NEW
        self.model.enable_input_require_grads() # NEW
        # --- END OF CHANGE ---

        # tokenizer (pad_token may need fixing)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.model_name) # Changed to use self.model_cfg
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.model.config.pad_token_id
                or self.tokenizer.eos_token_id
            )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # learning rate (removed self.lr as it's now handled by optimizers)
        self._setup_parameter_groups() # NEW: Setup parameter groups for optimizers

        # Gate logging utility
        self.gate_logger = GateLogger(self.model.config.num_hidden_layers) # NEW
        self.per_layer_gate_activation_rolling_history = ( # NEW
            self.gate_logger.per_layer_gate_activation_rolling_history
        )

    def _setup_parameter_groups(self): # NEW METHOD (copied from d_llama_trainer)
        log.info(
            "Setting up parameter groups for differential learning rates and LoRA."
        )
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
            log.info(
                f"LoRA enabled for main decoder path. Training {len(self.original_params)} LoRA parameters."
            )
            if len(self.original_params) == 0:
                log.warning(
                    "No trainable parameters found for the main decoder path despite LoRA being enabled. Check target modules."
                )
        else:
            log.info(
                f"Full training for main decoder path. Training {len(self.original_params)} parameters."
            )

        if enable_lora_prior:
            log.info(
                f"LoRA enabled for prior FFN. Training {len(self.new_prior_params)} LoRA parameters."
            )
            if len(self.new_prior_params) == 0:
                log.warning(
                    "No trainable parameters found for the prior FFN despite LoRA being enabled. Check target modules."
                )
        else:
            log.info(
                f"Full training for prior FFN. Training {len(self.new_prior_params)} parameters."
            )

        total_trainable = sum(p.numel() for p in self.original_params) + sum(p.numel() for p in self.new_prior_params)
        model_total_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        if total_trainable != model_total_trainable:
            log.warning(
                f"Mismatch in trainable parameter count! _setup_parameter_groups found {total_trainable} (numel), model reports {model_total_trainable} (numel). This might indicate a miscategorization."
            )

    def forward(self, **inputs):
        if "input_ids" not in inputs:
            raise ValueError("input_ids must be provided.")
        inputs.update(
            {
                "current_iter": self.global_step,
                "return_metrics": True, # NEW: Ensure metrics are returned
            }
        )
        return self.model(**inputs)
    
    def _calculate_loss(self, batch): # NEW METHOD (copied from d_llama_trainer)
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
    
    def _log_step_metrics( # NEW METHOD (copied from d_llama_trainer)
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
        opt_base, opt_prior = self.optimizers() # NEW: Get multiple optimizers
        opt_base.zero_grad() # NEW
        opt_prior.zero_grad() # NEW

        outputs = self._calculate_loss(batch) # NEW: Call _calculate_loss
        total_loss = outputs[0] # NEW: Get total_loss from the tuple
        self.manual_backward(total_loss) # NEW: Manual backward
        opt_base.step() # NEW
        opt_prior.step() # NEW

        self._log_step_metrics("train", outputs, on_step=True, on_epoch=True) # NEW: Use _log_step_metrics

        # Gate rolling stats update & logging
        per_layer_gate_stats = outputs[5] # NEW
        if self.trainer.global_step > 0: # NEW
            self.gate_logger.update_rolling_history(per_layer_gate_stats) # NEW
            self.gate_logger.log_rolling_history( # NEW
                self.trainer.global_step,
                self.trainer.log_every_n_steps,
            )

    def validation_step(self, batch, batch_idx):
        outputs = self._calculate_loss(batch) # NEW: Call _calculate_loss
        self._log_step_metrics("val", outputs, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        outputs = self._calculate_loss(batch) # NEW: Call _calculate_loss
        self._log_step_metrics("test", outputs, on_step=False, on_epoch=True) # NEW: Use _log_step_metrics

    def configure_optimizers(self):
        optimizers = [ # NEW: Configure two optimizers
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
        schedulers = [] # NEW: Add schedulers
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