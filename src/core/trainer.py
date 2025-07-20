import logging
import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from src.models.dynamic_llama import DynamicLlamaDecoderLayer
from typing import Tuple

log = logging.getLogger(__name__)


class LightningModel(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, training_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg

        log.info(f"Loading pre-trained model: {self.model_cfg.model_name}")
        
        # Load the configuration first
        config = AutoConfig.from_pretrained(self.model_cfg.model_name)

        # --- FIX: Handle potential missing 'type' in 'rope_scaling' ---
        # The KeyError occurs if config.rope_scaling is a dict but lacks "type" or "rope_type".
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            if "type" not in config.rope_scaling and "rope_type" not in config.rope_scaling:
                log.warning(
                    f"Config for {self.model_cfg.model_name} has `rope_scaling` but no `type` or `rope_type`."
                    " Defaulting `rope_scaling.type` to 'linear' and `factor` to 1.0."
                )
                config.rope_scaling["type"] = "linear"
                # Ensure 'factor' is also present, commonly 1.0 for linear scaling
                config.rope_scaling["factor"] = config.rope_scaling.get("factor", 1.0)
        elif not hasattr(config, "rope_scaling") or config.rope_scaling is None:
            # If rope_scaling attribute doesn't exist or is None, initialize it with defaults
            log.warning(
                f"Config for {self.model_cfg.model_name} does not have `rope_scaling` attribute."
                " Initializing `rope_scaling` with default 'linear' type and factor 1.0."
            )
            config.rope_scaling = {"type": "linear", "factor": 1.0}
        # --- END FIX --- 


        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.model_name, torch_dtype=torch.bfloat16, config=config # Pass the potentially modified config
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
                param.requires_grad = True # Ensure original parameters are trainable
                self.original_params.append(param)
        log.info(
            f"Found {len(self.original_params)} original parameters and "
            f"{len(self.new_prior_params)} new prior parameters."
        )

    def forward(self, **inputs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get current training iteration for dynamic gating
        current_iter = self.global_step
        gate_warmup_iters = self.training_cfg.gate_warmup_iters
        dynamic_k = self.model_cfg.dynamic_k

        hidden_states = self.model.model.embed_tokens(inputs["input_ids"])
        attention_mask = inputs.get("attention_mask")
        
        prior_losses_per_layer = []
        gate_vecs_per_layer = []

        # Iterate through the dynamically modified layers
        for layer in self.model.model.layers:
            # Pass new gating parameters to the forward pass of each layer
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=inputs.get("position_ids"), # Llama Decoder layer expects position_ids
                current_iter=current_iter,
                gate_warmup_iters=gate_warmup_iters,
                dynamic_k=dynamic_k,
            )
            hidden_states = layer_outputs[0] # The final output of the block
            prior_losses_per_layer.append(layer_outputs[-2]) # Second to last is prior_loss
            gate_vecs_per_layer.append(layer_outputs[-1]) # Last is gate_vec

        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
        
        # Aggregate prior losses and gate statistics
        avg_prior_loss = torch.stack(prior_losses_per_layer).mean()
        avg_gate_activation = torch.stack(gate_vecs_per_layer).mean() # Mean across layers and batch

        return logits, avg_prior_loss, avg_gate_activation

    def _calculate_loss(self, batch) -> Tuple[torch.Tensor, ...]:
        logits, prior_loss, gate_activation = self.forward(**batch)
        
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

        return total_loss, lm_loss, prior_loss, perplexity, gate_activation

    def training_step(self, batch, batch_idx):
        total_loss, lm_loss, prior_loss, perplexity, gate_activation = self._calculate_loss(batch)
        self.log("train/loss", total_loss)
        self.log("train/lm_loss", lm_loss, prog_bar=True)
        self.log("train/prior_loss", prior_loss, prog_bar=True)
        self.log("train/perplexity", perplexity)
        self.log("train/gate_activation", gate_activation, prog_bar=True) # Log gate activation
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, lm_loss, prior_loss, perplexity, gate_activation = self._calculate_loss(batch)
        self.log("val/loss", total_loss)
        self.log("val/lm_loss", lm_loss, prog_bar=True)
        self.log("val/perplexity", perplexity)
        self.log("val/prior_loss", prior_loss, prog_bar=False)
        self.log("val/gate_activation", gate_activation, prog_bar=True)

    def test_step(self, batch, batch_idx):
        total_loss, lm_loss, prior_loss, perplexity, gate_activation = self._calculate_loss(batch)
        self.log("test/loss", total_loss)
        self.log("test/lm_loss", lm_loss)
        self.log("test/perplexity", perplexity)
        self.log("test/prior_loss", prior_loss)
        self.log("test/gate_activation", gate_activation)

    def configure_optimizers(self):
        param_groups = [
            {"params": self.original_params, "lr": self.training_cfg.optimizer.base_lr},
            {"params": self.new_prior_params, "lr": self.training_cfg.optimizer.prior_ffn_lr},
        ]
        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.training_cfg.optimizer.weight_decay
        )
        total_training_steps = self.trainer.estimated_stepping_batches
        if total_training_steps is None:
            # Fallback for scheduler if estimated_stepping_batches is not available at this stage.
            # This happens if `trainer.fit` hasn't been called.
            # We assume `max_iters` represents the total training steps.
            log.warning("`trainer.estimated_stepping_batches` is None. Using `max_iters` for total training steps for scheduler setup.")
            total_training_steps = self.training_cfg.max_iters

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.training_cfg.scheduler.warmup_steps,
            num_training_steps=total_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }