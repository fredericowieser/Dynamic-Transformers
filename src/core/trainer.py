import logging
import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig
from src.models.dynamic_llama import DynamicLlamaDecoderLayer
from typing import Tuple, Optional, Dict, List
from collections import deque

log = logging.getLogger(__name__)

# Define the rolling window size
ROLLING_WINDOW_SIZE = 100

class LightningModel(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, training_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg

        log.info(f"Loading pre-trained model: {self.model_cfg.model_name}")
        
        # Load the configuration first
        config = AutoConfig.from_pretrained(self.model_cfg.model_name)
        log.info(f"Original config.json for {self.model_cfg.model_name}:")
        log.info(config.to_dict())

        # FIX: Handle potential missing 'type' in 'rope_scaling' with extra robustness
        # The LlamaRotaryEmbedding constructor uses:
        # self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling["type"])
        # This will KeyError if config.rope_scaling["type"] is missing AND config.rope_scaling.get("rope_type") is None.

        # Ensure 'rope_scaling' is a dictionary. If it's not present or is not a dict, create a default.
        if not hasattr(config, "rope_scaling") or config.rope_scaling is None or not isinstance(config.rope_scaling, dict):
            original_rope_scaling_value = getattr(config, "rope_scaling", "N/A")
            log.warning(
                f"Config for {self.model_cfg.model_name} `rope_scaling` is missing or not a dict (was: {original_rope_scaling_value}). "
                "Initializing `rope_scaling` with default 'linear' type and factor 1.0."
            )
            config.rope_scaling = {"type": "linear", "factor": 1.0}
        
        # If rope_scaling exists and is a dict, ensure it has both 'rope_type' and 'type' to be safe.
        # This is a defensive move against subtle behaviors of .get() or unexpected internal logic.
        if "rope_type" in config.rope_scaling and config.rope_scaling["rope_type"] is not None:
            # If rope_type is present and not None, ensure 'type' is also set to a consistent value
            # so the fallback in LlamaRotaryEmbedding never causes KeyError.
            if "type" not in config.rope_scaling:
                log.warning(
                    f"Config for {self.model_cfg.model_name} `rope_scaling` has 'rope_type' but no 'type'. "
                    f"Setting `rope_scaling.type` to be consistent with 'rope_type': {config.rope_scaling['rope_type']}."
                )
                config.rope_scaling["type"] = config.rope_scaling["rope_type"]
        else: # 'rope_type' is missing or None
            # If 'rope_type' is missing or None, then 'type' is the primary one. Ensure it exists.
            if "type" not in config.rope_scaling:
                log.warning(
                    f"Config for {self.model_cfg.model_name} `rope_scaling` lacks 'rope_type' (or it's None) and 'type'. "
                    "Setting `rope_scaling.type` to 'linear' and `factor` to 1.0."
                )
                config.rope_scaling["type"] = "linear"
                config.rope_scaling["factor"] = config.rope_scaling.get("factor", 1.0)
            
            # Ensure rope_type is also set if type is set but rope_type is missing/None
            if "rope_type" not in config.rope_scaling or config.rope_scaling["rope_type"] is None:
                config.rope_scaling["rope_type"] = config.rope_scaling["type"] # Make them consistent

        # Finally, ensure factor is present if it's a scaling config
        if "factor" not in config.rope_scaling:
            log.warning(
                f"Config for {self.model_cfg.model_name} `rope_scaling` lacks 'factor'. Setting to 1.0."
            )
            config.rope_scaling["factor"] = 1.0


        log.info(f"Modified config.rope_scaling before model loading: {config.rope_scaling}")
        


        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.model_name, torch_dtype=torch.bfloat16, config=config # Pass the potentially modified config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self._modify_model_architecture()
        self._setup_parameter_groups()

        self.per_layer_gate_activation_rolling_history = []
        for _ in range(self.model.config.num_hidden_layers):
            self.per_layer_gate_activation_rolling_history.append(
                {"mean": deque(maxlen=ROLLING_WINDOW_SIZE), "std": deque(maxlen=ROLLING_WINDOW_SIZE)}
            )

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

    def forward(self, **inputs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_iter = self.global_step
        gate_warmup_iters = self.training_cfg.gate_warmup_iters
        dynamic_k = self.model_cfg.dynamic_k

        input_ids = inputs["input_ids"]
        hidden_states = self.model.model.embed_tokens(inputs["input_ids"])
        # attention_mask = inputs.get("attention_mask")

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = hidden_states.dtype # Define dtype here

        # FIX: Generate position_ids if not provided by the batch
        position_ids = inputs.get("position_ids")
        if position_ids is None:
            # Generate default position_ids: a tensor of increasing integers (0, 1, 2, ...)
            # for each sequence in the batch.
            seq_len = input_ids.shape[1]
            position_ids = (
                torch.arange(
                    seq_len,
                    dtype=torch.long,
                    device=input_ids.device,
                )
                .unsqueeze(0) # Add batch dimension
                .expand(input_ids.shape[0], -1) # Expand to match batch size
            )

        # FIX: Prepare the 4D attention mask
        # Get the original 2D attention_mask (padding mask) from inputs
        padding_attention_mask_2d = inputs.get("attention_mask")

        # Create a base causal mask (lower triangular)
        # It's (seq_len, seq_len) with -inf for future tokens
        causal_mask_base = torch.full(
            (seq_len, seq_len), 
            torch.finfo(dtype).min, 
            device=device
        )
        causal_mask_base = torch.triu(causal_mask_base, diagonal=1)

        # Expand padding_attention_mask_2d to (batch_size, 1, 1, seq_len)
        # This will contain 0 for actual tokens and -inf for padded tokens
        # We need to broadcast it with the causal mask.
        if padding_attention_mask_2d is not None:
            expanded_padding_mask = (
                (1 - padding_attention_mask_2d)
                .bool()
                .to(dtype)
                .masked_fill_((1 - padding_attention_mask_2d).bool(), torch.finfo(dtype).min)
                .unsqueeze(1) # Add head dim
                .unsqueeze(1) # Add query sequence dim
            ) # Shape: (batch_size, 1, 1, seq_len)
            
            # Combine causal mask with padding mask using broadcasting
            # The result will be (batch_size, 1, seq_len, seq_len)
            attention_mask_4d = causal_mask_base + expanded_padding_mask
        else:
            # If no padding mask is provided, use only the causal mask
            attention_mask_4d = causal_mask_base.unsqueeze(0).unsqueeze(0) # -> (1, 1, seq_len, seq_len)
            attention_mask_4d = attention_mask_4d.expand(batch_size, -1, -1, -1) # -> (batch_size, 1, seq_len, seq_len)
        
        # NOTE: LlamaAttention might expect a specific dimension for num_heads or just 1.
        # It typically expands the mask to num_heads internally if it's 1.
        # So, (batch_size, 1, seq_len, seq_len) should work.
        
        prior_losses_per_layer = []
        gate_vecs_per_layer = []

        for layer in self.model.model.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                current_iter=current_iter,
                gate_warmup_iters=gate_warmup_iters,
                dynamic_k=dynamic_k,
            )
            hidden_states = layer_outputs[0]
            prior_losses_per_layer.append(layer_outputs[-2])
            gate_vecs_per_layer.append(layer_outputs[-1])

        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
        
        avg_prior_loss = torch.stack(prior_losses_per_layer).mean()
        
        return logits, avg_prior_loss, gate_vecs_per_layer


    def _calculate_loss(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
        # forward now returns gate_vecs_per_layer instead of aggregated gate stats
        logits, prior_loss, gate_vecs_per_layer = self.forward(**batch)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()

        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )

        total_loss = lm_loss + self.model_cfg.prior_loss_weight * prior_loss
        
        perplexity = torch.exp(lm_loss)

        # Calculate overall mean gate activation for the progress bar
        # This is essentially what `avg_gate_activation` was before
        if gate_vecs_per_layer:
            overall_gate_activation_mean = torch.stack(gate_vecs_per_layer).mean()
        else:
            overall_gate_activation_mean = torch.tensor(0.0, device=self.device)

        # Collect per-layer gate statistics (mean and std)
        per_layer_gate_stats = []
        for i, gate_vec in enumerate(gate_vecs_per_layer):
            # gate_vec is (B,), so mean and std are taken across the batch dimension
            layer_mean = gate_vec.mean()
            layer_std = gate_vec.std() if gate_vec.numel() > 1 else torch.tensor(0.0, device=self.device) # Handle single-element case
            per_layer_gate_stats.append(
                {"mean": layer_mean, "std": layer_std}
            )

        # Return the overall mean for prog_bar, and the detailed stats separately
        return total_loss, lm_loss, prior_loss, perplexity, overall_gate_activation_mean, per_layer_gate_stats

    def training_step(self, batch, batch_idx):
        total_loss, lm_loss, prior_loss, perplexity, overall_gate_activation_mean, per_layer_gate_stats = self._calculate_loss(batch)
        
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/lm_loss", lm_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/prior_loss", prior_loss, on_step=True, on_epoch=True)
        self.log("train/perplexity", perplexity, on_step=True, on_epoch=True)
        self.log("train/overall_gate_activation_mean", overall_gate_activation_mean, on_step=True, on_epoch=True, prog_bar=True)
        
        # Update and log rolling averages for per-layer stats
        if self.trainer.global_step > 0: # Ensure global_step is initialized
            log_interval = self.trainer.log_every_n_steps
            
            # Update history deques for each layer
            for i, stats in enumerate(per_layer_gate_stats):
                self.per_layer_gate_activation_rolling_history[i]["mean"].append(stats["mean"].item())
                self.per_layer_gate_activation_rolling_history[i]["std"].append(stats["std"].item())

            # Log rolling averages if it's a logging step
            if self.trainer.global_step % log_interval == 0:
                log_lines = [f"--- Per-Layer Gate Activations (Training, Rolling Avg over last {ROLLING_WINDOW_SIZE} steps) ---"]
                for i in range(len(self.per_layer_gate_activation_rolling_history)):
                    history = self.per_layer_gate_activation_rolling_history[i]
                    
                    if len(history["mean"]) > 0:
                        rolling_mean = sum(history["mean"]) / len(history["mean"])
                        # Calculate rolling std. Convert deque to tensor for torch.std
                        rolling_std = torch.tensor(list(history["std"])).std().item() if len(history["std"]) > 1 else 0.0
                        log_lines.append(f"  Layer {i}: Mean = {rolling_mean:.3f}, Std = {rolling_std:.3f}")
                    else:
                        log_lines.append(f"  Layer {i}: No data yet for rolling average.")
                
                log_lines.append(f"  (Global Step: {self.trainer.global_step})")
                log.info("\n".join(log_lines))

        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, lm_loss, prior_loss, perplexity, overall_gate_activation_mean, per_layer_gate_stats = self._calculate_loss(batch)
        
        self.log("val/loss", total_loss, on_epoch=True, prog_bar=True)
        self.log("val/lm_loss", lm_loss, on_epoch=True, prog_bar=True)
        self.log("val/perplexity", perplexity, on_epoch=True)
        self.log("val/prior_loss", prior_loss, on_epoch=True)
        self.log("val/overall_gate_activation_mean", overall_gate_activation_mean, on_epoch=True, prog_bar=True)

        # Log per-layer gate activation stats to the console (at epoch end for validation)
        if batch_idx == len(self.trainer.val_dataloaders) - 1:
            log_lines = ["--- Per-Layer Gate Activations (Validation) ---"]
            for i, stats in enumerate(per_layer_gate_stats):
                log_lines.append(f"  Layer {i}: Mean = {stats['mean']:.3f}, Std = {stats['std']:.3f}")
            log_lines.append(f"  (Global Step: {self.trainer.global_step})")
            log.info("\n".join(log_lines))

    def test_step(self, batch, batch_idx):
        total_loss, lm_loss, prior_loss, perplexity, overall_gate_activation_mean, per_layer_gate_stats = self._calculate_loss(batch)
        
        self.log("test/loss", total_loss, on_epoch=True)
        self.log("test/lm_loss", lm_loss, on_epoch=True)
        self.log("test/perplexity", perplexity, on_epoch=True)
        self.log("test/prior_loss", prior_loss, on_epoch=True)
        self.log("test/overall_gate_activation_mean", overall_gate_activation_mean, on_epoch=True)

        if batch_idx == len(self.trainer.test_dataloaders) - 1:
            log_lines = ["--- Per-Layer Gate Activations (Test) ---"]
            for i, stats in enumerate(per_layer_gate_stats):
                log_lines.append(f"  Layer {i}: Mean = {stats['mean']:.3f}, Std = {stats['std']:.3f}")
            log_lines.append(f"  (Global Step: {self.trainer.global_step})")
            log.info("\n".join(log_lines))


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