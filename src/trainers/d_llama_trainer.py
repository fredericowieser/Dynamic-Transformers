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

        self.per_layer_gate_activation_rolling_history = []
        for _ in range(self.model.config.num_hidden_layers):
            self.per_layer_gate_activation_rolling_history.append(
                {
                    "mean": deque(maxlen=ROLLING_WINDOW_SIZE),
                    "std": deque(maxlen=ROLLING_WINDOW_SIZE),
                }
            )

    def _modify_model_architecture(self):
        """Replace standard Llama layers with Dynamic layers."""
        log.info("Replacing LlamaDecoderLayer with DynamicLlamaDecoderLayer...")

        # Add token_wise config to model config if not present
        if not hasattr(self.model.config, "token_wise"):
            self.model.config.token_wise = self.model_cfg.token_wise

        # Add other dynamic configs
        self.model.config.dynamic_k = getattr(self.model_cfg, "dynamic_k", 0.5)
        self.model.config.ce_bias = getattr(self.model_cfg, "ce_bias", 0.0)

        new_layers = nn.ModuleList()
        for i, layer in enumerate(self.model.model.layers):
            # Single unified layer creation - config determines behavior
            custom_layer = DynamicLlamaDecoderLayer(self.model.config, i)
            custom_layer.load_state_dict(layer.state_dict(), strict=False)
            new_layers.append(custom_layer)

        self.model.model.layers = new_layers
        log.info(
            f"All {len(new_layers)} Llama decoder layers replaced with Dynamic layers (token_wise={self.model.config.token_wise})"
        )

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

    def forward(self, **inputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_iter = self.global_step
        gate_warmup_iters = self.training_cfg.gate_warmup_iters
        dynamic_k = self.model_cfg.dynamic_k
        ce_bias = self.model_cfg.ce_bias

        input_ids = inputs["input_ids"]
        hidden_states = self.model.model.embed_tokens(inputs["input_ids"])
        # attention_mask = inputs.get("attention_mask")

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = hidden_states.dtype

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
                .unsqueeze(0)  # Add batch dimension
                .expand(input_ids.shape[0], -1)  # Expand to match batch size
            )

        # FIX: Prepare the 4D attention mask
        # Get the original 2D attention_mask (padding mask) from inputs
        padding_attention_mask_2d = inputs.get("attention_mask")

        # Create a base causal mask (lower triangular)
        # It's (seq_len, seq_len) with -inf for future tokens
        causal_mask_base = torch.full(
            (seq_len, seq_len), torch.finfo(dtype).min, device=device
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
                .masked_fill_(
                    (1 - padding_attention_mask_2d).bool(), torch.finfo(dtype).min
                )
                .unsqueeze(1)  # Add head dim
                .unsqueeze(1)  # Add query sequence dim
            )  # Shape: (batch_size, 1, 1, seq_len)

            # Combine causal mask with padding mask using broadcasting
            # The result will be (batch_size, 1, seq_len, seq_len)
            attention_mask_4d = causal_mask_base + expanded_padding_mask
        else:
            # If no padding mask is provided, use only the causal mask
            attention_mask_4d = causal_mask_base.unsqueeze(0).unsqueeze(
                0
            )  # -> (1, 1, seq_len, seq_len)
            attention_mask_4d = attention_mask_4d.expand(
                batch_size, -1, -1, -1
            )  # -> (batch_size, 1, seq_len, seq_len)

        # NOTE: LlamaAttention might expect a specific dimension for num_heads or just 1.
        # It typically expands the mask to num_heads internally if it's 1.
        # So, (batch_size, 1, seq_len, seq_len) should work.

        prior_losses, gate_vecs, ce_proportions, cu_proportions = [], [], [], []

        for layer in self.model.model.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                current_iter=current_iter,
                gate_warmup_iters=gate_warmup_iters,
                dynamic_k=dynamic_k,
                ce_bias=ce_bias,
            )
            hidden_states = layer_outputs[0]
            # Unpack new metrics: (..., avg_ce, avg_cu, prior_loss, gate_vec)
            ce_proportions.append(layer_outputs[-4])
            cu_proportions.append(layer_outputs[-3])
            prior_losses.append(layer_outputs[-2])
            gate_vecs.append(layer_outputs[-1])

        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)

        avg_prior_loss = torch.stack(prior_losses).mean()

        return logits, avg_prior_loss, gate_vecs, ce_proportions, cu_proportions

    def _calculate_loss(self, batch) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, torch.Tensor]],
    ]:
        # --- NEW: Unpack per-layer lists ---
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

        # --- NEW: Calculate overall averages from the per-layer lists ---
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

        per_layer_gate_stats = []
        for gate_vec in gate_vecs_per_layer:
            layer_mean = gate_vec.mean()
            layer_std = (
                gate_vec.std()
                if gate_vec.numel() > 1
                else torch.tensor(0.0, device=self.device)
            )
            per_layer_gate_stats.append({"mean": layer_mean, "std": layer_std})

        # --- NEW: Return all metrics for logging ---
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
        self.log(f"{prefix}/perplexity", perplexity, on_step=on_step, on_epoch=on_epoch)

        # Log overall gate, CE, and CU metrics
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
        """
        Configure two AdamW optimizers with separate ReduceLROnPlateau schedulers.
        - One for the base model parameters.
        - One for the new prior FFN parameters.
        """
        # Optimizer for the base model (e.g., initial LR 1e-4)
        optimizer_base = torch.optim.AdamW(
            self.original_params,
            lr=self.training_cfg.optimizer.base_lr,
            weight_decay=self.training_cfg.optimizer.weight_decay,
        )

        # Optimizer for the new prior FFN components (e.g., initial LR 1e-2)
        optimizer_prior = torch.optim.AdamW(
            self.new_prior_params,
            lr=self.training_cfg.optimizer.prior_ffn_lr,
            weight_decay=self.training_cfg.optimizer.weight_decay,
        )

        # Schedulers that reduce LR when validation loss plateaus
        scheduler_base = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_base,
            mode="min",
            factor=self.training_cfg.scheduler.factor,
            patience=self.training_cfg.scheduler.patience,
            min_lr=1e-5,  # Shared minimum LR
        )
        scheduler_prior = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_prior,
            mode="min",
            factor=self.training_cfg.scheduler.factor,
            patience=self.training_cfg.scheduler.patience,
            min_lr=1e-5,  # Shared minimum LR
        )

        return (
            [optimizer_base, optimizer_prior],
            [
                {
                    "scheduler": scheduler_base,
                    "monitor": "val/loss",
                    "interval": "epoch",
                },
                {
                    "scheduler": scheduler_prior,
                    "monitor": "val/loss",
                    "interval": "epoch",
                },
            ],
        )
