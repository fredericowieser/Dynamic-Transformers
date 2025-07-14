import logging

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

log = logging.getLogger(__name__)


class FeedForward(nn.Module):
    """A standard Feed-Forward Network, as used in Llama"""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        # This structure is based on the Llama architecture's FFN
        config = {"hidden_size": n_embd, "intermediate_size": 4 * n_embd}
        self.c_fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.c_fc2 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.c_proj = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_proj(self.act(self.c_fc1(x)) * self.c_fc2(x))
        return self.dropout(x)


class DynamicLlamaDecoderLayer(LlamaDecoderLayer):
    """
    A custom version of the LlamaDecoderLayer that inserts our Prior FFN.
    We inherit from the original to reuse as much of the existing logic as possible.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        # Add the new components: the prior FFN and its own LayerNorm
        self.prior_ffn = FeedForward(config.hidden_size, config.hidden_dropout_prob)
        self.prior_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Initialize the new components' weights.
        # The rest of the layer will have its weights copied from the pre-trained model.
        log.info(f"Initializing new prior_ffn for layer {layer_idx}")
        for module in [self.prior_ffn, self.prior_layernorm]:
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.normal_(param, mean=0.0, std=0.02)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_value: tuple[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # Standard Llama Decoder Path
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            **kwargs,
        )
        attention_output = attn_outputs[0]
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # This is the final "posterior" state for this block
        posterior_mlp_output = self.mlp(hidden_states)
        hidden_states = residual + posterior_mlp_output

        # The prior predicts the current state based on the *previous* attention output.
        # We use F.pad to shift the attention_output by one timestep.
        prev_attention_output = F.pad(attention_output[:, :-1, :], (0, 0, 1, 0))
        prior_input = self.prior_layernorm(prev_attention_output)
        prior_prediction = self.prior_ffn(prior_input)

        # Calculate the loss for the prior FFN.
        # We detach the posterior so gradients from this loss only flow into the prior_ffn.
        prior_loss = F.mse_loss(prior_prediction, posterior_mlp_output.detach())

        # Outputs
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[2],)

        # We need to pass the prior_loss up to the main model for the final loss calculation.
        # We store it in the outputs tuple.
        outputs += (prior_loss,)

        return outputs


class DynamicLlama(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        log.info(f"Loading pre-trained model: {self.cfg.model.name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.name, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self._modify_model_and_setup_param_groups()

    def _modify_model_and_setup_param_groups(self):
        log.info("Starting architectural modification of the Llama model...")
        original_layers = self.model.model.layers
        new_layers = nn.ModuleList()

        for i, original_layer in enumerate(original_layers):
            log.info(f"Replacing decoder layer {i} with custom layer...")
            custom_layer = DynamicLlamaDecoderLayer(self.model.config, i)
            custom_layer.load_state_dict(original_layer.state_dict(), strict=False)
            new_layers.append(custom_layer)

        self.model.model.layers = new_layers
        log.info("All Llama decoder layers have been replaced.")

        # Setup for Differential Learning Rates
        log.info("Setting up parameter groups for differential learning rates.")
        self.original_params = []
        self.new_prior_params = []
        for name, param in self.model.named_parameters():
            if "prior_ffn" in name or "prior_layernorm" in name:
                self.new_prior_params.append(param)
                log.info(f"  - Found new parameter for high LR group: {name}")
            else:
                self.original_params.append(param)
        log.info(
            f"Found {len(self.original_params)} original parameters and "
            f"{len(self.new_prior_params)} new parameters."
        )

    def forward(self, **inputs):
        # The custom layer now returns an extra element (prior_loss)
        # We need to handle this in the forward pass.
        # The base `AutoModelForCausalLM` forward pass doesn't expect this.
        # So we call the `model.model` (the base LlamaModel) directly.
        transformer_outputs = self.model.model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            position_ids=inputs.get("position_ids"),
        )
        hidden_states = transformer_outputs[0]
        logits = self.model.lm_head(hidden_states)

        # Collect prior losses from each layer's output
        prior_losses = [layer_output[-1] for layer_output in transformer_outputs[1]]
        avg_prior_loss = torch.stack(prior_losses).mean()

        return logits, avg_prior_loss

    def training_step(self, batch, batch_idx):
        logits, prior_loss = self.forward(**batch)

        # Standard Language Modeling Loss
        lm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), batch["labels"].view(-1)
        )

        # Total loss is a combination of LM loss and the new prior loss
        total_loss = lm_loss + self.cfg.model.prior_loss_weight * prior_loss

        self.log("train_lm_loss", lm_loss, prog_bar=True)
        self.log("train_prior_loss", prior_loss, prog_bar=True)
        self.log("train_total_loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        # Set up two parameter groups with different learning rates
        param_groups = [
            {
                "params": self.original_params,
                "lr": self.cfg.optimizer.base_lr,
            },
            {
                "params": self.new_prior_params,
                "lr": self.cfg.optimizer.prior_ffn_lr,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups)
        return optimizer