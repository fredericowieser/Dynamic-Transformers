import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    Qwen2MLP,
    Qwen2Attention
)
from transformers.modeling_outputs import CausalLMOutputWithPast


class BaseDynamicModel(nn.Module):
    """Base class for dynamic models using Qwen2 components."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Ensure config has required attributes
        if not hasattr(config, '_attn_implementation'):
            config._attn_implementation = 'eager'  # Default attention implementation

        # Use Qwen2 embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Initialize layers (to be overridden by subclasses)
        self.layers = nn.ModuleList()

        # Use Qwen2 RMSNorm
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Use rotary embeddings
        self.rotary_emb = Qwen2RotaryEmbedding(config)

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights if training from scratch
        if hasattr(config, 'init_from_scratch') and config.init_from_scratch:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Qwen's approach."""
        std = getattr(self.config, 'initializer_range', 0.02)

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen2RMSNorm):
            module.weight.data.fill_(1.0)

    def create_standard_layer(self, layer_idx):
        """Create a standard Qwen2 decoder layer."""
        return Qwen2DecoderLayer(self.config, layer_idx)

    def copy_weights_from_pretrained(self, pretrained_model):
        """Copy weights from a pretrained Qwen2 model."""
        # Copy embeddings
        self.embed_tokens.weight.data = pretrained_model.model.embed_tokens.weight.data.clone()

        # Copy final norm
        self.norm.weight.data = pretrained_model.model.norm.weight.data.clone()

        # Copy LM head if not tied
        if not self.config.tie_word_embeddings:
            self.lm_head.weight.data = pretrained_model.lm_head.weight.data.clone()

        # Copy rotary embeddings if they have learnable parameters
        if hasattr(pretrained_model.model, 'rotary_emb'):
            if hasattr(self.rotary_emb, 'inv_freq') and hasattr(pretrained_model.model.rotary_emb, 'inv_freq'):
                self.rotary_emb.inv_freq = pretrained_model.model.rotary_emb.inv_freq.clone()

        # Subclasses will handle copying layer weights

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        """Forward pass to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")

    def compute_loss(self, logits, labels):
        """Compute language modeling loss."""
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten and compute loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            return loss
        return None

    def get_trainable_parameters(self):
        """Get parameter groups for training (to be overridden by subclasses)."""
        return [
            {
                'params': self.parameters(),
                'lr_scale': 1.0,
                'name': 'all'
            }
        ]

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def enable_input_require_grads(self):
        """Enable gradients for input embeddings (needed for gradient checkpointing)."""
        self.embed_tokens.requires_grad_(True)

    def freeze_main_transformer_blocks(self):
        """Freezes the parameters of the main transformer blocks.
        This method should be called after model initialization and weight copying.
        Parameters that are part of the dynamic components (PriorFFN, Routers)
        should be explicitly unfrozen by their respective get_trainable_parameters methods
        or by setting requires_grad=True after this call.
        """
        for name, param in self.named_parameters():
            # Freeze all parameters by default
            param.requires_grad = False
            # Log which parameters are being frozen
            # print(f"Frozen: {name}")