import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from omegaconf import DictConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2RotaryEmbedding, Qwen2DecoderLayer, Qwen2Config
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from .block import DynamicBlock
import logging
import copy

log = logging.getLogger(__name__)

from transformers import PreTrainedModel, Qwen2Config

class BaseForCausalLM(PreTrainedModel):
    config_class = Qwen2Config 

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # ————————————————
        # Mirror Qwen2Model: own a single rotary embedding
        self.rotary_emb = Qwen2RotaryEmbedding(config)
        # ————————————————
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _setup_layers(self):
        raise NotImplementedError("Subclasses must implement `_setup_layers`")

    def _forward_layers(self, hidden_states: torch.Tensor, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement `_forward_layers`")

    def forward(
        self, input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        B, T, D = hidden_states.shape

        if position_ids is None:
            position_ids = torch.arange(T, device=hidden_states.device).unsqueeze(0).expand(B, -1)
        
        causal_mask = _prepare_4d_causal_attention_mask(attention_mask, (B, T), hidden_states, 0)
        
        # compute RoPE once (partial dims) and share to every layer
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        layer_kwargs = {
            **kwargs,
            "attention_mask": causal_mask,
            "position_ids": position_ids,
            "position_embeddings": (cos, sin),
        }
        
        layer_outputs = self._forward_layers(hidden_states, **layer_kwargs)
        
        final_hidden_states = self.norm(layer_outputs["hidden_states"])
        logits = self.lm_head(final_hidden_states)
        
        lm_loss = self.compute_loss(logits, labels)
        
        total_loss = lm_loss
        if "aux_loss" in layer_outputs and layer_outputs["aux_loss"] is not None and self.training:
            total_loss += layer_outputs["aux_loss"]
        
        final_outputs = {"logits": logits, "loss": total_loss, "lm_loss": lm_loss}
        final_outputs.update(layer_outputs)
        
        return final_outputs

    def compute_loss(self, logits, labels):
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            return loss_fct(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
        return None

    def gradient_checkpointing_enable(self):
        """
        Enables gradient checkpointing for all `DynamicBlock` and `Qwen2DecoderLayer`
        instances within the model's layers.
        """
        for layer in self.layers:
            if isinstance(layer, DynamicBlock):
                if hasattr(layer.layer, 'gradient_checkpointing_enable'):
                    layer.layer.gradient_checkpointing_enable()
            elif isinstance(layer, nn.ModuleDict):
                # Handle cases where DynamicBlock is nested in ModuleDict (e.g., MoD, SDT, STT)
                for sub_module in layer.values():
                    if isinstance(sub_module, DynamicBlock):
                        if hasattr(sub_module.layer, 'gradient_checkpointing_enable'):
                            sub_module.layer.gradient_checkpointing_enable()
                    elif hasattr(sub_module, 'block') and isinstance(sub_module.block, DynamicBlock):
                        if hasattr(sub_module.block.layer, 'gradient_checkpointing_enable'):
                            sub_module.block.layer.gradient_checkpointing_enable()
                    elif hasattr(sub_module, 'decision') and hasattr(sub_module.decision, 'block'): # For SDT
                        if hasattr(sub_module.decision.block.layer, 'gradient_checkpointing_enable'):
                            sub_module.decision.block.layer.gradient_checkpointing_enable()
                    elif isinstance(sub_module, Qwen2DecoderLayer): # For standard layers in ModuleDict
                        if hasattr(sub_module, 'gradient_checkpointing_enable'):
                            sub_module.gradient_checkpointing_enable()
                    elif hasattr(sub_module, 'block') and isinstance(sub_module.block, Qwen2DecoderLayer): # For STTLayer
                        if hasattr(sub_module.block, 'gradient_checkpointing_enable'):
                            sub_module.block.gradient_checkpointing_enable()
            elif isinstance(layer, Qwen2DecoderLayer): # For standard layers directly in self.layers
                if hasattr(layer, 'gradient_checkpointing_enable'):
                    layer.gradient_checkpointing_enable()
        # Also enable for the base model if it has the method (e.g., if it's a Qwen2ForCausalLM)
        if hasattr(super(), 'gradient_checkpointing_enable'):
            super().gradient_checkpointing_enable()

    def enable_input_require_grads(self):
        """
        Enables `requires_grad` for the input embeddings, which is necessary
        when using gradient checkpointing.
        """
        if hasattr(self, 'embed_tokens') and hasattr(self.embed_tokens, 'weight'):
            self.embed_tokens.weight.requires_grad_(True)
        # If the model has a get_input_embeddings method (like PreTrainedModel), use it
        elif hasattr(self, 'get_input_embeddings'):
            input_embeddings = self.get_input_embeddings()
            if input_embeddings is not None and hasattr(input_embeddings, 'weight'):
                input_embeddings.weight.requires_grad_(True)

    def copy_weights_from_pretrained(self, pretrained_model):
        """
        Copies weights from a pretrained Qwen2 model to this model.
        """
        self.embed_tokens.load_state_dict(pretrained_model.model.embed_tokens.state_dict())
        self.norm.load_state_dict(pretrained_model.model.norm.state_dict())
        self.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())

        for i, layer in enumerate(self.layers):
            if i >= len(pretrained_model.model.layers): break
            pretrained_layer = pretrained_model.model.layers[i]
            
            if isinstance(layer, DynamicBlock):
                layer.layer.load_state_dict(pretrained_layer.state_dict())
            elif isinstance(layer, nn.ModuleDict):
                if 'block' in layer:
                    layer.block.layer.load_state_dict(pretrained_layer.state_dict())
                if 'decision' in layer and hasattr(layer.decision, 'block'):
                    layer.decision.block.load_state_dict(pretrained_layer.state_dict())
                if 'dynamic_block' in layer and i + 1 < len(pretrained_model.model.layers):
                    pretrained_dynamic_layer = pretrained_model.model.layers[i + 1]
                    layer.dynamic_block.layer.load_state_dict(pretrained_dynamic_layer.state_dict())

    def get_trainable_parameters(self) -> List[Dict[str, Any]]:
        return [{'name': 'base_model', 'params': list(p for p in self.parameters() if p.requires_grad)}]

    def _create_param_groups(self, component_map: Dict[str, str]) -> List[Dict[str, Any]]:
        param_groups = {}
        base_model_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            assigned = False
            for group_name, keyword in component_map.items():
                if keyword in name:
                    param_groups.setdefault(group_name, []).append(param)
                    assigned = True
                    break
            if not assigned:
                base_model_params.append(param)
        
        groups_list = [{'name': 'base_model', 'params': base_model_params}]
        groups_list.extend([{'name': name, 'params': params} for name, params in param_groups.items()])
        return [g for g in groups_list if g['params']]
