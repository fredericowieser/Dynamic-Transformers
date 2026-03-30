import torch
import torch.nn as nn
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from ..base.block import DynamicBlock
from ..base.causal_lm import BaseForCausalLM
from ..configs import StandardConfig


class StandardLayer(nn.Module):
    """
    A simple wrapper around DynamicBlock that always selects 100% of tokens.
    This allows the standard model to use the same optimized Triton kernels.
    """

    def __init__(self, hf_layer: Qwen2DecoderLayer, config, model_params: dict):
        super().__init__()
        self.block = DynamicBlock(hf_layer)
        self.config = config

    def forward(self, hidden_states, training: bool, **kwargs):
        B, T, D = hidden_states.shape
        
        # Select ALL tokens
        batch_idx = torch.arange(B, device=hidden_states.device).unsqueeze(1).expand(-1, T).reshape(-1)
        token_idx = torch.arange(T, device=hidden_states.device).unsqueeze(0).expand(B, -1).reshape(-1)
        
        # Standard model doesn't use gating, so we pass None
        new_states, _, _ = self.block.process_selected(
            hidden_states,
            batch_indices=batch_idx,
            token_indices=token_idx,
            gating_scores=None,
            use_soft_gating=False,
            **kwargs,
        )
        
        return new_states, {}, {}


class StandardTransformerForCausalLM(BaseForCausalLM):
    config_class = StandardConfig
    _supports_flash_attn_2 = False

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # Replace standard layers with StandardLayers that wrap DynamicBlock
        for i in range(self.config.num_hidden_layers):
            self.model.layers[i] = StandardLayer(self.model.layers[i], config, self.model_params)

    def _run_layers(
        self,
        hidden_states,
        mask_mapping,
        position_ids,
        past_key_values,
        use_cache,
        cache_position,
        position_embeddings,
        output_attentions,
        **kwargs,
    ):
        if self.training and self.gradient_checkpointing:
            if not hidden_states.requires_grad:
                hidden_states = hidden_states + (self.gradient_checkpointing_trigger * 0.0)

        all_losses = []
        
        for layer in self.model.layers:
            # All layers are now StandardLayers wrapping DynamicBlock
            layer_attn_mask = mask_mapping[layer.block.layer.attention_type]
            
            if self.gradient_checkpointing and self.training:
                # StandardLayer.forward is simplified to just process_selected
                hidden_states, losses, metrics = torch.utils.checkpoint.checkpoint(
                    layer.__call__,
                    hidden_states,
                    use_reentrant=False,
                    training=self.training,
                    attention_mask=layer_attn_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    output_attentions=output_attentions,
                    **kwargs,
                )
            else:
                hidden_states, losses, metrics = layer(
                    hidden_states,
                    training=self.training,
                    attention_mask=layer_attn_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    output_attentions=output_attentions,
                    **kwargs,
                )
            all_losses.append(losses)

        return hidden_states, {"unscaled_losses": {}}
