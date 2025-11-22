import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen2RMSNorm,
    create_causal_mask,
    create_sliding_window_causal_mask,
)

log = logging.getLogger(__name__)


class BaseForCausalLM(PreTrainedModel):

    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model_params = kwargs
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
            # Explicitly declare tied weights for the saving mechanism
            self._tied_weights_keys = ["lm_head.weight", "model.embed_tokens.weight"]

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def copy_weights_from_pretrained(self, base_model: Qwen2ForCausalLM):
        log.info("Copying weights from pretrained model...")
        self.model.load_state_dict(base_model.model.state_dict(), strict=False)
        self.lm_head.load_state_dict(base_model.lm_head.state_dict(), strict=False)
        log.info("Weight copy complete.")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        cfg = self.config
        use_cache = use_cache if use_cache is not None else cfg.use_cache

        # Input handling (match HF logic)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        # Past KV and cache positions
        if use_cache and past_key_values is None:
            pass

        if cache_position is None:
            past_seen = 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(inputs_embeds.shape[0], -1)

        # Create mask mapping
        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.model.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if getattr(self.model, "has_sliding_layers", False):
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
                    **mask_kwargs
                )
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        hidden_states, aux = self._run_layers(
            hidden_states=hidden_states,
            mask_mapping=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # If no labels, we are in inference mode (e.g., for lm-eval)
        if labels is None:
            return CausalLMOutputWithPast(logits=logits)

        # Otherwise, we are in training mode, calculate loss and return custom dict
        lm_loss = None
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        total_loss = lm_loss
        out = {"logits": logits, "lm_loss": lm_loss}

        aux_metrics = {}
        if self.training and aux:
            if "unscaled_losses" in aux:
                unscaled_losses = aux.pop("unscaled_losses")
                for loss_name, unscaled_loss in unscaled_losses.items():
                    model_type = loss_name.split("_")[0]
                    weight_key = loss_name.replace(f"{model_type}_", "").replace(
                        "_loss", "_loss_weight"
                    )
                    loss_weight = self.model_params.get(model_type, {}).get(weight_key, 0.0)

                    if unscaled_loss is not None and loss_weight > 0:
                        scaled_loss = unscaled_loss * loss_weight
                        total_loss += scaled_loss
                        aux_metrics[f"loss/{loss_name}_scaled"] = scaled_loss
                        aux_metrics[f"loss_weight/{loss_name}"] = loss_weight

                    aux_metrics[f"loss/{loss_name}_unscaled"] = unscaled_loss

            for key, value in aux.items():
                if key.startswith("router_stats"):
                    for stat_key, stat_value in value.items():
                        aux_metrics[f"router/{stat_key}"] = stat_value
                elif key.startswith("beta"):
                    aux_metrics[f"beta/{key.replace('beta_', '')}"] = value

        out["loss"] = total_loss
        if aux_metrics:
            out["aux_metrics"] = aux_metrics

        return out

    def _run_layers(
        self,
        hidden_states: torch.Tensor,
        mask_mapping: Dict[str, torch.Tensor],
        position_ids: torch.LongTensor,
        past_key_values: Optional[Any],
        use_cache: bool,
        cache_position: Optional[torch.LongTensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        output_attentions: bool,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Default: run all stock HF layers with the right mask/pos embeddings.
        Dynamic subclasses override this and interpose custom routing.
        """
        for layer in self.model.layers:
            attn_mask = mask_mapping[layer.attention_type]
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        return hidden_states, {}
