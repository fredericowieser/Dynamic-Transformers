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
    supports_gradient_checkpointing = True

    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model_params = kwargs
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Dummy parameter to force gradient checkpointing to trigger
        self.gradient_checkpointing_trigger = nn.Parameter(torch.zeros(1))

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
            # Explicitly declare tied weights for the saving mechanism
            self._tied_weights_keys = ["lm_head.weight", "model.embed_tokens.weight"]

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func=None):
        """
        Override to prevent double-checkpointing. HF's default propagates gradient_checkpointing=True
        to all GradientCheckpointingLayer submodules (e.g. Qwen2DecoderLayer), so when our _run_layers
        calls checkpoint(layer.__call__, ..., use_reentrant=False), the layer's __call__ also applies
        its own inner checkpoint(use_reentrant=True). The interaction between the two modes corrupts
        requires_grad on the output, triggering the 'None of the inputs have requires_grad=True' warning
        and causing DDP NCCL deadlocks. By only setting the flag on self, we ensure a single checkpoint
        level applied in _run_layers.
        """
        self.gradient_checkpointing = enable
        if gradient_checkpointing_func is not None:
            self._gradient_checkpointing_func = gradient_checkpointing_func

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

    @property
    def gradient_checkpointing(self) -> bool:
        """Robustly check if gradient checkpointing is enabled on this model."""
        if getattr(self.model, "gradient_checkpointing", False):
            return True
        if getattr(self.config, "gradient_checkpointing", False):
            return True
        return getattr(self, "_gradient_checkpointing", False)

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value: bool):
        self._gradient_checkpointing = value

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
            
        hidden_states = inputs_embeds
        
        if self.training and self.gradient_checkpointing:
            # Critical: This is a foolproof trigger for gradient checkpointing.
            # Adding a parameter-dependent zero ensures hidden_states.requires_grad is True.
            # This is MANDATORY to prevent DDP NCCL deadlocks when checkpointing is skipped.
            if not hidden_states.requires_grad:
                hidden_states = hidden_states + (self.gradient_checkpointing_trigger * 0.0)
            # Explicitly disable cache
            use_cache = False

        if cache_position is None:
            past_seen = 0
            if past_key_values is not None:
                past_seen = past_key_values.get_seq_length()
            cache_position = torch.arange(
                past_seen, past_seen + hidden_states.shape[1], device=hidden_states.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(hidden_states.shape[0], -1)

        # Create mask mapping
        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.model.config,
                "input_embeds": hidden_states,
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
        # Don't include logits in training output — the [B, T, V] tensor is never
        # used by the training loop, but Accelerate's DDP wrapper converts all
        # outputs to fp32, which allocates an extra B*T*V*4 bytes (~18 GB at B=32).
        out = {"lm_loss": lm_loss}

        aux_metrics = {}
        if aux:
            if "unscaled_losses" in aux:
                unscaled_losses = aux.pop("unscaled_losses")
                model_type = getattr(self.config, "model_type", None)
                # Ensure model_weights is a dict (config sub-objects might be objects or dicts)
                model_weights = getattr(self.config, model_type, {}) if model_type else {}
                if hasattr(model_weights, "to_dict"):
                    model_weights = model_weights.to_dict()
                elif not isinstance(model_weights, dict):
                    # Handle case where it might be a Namespace or similar
                    model_weights = vars(model_weights) if hasattr(model_weights, "__dict__") else {}

                # 1. Causal Router Loss (Shared across MoD, SDT, STT)
                if "causal_router_loss" in unscaled_losses:
                    u_loss = unscaled_losses["causal_router_loss"]
                    weight = model_weights.get("causal_router_loss_weight", 0.0)
                    if u_loss is not None:
                        scaled_loss = u_loss * weight
                        if self.training:
                            total_loss += scaled_loss
                        aux_metrics["loss/causal_router_scaled"] = scaled_loss
                        aux_metrics["loss/causal_router_unscaled"] = u_loss

                # 2. Model-Specific Auxiliary Losses
                if model_type == "sdt" and "sdt_prior_loss" in unscaled_losses:
                    u_loss = unscaled_losses["sdt_prior_loss"]
                    weight = model_weights.get("prior_loss_weight", 0.0)
                    if u_loss is not None:
                        scaled_loss = u_loss * weight
                        if self.training:
                            total_loss += scaled_loss
                        aux_metrics["loss/sdt_prior_scaled"] = scaled_loss
                        aux_metrics["loss/sdt_prior_unscaled"] = u_loss
                        
                elif model_type == "stt" and "stt_tpn_loss" in unscaled_losses:
                    u_loss = unscaled_losses["stt_tpn_loss"]
                    weight = model_weights.get("tpn_loss_weight", 0.0)
                    if u_loss is not None:
                        scaled_loss = u_loss * weight
                        if self.training:
                            total_loss += scaled_loss
                        aux_metrics["loss/stt_tpn_scaled"] = scaled_loss
                        aux_metrics["loss/stt_tpn_unscaled"] = u_loss
                        
                elif model_type == "mod":
                    if "mod_router_aux_loss" in unscaled_losses:
                        u_loss = unscaled_losses["mod_router_aux_loss"]
                        weight = model_weights.get("router_aux_loss_weight", 0.0)
                        if u_loss is not None:
                            scaled_loss = u_loss * weight
                            if self.training:
                                total_loss += scaled_loss
                            aux_metrics["loss/mod_router_aux_scaled"] = scaled_loss
                            aux_metrics["loss/mod_router_aux_unscaled"] = u_loss
                    
                    if "mod_z_loss" in unscaled_losses:
                        u_loss = unscaled_losses["mod_z_loss"]
                        weight = model_weights.get("z_loss_weight", 1e-4)
                        if u_loss is not None:
                            scaled_loss = u_loss * weight
                            if self.training:
                                total_loss += scaled_loss
                            aux_metrics["loss/mod_z_scaled"] = scaled_loss
                            aux_metrics["loss/mod_z_unscaled"] = u_loss

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
        if self.training and self.gradient_checkpointing:
            if not hidden_states.requires_grad:
                hidden_states = hidden_states + (self.gradient_checkpointing_trigger * 0.0)

        for layer in self.model.layers:
            attn_mask = mask_mapping[layer.attention_type]
            
            if self.gradient_checkpointing and self.training:
                # Clean up checkpoint call to ensure hidden_states is the primary positional arg
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer.__call__,
                    hidden_states,
                    use_reentrant=False,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            else:
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
