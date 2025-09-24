import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import logging

from transformers import PreTrainedModel, PretrainedConfig
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config,
    Qwen2Model,
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
    create_causal_mask,
    create_sliding_window_causal_mask,
    Qwen2ForCausalLM,
)

log = logging.getLogger(__name__)

class BaseForCausalLM(PreTrainedModel):
    config_class = Qwen2Config

    def __init__(self, config: PretrainedConfig, model_type: str = None, **kwargs):
        super().__init__(config, **kwargs)
        # Make model compatible with HF from_pretrained by allowing model_type to be optional
        self.config.model_type = model_type or config.model_type
        self.model_params = kwargs
        self.model = Qwen2Model(config)
        self.model.config.model_type = model_type
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
            # We don’t create DynamicCache explicitly here; Qwen2Model does this
            # but for training we typically keep use_cache=False.
            pass

        if cache_position is None:
            past_seen = 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            # unsqueezed to [1, T], now expand to [B, T] for all batches
            position_ids = cache_position.unsqueeze(0)
            batch_size = inputs_embeds.shape[0]
            position_ids = position_ids.expand(batch_size, -1)

        # Create mask mapping (same contract as Qwen2Model)
        if not isinstance(causal_mask_mapping := attention_mask, dict):
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

        hidden_states = inputs_embeds

        # Create position embeddings once (stock behavior)
        # Qwen2Model stores rotary_emb internally
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        # Run layers (allow subclasses to interpose dynamic logic)
        layer_kwargs = {**kwargs, "attention_mask": causal_mask_mapping["full_attention"], "position_ids": position_ids, "position_embeddings": position_embeddings}
        
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

        # Cross-entropy loss for the main language modeling task
        lm_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Start with the base LM loss
        total_loss = lm_loss
        
        # Prepare output dictionary
        out = {"logits": logits, "lm_loss": lm_loss}
        if aux:
            out.update(aux)

        # Add auxiliary losses to the total loss and prepare for logging
        if self.training and 'unscaled_losses' in out:
            log.debug("--- LOSS DEBUG: ENTERING AUX LOSS BLOCK ---")
            unscaled_losses = out.pop('unscaled_losses') # Pop from the final output dict
            log.debug(f"--- LOSS DEBUG: unscaled_losses dict: {unscaled_losses}")
            for loss_name, unscaled_loss in unscaled_losses.items():
                # Determine the weight (lambda) for this loss
                # e.g., for 'mod_router_bce_loss', look for 'mod.aux_loss_weight' in config
                model_type = loss_name.split('_')[0] # mod, sdt, stt
                weight_key = loss_name.replace(f"{model_type}_", "").replace("_loss", "_loss_weight")
                loss_weight = self.model_params.get(model_type, {}).get(weight_key, 0.0)
                log.debug(f"--- LOSS DEBUG: Processing {loss_name} with weight {loss_weight}")

                # Add to total loss
                if unscaled_loss is not None and loss_weight > 0:
                    scaled_loss = unscaled_loss * loss_weight
                    log.debug(f"--- LOSS DEBUG: lm_loss={total_loss}, unscaled={unscaled_loss}, scaled={scaled_loss}")
                    total_loss = total_loss + scaled_loss
                    log.debug(f"--- LOSS DEBUG: new total_loss={total_loss}")
                    out[f"loss/{loss_name}_scaled"] = scaled_loss
                    out[f"loss_weight/{loss_name}"] = loss_weight
                
                out[f"loss/{loss_name}_unscaled"] = unscaled_loss
        else:
            log.debug("--- LOSS DEBUG: SKIPPING AUX LOSS BLOCK ---")
            if not self.training:
                log.debug("--- LOSS DEBUG: Reason: not self.training")
            if 'unscaled_losses' not in out:
                log.debug(f"--- LOSS DEBUG: Reason: 'unscaled_losses' not in output dict. Keys: {out.keys()}")


        # Final total loss
        out['loss'] = total_loss
        log.debug(f"--- LOSS DEBUG: Final lm_loss={lm_loss}, final total_loss={out['loss']}")

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