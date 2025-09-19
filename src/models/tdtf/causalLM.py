import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Dict, Any

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from ..base.dynamic_model import BaseDynamicModel
from .layers import TDTFLayer

log = logging.getLogger(__name__)


class TDTFForCausalLM(BaseDynamicModel):
    """TDTF model for causal language modeling."""

    def __init__(self, config):
        super().__init__(config)

        # Strictly require loss weights from config
        if not hasattr(config, "tpn_loss_weight"):
            raise ValueError("Missing config.tpn_loss_weight")
        if not hasattr(config, "causal_loss_weight"):
            raise ValueError("Missing config.causal_loss_weight")
        self.tpn_loss_weight = float(config.tpn_loss_weight)
        self.causal_loss_weight = float(config.causal_loss_weight)

        self._setup_layers()

        if getattr(config, "freeze_base_model", False):
            self.freeze_main_transformer_blocks()

    def _setup_layers(self):
        self.layers = nn.ModuleList()
        for i in range(self.config.num_hidden_layers):
            if i % 2 == 0:
                self.layers.append(Qwen2DecoderLayer(self.config, i))
            else:
                self.layers.append(TDTFLayer(self.config, i))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass with temporal dynamic routing.

        Training mode requires kwargs['beta_ce'] and kwargs['beta_cu'] (scheduled).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Optional β schedule values (required in training)
        beta_ce = kwargs.get("beta_ce", None)
        beta_cu = kwargs.get("beta_cu", None)
        if self.training and (beta_ce is None or beta_cu is None):
            raise ValueError("beta_ce and beta_cu must be provided in training mode")

        # Embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        B, T, D = hidden_states.shape

        if position_ids is None:
            position_ids = torch.arange(T, device=hidden_states.device).unsqueeze(0).expand(B, -1)

        # Rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        total_tpn_loss = torch.tensor(0.0, device=hidden_states.device)
        total_causal_loss = torch.tensor(0.0, device=hidden_states.device)
        total_router_stats: Dict[str, Any] = {}

        for i, layer in enumerate(self.layers):
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            # Prepare attention mask for current layer
            current_attention_mask = attention_mask
            if current_attention_mask is not None and self.config.attn_implementation != "flash_attention_2":
                current_attention_mask = _prepare_4d_causal_attention_mask(
                    current_attention_mask, (B, T), hidden_states, 0
                )

            if isinstance(layer, TDTFLayer):
                layer_output = layer(
                    hidden_states,
                    attention_mask=current_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    position_embeddings=position_embeddings,
                    beta_ce=beta_ce,
                    beta_cu=beta_cu,
                )
                hidden_states = layer_output["hidden_states"]

                if self.training:
                    if layer_output.get("tpn_loss") is not None:
                        total_tpn_loss = total_tpn_loss + layer_output["tpn_loss"]
                    if layer_output.get("causal_loss") is not None:
                        total_causal_loss = total_causal_loss + layer_output["causal_loss"]

                if "router_stats" in layer_output:
                    for k, v in layer_output["router_stats"].items():
                        total_router_stats.setdefault(k, []).append(v)

                if use_cache and "past_key_value" in layer_output:
                    next_decoder_cache += (layer_output["past_key_value"],)

                if output_attentions and "attention_weights" in layer_output:
                    if layer_output["attention_weights"] is not None:
                        all_attentions += (layer_output["attention_weights"],)

            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=current_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    position_embeddings=position_embeddings,
                )
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                    if use_cache:
                        next_decoder_cache += (layer_outputs[1],)
                    if output_attentions:
                        all_attentions += (layer_outputs[2],)
                else:
                    hidden_states = layer_outputs

        # Final norm
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = self.compute_loss(logits, labels)

        # Average aux losses across TDTF layers to stabilize magnitude, then apply weights
        if loss is not None and self.training:
            tdtf_layers = sum(isinstance(l, TDTFLayer) for l in self.layers)
            denom = max(1, tdtf_layers)
            if total_tpn_loss > 0:
                loss = loss + self.tpn_loss_weight * (total_tpn_loss / denom)
            if total_causal_loss > 0:
                loss = loss + self.causal_loss_weight * (total_causal_loss / denom)

        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "tpn_loss": total_tpn_loss.detach(),
            "causal_loss": total_causal_loss.detach(),
            "router_stats": total_router_stats,
        }