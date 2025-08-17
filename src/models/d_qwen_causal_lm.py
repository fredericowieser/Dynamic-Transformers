# src/models/d_qwen_causal_lm.py

import logging
import torch
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from src.models.d_qwen_config import DynamicQwenConfig
# Import the new layer types
from src.models.dec_qwen_layers import DecisionQwenDecoderLayer
from src.models.dyn_qwen_layers import DynamicQwenDecoderLayer # This is now the "Dynamic Sub-Layer"

logger = logging.getLogger(__name__)

class DynamicQwenForCausalLM(Qwen2ForCausalLM):
    config_class = DynamicQwenConfig

    def __init__(self, config: DynamicQwenConfig):
        super().__init__(config)

        # New training control parameter from config
        self._freeze_main_transformer_blocks = getattr(config, "freeze_main_transformer_blocks", False)

        # Now, patch the layers. This replaces the default Qwen2DecoderLayers with our
        # DecisionQwenDecoderLayer followed by a DynamicQwenDecoderLayer.
        # This function will also handle weight transfer and freezing.
        self._patch_qwen_layers(self, config) # Call internal patching method

        self._log_gates = False
        self._last_gate_means = None

    @property
    def freeze_main_transformer_blocks(self):
        return self._freeze_main_transformer_blocks

    @freeze_main_transformer_blocks.setter
    def freeze_main_transformer_blocks(self, v: bool):
        self._freeze_main_transformer_blocks = v
        # Apply freezing/unfreezing immediately
        self._apply_main_block_freezing()
        logger.info(f"Set freeze_main_transformer_blocks = {v}. Parameters updated.")

    def _apply_main_block_freezing(self):
        """Applies or removes gradient requirement for main transformer block parameters."""
        for layer in self.model.layers: # Iterate over our custom Macro-Layers
            if hasattr(layer, 'decision_layer') and isinstance(layer.decision_layer, DecisionQwenDecoderLayer):
                # Parameters of the original Qwen2 attention and MLP within DecisionLayer
                for param_name, param in layer.decision_layer.named_parameters():
                    if "prior_ffn" not in param_name and "prior_layernorm" not in param_name:
                        param.requires_grad = not self._freeze_main_transformer_blocks
                        if not param.requires_grad:
                            param.data = param.data.contiguous() # Ensure contiguous for frozen params

    def enable_gate_logging(self, flag: bool = True):
        self._log_gates = flag
        self._last_gate_means = None

    def get_last_gate_means(self):
        return self._last_gate_means

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> dict:
        """
        Generates position_ids if missing, creates 4D causal + padding mask.
        (Copied largely from DynamicLlamaForCausalLM, adapted for Qwen2 specifics)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        mask_dtype = torch.float32

        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        if position_ids.shape != (batch_size, seq_len):
            raise ValueError(
                f"position_ids shape {position_ids.shape} does not match input_ids {input_ids.shape}"
            )

        causal_mask_base = torch.full(
            (seq_len, seq_len),
            torch.finfo(mask_dtype).min,
            dtype=mask_dtype,
            device=device,
        )
        causal_mask_base = torch.triu(causal_mask_base, diagonal=1)

        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.shape != (
                batch_size,
                seq_len,
            ):
                raise ValueError(
                    f"attention_mask must be (batch_size, seq_len), got {attention_mask.shape}"
                )
            expanded_padding_mask = (
                (1 - attention_mask)
                .bool()
                .to(mask_dtype)
                .masked_fill_((1 - attention_mask).bool(), torch.finfo(mask_dtype).min)
                .unsqueeze(1)
                .unsqueeze(1)
            )
            attention_mask_4d = (
                causal_mask_base.unsqueeze(0).unsqueeze(0) + expanded_padding_mask
            )
            attention_mask_4d = attention_mask_4d.expand(
                batch_size, 1, seq_len, seq_len
            )
        else:
            attention_mask_4d = (
                causal_mask_base.unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, 1, seq_len, seq_len)
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask_4d,
            "position_ids": position_ids,
            **kwargs,
        }

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )
        # Only current_iter is relevant here; VPR router parameters are in config
        model_inputs["current_iter"] = kwargs.get("current_iter", 0)
        return model_inputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        labels=None, # Labels for CE loss, handled by trainer
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False, # We get hidden states from layers
        return_dict=True,
        **kwargs, # Accept extra kwargs, for current_iter and return_metrics
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided.")

        current_iter = kwargs.pop("current_iter", 0)
        return_metrics = kwargs.pop("return_metrics", self.training)

        prepared = self._prepare_inputs(input_ids, attention_mask, position_ids)
        hidden_states = self.model.embed_tokens(prepared["input_ids"])
        attention_mask = prepared["attention_mask"]
        position_ids = prepared["position_ids"]

        if self._log_gates:
            self._gate_means_tmp = []
            def _collect_gate_stats(_, __, outputs):
                # outputs[-1] is gate_vec_for_stats from DynamicQwenDecoderLayer
                gate_vec = outputs[-1]
                self._gate_means_tmp.append(gate_vec.mean().item())
                return outputs
            hooks = [l.register_forward_hook(_collect_gate_stats) for l in self.model.layers]
        else:
            hooks = []

        all_self_attns = [] if output_attentions else None
        all_past_key_values = [] if use_cache else None
        gate_vecs, ce_proportions, cu_proportions = [], [], []

        try:
            # Loop through our new Macro-Layers
            for layer_idx, macro_layer in enumerate(self.model.layers):
                # Each macro_layer is an nn.Module with decision_layer and dynamic_layer
                decision_layer_outputs = macro_layer.decision_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values[layer_idx] if past_key_values else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                (
                    original_input_to_decision, # Z^{n-1}
                    posterior_full_path_output, # H^{D_n}_{trans}
                    prior_hidden_states,        # H^{D_n}_{prior}
                    present_key_value,
                    attn_weights,
                ) = decision_layer_outputs

                dynamic_layer_outputs = macro_layer.dynamic_layer(
                    original_input_to_block=original_input_to_decision,
                    posterior_full_path_output=posterior_full_path_output,
                    prior_hidden_states=prior_hidden_states,
                    attention_mask=attention_mask, # Pass through for compatibility
                    position_ids=position_ids,     # Pass through for compatibility
                    past_key_values=present_key_value, # This is the KV cache from decision_layer
                    output_attentions=output_attentions, # Boolean flag
                    use_cache=use_cache,
                    current_iter=current_iter,
                    decision_attn_weights=attn_weights, # Pass actual attention weights from decision layer
                )

                # Unpack outputs from dynamic layer
                hidden_states = dynamic_layer_outputs[0] # Updated hidden states (Z^n)

                # Conditional unpacking of present_key_value and attention_weights
                # The order in dynamic_layer_outputs will be:
                # (hidden_states_final, [present_key_value if use_cache], [attn_weights if output_attentions], avg_ce, avg_cu, gate_vec)
                idx_offset = 1 # Start with 1 for optional present_key_value
                if use_cache:
                    if all_past_key_values is None:
                        all_past_key_values = (dynamic_layer_outputs[idx_offset],)
                    else:
                        all_past_key_values += (dynamic_layer_outputs[idx_offset],)
                    idx_offset += 1
                if output_attentions:
                    if all_self_attns is None:
                        all_self_attns = (dynamic_layer_outputs[idx_offset],)
                    else:
                        all_self_attns += (dynamic_layer_outputs[idx_offset],)
                    idx_offset += 1

                # Extract custom metrics from DynamicQwenDecoderLayer's end
                if return_metrics:
                    ce_proportions.append(dynamic_layer_outputs[idx_offset])
                    cu_proportions.append(dynamic_layer_outputs[idx_offset + 1])
                    gate_vecs.append(dynamic_layer_outputs[idx_offset + 2])


            hidden_states = self.model.norm(hidden_states)
            logits = self.lm_head(hidden_states)

            if return_metrics:
                overall_avg_ce = (
                    torch.stack(ce_proportions).mean()
                    if ce_proportions
                    else torch.tensor(0.0, device=logits.device)
                )
                overall_avg_cu = (
                    torch.stack(cu_proportions).mean()
                    if cu_proportions
                    else torch.tensor(0.0, device=logits.device)
                )

                # Return tuple for DynamicQwenTrainer's _calculate_loss
                return (
                    logits,
                    None, # No prior_loss anymore
                    gate_vecs,
                    overall_avg_ce,
                    overall_avg_cu,
                    ce_proportions, # Individual layer proportions for logging
                    cu_proportions, # Individual layer proportions for logging
                )
            else:
                return CausalLMOutputWithPast(
                    logits=logits,
                    past_key_values=all_past_key_values,
                    attentions=all_self_attns,
                )
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise
        finally:
            for h in hooks:
                h.remove()
            if self._log_gates:
                self._last_gate_means = self._gate_means_tmp


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load the base DynamicQwenConfig first and apply kwargs overrides
        config = DynamicQwenConfig.from_pretrained(pretrained_model_name_or_path)

        # Apply new dynamic/freezing params from kwargs to config
        config.capacity_gamma = kwargs.pop("capacity_gamma", getattr(config, "capacity_gamma", 1.0))
        config.beta_ce_init = kwargs.pop("beta_ce_init", getattr(config, "beta_ce_init", 1.0))
        config.beta_cu_init = kwargs.pop("beta_cu_init", getattr(config, "beta_cu_init", 1.0))
        config.cu_detection_multiplier_init = kwargs.pop("cu_detection_multiplier_init", getattr(config, "cu_detection_multiplier_init", 1.0))
        config.ce_criterion_offset_init = kwargs.pop("ce_criterion_offset_init", getattr(config, "ce_criterion_offset_init", 0.0))
        config.token_wise_gating = kwargs.pop("token_wise_gating", getattr(config, "token_wise_gating", True))
        config.moving_average_window_size = kwargs.pop("moving_average_window_size", getattr(config, "moving_average_window_size", 100))
        config.prior_ffn_intermediate_size_factor = kwargs.pop("prior_ffn_intermediate_size_factor", getattr(config, "prior_ffn_intermediate_size_factor", 2.0))
        config.freeze_main_transformer_blocks = kwargs.pop("freeze_main_transformer_blocks", getattr(config, "freeze_main_transformer_blocks", False))

        # Load the *base* Qwen2ForCausalLM model with our extended config.
        # This will load the standard Qwen2DecoderLayers.
        model = super().from_pretrained(
            pretrained_model_name_or_path, config=config, *model_args, **kwargs
        )

        # Now, patch the layers. This will iterate through the loaded
        # Qwen2DecoderLayers, extract their state_dicts, and replace them
        # with our DecisionQwenDecoderLayer and DynamicQwenDecoderLayer instances.
        patched_model = cls._patch_qwen_layers(model, config)

        # Explicitly apply freezing based on the config
        patched_model._apply_main_block_freezing()

        return patched_model

    @staticmethod
    def _patch_qwen_layers(model, config):
        """
        Replaces each Qwen2DecoderLayer with a DecisionQwenDecoderLayer
        followed by a DynamicQwenDecoderLayer.
        It transfers the state_dict from the original layers to the new decision layer.
        """
        logger.info("Patching Qwen model layers to hierarchical Decision/Dynamic layers.")
        new_macro_layers = nn.ModuleList()
        for i, original_layer in enumerate(model.model.layers):
            original_layer_state_dict = original_layer.state_dict()
            # Ensure state_dict is on CPU before passing, constructor handles device move
            original_layer_state_dict = {k: v.cpu() for k, v in original_layer_state_dict.items()}

            # Create Decision Layer (this layer gets the original Qwen2 weights)
            decision_layer = DecisionQwenDecoderLayer(
                config,
                layer_idx=i,
                load_from_pretrained=True,
                original_layer_state_dict=original_layer_state_dict,
            )

            # Create Dynamic Layer (this layer is "new" and will have VPRRouter)
            dynamic_layer = DynamicQwenDecoderLayer(config, layer_idx=i)

            # Define a container for the Macro-Layer to house both components
            # This allows treating `model.layers[i]` as a single conceptual unit
            # that encapsulates both Decision and Dynamic behavior.
            class MacroLayer(nn.Module):
                def __init__(self, decision_l, dynamic_l):
                    super().__init__()
                    self.decision_layer = decision_l
                    self.dynamic_layer = dynamic_l
                    self.layer_idx = decision_l.layer_idx # For logging/debugging

                def forward(self, *args, **kwargs):
                    # Forward pass for MacroLayer itself won't be called directly by self.model.layers
                    # This is just a container for `model.model.layers`
                    # The `DynamicQwenForCausalLM.forward` will call `macro_layer.decision_layer`
                    # and `macro_layer.dynamic_layer` explicitly.
                    raise NotImplementedError("MacroLayer's forward should not be called directly.")

            macro_layer = MacroLayer(decision_layer, dynamic_layer)

            # Move components to the correct device.
            # `model.parameters()` gives us a reference to the actual device.
            device = next(model.parameters()).device
            # Ensure layers are moved to device after initialization, especially if `to_empty` was used.
            macro_layer.decision_layer = macro_layer.decision_layer.to(device)
            macro_layer.dynamic_layer = macro_layer.dynamic_layer.to(device)


            new_macro_layers.append(macro_layer)
            logger.info(f"Successfully re-instantiated layer {i} as Macro-Layer (Decision + Dynamic).")

        model.model.layers = new_macro_layers
        return model