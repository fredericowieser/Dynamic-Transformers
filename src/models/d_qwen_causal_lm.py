# src/models/d_qwen_causal_lm.py

import logging
import torch
import torch.nn as nn
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

        self._freeze_main_transformer_blocks = getattr(config, "freeze_main_transformer_blocks", False)

        # Patch the layers: this now creates an alternating sequence of Decision and Dynamic layers.
        self._patch_qwen_layers(self, config)

        # Apply freezing based on the initial config setting
        self._apply_main_block_freezing()

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
        """Applies or removes gradient requirement for main transformer block parameters (attn, mlp, layernorms)."""
        for layer_idx, layer in enumerate(self.model.layers):
            if layer_idx % 2 == 0:  # Decision Layer
                # Freeze Decision Layer's core Qwen2 components
                for n, p in layer.named_parameters():
                    if "prior_ffn" not in n and "prior_layernorm" not in n: # Exclude Prior FFN params
                        p.requires_grad = not self._freeze_main_transformer_blocks
                        if not p.requires_grad:
                            p.data = p.data.contiguous() # Ensure contiguous for frozen params
            else:  # Dynamic Layer
                # Freeze Dynamic Layer's core Qwen2 components
                for n, p in layer.named_parameters():
                    if "vpr_router" not in n: # Exclude VPR Router params
                        p.requires_grad = not self._freeze_main_transformer_blocks
                        if not p.requires_grad:
                            p.data = p.data.contiguous()

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
        (Copied from previous iteration, largely from DynamicLlamaForCausalLM, adapted for Qwen2 specifics)
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
            # Hooks are now on DynamicQwenDecoderLayer instances
            def _collect_gate_stats(_, __, outputs):
                # outputs[-1] is gate_vec_for_stats from DynamicQwenDecoderLayer
                gate_vec = outputs[-1]
                self._gate_means_tmp.append(gate_vec.mean().item())
                return outputs
            # Only attach hooks to Dynamic layers (odd indices)
            hooks = [
                self.model.layers[i].register_forward_hook(_collect_gate_stats)
                for i in range(1, len(self.model.layers), 2)
            ]
        else:
            hooks = []

        all_self_attns = [] if output_attentions else None
        all_past_key_values = [] if use_cache else None
        gate_vecs, ce_proportions, cu_proportions = [], [], []

        # Variables to pass VPR signals from Decision to next Dynamic layer
        # Initialized to None; the first Decision Layer (idx 0) will populate them.
        vpr_signal_original_input = None
        vpr_signal_posterior_output = None
        vpr_signal_prior_hidden_states = None

        try:
            for layer_idx, layer in enumerate(self.model.layers):
                if layer_idx % 2 == 0:  # Decision Layer
                    # Call Decision layer. It takes current hidden_states as input.
                    # It returns its output hidden_states AND VPR signals for the NEXT layer.
                    decision_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values[layer_idx // 2] if past_key_values else None, # Index past_key_values by macro-layer index
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                    (
                        hidden_states, # This is the output of the Decision Layer, becomes input to next layer
                        vpr_signal_original_input, # Signal 1 for next Dynamic Layer's router
                        vpr_signal_posterior_output, # Signal 2 for next Dynamic Layer's router
                        vpr_signal_prior_hidden_states, # Signal 3 for next Dynamic Layer's router
                        present_key_value,
                        attn_weights,
                    ) = decision_outputs

                    # Collect KV cache and attentions from this Decision layer
                    if use_cache:
                        if all_past_key_values is None:
                            all_past_key_values = (present_key_value,)
                        else:
                            all_past_key_values += (present_key_value,)
                    if output_attentions:
                        if all_self_attns is None:
                            all_self_attns = (attn_weights,)
                        else:
                            all_self_attns += (attn_weights,)

                else:  # Dynamic Layer (layer_idx % 2 == 1)
                    # For a Dynamic layer, the VPR signals MUST be available from the preceding Decision layer.
                    if vpr_signal_original_input is None or vpr_signal_posterior_output is None or vpr_signal_prior_hidden_states is None:
                        raise ValueError(f"VPR signals not available for Dynamic Layer {layer_idx}. Ensure previous layer was a Decision Layer.")

                    # Call Dynamic layer. It takes current hidden_states as input.
                    # It also takes the VPR signals from the *preceding* Decision Layer.
                    dynamic_outputs = layer(
                        hidden_states, # This is the hidden_states output from the *previous* Decision Layer
                        prev_decision_original_input=vpr_signal_original_input,
                        prev_decision_posterior_output=vpr_signal_posterior_output,
                        prev_decision_prior_output=vpr_signal_prior_hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values[layer_idx // 2] if past_key_values else None, # Index past_key_values by macro-layer index
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        current_iter=current_iter,
                    )

                    # Dynamic layer returns: hidden_states_final, [present_kv], [attn_wts], avg_ce, avg_cu, gate_vec
                    hidden_states = dynamic_outputs[0] # This is the output of the Dynamic Layer, becomes input to next layer

                    # Unpack optional outputs and VPR metrics
                    output_idx_offset = 1
                    if use_cache:
                        present_key_value = dynamic_outputs[output_idx_offset]
                        if all_past_key_values is None:
                            all_past_key_values = (present_key_value,)
                        else:
                            all_past_key_values += (present_key_value,)
                        output_idx_offset += 1
                    if output_attentions:
                        attn_weights = dynamic_outputs[output_idx_offset]
                        if all_self_attns is None:
                            all_self_attns = (attn_weights,)
                        else:
                            all_self_attns += (attn_weights,)
                        output_idx_offset += 1

                    if return_metrics:
                        ce_proportions.append(dynamic_outputs[output_idx_offset])
                        cu_proportions.append(dynamic_outputs[output_idx_offset + 1])
                        gate_vecs.append(dynamic_outputs[output_idx_offset + 2])


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
                    gate_vecs, # Gate vecs from Dynamic layers only
                    overall_avg_ce,
                    overall_avg_cu,
                    ce_proportions, # Per-layer CE proportions from Dynamic layers
                    cu_proportions, # Per-layer CU proportions from Dynamic layers
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


        # Load the *base* Qwen2ForCausalLM model to get its original layers.
        # This model object will be passed to _patch_qwen_layers.
        base_model = super().from_pretrained(
            pretrained_model_name_or_path, config=config, *model_args, **kwargs
        )

        # Create an instance of our custom CausalLM class which will then patch the layers
        # The base_model's original layers will be replaced.
        custom_model = cls(config) # Instantiate custom class to trigger its __init__ and _patch_qwen_layers

        # Transfer relevant parts from the base_model to the custom_model
        # This effectively copies the embedded_tokens, lm_head, and norm layers
        # and also ensures the patched layers (Decision/Dynamic) get their weights.
        custom_model.model.embed_tokens = base_model.model.embed_tokens
        custom_model.model.norm = base_model.model.norm
        custom_model.lm_head = base_model.lm_head

        # Ensure the patched layers in `custom_model` have their weights transferred from `base_model`'s original layers.
        # This is handled within _patch_qwen_layers during custom_model's __init__,
        # which loads state_dicts from the original base_model.layers.
        # However, to explicitly handle this and be robust, let's pass the original layers to the patcher.
        # Re-calling patch_qwen_layers with base_model's original layers as source.
        cls._patch_qwen_layers_from_source(custom_model, config, base_model.model.layers)

        # Ensure dynamic parameters are set on the instance (as @property setters are used)
        # These are now attributes of the custom_model instance, set via its config
        custom_model._freeze_main_transformer_blocks = config.freeze_main_transformer_blocks
        custom_model._apply_main_block_freezing()


        return custom_model


    @staticmethod
    def _patch_qwen_layers(model_to_patch, config, source_layers):
        """
        Replaces layers in model_to_patch.model.layers with alternating Decision/Dynamic layers,
        transferring weights from source_layers.
        """
        logger.info("Patching Qwen model layers to alternating Decision/Dynamic layers.")
        new_layers = nn.ModuleList()
        # The number of new layers will be the same as original layers.
        # We assume an even number of layers for strict alternation,
        # or handle odd numbers by making the last one a Decision layer.
        # For simplicity, let's assume original num_hidden_layers is even or it's implicitly handled.
        # If it's 24 layers, we'll have 12 Decision, 12 Dynamic.
        for i, original_layer in enumerate(source_layers):
            original_layer_state_dict = original_layer.state_dict()
            original_layer_state_dict = {k: v.cpu() for k, v in original_layer_state_dict.items()}
            device = next(model_to_patch.parameters()).device # Get current device of the model

            if i % 2 == 0:  # Even index: Decision Layer
                new_layer_instance = DecisionQwenDecoderLayer(
                    config,
                    layer_idx=i,
                    load_from_pretrained=True,
                    original_layer_state_dict=original_layer_state_dict,
                )
                logger.info(f"Instantiated layer {i} as DecisionQwenDecoderLayer.")
            else:  # Odd index: Dynamic Layer
                new_layer_instance = DynamicQwenDecoderLayer(
                    config,
                    layer_idx=i,
                    load_from_pretrained=True,
                    original_layer_state_dict=original_layer_state_dict, # Dynamic layer also loads its own MLP/Attention weights
                )
                logger.info(f"Instantiated layer {i} as DynamicQwenDecoderLayer.")

            new_layers.append(new_layer_instance.to(device))

        model_to_patch.model.layers = new_layers
        return model_to_patch