# src/models/d_qwen_causal_lm.py

import logging
import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM, Qwen2Model
from transformers.modeling_outputs import CausalLMOutputWithPast
from src.models.d_qwen_config import DynamicQwenConfig
# Import the new layer types
from src.models.dec_qwen_layers import DecisionQwenDecoderLayer
from src.models.dyn_qwen_layers import DynamicQwenDecoderLayer # This is now the "Dynamic Sub-Layer"

logger = logging.getLogger(__name__)


class DynamicQwenForCausalLM(Qwen2ForCausalLM): # Inherit from nn.Module, not Qwen2ForCausalLM directly
    config_class = DynamicQwenConfig

    def __init__(self, config: DynamicQwenConfig):
        super().__init__()
        self.config = config # Store our custom config

        # Explicitly initialize the components as Qwen2ForCausalLM would, but without its layers
        self.model = Qwen2Model(config) # This creates embed_tokens and model.norm
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Layers will be populated by from_pretrained
        self.model.layers = nn.ModuleList()

        # Set freezing flag, this will be applied AFTER layers are loaded/patched
        self._freeze_main_transformer_blocks = getattr(config, "freeze_main_transformer_blocks", False)

        # Gate logging utility setup
        self._log_gates = False
        self._last_gate_means = None

        # Link lm_head weights to embeddings for weight tying if applicable
        # (This is standard practice for LMs)
        self.tie_weights()

    def tie_weights(self):
        # Implement weight tying logic manually, as Qwen2ForCausalLM does this in its init
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
            logger.info("Tying weights between lm_head and embed_tokens.")

    @property
    def freeze_main_transformer_blocks(self):
        return self._freeze_main_transformer_blocks

    @freeze_main_transformer_blocks.setter
    def freeze_main_transformer_blocks(self, v: bool):
        self._freeze_main_transformer_blocks = v
        self._apply_main_block_freezing()
        logger.info(f"Set freeze_main_transformer_blocks = {v}. Parameters updated.")

    def _apply_main_block_freezing(self):
        """Applies or removes gradient requirement for main transformer block parameters (attn, mlp, layernorms)."""
        for layer_idx, layer in enumerate(self.model.layers):
            if layer_idx % 2 == 0: # Decision Layer
                for n, p in layer.named_parameters():
                    if "prior_ffn" not in n and "prior_layernorm" not in n:
                        p.requires_grad = not self._freeze_main_transformer_blocks
                        if not p.requires_grad:
                            p.data = p.data.contiguous()
            else: # Dynamic Layer
                for n, p in layer.named_parameters():
                    if "vpr_router" not in n:
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
        (Copied from previous iteration)
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

    # Override prepare_inputs_for_generation, now inheriting from PreTrainedModel via our manual inheritance path.
    # We must explicitly define it since we don't directly inherit from Qwen2ForCausalLM as a superclass for __init__.
    # This also means we need to call _prepare_inputs from here.
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        # If past_key_values is provided, only the last token is needed as input_ids
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # Create position_ids for new token, relative to past_key_values
        # If no past_key_values, position_ids start from 0
        position_ids = kwargs.get("position_ids", None)
        if position_ids is None and past_key_values is not None:
            # Assuming past_key_values is a tuple of tuples, e.g., ( (k,v), (k,v) ...)
            # seq_len_past is the length of the sequence already generated.
            seq_len_past = past_key_values[0][0].shape[2] # Shape is (B, N_H, S, H_D)
            position_ids = torch.tensor(
                [[seq_len_past]], dtype=torch.long, device=input_ids.device
            ).repeat(input_ids.shape[0], 1)

        # Prepare full inputs (handles attention_mask expansion too)
        model_inputs = self._prepare_inputs(input_ids, attention_mask, position_ids, **kwargs)

        # Add `past_key_values` back to the dictionary for the model forward pass
        model_inputs["past_key_values"] = past_key_values
        model_inputs["use_cache"] = kwargs.get("use_cache", self.config.use_cache)
        model_inputs["current_iter"] = kwargs.get("current_iter", 0) # For router in generation
        return model_inputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        labels=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided.")

        current_iter = kwargs.pop("current_iter", 0)
        return_metrics = kwargs.pop("return_metrics", self.training)

        # The embed_tokens and model.norm are part of `self.model`
        prepared = self._prepare_inputs(input_ids, attention_mask, position_ids)
        hidden_states = self.model.embed_tokens(prepared["input_ids"])
        attention_mask = prepared["attention_mask"]
        position_ids = prepared["position_ids"]

        if self._log_gates:
            self._gate_means_tmp = []
            def _collect_gate_stats(_, __, outputs):
                gate_vec = outputs[-1]
                self._gate_means_tmp.append(gate_vec.mean().item())
                return outputs
            # Attach hooks only to Dynamic layers (odd indices)
            hooks = [
                self.model.layers[i].register_forward_hook(_collect_gate_stats)
                for i in range(1, len(self.model.layers), 2)
            ]
        else:
            hooks = []

        all_self_attns = [] if output_attentions else None
        all_past_key_values = [] if use_cache else None
        gate_vecs, ce_proportions, cu_proportions = [], [], []

        # Variables to store VPR signals from the *last executed Decision Layer*
        vpr_signal_original_input = None
        vpr_signal_posterior_output = None
        vpr_signal_prior_hidden_states = None

        try:
            # We iterate through the layers. Each pair is a macro-layer (Decision -> Dynamic)
            # We have config.num_hidden_layers total, so total_macro_layers = config.num_hidden_layers / 2
            # Each original Qwen2 layer is replaced by EITHER a Decision OR a Dynamic layer.
            # So, total layers in self.model.layers is still config.num_hidden_layers.
            for layer_idx, layer in enumerate(self.model.layers):
                is_decision_layer = (layer_idx % 2 == 0)

                if is_decision_layer:
                    # Decision Layer: Processes hidden_states, produces next hidden_states AND VPR signals
                    decision_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values[layer_idx] if past_key_values else None,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                    (
                        hidden_states, # Output of Decision Layer, becomes input to next (Dynamic) layer
                        vpr_signal_original_input, # Signal 1 for next Dynamic Layer's router
                        vpr_signal_posterior_output, # Signal 2 for next Dynamic Layer's router
                        vpr_signal_prior_hidden_states, # Signal 3 for next Dynamic Layer's router
                        present_key_value, # KV from this Decision Layer
                        attn_weights, # Attentions from this Decision Layer
                    ) = decision_outputs

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

                else: # Dynamic Layer: Processes hidden_states (from prev Decision), uses VPR signals (from prev Decision)
                    if vpr_signal_original_input is None or vpr_signal_posterior_output is None or vpr_signal_prior_hidden_states is None:
                        raise ValueError(f"VPR signals not available for Dynamic Layer {layer_idx}. Preceding layer was not a Decision Layer or did not pass signals correctly.")

                    dynamic_outputs = layer(
                        hidden_states, # Input to this Dynamic layer (output from previous Decision)
                        prev_decision_original_input=vpr_signal_original_input,
                        prev_decision_posterior_output=vpr_signal_posterior_output,
                        prev_decision_prior_output=vpr_signal_prior_hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values[layer_idx] if past_key_values else None,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        current_iter=current_iter,
                    )

                    hidden_states = dynamic_outputs[0] # Output of Dynamic Layer, becomes input to next layer

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

                return (
                    logits,
                    None, # prior_loss is removed
                    gate_vecs,
                    overall_avg_ce,
                    overall_avg_cu,
                    ce_proportions,
                    cu_proportions,
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
        # 1. Handle `config` from kwargs: Pop it so it's not passed twice if kwargs includes it.
        config_from_kwargs = kwargs.pop("config", None)

        # 2. Prepare the config object for our custom model.
        if config_from_kwargs is None:
            config = DynamicQwenConfig.from_pretrained(pretrained_model_name_or_path)
        else:
            if not isinstance(config_from_kwargs, DynamicQwenConfig):
                logger.warning(
                    "Provided config is not DynamicQwenConfig. Attempting to load as DynamicQwenConfig."
                )
                config = DynamicQwenConfig.from_pretrained(
                    pretrained_model_name_or_path, **config_from_kwargs.to_dict()
                )
            else:
                config = config_from_kwargs

        # 3. Apply any *other* custom kwargs (e.g., from Hydra `model_cfg`) to this `config` object.
        # These are specific parameters for our custom layers/router.
        for key, default_val in [
            ("capacity_gamma", 1.0), ("beta_ce_init", 1.0), ("beta_cu_init", 1.0),
            ("cu_detection_multiplier_init", 1.0), ("ce_criterion_offset_init", 0.0),
            ("token_wise_gating", True), ("moving_average_window_size", 100),
            ("prior_ffn_intermediate_size_factor", 2.0), ("freeze_main_transformer_blocks", False)
        ]:
            setattr(config, key, kwargs.pop(key, getattr(config, key, default_val)))

        # --- START OF FIX: Load base HF model first, THEN initialize custom_model with its components ---
        # 4. Load the base Qwen2ForCausalLM model. This will instantiate it with its original layers and weights.
        # This is where the `gradient_checkpointing_enable` functionality comes from.
        base_hf_model = Qwen2ForCausalLM.from_pretrained(
            pretrained_model_name_or_path, config=config, *model_args, **kwargs
        )

        # 5. Create an instance of our custom DynamicQwenForCausalLM.
        # We pass the same config. Its __init__ will set up empty layers.
        custom_model = cls(config) # This calls our DynamicQwenForCausalLM.__init__

        # 6. Transfer core model components (embeddings, final norm, lm_head) from the base HF model
        # to our custom model. This ensures we have the correct weight values.
        custom_model.model.embed_tokens = base_hf_model.model.embed_tokens
        custom_model.model.norm = base_hf_model.model.norm
        custom_model.lm_head = base_hf_model.lm_head

        # 7. Populate custom_model.model.layers with our alternating Decision/Dynamic layers,
        # transferring weights from the original layers of base_hf_model.
        # Pass the original layers from base_hf_model to _patch_and_populate_layers
        cls._patch_and_populate_layers(custom_model, config, base_hf_model.model.layers)

        # 8. Apply freezing configuration.
        custom_model._apply_main_block_freezing()

        return custom_model
        # --- END OF FIX ---


    @staticmethod
    def _patch_and_populate_layers(model_to_patch, config, source_hf_layers):
        """
        Replaces layers in model_to_patch.model.layers with alternating Decision/Dynamic layers,
        transferring weights from source_hf_layers.
        """
        logger.info(f"Patching {len(source_hf_layers)} Qwen model layers into alternating Decision/Dynamic layers.")
        new_layers = nn.ModuleList()
        device = next(model_to_patch.parameters()).device

        for i, original_layer in enumerate(source_hf_layers):
            original_layer_state_dict = original_layer.state_dict()
            original_layer_state_dict = {k: v.cpu() for k, v in original_layer_state_dict.items()}

            if i % 2 == 0:  # Even index: Decision Layer
                new_layer_instance = DecisionQwenDecoderLayer(
                    config,
                    layer_idx=i,
                    load_from_pretrained=True,
                    original_layer_state_dict=original_layer_state_dict,
                ).to(device)
                logger.info(f"Instantiated layer {i} as DecisionQwenDecoderLayer.")
            else:  # Odd index: Dynamic Layer
                new_layer_instance = DynamicQwenDecoderLayer(
                    config,
                    layer_idx=i,
                    load_from_pretrained=True,
                    original_layer_state_dict=original_layer_state_dict,
                ).to(device)
                logger.info(f"Instantiated layer {i} as DynamicQwenDecoderLayer.")

            new_layers.append(new_layer_instance)

        model_to_patch.model.layers = new_layers
        return model_to_patch