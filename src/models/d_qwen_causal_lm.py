import logging
import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.modeling_outputs import CausalLMOutputWithPast
from src.models.d_qwen_config import DynamicQwenConfig
# Import the new layer types
from src.models.dec_qwen_layers import DecisionQwenDecoderLayer
from src.models.dyn_qwen_layers import DynamicQwenDecoderLayer # This is now the "Dynamic Sub-Layer"

logger = logging.getLogger(__name__)


class DynamicQwenForCausalLM(Qwen2ForCausalLM):
    config_class = DynamicQwenConfig

    def __init__(self, config: DynamicQwenConfig):
        super().__init__(config) # Pass the config object to the superclass __init__

        # self.config is already set by the super().__init__(config) call

        # Set freezing flag, this will be applied AFTER layers are loaded/patched
        self._freeze_main_transformer_blocks = getattr(config, "freeze_main_transformer_blocks", False)

        # Gate logging utility setup
        self._log_gates = False
        self._last_gate_means = None

        # self.tie_weights() is called by Qwen2ForCausalLM.__init__

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
                # Decision layers contain Qwen2Attention, Qwen2MLP, and RMSNorms, plus PriorFFN.
                # Only freeze the *main* Qwen2 components.
                for n, p in layer.named_parameters():
                    # Check if parameter name does NOT contain prior_ffn or prior_layernorm
                    if "prior_ffn" not in n and "prior_layernorm" not in n:
                        p.requires_grad = not self._freeze_main_transformer_blocks
                        if not p.requires_grad:
                            p.data = p.data.contiguous() # Ensure data is contiguous if not requiring grad
            else: # Dynamic Layer
                # Dynamic layers contain Qwen2Attention, Qwen2MLP, RMSNorms, plus VPRRouter.
                # Only freeze the *main* Qwen2 components.
                for n, p in layer.named_parameters():
                    # Check if parameter name does NOT contain vpr_router
                    # FIX: Now that ce_criterion_offset is also a parameter of vpr_router,
                    # ensure vpr_router params are NOT frozen by this logic.
                    if "vpr_router" not in n: # Correct, this check already excludes the router
                        p.requires_grad = not self._freeze_main_transformer_blocks
                        if not p.requires_grad:
                            p.data = p.data.contiguous() # Ensure data is contiguous if not requiring grad


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

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        # If past_key_values is provided, only the last token is needed as input_ids
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # Create position_ids for new token, relative to past_key_values
        position_ids = kwargs.get("position_ids", None)
        if position_ids is None and past_key_values is not None:
            seq_len_past = past_key_values[0][0].shape[2]
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
        labels=None, # Labels for CE loss, handled by trainer
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False, # We get hidden states from layers, but don't explicitly pass this down for Qwen2Model
        return_dict=True,
        **kwargs, # Accept extra kwargs, for current_iter and return_metrics
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided.")

        current_iter = kwargs.pop("current_iter", 0)
        return_metrics = kwargs.pop(
            "return_metrics", self.training
        )  # Default to training mode

        # The embed_tokens and model.norm are part of `self.model`
        prepared = self._prepare_inputs(input_ids, attention_mask, position_ids)
        hidden_states = self.model.embed_tokens(prepared["input_ids"])
        attention_mask = prepared["attention_mask"]
        position_ids = prepared["position_ids"]

        if self._log_gates:
            self._gate_means_tmp = []
            def _collect_gate_stats(_, __, outputs):
                # Gate stats (gate_vec_binary) are the *last* element in DynamicQwenDecoderLayer output
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

        # Initialize lists for metrics collected from Dynamic layers
        gate_vecs, ce_proportions, cu_proportions = [], [], []
        # Initialize lists for learnable parameters from VPRRouter
        router_beta_ces, router_beta_cus, router_cu_detection_multipliers, router_ce_criterion_offsets = [], [], [], []
        # Initialize list for prior_loss from Decision layers
        prior_losses = []
        all_combined_gating_signals = []

        # Initialize for KV cache and attentions. Use None to signify not collected/returned.
        all_past_key_values = [None] * len(self.model.layers) if use_cache else None
        all_self_attns = [None] * len(self.model.layers) if output_attentions else None

        # Variables to store VPR signals from the *last executed Decision Layer*
        vpr_signal_original_input = None
        vpr_signal_posterior_output = None
        vpr_signal_prior_hidden_states = None

        try:
            # We iterate through the layers. Each pair is a macro-layer (Decision -> Dynamic)
            # The total number of layers in self.model.layers is still config.num_hidden_layers.
            for layer_idx, layer in enumerate(self.model.layers):
                is_decision_layer = (layer_idx % 2 == 0)

                if is_decision_layer:
                    # Decision Layer: Processes hidden_states, produces next hidden_states AND VPR signals
                    decision_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        # Pass the specific past_key_values tuple for THIS layer
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
                        current_prior_loss, # Prior loss from this Decision Layer (for monitoring)
                    ) = decision_outputs

                    if use_cache:
                        all_past_key_values[layer_idx] = present_key_value
                    if output_attentions:
                        all_self_attns[layer_idx] = attn_weights
                    if return_metrics: # Prior loss is for monitoring, so always collect if return_metrics is True
                        prior_losses.append(current_prior_loss)

                else: # Dynamic Layer: Processes hidden_states (from prev Decision), uses VPR signals (from prev Decision)
                    if (vpr_signal_original_input is None or vpr_signal_posterior_output is None
                        or vpr_signal_prior_hidden_states is None or current_prior_loss is None): # Need prior_loss also
                        # This should ideally not happen if layers are correctly alternated
                        raise ValueError(f"VPR signals or prior_loss not available for Dynamic Layer {layer_idx}. Preceding layer was not a Decision Layer or did not pass signals correctly.")

                    dynamic_outputs = layer(
                        hidden_states, # Input to this Dynamic layer (output from previous Decision)
                        prev_decision_original_input=vpr_signal_original_input,
                        prev_decision_posterior_output=vpr_signal_posterior_output,
                        prev_decision_prior_output=vpr_signal_prior_hidden_states,
                        prior_loss_from_decision=current_prior_loss, # Pass prior_loss to dynamic layer
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values[layer_idx] if past_key_values else None,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        current_iter=current_iter,
                    )

                    hidden_states = dynamic_outputs[0] # Output of Dynamic Layer, becomes input to next layer

                    output_idx_offset = 1 # Start checking from here for optional outputs
                    current_present_key_value = None
                    current_attn_weights = None

                    # Extract present_key_value
                    if use_cache:
                        current_present_key_value = dynamic_outputs[output_idx_offset]
                        all_past_key_values[layer_idx] = current_present_key_value
                        output_idx_offset += 1
                    # Extract attn_weights
                    if output_attentions:
                        current_attn_weights = dynamic_outputs[output_idx_offset]
                        all_self_attns[layer_idx] = current_attn_weights
                        output_idx_offset += 1

                    if return_metrics:
                        # These are the *last* elements in the DynamicQwenDecoderLayer's output tuple
                        # (output_hidden_states, ..., avg_ce_prop, avg_cu_prop, gate_vec_final, prior_loss_from_decision, router_beta_ce, router_beta_cu, router_cu_detection_multiplier, router_ce_criterion_offset)
                        ce_proportions.append(dynamic_outputs[output_idx_offset])
                        cu_proportions.append(dynamic_outputs[output_idx_offset + 1])
                        # NEW: Collect combined_gating_signal_continuous
                        all_combined_gating_signals.append(dynamic_outputs[output_idx_offset + 2])
                        gate_vecs.append(dynamic_outputs[output_idx_offset + 3]) # Shifted by 1
                        # Prior loss from Decision layer is *already* in prior_losses list
                        router_beta_ces.append(dynamic_outputs[output_idx_offset + 4]) # 3 for prior_loss, +1 for prev 
                        router_beta_cus.append(dynamic_outputs[output_idx_offset + 5])
                        router_cu_detection_multipliers.append(dynamic_outputs[output_idx_offset + 6])
                        router_ce_criterion_offsets.append(dynamic_outputs[output_idx_offset + 7])


            hidden_states = self.model.norm(hidden_states)
            logits = self.lm_head(hidden_states)

            # Convert all_past_key_values and all_self_attns to tuples of tuples for HF output
            final_past_key_values = tuple(all_past_key_values) if use_cache else None
            final_self_attns = tuple(all_self_attns) if output_attentions else None


            if return_metrics:
                # Aggregate collected metrics
                overall_avg_ce_return = (
                    torch.stack(ce_proportions).mean()
                    if ce_proportions
                    else torch.tensor(0.0, device=logits.device)
                )
                overall_avg_cu_return = (
                    torch.stack(cu_proportions).mean()
                    if cu_proportions
                    else torch.tensor(0.0, device=logits.device)
                )
                overall_prior_loss = (
                    torch.stack(prior_losses).mean()
                    if prior_losses
                    else torch.tensor(0.0, device=logits.device)
                )
                # NEW: Overall mean of combined_gating_signal_continuous
                overall_combined_gating_signal_mean = (
                    torch.stack([s.mean() for s in all_combined_gating_signals]).mean()
                    if all_combined_gating_signals
                    else torch.tensor(0.0, device=logits.device)
                )
                
                # Convert list of floats to tensors for mean calculation and logging
                overall_beta_ce = (
                    torch.tensor(router_beta_ces).mean()
                    if router_beta_ces
                    else torch.tensor(0.0, device=logits.device)
                )
                overall_beta_cu = (
                    torch.tensor(router_beta_cus).mean()
                    if router_beta_cus
                    else torch.tensor(0.0, device=logits.device)
                )
                overall_cu_detection_multiplier = (
                    torch.tensor(router_cu_detection_multipliers).mean()
                    if router_cu_detection_multipliers
                    else torch.tensor(0.0, device=logits.device)
                )
                overall_ce_criterion_offset = (
                    torch.tensor(router_ce_criterion_offsets).mean()
                    if router_ce_criterion_offsets
                    else torch.tensor(0.0, device=logits.device)
                )


                # Return tuple as expected by DynamicQwenTrainer's _calculate_loss
                return (
                    logits,
                    overall_prior_loss, # Renamed from None, now contains avg prior loss
                    gate_vecs, # List of gate_vec_for_stats (B,T) or (B,) per Dynamic layer
                    overall_avg_ce_return, # Scalar mean across Dynamic layers
                    overall_avg_cu_return, # Scalar mean across Dynamic layers
                    overall_combined_gating_signal_mean,
                    ce_proportions, # List of scalar means per Dynamic layer
                    cu_proportions, # List of scalar means per Dynamic layer
                    overall_beta_ce,
                    overall_beta_cu,
                    overall_cu_detection_multiplier,
                    overall_ce_criterion_offset,
                )
            else:
                return CausalLMOutputWithPast(
                    logits=logits,
                    past_key_values=final_past_key_values,
                    attentions=final_self_attns,
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
            # Load default Qwen2Config, then upgrade to DynamicQwenConfig.
            base_config = Qwen2Config.from_pretrained(pretrained_model_name_or_path)
            config = DynamicQwenConfig(**base_config.to_dict())
        else:
            if not isinstance(config_from_kwargs, DynamicQwenConfig):
                logger.warning(
                    "Provided config is not DynamicQwenConfig. Attempting to load as DynamicQwenConfig."
                )
                # Convert provided config to dict, then initialize DynamicQwenConfig
                config = DynamicQwenConfig(**config_from_kwargs.to_dict())
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
            # Use kwargs.get with default from current config value if it exists, else use hardcoded default.
            setattr(config, key, kwargs.pop(key, getattr(config, key, default_val)))

        # 4. Load the base Qwen2ForCausalLM model. This will instantiate it with its original layers and weights.
        base_hf_model = Qwen2ForCausalLM.from_pretrained(
            pretrained_model_name_or_path, config=config, *model_args, **kwargs
        )

        # 5. Create an instance of our custom DynamicQwenForCausalLM.
        # We pass the same config. Its __init__ will use this config for base model setup.
        custom_model = cls(config) # This calls our DynamicQwenForCausalLM.__init__

        # 6. Transfer core model components (embeddings, final norm, lm_head) from the base HF model
        # to our custom model. This ensures we have the correct weight values.
        custom_model.model.embed_tokens = base_hf_model.model.embed_tokens
        custom_model.model.norm = base_hf_model.model.norm
        custom_model.lm_head = base_hf_model.lm_head

        # 7. Populate custom_model.model.layers with our alternating Decision/Dynamic layers,
        # transferring weights from the original layers of base_hf_model.
        cls._patch_and_populate_layers(custom_model, config, base_hf_model.model.layers)

        # 8. Apply freezing configuration.
        custom_model._apply_main_block_freezing()

        # Clean up the base_hf_model to free memory if it's no longer needed
        del base_hf_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return custom_model


    @staticmethod
    def _patch_and_populate_layers(model_to_patch, config, source_hf_layers):
        """
        Replaces layers in model_to_patch.model.layers with alternating Decision/Dynamic layers,
        transferring weights from source_hf_layers.
        """
        logger.info(f"Patching {len(source_hf_layers)} Qwen model layers into alternating Decision/Dynamic layers.")
        new_layers = nn.ModuleList()
        # Ensure we get the device from existing model parts or from the config/kwargs
        device = next(model_to_patch.parameters()).device if next(model_to_patch.parameters(), None) is not None else torch.device("cpu") # Fallback to CPU

        for i, original_layer in enumerate(source_hf_layers):
            original_layer_state_dict = original_layer.state_dict()
            # Ensure state_dict is on CPU before passing, constructor handles device move
            original_layer_state_dict = {k: v.cpu() for k, v in original_layer_state_dict.items()}

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
                    original_layer_state_dict=original_layer_state_dict,
                )
                logger.info(f"Instantiated layer {i} as DynamicQwenDecoderLayer.")
            
            # Move the newly created layer to the target device
            new_layers.append(new_layer_instance.to(device))

        model_to_patch.model.layers = new_layers
        # Ensure the model is on the correct device after all layers are patched
        model_to_patch.to(device)
        return model_to_patch