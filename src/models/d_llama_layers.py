import logging
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaMLP
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

log = logging.getLogger(__name__)

class FeedForward(nn.Module):
    def __init__(self, config, skip_init=False, state_dict=None, device=None,
                 init_from_other_mlp_weights=False, other_mlp_state_dict=None,
                 enable_lora=False, lora_params=None):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False).to(device)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False).to(device)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False).to(device)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, "hidden_dropout_prob") else 0.0)
        
        if init_from_other_mlp_weights and other_mlp_state_dict is not None:
            log.info("Initializing prior_ffn from LlamaMLP weights for FeedForward on device %s", device)
            self.w1.weight.data.copy_(other_mlp_state_dict["gate_proj.weight"].to(device))
            self.w3.weight.data.copy_(other_mlp_state_dict["up_proj.weight"].to(device))
            self.w2.weight.data.copy_(other_mlp_state_dict["down_proj.weight"].to(device))
            if "gate_proj.bias" in other_mlp_state_dict and self.w1.bias is not None:
                self.w1.bias.data.copy_(other_mlp_state_dict["gate_proj.bias"].to(device))
            if "up_proj.bias" in other_mlp_state_dict and self.w3.bias is not None:
                self.w3.bias.data.copy_(other_mlp_state_dict["up_proj.bias"].to(device))
            if "down_proj.bias" in other_mlp_state_dict and self.w2.bias is not None:
                self.w2.bias.data.copy_(other_mlp_state_dict["down_proj.bias"].to(device))
            self.to(device)
            skip_init = True

        if not skip_init and state_dict is None:
            self._initialize_weights()
            log.info("Initialized weights for FeedForward on device %s", device)
        elif state_dict is not None:
            self.load_state_dict(state_dict, strict=True)
            if device:
                self.to(device)  # Move to device after loading
            log.info("Loaded pre-trained weights for FeedForward on device %s", device)
        
        if enable_lora and lora_params:
            lora_config = LoraConfig(
                r=lora_params["lora_r"],
                lora_alpha=lora_params["lora_alpha"],
                target_modules=lora_params["lora_target_modules_prior_ffn"],
                lora_dropout=lora_params["lora_dropout"],
                bias=lora_params["lora_bias"],
                task_type=TaskType.CAUSAL_LM,
            )
            self = get_peft_model(self, lora_config)
            log.info("LoRA applied to FeedForward for prior_ffn.")

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device  # Ensure input is on the same device
        if x.device != device:
            x = x.to(device)
        x = self.w2(self.act_fn(self.w1(x)) * self.w3(x))
        return self.dropout(x)

class DynamicLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx: int, load_from_pretrained=False, state_dict=None, device=None,
                 init_prior_from_mlp: bool = False, original_mlp_state_dict_for_prior_init: dict = None): # New params
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        self.token_wise_gating = getattr(config, "token_wise", True)
        
        # Apply LoRA to self_attn and mlp (main decoder path) if enabled
        enable_lora_main = getattr(config, "enable_lora_main_path", False)
        if enable_lora_main:
            lora_config_main = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules_main,
                lora_dropout=config.lora_dropout,
                bias=config.lora_bias,
                task_type=TaskType.CAUSAL_LM,
            )
            self.self_attn = get_peft_model(self.self_attn, lora_config_main)
            self.mlp = get_peft_model(self.mlp, lora_config_main)
            log.info(f"LoRA applied to main path of Layer {self.layer_idx}.")

        # Determine if prior_ffn itself should be LoRA-enabled
        enable_lora_prior_ffn = getattr(config, "enable_lora_prior_ffn", False)
        lora_params_prior_ffn = {
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "lora_bias": config.lora_bias,
            "lora_target_modules_prior_ffn": config.lora_target_modules_prior_ffn,
        }

        # Conditional initialization of prior_ffn
        if init_prior_from_mlp:
            self.prior_ffn = FeedForward(config, skip_init=True, device=device,
                                         init_from_other_mlp_weights=True,
                                         other_mlp_state_dict=original_mlp_state_dict_for_prior_init,
                                         enable_lora=enable_lora_prior_ffn,
                                         lora_params=lora_params_prior_ffn)
            self.prior_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps).to(device)
        else:
            self.prior_ffn = FeedForward(config, skip_init=load_from_pretrained, state_dict=None, device=device,
                                         enable_lora=enable_lora_prior_ffn,
                                         lora_params=lora_params_prior_ffn)
            self.prior_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps).to(device)

        if device:
            self.to(device)
            log.info(f"Layer {self.layer_idx} created (token_wise={self.token_wise_gating}) on device {device}")

    def load_prior_components(self, state_dict, device=None):
        if self.prior_ffn is None:
            # Need to re-instantiate, considering LoRA and init_from_mlp if applicable
            # This method needs more context if it's meant to re-initialize from scratch
            # or just load into an existing structure with potential LoRA adapters.
            # For now, assuming it's for loading into an already set up structure.
            log.warning("load_prior_components might behave unexpectedly with LoRA. Ensure prior_ffn is correctly initialized before calling this.")
            # Placeholder: a more robust implementation might involve passing config and checking
            # current LoRA state or re-instantiating FeedForward with full parameters.
            self.prior_ffn = FeedForward(self.config, skip_init=True, device=device)
        if self.prior_layernorm is None:
            self.prior_layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.rms_norm_eps).to(device)
        self.load_state_dict(state_dict, strict=True)
        if device:
            self.to(device)

    def _prepare_attention_inputs(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
    ):
        """
        Prepare inputs for the attention mechanism, handling position embeddings if needed.

        Returns:
            dict: Arguments ready to be passed to self.self_attn
        """
        attn_args = {
            "hidden_states": hidden_states,
            "output_attentions": output_attentions,
            "use_cache": use_cache,
        }

        if attention_mask is not None:
            attn_args["attention_mask"] = attention_mask
        if past_key_value is not None:
            attn_args["past_key_value"] = past_key_value

        # Handle position embeddings if position_ids are provided
        if position_ids is not None:
            batch_size, seq_len, hidden_size = hidden_states.shape
            num_heads = self.self_attn.num_heads
            head_dim = self.self_attn.head_dim

            # Create dummy tensor for rotary embedding computation
            dummy_value_states = hidden_states.reshape(
                batch_size, seq_len, num_heads, head_dim
            ).transpose(
                1, 2
            )  # -> (B, num_heads, T, head_dim)

            # Compute cos and sin using the attention layer's rotary embedding
            cos, sin = self.self_attn.rotary_emb(dummy_value_states, position_ids)
            attn_args["position_embeddings"] = (cos, sin)

        return attn_args

    def _compute_gate_signals(self, posterior_output, prior_prediction, original_input):
        """
        Compute the fundamental gate signals (d_st and d_ch) that drive the gating decision.

        Args:
            posterior_output: Output from the full transformer block (B, T, C)
            prior_prediction: Prediction from the prior FFN (B, T, C)
            original_input: Original input to the block (B, T, C)

        Returns:
            tuple: (d_st_tok, d_ch_tok) both with shape (B, T)
        """
        d_st_tok = F.mse_loss(posterior_output, original_input, reduction="none").mean(
            -1
        )  # (B, T)

        d_ch_tok = F.mse_loss(
            posterior_output, prior_prediction, reduction="none"
        ).mean(
            -1
        )  # (B, T)

        return d_st_tok, d_ch_tok

    def _apply_gating_strategy(self, d_st_tok, d_ch_tok, gate_config):
        """
        Apply the gating strategy (token-wise vs block-wise) and compute final gates.

        Args:
            d_st_tok: Per-token status quo loss (B, T)
            d_ch_tok: Per-token change loss (B, T)
            gate_config: Dictionary containing gating parameters

        Returns:
            tuple: (gate_vec, gate_metrics) where gate_vec has appropriate shape for broadcasting
        """
        # Apply gating strategy
        if self.token_wise_gating:
            D_st, D_ch = d_st_tok, d_ch_tok  # Shape: (B, T)
        else:
            D_st = d_st_tok.mean(dim=1)  # Shape: (B,)
            D_ch = d_ch_tok.mean(dim=1)  # Shape: (B,)

        # Apply warmup bias if configured
        if gate_config["gate_warmup_iters"] > 0:
            bias_scale = max(
                0.0,
                1.0 - gate_config["current_iter"] / gate_config["gate_warmup_iters"],
            )
            beta = D_ch.detach().mean() * bias_scale
            D_ch = D_ch - beta

        # Compute gate conditions
        CE = D_st > D_ch - gate_config["ce_bias"]  # Complexity Enhancement
        CU = (
            D_st > gate_config["dynamic_k"] * D_st.detach().mean()
        )  # Complexity Understanding

        # Combine conditions
        gate_vec = (CE | CU).float()

        # Compute metrics
        gate_metrics = {
            "avg_ce_proportion": CE.float().mean(),
            "avg_cu_proportion": CU.float().mean(),
            # Ensure consistent output shape for gate statistics
            "gate_vec_for_stats": (
                gate_vec.mean(dim=1) if self.token_wise_gating else gate_vec
            ),
        }

        return gate_vec, gate_metrics

    def _apply_gate_to_outputs(self, gate_vec, posterior_output, original_input):
        """
        Apply the computed gate to mix posterior and original outputs.

        Args:
            gate_vec: Gate values, shape depends on gating strategy
            posterior_output: Full transformer block output (B, T, C)
            original_input: Original input to block (B, T, C)

        Returns:
            torch.Tensor: Mixed output (B, T, C)
        """
        if self.token_wise_gating:
            # gate_vec: (B, T) -> (B, T, 1)
            gate = gate_vec.unsqueeze(-1)
        else:
            # gate_vec: (B,) -> (B, 1, 1)
            gate = gate_vec.view(-1, 1, 1)

        return gate * posterior_output + (1.0 - gate) * original_input

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        current_iter: int = 0,
        gate_warmup_iters: int = 1000,
        dynamic_k: float = 0.5,
        ce_bias: float = 0.0,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """
        Forward pass of the Dynamic Llama Decoder Layer.

        This method implements the core dynamic gating mechanism where each layer
        can choose between using its full computation (posterior) or bypassing it
        based on complexity measures.
        """

        # Override parameters from config if available (used at inference)
        gate_config = {
            "dynamic_k": getattr(self.config, "dynamic_k", dynamic_k),
            "gate_warmup_iters": getattr(
                self.config, "gate_warmup_iters", gate_warmup_iters
            ),
            "ce_bias": getattr(self.config, "ce_bias", ce_bias),
            "current_iter": current_iter,
        }

        # Store original input for gating decision
        original_input_to_block = hidden_states  # (B, T, C)

        # ===== Standard Llama Decoder Path =====

        # Self-attention
        residual_attn = hidden_states
        hidden_states_pre_attn_ln = self.input_layernorm(hidden_states)

        # Prepare and execute attention
        attn_args = self._prepare_attention_inputs(
            hidden_states_pre_attn_ln,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
        )
        attn_outputs = self.self_attn(**attn_args)

        attention_output = attn_outputs[0]  # (B, T, C)
        hidden_states_after_attn = residual_attn + attention_output

        # MLP (Posterior path)
        residual_mlp = hidden_states_after_attn
        hidden_states_pre_mlp_ln = self.post_attention_layernorm(
            hidden_states_after_attn
        )
        posterior_mlp_output = self.mlp(hidden_states_pre_mlp_ln)  # (B, T, C)
        posterior_full_path_output = residual_mlp + posterior_mlp_output  # (B, T, C)

        # ===== Dynamic Prior Prediction =====

        # Use previous attention output (shifted) as input to prior FFN
        prev_attention_output = F.pad(attention_output[:, :-1, :], (0, 0, 1, 0))
        prior_input = self.prior_layernorm(prev_attention_output)
        prior_prediction = self.prior_ffn(prior_input)  # (B, T, C)

        # ===== Dynamic Gating Logic =====

        # Compute fundamental gate signals
        d_st_tok, d_ch_tok = self._compute_gate_signals(
            posterior_full_path_output, prior_prediction, original_input_to_block
        )

        # Apply gating strategy and compute gates
        gate_vec, gate_metrics = self._apply_gating_strategy(
            d_st_tok, d_ch_tok, gate_config
        )

        # Apply gates to produce final output
        hidden_states_final = self._apply_gate_to_outputs(
            gate_vec, posterior_full_path_output, original_input_to_block
        )

        # ===== Compute Prior Loss =====

        prior_loss = F.mse_loss(prior_prediction, posterior_full_path_output.detach())

        # ===== Prepare Outputs =====

        outputs = (hidden_states_final,)

        # Add attention outputs if requested
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[2],)

        # Add dynamic gating metrics
        outputs += (
            gate_metrics["avg_ce_proportion"],
            gate_metrics["avg_cu_proportion"],
            prior_loss,
            gate_metrics["gate_vec_for_stats"],
        )

        return outputs
