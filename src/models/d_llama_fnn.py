import logging
import torch
import torch.nn.functional as F
from torch import nn

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