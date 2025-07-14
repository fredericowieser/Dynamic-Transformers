import torch
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
import logging

log = logging.getLogger(__name__)

def create_model(cfg):
    """
    Factory function to create a model based on the configuration.
    """
    model_name = cfg.model_params.model_name
    print(f"Loading base model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if cfg.lora_params.enabled:
        print("LoRA is enabled. Applying PEFT...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_params.r,
            lora_alpha=cfg.lora_params.lora_alpha,
            lora_dropout=cfg.lora_params.lora_dropout,
            target_modules=cfg.lora_params.target_modules,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        print("PEFT model created. Trainable parameters:")
        model.print_trainable_parameters()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal params: {total_params/1e6:.2f}M")
    print(f"Trainable params: {trainable_params/1e6:.2f}M")

    return model.to(cfg.run_params.device)