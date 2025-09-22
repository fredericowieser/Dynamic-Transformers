import torch
from omegaconf import OmegaConf
from src.training.utils import create_model
import hydra

# Initialize Hydra and compose the config
# The path is relative to the current working directory where the script is run
with hydra.initialize(config_path="config", version_base=None):
    cfg = hydra.compose(config_name="laptop_10m_wikitext.yaml")

cfg.training.mode = "scratch"
cfg.model.size = "10M"

# Allow dynamic addition of keys to cfg.model
OmegaConf.set_struct(cfg.model, False)

model_types = ["stt", "sdt", "mod", "standard"]

for model_type in model_types:
    print(f"\n--- Testing model type: {model_type.upper()} ---")
    cfg.model.type = model_type

    # Set model-specific parameters if needed
    if model_type == "stt":
        cfg.model.prior_ffn_intermediate_size_factor = 0.25
        cfg.model.stt_capacity = 0.5
        cfg.model.tpn_loss_weight = 0.05 # From prior.yaml
        cfg.model.causal_loss_weight = 0.05 # From causal_router.yaml
    elif model_type == "sdt":
        cfg.model.prior_ffn_intermediate_size_factor = 0.25
        cfg.model.sdt_capacity = 0.5
        cfg.model.prior_loss_weight = 0.05 # From prior.yaml
        cfg.model.causal_loss_weight = 0.05 # From causal_router.yaml
    elif model_type == "mod":
        cfg.model.mod_capacity = 0.5
        cfg.model.mod_aux_loss_weight = 0.01
        cfg.model.causal_loss_weight = 0.05 # From causal_router.yaml
    # For "standard" model, no extra parameters are needed for this test

    m = create_model(cfg.model.type, cfg.model.size, True, cfg)
    m.eval()
    B,T = 2, 64
    x = torch.randint(0, cfg.model.scratch_config.vocab_size, (B,T))
    with torch.no_grad():
        out = m(input_ids=x, labels=x)
    print(f"OK ({model_type.upper()}): logits shape: {out['logits'].shape}, loss: {out['loss'].item()}")