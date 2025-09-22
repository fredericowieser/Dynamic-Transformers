import torch
from omegaconf import OmegaConf
from src.training.utils import create_model

# Simplified and unified config for smoke test
cfg = OmegaConf.create({
  "system": {
      "torch_dtype": "float32", 
      "use_flash_attention": False
  },
  "model": {
    "type": "stt",
    "size": "10M",
    "from_scratch": True,
    "scratch_config": {
      "vocab_size": 32000,
      "max_position_embeddings": 1024,
      "rope_theta": 1000000.0,
      "sliding_window": 1024,
      "10M": {"hidden_size": 32, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2},
    },
    "capacity": 0.5,
    "o_ce_init": 1.025, 
    "m_cu_init": 1.1, 
    "ma_window": 100,
    "attn_implementation": "eager", 
    "use_cache": False, 
    "tie_word_embeddings": True,
    "stt": {
        "tpn_loss_weight": 0.05, 
        "causal_loss_weight": 0.01,
    },
    "sdt": {
        "prior_loss_weight": 0.05,
    },
    "mod": {
        "aux_loss_weight": 0.01,
    }
  },
})

# --- Test STT ---
cfg.model.type = "stt"
print(f"Creating model {cfg.model.type}")
m = create_model(cfg.model.type, cfg)
m.eval()
B, T = 2, 64
x = torch.randint(0, cfg.model.scratch_config.vocab_size, (B, T))
with torch.no_grad():
    out = m(input_ids=x, labels=x)
print("OK STT:", out["logits"].shape, "loss:", float(out["loss"]))

# --- Test other models ---
for typ in ["sdt", "mod"]:
    cfg.model.type = typ
    print(f"Creating model {cfg.model.type}")
    m = create_model(cfg.model.type, cfg)
    m.eval()
    with torch.no_grad():
        out = m(input_ids=x, labels=x)
    print("OK", typ.upper(), ":", out["logits"].shape, "loss:", float(out["loss"]))