import sys
import os
import pytest
import torch
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def dtype():
    return torch.float32

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def seq_length():
    return 128

@pytest.fixture
def hidden_size():
    return 768

@pytest.fixture
def num_attention_heads():
    return 12

@pytest.fixture
def num_hidden_layers():
    return 4

@pytest.fixture
def vocab_size():
    return 32000

@pytest.fixture
def intermediate_size():
    return 3072

@pytest.fixture
def base_config():
    from src.models.qwen.config import DynamicQwenConfig
    return DynamicQwenConfig(
        hidden_size=768,
        num_hidden_layers=4,
        num_attention_heads=12,
        num_key_value_heads=12,  # Add key-value heads
        intermediate_size=3072,
        vocab_size=32000,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        dynamic_architecture="vpr",
        beta_ce_init=0.5,
        beta_cu_init=0.5,
        cu_detection_multiplier_init=1.0,
        ce_criterion_offset_init=0.0,
        token_wise_gating=True,
        rope_theta=10000.0,  # Add RoPE theta
        _attn_implementation="eager",  # Set attention implementation
    )

@pytest.fixture
def mod_config():
    from src.models.qwen.config import DynamicQwenConfig
    return DynamicQwenConfig(
        hidden_size=768,
        num_hidden_layers=4,
        num_attention_heads=12,
        num_key_value_heads=12,  # Add key-value heads
        intermediate_size=3072,
        vocab_size=32000,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        dynamic_architecture="mod",
        capacity_gamma=0.5,
        mod_capacity_factor=1.25,
        mod_topk=2,
        mod_num_experts=8,
        mod_aux_loss_weight=0.01,
        rope_theta=10000.0,  # Add RoPE theta
        _attn_implementation="eager",  # Set attention implementation
    )

@pytest.fixture
def sample_input_ids(batch_size, seq_length, vocab_size, device):
    return torch.randint(0, vocab_size, (batch_size, seq_length), device=device)

@pytest.fixture
def sample_hidden_states(batch_size, seq_length, hidden_size, device, dtype):
    return torch.randn(batch_size, seq_length, hidden_size, device=device, dtype=dtype)

@pytest.fixture
def sample_attention_mask(batch_size, seq_length, device):
    return torch.ones(batch_size, seq_length, device=device, dtype=torch.long)