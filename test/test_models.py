# tests/test_models.py
import torch
import pytest
from omegaconf import OmegaConf

from dynamic_transformer.models.gpt_2 import GPT


# We use pytest fixtures to create a reusable, minimal model configuration.
@pytest.fixture
def gpt_config():
    """Provides a minimal configuration for a small GPT model for testing."""
    return OmegaConf.create(
        {
            "block_size": 64,
            "vocab_size": 500,
            "n_layer": 4,
            "n_head": 4,
            "n_embd": 128,
            "dropout": 0.1,
        }
    )


def test_gpt_model_initialization(gpt_config):
    """
    Tests if the custom GPT model can be initialized without errors.
    """
    try:
        model = GPT(gpt_config)
        assert model is not None, "Model should not be None after initialization."
        # Check if the number of parameters is greater than zero
        assert (
            model.get_num_params() > 0
        ), "Model should have a positive number of parameters."
    except Exception as e:
        pytest.fail(f"GPT model initialization failed with an exception: {e}")


def test_gpt_forward_pass_with_targets(gpt_config):
    """
    Tests the forward pass during a training-like scenario (with targets).
    """
    model = GPT(gpt_config)
    batch_size = 4
    seq_length = gpt_config.block_size

    # Create dummy input tensors
    input_indices = torch.randint(
        0, gpt_config.vocab_size, (batch_size, seq_length)
    )
    targets = torch.randint(
        0, gpt_config.vocab_size, (batch_size, seq_length)
    )

    logits, loss = model(input_indices, targets)

    # --- Assertions ---
    # 1. Check output shapes
    assert logits.shape == (
        batch_size,
        seq_length,
        gpt_config.vocab_size,
    ), "Logits shape is incorrect."
    assert loss.shape == torch.Size(
        []
    ), "Loss should be a scalar tensor."

    # 2. Check that loss is a valid number
    assert not torch.isnan(loss), "Loss should not be NaN."
    assert not torch.isinf(loss), "Loss should not be infinite."

    # 3. Check that backpropagation can be initiated
    try:
        loss.backward()
    except Exception as e:
        pytest.fail(f"loss.backward() failed with an exception: {e}")


def test_gpt_forward_pass_no_targets(gpt_config):
    """
    Tests the forward pass during an inference-like scenario (without targets).
    """
    model = GPT(gpt_config)
    model.eval()  # Set model to evaluation mode

    batch_size = 4
    seq_length = 32  # Use a shorter sequence for inference testing

    input_indices = torch.randint(
        0, gpt_config.vocab_size, (batch_size, seq_length)
    )

    with torch.no_grad():
        logits, loss = model(input_indices, targets=None)

    # --- Assertions ---
    # 1. Check output shapes
    # For inference, we only expect logits for the very last token
    assert logits.shape == (
        batch_size,
        1,
        gpt_config.vocab_size,
    ), "Inference logits shape is incorrect."

    # 2. Check that loss is None
    assert loss is None, "Loss should be None during inference."


def test_gpt_sequence_length_assertion(gpt_config):
    """
    Tests that the model raises an error if the input sequence is too long.
    """
    model = GPT(gpt_config)
    too_long_seq = torch.randint(
        0, gpt_config.vocab_size, (1, gpt_config.block_size + 1)
    )

    # Use pytest.raises to assert that a specific error is thrown
    with pytest.raises(AssertionError, match="Cannot forward sequence of length"):
        model(too_long_seq)