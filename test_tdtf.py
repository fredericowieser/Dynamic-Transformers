#!/usr/bin/env python3
"""Test script for TDTF model integration."""

import torch
from transformers import Qwen2Config

# Test imports
try:
    from src.models.tdtf.model import TDTFForCausalLM
    print("✅ TDTF model import successful")
except ImportError as e:
    print(f"❌ TDTF model import failed: {e}")
    exit(1)

# Test model creation
def test_tdtf_model():
    """Test TDTF model creation and basic functionality."""
    print("\n🧪 Testing TDTF model creation...")

    # Create a small config for testing
    config = Qwen2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        # TDTF-specific config
        tpn_intermediate_size_factor=0.25,
        tpn_loss_weight=1.0,
        causal_loss_weight=1.0,
        tdtf_capacity=0.5,
        o_ce_init=1.025,
        m_cu_init=1.1,
        beta_ce_init=-0.3,
        beta_cu_init=-0.6,
        ma_window=10,  # Smaller for testing
    )

    # Fix attention implementation
    config._attn_implementation = 'eager'

    try:
        model = TDTFForCausalLM(config)
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")

        # Test forward pass
        batch_size = 2
        seq_len = 16

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        print(f"🔄 Testing forward pass with input shape: {input_ids.shape}")

        # Training mode (should use teacher model)
        model.train()
        outputs = model(input_ids=input_ids, labels=labels)

        print(f"✅ Training forward pass successful")
        print(f"   Loss: {outputs.loss.item():.4f}")
        print(f"   Logits shape: {outputs.logits.shape}")

        # Inference mode (should use student model)
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        print(f"✅ Inference forward pass successful")
        print(f"   Logits shape: {outputs.logits.shape}")

        # Test parameter groups
        param_groups = model.get_trainable_parameters()
        print(f"✅ Parameter groups: {len(param_groups)}")
        for group in param_groups:
            print(f"   {group['name']}: {sum(p.numel() for p in group['params'])} params")

        return True

    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_integration():
    """Test integration with training utils."""
    print("\n🧪 Testing training integration...")

    try:
        from src.training.utils import create_model, PlatformOptimizer

        # Mock config
        class MockConfig:
            def __init__(self):
                self.training = self
                self.model = self
                self.system = self

            def get(self, key, default):
                return default

            def __contains__(self, item):
                return False

        config = MockConfig()
        platform_settings = PlatformOptimizer.get_optimal_settings('cpu', config)

        # Test model creation
        model = create_model('tdtf', '0.5B', from_scratch=True, platform_settings=platform_settings)

        print(f"✅ Training integration successful")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

        return True

    except Exception as e:
        print(f"❌ Training integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 TDTF Model Integration Test")
    print("=" * 50)

    success = True
    success &= test_tdtf_model()
    success &= test_training_integration()

    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! TDTF model is ready for training.")
    else:
        print("❌ Some tests failed. Please check the errors above.")

    exit(0 if success else 1)