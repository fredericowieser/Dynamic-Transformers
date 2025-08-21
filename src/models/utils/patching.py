import logging

import torch.nn as nn

from ..layers.decision_layer import DecisionLayer
from ..layers.dynamic_layer import DynamicLayer
from ..layers.mod_layer import MoDLayer
from ..blocks.qwen_block import Qwen2Block

log = logging.getLogger(__name__)


def patch_and_populate_layers(model_to_patch, config, source_hf_layers):
    """
    Replaces layers in a model with dynamic architecture layers (VPR or MoD),
    transferring weights from the source layers.

    Args:
        model_to_patch: The custom model instance to modify.
        config: The dynamic model configuration object.
        source_hf_layers: The list of original transformer layers from the pretrained model.

    Returns:
        The model with its layers replaced.
    """
    log.info(
        f"Patching {len(source_hf_layers)} layers into '{config.dynamic_architecture}' architecture."
    )
    new_layers = nn.ModuleList()
    device = next(model_to_patch.parameters()).device

    for i, original_layer in enumerate(source_hf_layers):
        original_layer_state_dict = original_layer.state_dict()
        
        if config.dynamic_architecture == "vpr":
            # VPR architecture uses alternating Decision and Dynamic layers.
            if i % 2 == 0:
                new_layer = DecisionLayer(config, layer_idx=i)
            else:
                new_layer = DynamicLayer(config, layer_idx=i)
            # For VPR, the entire original state dict can be loaded,
            # as both layers contain a full transformer block.
            new_layer.block.load_state_dict(original_layer_state_dict, strict=False)

        elif config.dynamic_architecture == "mod":
            # MoD architecture interleaves standard blocks with MoD blocks.
            # Every other layer (odd indices) becomes a MoD layer.
            if (i + 1) % 2 == 0:
                new_layer = MoDLayer(config, layer_idx=i)
                # Load weights into the MoD layer's internal block
                new_layer.block.load_state_dict(original_layer_state_dict)
            else:
                # Even layers are standard, non-dynamic blocks
                new_layer = Qwen2Block(config, layer_idx=i)
                new_layer.load_state_dict(original_layer_state_dict)
        else:
            raise ValueError(f"Unknown dynamic_architecture: '{config.dynamic_architecture}'")

        new_layers.append(new_layer.to(device))

    model_to_patch.model.layers = new_layers
    log.info("Model patching complete.")
    return model_to_patch