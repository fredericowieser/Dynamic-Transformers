import logging
import torch.nn as nn

from ..layers.decision_layer import DecisionLayer
from ..layers.dynamic_layer import DynamicLayer
from ..layers.mod_layer import MoDLayer
from ..blocks.qwen_block import Qwen2Block

log = logging.getLogger(__name__)

def populate_weights_from_source_layers(custom_model, source_hf_layers):
    """
    Populates the weights of a custom dynamic model from a list of source
    transformer layers. Assumes the custom model's layer structure is already built.
    """
    log.info("Populating weights from source layers into custom model...")
    
    for i, source_layer in enumerate(source_hf_layers):
        target_layer = custom_model.model.layers[i]
        source_state_dict = source_layer.state_dict()
        
        # Load weights into the '.block' attribute of custom layers
        if hasattr(target_layer, 'block') and isinstance(target_layer.block, Qwen2Block):
            target_layer.block.load_state_dict(source_state_dict, strict=True)
        # Load weights directly if the target is a standard block
        elif isinstance(target_layer, Qwen2Block):
            target_layer.load_state_dict(source_state_dict, strict=True)
        else:
            log.warning(f"Could not load weights for layer {i} of type {type(target_layer)}.")
            
    log.info("Weight population complete.")
    return custom_model