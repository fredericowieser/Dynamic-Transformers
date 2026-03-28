from transformers import AutoConfig, AutoModelForCausalLM

from .base.causal_lm import BaseForCausalLM
from .configs import MoDConfig, SDTConfig, StandardConfig, STTConfig
from .mod.model import MoDForCausalLM
from .sdt.model import SDTForCausalLM
from .standard.model import StandardTransformerForCausalLM
from .stt.model import STTForCausalLM

# Register custom model types with their corresponding config class.
AutoConfig.register("standard", StandardConfig)
AutoConfig.register("mod", MoDConfig)
AutoConfig.register("sdt", SDTConfig)
AutoConfig.register("stt", STTConfig)

AutoModelForCausalLM.register(StandardConfig, StandardTransformerForCausalLM)
AutoModelForCausalLM.register(MoDConfig, MoDForCausalLM)
AutoModelForCausalLM.register(SDTConfig, SDTForCausalLM)
AutoModelForCausalLM.register(STTConfig, STTForCausalLM)


__all__ = [
    "BaseForCausalLM",
    "StandardTransformerForCausalLM",
    "MoDForCausalLM",
    "SDTForCausalLM",
    "STTForCausalLM",
    "StandardConfig",
    "MoDConfig",
    "SDTConfig",
    "STTConfig",
]
