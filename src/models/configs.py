from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class StandardConfig(Qwen2Config):
    model_type = "standard"


class MoDConfig(Qwen2Config):
    model_type = "mod"


class SDTConfig(Qwen2Config):
    model_type = "sdt"


class STTConfig(Qwen2Config):
    model_type = "stt"
