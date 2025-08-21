from transformers.models.qwen2.tokenization_qwen2 import Qwen2TokenizerFast


class DynamicQwenTokenizer(Qwen2TokenizerFast):
    """
    A thin subclass of the Qwen fast tokenizer.
    Use this if you need to inject any special fixes (e.g. pad_token_id)
    or methods (e.g. custom chat templates). For now, it inherits unchanged.
    """

    pass
