import torch
from typing import List

class LMEvalAdaptor:
    """
    A wrapper class to make a Hugging Face-style model compatible with the lm-eval harness.
    """
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self._batch_size = 1 # Default, can be overridden by lm-eval
    
    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # Assumes model has a config with max_position_embeddings
        return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256 # A reasonable default for generation

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(string, **kwargs)

    def tok_decode(self, tokens: List[int], **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)

    def _model_call(self, inps: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        The main model call function for lm-eval.
        :param inps: a torch.Tensor of shape [batch, sequence_length]
        :return: a torch.Tensor of shape [batch, sequence_length, vocab_size]
        """
        with torch.no_grad():
            return self.model(inps, **kwargs).get("logits")

    def _model_generate(self, context, max_length, eos_token_id):
        # lm-eval uses this for generation tasks
        return self.model.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False, # Use greedy decoding for determinism
        )
