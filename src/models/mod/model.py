import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from omegaconf import DictConfig
from ..base.routers import BaseRouter
import torch.nn as nn
from ..base.causal_lm import BaseForCausalLM
from ..base.block import DynamicBlock
from ..base.routers import CausalRouter

class MoDRouter(BaseRouter):
    """Implements the MoD router, which is a simple linear layer."""
    def __init__(self, config, layer_idx: int, model_cfg: Dict = None):
        super().__init__(config, capacity_attr='mod_capacity')
        self.router = nn.Linear(config.hidden_size, 1, bias=False)
        self.aux_loss_weight = config.mod_aux_loss_weight

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        logits = self.router(hidden_states).squeeze(-1)
        return logits, None, {}

class MoDForCausalLM(BaseForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self._setup_layers()
    
    def _setup_layers(self):
        for i in range(self.config.num_hidden_layers):
            block = DynamicBlock(self.config, i)
            if i % 2 == 1: # Dynamic MoD layer
                self.layers.append(nn.ModuleDict({
                    'block': block,
                    'router': MoDRouter(self.config, i),
                    'causal_router': CausalRouter(self.config, i, 'mod_capacity'),
                }))

    def _forward_layers(self, hidden_states, **kwargs):
        total_aux_loss = 0
        for layer in self.layers:
            if isinstance(layer, nn.ModuleDict):
                router = layer['router'] if self.training else layer['causal_router']
                scores, aux_loss, _ = router(hidden_states)
                if aux_loss is not None:
                    total_aux_loss += aux_loss
                
                _, batch_idx, token_idx, gating_scores = router.select_tokens(scores, hidden_states)
                hidden_states, _, _ = layer['block'].process_selected(
                    hidden_states, batch_idx, token_idx, gating_scores, use_soft_gating=self.training, **kwargs
                )
            else:
                hidden_states = layer(hidden_states, **kwargs)[0]
            
        return {"hidden_states": hidden_states, "aux_loss": total_aux_loss}
        
    def get_trainable_parameters(self):
        return self._create_param_groups({'router': 'router', 'causal_router': 'causal_router'})