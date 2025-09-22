import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
from ..base.priors import BasePriorNetwork
from ..base.routers import BaseSurpriseRouter
from ..base.block import DynamicBlock
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

class SDTPriorNetwork(BasePriorNetwork):
    """Implements the SDT prior: x + MLP(RMSNorm(x))."""
    def __init__(self, config):
        super().__init__(config)
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))

class SDTDecisionLayer(nn.Module):
    """Computes the original, posterior, and prior states needed for SDT routing."""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        log.debug(f"SDTDecisionLayer.__init__: config.use_cache={config.use_cache}, config.attn_implementation={config.attn_implementation}")
        self.block = DynamicBlock(config, layer_idx)
        self.prior_network = SDTPriorNetwork(config)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        original = hidden_states
        log.debug(f"SDTDecisionLayer.forward: hidden_states shape before block: {hidden_states.shape}")

        # Flatten hidden_states to (B*T, D) before passing to Qwen2DecoderLayer
        outputs = self.block(hidden_states, **kwargs)
        # Qwen2DecoderLayer returns a tuple; DynamicBlock returns it directly
        posterior = outputs[0]
        log.debug(f"SDTDecisionLayer.forward: posterior shape: {posterior.shape}")
        prior = self.prior_network(original)
        prior_loss = F.mse_loss(prior, posterior.detach())
        return {'original': original, 'posterior': posterior, 'prior': prior, 'prior_loss': prior_loss}

class SDTRouter(BaseSurpriseRouter):
    """Implements the SDT surprise calculation by inheriting VPR logic."""
    def __init__(self, config, layer_idx: int):
        super().__init__(config, capacity_attr='sdt_capacity')

    def forward(self, original: torch.Tensor, posterior: torch.Tensor, prior: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        d = float(original.shape[-1])
        D_st = torch.sum((posterior - original).pow(2), dim=-1) / d
        D_ch = torch.sum((posterior - prior).pow(2), dim=-1) / d
        
        beta_ce = kwargs.get('beta_ce', self.config.beta_ce_init)
        beta_cu = kwargs.get('beta_cu', self.config.beta_cu_init)

        # Use betas passed from the training loop
        g_cont, stats = self._get_vpr_signals(D_st, D_ch, beta_ce, beta_cu)
        return g_cont, None, stats

import torch.nn as nn
from ..base.causal_lm import BaseForCausalLM
from ..base.block import DynamicBlock
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

class SDTForCausalLM(BaseForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self._setup_layers()
    
    def _setup_layers(self):
        for i in range(0, self.config.num_hidden_layers, 2):
            self.layers.append(nn.ModuleDict({
                'decision': SDTDecisionLayer(self.config, i),
                'dynamic_block': DynamicBlock(self.config, i + 1),
                'router': SDTRouter(self.config, i + 1)
            }))

    def _forward_layers(self, hidden_states, **kwargs):
        total_aux_loss = 0
        for layer_pair in self.layers:
            decision_output = layer_pair['decision'](hidden_states, **kwargs)
            hidden_states = decision_output['posterior']
            if self.training and decision_output['prior_loss'] is not None:
                # The weight is applied in the main forward pass of the base class
                total_aux_loss += self.config.prior_loss_weight * decision_output['prior_loss']
            
            router = layer_pair['router']
            scores, _, stats = router(**decision_output, **kwargs) # Capture stats
            log.debug(f"SDTForCausalLM: scores shape: {scores.shape}, hidden_states shape: {hidden_states.shape}")
            _, batch_idx, token_idx, gating_scores = router.select_tokens(scores, hidden_states)
            hidden_states, _, _ = layer_pair['dynamic_block'].process_selected(
                hidden_states, batch_idx, token_idx, gating_scores, use_soft_gating=self.training, **kwargs
            )
        
        return {"hidden_states": hidden_states, "aux_loss": total_aux_loss, "router_stats": stats} # Return stats

    def get_trainable_parameters(self):
        return self._create_param_groups({'router': 'router', 'prior': 'decision.prior_network'})