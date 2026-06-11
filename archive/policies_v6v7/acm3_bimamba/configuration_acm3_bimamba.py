from dataclasses import dataclass

from lerobot.policies.acm3.configuration_acm3 import ACM3Config
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("acm3_bimamba")
@dataclass
class ACM3BiMambaConfig(ACM3Config):
    """ACM3 with Bidirectional Mamba3 (Vim-style: forward + backward, averaged).

    Identical to ACM3Config — the bidirectional structure is entirely in the decoder
    and requires no extra hyperparameters.  Two independent Mamba3 layer stacks are
    created (forward_layers, backward_layers), doubling decoder parameter count.
    """
    pass
