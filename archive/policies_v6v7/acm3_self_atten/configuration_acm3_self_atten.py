from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm3.configuration_acm3 import ACM3Config


@PreTrainedConfig.register_subclass("acm3_self_atten")
@dataclass
class ACM3SelfAttenConfig(ACM3Config):
    """ACM3 with post-Mamba self-attention on action tokens.

    After the Mamba3 decoder produces K action token representations, a
    MultiheadAttention layer lets them attend to each other.  A tanh-gated
    residual (γ initialised near 0) ensures training starts effectively
    identical to vanilla ACM3 and the gate opens gradually.
    """
    self_atten_nhead: int = 8
    self_atten_gamma_init: float = 1e-4
