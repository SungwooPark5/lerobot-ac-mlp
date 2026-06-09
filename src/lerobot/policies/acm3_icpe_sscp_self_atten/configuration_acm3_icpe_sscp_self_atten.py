from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm3_icpe_sscp.configuration_acm3_icpe_sscp import ACM3ICPESSCPConfig


@PreTrainedConfig.register_subclass("acm3_icpe_sscp_self_atten")
@dataclass
class ACM3ICPESSCPSelfAttenConfig(ACM3ICPESSCPConfig):
    """ACM3 + ICPE + SSCP with post-Mamba self-attention on action tokens.

    Adds MultiheadAttention on the K decoder output tokens after the Mamba3
    pass, before the action head.  tanh(γ) gated residual keeps initialisation
    near ACM3ICPESECP baseline.  The carry for SSCP is extracted from the
    post-attention representation so it encodes refined cross-token context.
    """
    self_atten_nhead: int = 8
    self_atten_gamma_init: float = 1e-4
