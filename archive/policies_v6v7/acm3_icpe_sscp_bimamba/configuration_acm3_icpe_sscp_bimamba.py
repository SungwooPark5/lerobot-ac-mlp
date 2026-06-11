from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm3_icpe_sscp.configuration_acm3_icpe_sscp import ACM3ICPESSCPConfig


@PreTrainedConfig.register_subclass("acm3_icpe_sscp_bimamba")
@dataclass
class ACM3ICPESSCPBiMambaConfig(ACM3ICPESSCPConfig):
    """ACM3 + ICPE + SSCP with Bidirectional Mamba3 (C-series + BiMamba E1).

    Replaces the unidirectional Mamba3ICPEDecoder with Vim-style external BiMamba.
    Inherits all ICPE and SSCP hyperparameters unchanged.
    The carry token is prepended to both forward and backward sequences so both
    directions benefit from previous-chunk context warmup.
    """
    pass
