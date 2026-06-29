"""Configuration for ACM3 + SSCP + Bidirectional Mamba3 (BiMamba), no ICPE.

Combines:
  - SSCP (SSM State Carryover Protocol)  — chunk-boundary carry (our contribution)
  - BiMamba (bidirectional Mamba3 decoder) — eunji's contribution

The unidirectional Mamba3SSCPDecoder is replaced by a forward+backward BiMamba
decoder. All SSCP hyperparameters (sscp_enabled / sscp_p_carry / sscp_detach /
sscp_carry_position) are inherited unchanged. The carry token is prepended to
BOTH forward and backward sequences so both directions warm up from the previous
chunk's context.

No ICPE: decoder queries use only the learned positional embedding.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm3_sscp.configuration_acm3_sscp import ACM3SSCPConfig


@PreTrainedConfig.register_subclass("acm3_sscp_bimamba")
@dataclass
class ACM3SSCPBiMambaConfig(ACM3SSCPConfig):
    """ACM3 + SSCP + Bidirectional Mamba3 (no ICPE).

    Inherits every ACM3SSCPConfig field; only the decoder topology changes
    (unidirectional → BiMamba) in the modeling code.
    """

    pass
