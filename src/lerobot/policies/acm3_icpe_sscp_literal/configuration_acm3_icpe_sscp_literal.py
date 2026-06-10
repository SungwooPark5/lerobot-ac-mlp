"""Configuration for ACM3 + ICPE + LITERAL SSM State Carryover.

Same literal hidden-state carry as acm3_sscp_literal, but on top of the full
C-series (ICPE + SSCP). This is the literal counterpart of P1cc.

Compare:
  acm3_icpe_sscp        — summary-token warm-up (P1 / P1cc)
  acm3_icpe_sscp_literal — literal state handoff (P1cc_lit)  ← this file
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm3_icpe_sscp.configuration_acm3_icpe_sscp import ACM3ICPESSCPConfig


@PreTrainedConfig.register_subclass("acm3_icpe_sscp_literal")
@dataclass
class ACM3ICPESSCPLiteralConfig(ACM3ICPESSCPConfig):
    """ACM3 + ICPE + literal Mamba3 hidden-state carryover."""
    pass
