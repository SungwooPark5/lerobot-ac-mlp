"""Configuration for ACM3 + LITERAL SSM State Carryover.

Unlike acm3_sscp (which carries a *summary token* — the previous chunk's terminal
decoder output — and re-derives state by scanning it), this policy carries Mamba3's
**actual recurrent state tuple** (angle, ssm, k, v) across chunk boundaries via the
kernel's `Input_States` / `return_final_states` path. This is "true" SSM state
carryover: the scan literally continues from where the previous chunk ended.

The whole decoder sequence [encoder_out, queries] is treated as one continuous
stream across chunks (the SSM never resets). At training, the carried state is
detached at the chunk boundary (truncated BPTT) when sscp_detach=True.

Compare:
  acm3_sscp        — summary-token warm-up   (A4 / A4cc)
  acm3_sscp_literal — literal state handoff   (A4lit)   ← this file
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm3_sscp.configuration_acm3_sscp import ACM3SSCPConfig


@PreTrainedConfig.register_subclass("acm3_sscp_literal")
@dataclass
class ACM3SSCPLiteralConfig(ACM3SSCPConfig):
    """ACM3 + literal Mamba3 hidden-state carryover (no ICPE).

    Inherits sscp_enabled / sscp_p_carry / sscp_detach. `sscp_carry_position` is
    unused here (literal carry continues the full stream; there is no token to place).
    """
    pass
