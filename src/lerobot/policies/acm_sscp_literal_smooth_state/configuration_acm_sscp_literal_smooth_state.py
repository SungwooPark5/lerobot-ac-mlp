"""S3 — Config for ACM (Mamba-1) literal carry + SSM state-continuity smoothness.
Port of acm2_sscp_literal_smooth_state.

Adds a *state-continuity* term on top of the smooth family. Idea: the literal carried
(conv, ssm) state is what physically links chunk n -> n+1, so the boundary continuity should
be *supplied by the carry*, not merely regressed on the output. We run chunk n+1 twice — once
WITH the carried state (grad) and once fresh/memoryless (carry=None, detached target) — and
require the carried opening to bridge the seam at least as well as the memoryless restart, while
staying matched to the GT seam. This term is only definable for a state-carrying policy, so it
doubles as the family's mechanism-novelty axis.

sscp_state_weight = 0 AND sscp_smooth_weight = 0  ->  identical to acm_sscp_literal.
Inherits every acm_sscp_literal_smooth flag (seam C1/C2, warmup, free_latent, BiMamba, carry).
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm_sscp_literal_smooth.configuration_acm_sscp_literal_smooth import (
    ACMSSCPLiteralSmoothConfig,
)


@PreTrainedConfig.register_subclass("acm_sscp_literal_smooth_state")
@dataclass
class ACMSSCPLiteralSmoothStateConfig(ACMSSCPLiteralSmoothConfig):
    # Weight on the state-continuity (carry-vs-fresh) seam term. 0 = off. Costs one extra
    # (no-grad) chunk-n+1 forward per step when > 0. Follows the smooth warmup schedule.
    sscp_state_weight: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.sscp_state_weight < 0:
            raise ValueError("sscp_state_weight must be >= 0.")
