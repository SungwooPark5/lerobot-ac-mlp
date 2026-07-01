"""S5 — Config for ACM2 literal carry + velocity(delta)-integration head (v22 rank 5).

Structural: the action head output is reinterpreted as per-step *increments* (velocity), which
are integrated from an anchor:  a[t] = anchor + cumsum(delta)[:t+1].  With the anchor set to the
current proprioception (OBS_STATE) — natural for ALOHA position control, where actions are joint
targets ~ current pose — a[0] = anchor + delta[0], so a small first increment yields a[0] ~= the
current pose, guaranteeing a near-C0 seam at every chunk start *by construction* (no loss tuning).
Because proprioception at chunk n+1's start is continuous with chunk n's end, cross-chunk seams
are continuous too. The carried Mamba state still shapes the delta trajectory.

sscp_velint_anchor = "zero"  ->  plain absolute-delta cumsum (no anchor continuity).
Any inherited smooth seam loss still applies (on the integrated actions).
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2_sscp_literal_smooth.configuration_acm2_sscp_literal_smooth import (
    ACM2SSCPLiteralSmoothConfig,
)


@PreTrainedConfig.register_subclass("acm2_sscp_literal_smooth_velint")
@dataclass
class ACM2SSCPLiteralSmoothVelIntConfig(ACM2SSCPLiteralSmoothConfig):
    # Integration anchor:
    #   "state" — current proprioception OBS_STATE (falls back to zero if its dim != action dim).
    #   "zero"  — integrate from zero (no anchor-based seam continuity).
    sscp_velint_anchor: str = "state"

    def __post_init__(self):
        super().__post_init__()
        if self.sscp_velint_anchor not in ("state", "zero"):
            raise ValueError(f"sscp_velint_anchor must be 'state' or 'zero', got {self.sscp_velint_anchor}.")
