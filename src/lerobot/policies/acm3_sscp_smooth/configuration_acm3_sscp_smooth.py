"""Configuration for ACM3 + SSCP + boundary-smoothness loss.

Adds an explicit chunk-boundary continuity (C1/C2) loss on top of acm3_sscp's
chunk-continuation (CC) training. The carry already gives chunk n+1 the
*information* about chunk n; this loss gives the *incentive* to use it for a
smooth seam.

Key design (avoids hurting task success):
  The loss does NOT drive boundary jitter to zero (that would slow the motion and
  hurt SR). Instead it matches the PREDICTED seam finite-differences (velocity +
  acceleration straddling the chunk boundary) to the GROUND-TRUTH ones. Since the
  two CC chunks are consecutive real chunks, the GT seam is already smooth, so the
  objective is aligned with the task rather than fighting it.

sscp_smooth_weight = 0.0  →  identical behaviour to acm3_sscp (invariant / safety).
sscp_smooth_weight > 0.0  →  boundary-continuity loss active (only during CC
                             training, i.e. use_chunk_pairs + sscp_p_carry > 0).
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm3_sscp.configuration_acm3_sscp import ACM3SSCPConfig


@PreTrainedConfig.register_subclass("acm3_sscp_smooth")
@dataclass
class ACM3SSCPSmoothConfig(ACM3SSCPConfig):
    """ACM3 + SSCP with an explicit GT-matched chunk-boundary continuity loss.

    Architecture is identical to acm3_sscp (same Mamba3SSCPDecoder, same carry).
    Only the training objective gains the seam-smoothness term.
    """

    # λ for the boundary continuity loss. 0 = off (== acm3_sscp). Sweep this.
    sscp_smooth_weight: float = 0.0
