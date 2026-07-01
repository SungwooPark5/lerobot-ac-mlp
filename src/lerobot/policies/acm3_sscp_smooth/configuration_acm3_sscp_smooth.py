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

    # ── λ warmup ──────────────────────────────────────────────────────────────
    # Ramp λ from 0 → sscp_smooth_weight linearly over
    # [warmup_start, warmup_start + warmup_steps]. λ = 0 before warmup_start.
    # Rationale: let the model learn the task (grasp) FIRST, then add smoothness,
    # so the seam loss doesn't kill the critical pre-grasp motion early on.
    # warmup_steps = 0 → no ramp (λ jumps to full at warmup_start).
    sscp_smooth_warmup_start: int = 0
    sscp_smooth_warmup_steps: int = 0

    # Finite-difference order of the seam loss:
    #   1 → C1 only (velocity match)                  — gentler, recommended for sweep
    #   2 → C1 + C2 (velocity + acceleration match)   — original, more aggressive
    sscp_smooth_order: int = 2

    # If True, compute the seam loss on INFERENCE-mode (VAE latent = 0) predictions
    # via an extra eval-mode forward, instead of the teacher-forced training
    # predictions. Targets the actual inference boundary (Phase-2 fix). Costs 2
    # extra forwards/step. Default False.
    sscp_smooth_free_latent: bool = False
