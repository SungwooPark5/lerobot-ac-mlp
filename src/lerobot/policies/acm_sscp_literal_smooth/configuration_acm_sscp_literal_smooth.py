"""Configuration for ACM (Mamba-1) + literal state carry + boundary-smoothness loss.
Port of acm2_sscp_literal_smooth.

Adds an explicit chunk-boundary continuity (C1/C2) loss on top of the literal-carry
chunk-continuation (CC) training. The literal state carry already gives chunk n+1 the
*information* about chunk n; this loss gives the *incentive* to use it for a smooth seam.

Key design (avoids hurting task success):
  The loss does NOT drive boundary jitter to zero (that would slow the motion and hurt
  SR). Instead it matches the PREDICTED seam finite-differences (velocity + acceleration
  straddling the chunk boundary) to the GROUND-TRUTH ones. Since the two CC chunks are
  consecutive real chunks, the GT seam is already smooth, so the objective is aligned with
  the task rather than fighting it.

sscp_smooth_weight = 0.0  ->  identical behaviour to acm_sscp_literal (invariant / safety).
sscp_smooth_weight > 0.0  ->  boundary-continuity loss active (only during CC training,
                              i.e. paired batches with sscp_enabled + sscp_p_carry > 0).

This config subclasses ACMSSCPLiteralConfig, so it inherits use_bimamba_decoder: set it
True to train BiMamba + literal carry + smoothness in one policy.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm_sscp_literal.configuration_acm_sscp_literal import ACMSSCPLiteralConfig


@PreTrainedConfig.register_subclass("acm_sscp_literal_smooth")
@dataclass
class ACMSSCPLiteralSmoothConfig(ACMSSCPLiteralConfig):
    """ACM literal carry + explicit GT-matched chunk-boundary continuity loss.

    Architecture / carry are identical to acm_sscp_literal (same stateful decoder, same
    literal (conv, ssm) handoff). Only the training objective gains the seam term.
    """

    # lambda for the boundary continuity loss. 0 = off (== acm_sscp_literal). Sweep this.
    sscp_smooth_weight: float = 0.0

    # ── lambda warmup ──────────────────────────────────────────────────────────
    # Ramp lambda from 0 -> sscp_smooth_weight linearly over
    # [warmup_start, warmup_start + warmup_steps]. lambda = 0 before warmup_start.
    # Rationale: let the model learn the task (grasp) FIRST, then add smoothness, so the
    # seam loss doesn't kill the critical pre-grasp motion early on. warmup_steps = 0 ->
    # no ramp (lambda jumps to full at warmup_start).
    sscp_smooth_warmup_start: int = 0
    sscp_smooth_warmup_steps: int = 0

    # Finite-difference order of the seam loss:
    #   1 -> C1 only (velocity match)                  — gentler, recommended for sweep
    #   2 -> C1 + C2 (velocity + acceleration match)   — original, more aggressive
    sscp_smooth_order: int = 2

    # If True, compute the seam loss on INFERENCE-mode (VAE latent = 0) predictions via an
    # extra eval-mode forward, instead of the teacher-forced training predictions. Targets
    # the actual inference boundary. Costs 2 extra forwards/step. Default False.
    sscp_smooth_free_latent: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.sscp_smooth_weight < 0:
            raise ValueError("sscp_smooth_weight must be >= 0.")
        if self.sscp_smooth_order not in (1, 2):
            raise ValueError(f"sscp_smooth_order must be 1 or 2, got {self.sscp_smooth_order}.")
