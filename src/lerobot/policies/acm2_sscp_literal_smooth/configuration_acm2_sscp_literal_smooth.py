"""Configuration for ACM2 literal SSD state carry + chunk-boundary smoothness (v20).

Base = `acm2_sscp_literal` (the v10 `m2_lit` policy that trained well: literal Mamba-2 SSD
recurrent-state handoff across chunk boundaries). v20 inherits that PROVEN carry untouched
and adds ONLY a chunk-boundary smoothness objective:

  • boundary-continuity loss : match chunk n's last action ↔ chunk n+1's first action in
                               position (C0) and optionally velocity (C1).
  • interior jerk penalty    : suppress the 2nd action-difference inside each chunk
                               (excess over the demonstration's own jerk only).

Pure-carry framing (v15 paper): with the inherited `carry_fusion="none"` and
`carry_noise_std=0.0` (both defaults), this is a literal carry with NO disturbance
machinery — gate / noise / fusion are inherited knobs left OFF. carry is used solely to
stitch the chunk boundary smoothly, not to correct for disturbance.

Property: with smooth_boundary_weight = smooth_jerk_weight = 0 this policy reduces EXACTLY
to acm2_sscp_literal (same forward, same losses) — a clean superset for sanity-checking.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2_sscp_literal.configuration_acm2_sscp_literal import ACM2SSCPLiteralConfig

SMOOTH_JERK_MODES = ("l1", "l2")


@PreTrainedConfig.register_subclass("acm2_sscp_literal_smooth")
@dataclass
class ACM2SSCPLiteralSmoothConfig(ACM2SSCPLiteralConfig):
    """ACM2 literal carry + boundary-continuity / jerk smoothness losses.

    Inherits the full literal-carry config (sscp_enabled / sscp_p_carry / sscp_detach /
    carry_fusion / carry_noise_* / carry_gate_*). Defaults keep carry pure (fusion=none,
    noise=0): only the smoothness losses below are added on top.
    """

    # ── boundary-continuity loss (chunk n end ↔ n+1 start) ──────────────────────
    smooth_boundary_weight: float = 0.1      # 0 → off (then == lit, but jerk may still apply)
    smooth_boundary_velocity: bool = True    # True: position+velocity (C1) / False: position (C0)

    # ── interior jerk penalty (2nd difference within a chunk) ───────────────────
    smooth_jerk_weight: float = 0.02         # 0 → off
    smooth_jerk_mode: str = "l1"             # l1 | l2
    smooth_jerk_excess_only: bool = True     # penalize only jerk beyond the GT chunk's own jerk

    # ── dims excluded from the smoothness losses (e.g. grippers that must move sharply) ──
    # ALOHA 14-dim bimanual: left arm 6 + left gripper(6) + right arm 6 + right gripper(13).
    # [] = exclude nothing. Only affects boundary/jerk terms; the L1 reconstruction loss
    # always uses every dim.
    smooth_exclude_dims: list[int] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if self.smooth_jerk_mode not in SMOOTH_JERK_MODES:
            raise ValueError(
                f"smooth_jerk_mode must be one of {SMOOTH_JERK_MODES}, got '{self.smooth_jerk_mode}'."
            )
        if self.smooth_boundary_weight < 0 or self.smooth_jerk_weight < 0:
            raise ValueError("smooth_boundary_weight / smooth_jerk_weight must be >= 0.")
