#!/usr/bin/env python
"""ACM2 smooth — State-Carried BiMamba decoder with boundary-continuity + jerk training loss (v14).

Architecture == acm2_sscp_literal_bimamba (ACT Transformer encoder + bidirectional Mamba-2
decoder + recurrent state carry across chunk boundaries). `_build_model` is unchanged, so v12
BiMamba checkpoints load with zero key mismatch.

v14 adds ONLY training losses (no architecture change, inference unchanged):
  • boundary-continuity loss: match POSITION (and optionally VELOCITY = C1) between the end of
    chunk n and the start of chunk n+1, using the chunk-pair carry path. Removes the structural
    chunk-boundary jerk that plain behavior cloning leaves.
  • jerk penalty: minimize the 2nd difference (jerk proxy) of the predicted action chunk.

Headline: the carried recurrent state + a velocity-aware boundary loss give C1-continuous
(smooth) trajectories — vs ACT temporal ensembling which averages in OUTPUT space (C0, stale).
Trained with ChunkPairDataset (2 decodes); no reactive/per-step replanning (v13 archived).
"""
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2_sscp_literal_bimamba.configuration_acm2_sscp_literal_bimamba import (
    ACM2SSCPLiteralBiMambaConfig,
)

SMOOTH_JERK_MODES = ("l1", "l2")


@PreTrainedConfig.register_subclass("acm2_smooth")
@dataclass
class ACM2SmoothConfig(ACM2SSCPLiteralBiMambaConfig):
    # ── Boundary-continuity loss (needs chunk-pair: use_chunk_pairs + sscp_p_carry>0) ──
    # Weight on matching the chunk n→n+1 boundary. 0 disables (→ parent bimamba behavior).
    smooth_boundary_weight: float = 0.1
    # True: match position AND velocity at the boundary (C1). False: position only (C0).
    smooth_boundary_velocity: bool = True

    # ── Jerk penalty (intra-chunk 2nd-difference smoothness) ──────────────────────
    # Weight on the predicted-action jerk proxy. Keep SMALL — too large fights the L1/BC fit.
    smooth_jerk_weight: float = 0.02
    # "l1" (robust) or "l2" (penalizes spikes harder).
    smooth_jerk_mode: str = "l1"
    # If True, penalize only jerk ABOVE the ground-truth chunk's jerk (excess jerk), so the
    # policy is not forced smoother than the demonstrations. If False, penalize raw jerk.
    smooth_jerk_excess_only: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.smooth_jerk_mode not in SMOOTH_JERK_MODES:
            raise ValueError(f"smooth_jerk_mode must be one of {SMOOTH_JERK_MODES}, got '{self.smooth_jerk_mode}'.")
        if self.smooth_boundary_weight < 0 or self.smooth_jerk_weight < 0:
            raise ValueError("smooth_boundary_weight / smooth_jerk_weight must be >= 0.")
