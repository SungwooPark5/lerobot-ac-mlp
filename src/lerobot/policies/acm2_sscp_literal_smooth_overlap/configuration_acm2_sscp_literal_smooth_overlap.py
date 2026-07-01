"""S7 — Config for ACM2 literal carry + overlap-add crossfade at inference (v22 rank 3).

Structural, INFERENCE-ONLY smoothing (training is identical to acm2_sscp_literal_smooth, so any
inherited seam loss still applies). At each chunk boundary we crossfade the last `sscp_overlap`
executed steps of the previous chunk with the first `sscp_overlap` steps of the new chunk via a
rising window. Because the literal (conv, ssm) state is carried, the two overlapping predictions
are mutually consistent, so the blend is clean rather than a smeared average — the boundary
discontinuity is physically removed. This is only meaningful for a state-carrying policy.

sscp_overlap = 0  ->  identical inference to acm2_sscp_literal_smooth (hard chunk switch).
Disabled automatically when temporal_ensemble_coeff is set (the ensembler owns the queue).
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2_sscp_literal_smooth.configuration_acm2_sscp_literal_smooth import (
    ACM2SSCPLiteralSmoothConfig,
)


@PreTrainedConfig.register_subclass("acm2_sscp_literal_smooth_overlap")
@dataclass
class ACM2SSCPLiteralSmoothOverlapConfig(ACM2SSCPLiteralSmoothConfig):
    # Number of overlap steps crossfaded across each boundary. 0 = off. Must be <= chunk_size//2
    # so the hop (chunk_size - overlap) never underflows the overlap region.
    sscp_overlap: int = 0

    # Crossfade window shape for the rising blend weight: "linear" or "hann".
    sscp_overlap_window: str = "linear"

    def __post_init__(self):
        super().__post_init__()
        if self.sscp_overlap < 0:
            raise ValueError("sscp_overlap must be >= 0.")
        if self.sscp_overlap > self.chunk_size // 2:
            raise ValueError(
                f"sscp_overlap ({self.sscp_overlap}) must be <= chunk_size//2 ({self.chunk_size // 2})."
            )
        if self.sscp_overlap_window not in ("linear", "hann"):
            raise ValueError(f"sscp_overlap_window must be 'linear' or 'hann', got {self.sscp_overlap_window}.")
