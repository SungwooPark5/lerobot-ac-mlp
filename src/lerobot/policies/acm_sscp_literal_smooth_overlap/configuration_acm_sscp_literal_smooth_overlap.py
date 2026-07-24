"""MOSAIC — Config for ACM (Mamba-1) literal carry + overlap-add crossfade.
Port of acm2_sscp_literal_smooth_overlap.

Inference (structural): at each chunk boundary the last `sscp_overlap` executed steps of the
previous chunk are crossfaded with the first `sscp_overlap` steps of the new chunk via a rising
window. Because the literal (conv, ssm) state is carried, the two overlapping predictions are
mutually consistent, so the blend is clean rather than a smeared average — the boundary
discontinuity is physically removed. This is only meaningful for a state-carrying policy.

Training: inherits the State objective (base l1 + optional GT-matched seam loss
`sscp_smooth_weight` + optional carry-vs-fresh `sscp_state_weight`). Additionally, when
`sscp_overlap_train_weight > 0`, an overlap-consistency term trains the model for the blend it
undergoes at inference (requires the chunk-pair offset = chunk_size - overlap; the train pipeline
sets this automatically). This turns overlap-add from an inference-only trick into a trained
property.

sscp_overlap = 0 -> inference identical to acm_sscp_literal_smooth (hard chunk switch).
All weights 0 + overlap 0 -> identical to plain literal-carry training/inference.
Disabled automatically at inference when temporal_ensemble_coeff is set.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm_sscp_literal_smooth_state.configuration_acm_sscp_literal_smooth_state import (
    ACMSSCPLiteralSmoothStateConfig,
)


@PreTrainedConfig.register_subclass("acm_sscp_literal_smooth_overlap")
@dataclass
class ACMSSCPLiteralSmoothOverlapConfig(ACMSSCPLiteralSmoothStateConfig):
    # Number of overlap steps crossfaded across each boundary. 0 = off. Must be <= chunk_size//2.
    sscp_overlap: int = 0

    # Crossfade window shape for the rising blend weight: "linear" or "hann".
    sscp_overlap_window: str = "linear"

    # Overlap-consistency training weight (the "mosaic-trained" objective). 0 = inference-only overlap.
    # >0 requires the chunk-pair offset = chunk_size - sscp_overlap (set by the train pipeline).
    sscp_overlap_train_weight: float = 0.0

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
        if self.sscp_overlap_train_weight < 0.0:
            raise ValueError("sscp_overlap_train_weight must be >= 0.")
