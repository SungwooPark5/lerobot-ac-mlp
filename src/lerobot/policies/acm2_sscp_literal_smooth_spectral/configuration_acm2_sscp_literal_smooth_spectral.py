"""S2 — Config for ACM2 literal carry + spectral (frequency-domain) seam loss (v22 rank 2).

Tremor is, by definition, high-frequency oscillation. This variant takes a short window
straddling the chunk boundary, rFFTs it, and penalizes only the predicted high-frequency
magnitude that *exceeds* the ground-truth high-frequency magnitude (one-sided, GT-matched).
Low-frequency task motion is preserved (not pushed to zero), so SR is protected while the
boundary tremor spike is flattened. This is a compact spectral summary of the time-domain
finite-difference seam loss and maps directly onto the jerk/SPARC headline metrics.

sscp_spectral_weight = 0 AND sscp_smooth_weight = 0  ->  identical to acm2_sscp_literal.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2_sscp_literal_smooth.configuration_acm2_sscp_literal_smooth import (
    ACM2SSCPLiteralSmoothConfig,
)


@PreTrainedConfig.register_subclass("acm2_sscp_literal_smooth_spectral")
@dataclass
class ACM2SSCPLiteralSmoothSpectralConfig(ACM2SSCPLiteralSmoothConfig):
    # Weight on the spectral seam term. 0 = off. Follows the smooth warmup schedule.
    sscp_spectral_weight: float = 0.0

    # Samples taken on EACH side of the seam; total FFT window = 2 * halfwin.
    sscp_spectral_halfwin: int = 8

    # Penalize frequency bins whose normalized index (bin / (nbins-1)) is >= this fraction —
    # i.e. the high band = tremor. 0.5 targets the upper half of the spectrum. 0 penalizes all
    # AC bins (>= DC). Must be in [0, 1).
    sscp_spectral_high_frac: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if self.sscp_spectral_weight < 0:
            raise ValueError("sscp_spectral_weight must be >= 0.")
        if self.sscp_spectral_halfwin < 2:
            raise ValueError("sscp_spectral_halfwin must be >= 2.")
        if not (0.0 <= self.sscp_spectral_high_frac < 1.0):
            raise ValueError("sscp_spectral_high_frac must be in [0, 1).")
