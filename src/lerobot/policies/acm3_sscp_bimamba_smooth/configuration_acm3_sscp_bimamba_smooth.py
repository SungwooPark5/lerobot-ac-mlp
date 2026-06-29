"""Configuration for ACM3 + SSCP + BiMamba + boundary-smoothness loss.

Triple combination:
  - SSCP carry (ours)                 — chunk-boundary state carry
  - BiMamba decoder (eunji's)         — bidirectional Mamba3 (forward + backward)
  - GT-matched boundary continuity loss — explicit seam smoothness (no ICPE)

Subclasses ACM3SSCPSmoothConfig, so it inherits `sscp_smooth_weight` (and all SSCP
hyperparameters). BiMamba needs no extra config fields (it reuses the same
mamba3_* / n_decoder_layers). The decoder topology change (uni → BiMamba) lives in
the modeling code.

sscp_smooth_weight = 0.0 → behaves like acm3_sscp_bimamba (no seam loss).
sscp_smooth_weight > 0.0 → BiMamba model + boundary-continuity loss (CC training).
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm3_sscp_smooth.configuration_acm3_sscp_smooth import ACM3SSCPSmoothConfig


@PreTrainedConfig.register_subclass("acm3_sscp_bimamba_smooth")
@dataclass
class ACM3SSCPBiMambaSmoothConfig(ACM3SSCPSmoothConfig):
    """ACM3 + SSCP + BiMamba decoder + GT-matched boundary continuity loss (no ICPE)."""

    pass
