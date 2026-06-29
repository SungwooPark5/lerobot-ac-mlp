"""ACM3 + SSCP + BiMamba + boundary-smoothness loss (no ICPE).

Composes two orthogonal extensions of acm3_sscp:
  - the BiMamba decoder (model side)   from acm3_sscp_bimamba
  - the GT-matched seam loss (loss side) from acm3_sscp_smooth

Implementation is trivial because the two are orthogonal:
  - Inherit ACM3SSCPSmoothPolicy → get `_forward_chunk_pair` + `_seam_loss`
    (model-agnostic: they only use self.model's forward / action_head).
  - Swap self.model to ACM3SSCPBiMamba (forward + backward Mamba3 stacks).

sscp_smooth_weight = 0 → identical to acm3_sscp_bimamba.
"""

from lerobot.policies.acm3_sscp_bimamba.modeling_acm3_sscp_bimamba import ACM3SSCPBiMamba
from lerobot.policies.acm3_sscp_bimamba_smooth.configuration_acm3_sscp_bimamba_smooth import (
    ACM3SSCPBiMambaSmoothConfig,
)
from lerobot.policies.acm3_sscp_smooth.modeling_acm3_sscp_smooth import ACM3SSCPSmoothPolicy


class ACM3SSCPBiMambaSmoothPolicy(ACM3SSCPSmoothPolicy):
    """SSCP policy with BiMamba decoder AND the boundary-continuity loss.

    Inherits carry tracking, chunk-pair training, and the seam-smoothness loss
    from ACM3SSCPSmoothPolicy. Only self.model is replaced with the BiMamba model.
    """

    config_class = ACM3SSCPBiMambaSmoothConfig
    name = "acm3_sscp_bimamba_smooth"

    def __init__(self, config: ACM3SSCPBiMambaSmoothConfig):
        super().__init__(config)
        # Parent built self.model = ACM3SSCP(config); swap to the BiMamba model.
        self.model = ACM3SSCPBiMamba(config)
