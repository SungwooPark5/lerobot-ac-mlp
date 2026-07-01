"""Integrated policy config: eunji's native bidirectional Mamba-2 (BiMamba) chunk decoder
+ literal SSD state carryover (SSCP-literal).

This is a thin preset over ACM2SSCPLiteralConfig that turns eunji's native BiMamba decoder
ON by default (use_bimamba_decoder=True). The model / carry logic is entirely shared with
acm2_sscp_literal — the only difference is this default and the registered policy name — so
acm2_sscp_literal (carry only) vs acm2_sscp_literal_bimamba (BiMamba + carry) form a clean
ablation pair.

Mechanism recap: the decoder scans [encoder_out, queries] both forward and backward.
  - forward scan  : carries the per-layer (conv, ssm) recurrent state across chunk boundaries.
  - backward scan : stateless, intra-chunk refinement (flip -> scan -> flip, fuse 0.5*(f+b)).
So BiMamba and the cross-chunk carry are orthogonal.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2_sscp_literal.configuration_acm2_sscp_literal import ACM2SSCPLiteralConfig


@PreTrainedConfig.register_subclass("acm2_sscp_literal_bimamba")
@dataclass
class ACM2SSCPLiteralBiMambaConfig(ACM2SSCPLiteralConfig):
    """ACM2SSCPLiteral with eunji's native BiMamba decoder enabled by default."""

    use_bimamba_decoder: bool = True
