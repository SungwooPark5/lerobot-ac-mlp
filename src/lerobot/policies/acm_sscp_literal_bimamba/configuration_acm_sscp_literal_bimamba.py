"""Integrated policy config: native bidirectional Mamba-1 (BiMamba) chunk decoder
+ literal recurrent state carryover (SSCP-literal). Port of acm2_sscp_literal_bimamba.

This is a thin preset over ACMSSCPLiteralConfig that turns the native BiMamba decoder
ON by default (use_bimamba_decoder=True). The model / carry logic is entirely shared with
acm_sscp_literal — the only difference is this default and the registered policy name — so
acm_sscp_literal (carry only) vs acm_sscp_literal_bimamba (BiMamba + carry) form a clean
ablation pair.

Mechanism recap: the decoder scans [encoder_out, queries] both forward and backward.
  - forward scan  : carries the per-layer (conv, ssm) recurrent state across chunk boundaries.
  - backward scan : stateless, intra-chunk refinement (flip -> scan -> flip, fuse 0.5*(f+b)).
So BiMamba and the cross-chunk carry are orthogonal.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm_sscp_literal.configuration_acm_sscp_literal import ACMSSCPLiteralConfig


@PreTrainedConfig.register_subclass("acm_sscp_literal_bimamba")
@dataclass
class ACMSSCPLiteralBiMambaConfig(ACMSSCPLiteralConfig):
    """ACMSSCPLiteral with the native BiMamba decoder enabled by default."""

    use_bimamba_decoder: bool = True
