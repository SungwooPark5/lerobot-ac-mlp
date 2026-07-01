"""Integrated policy modeling: eunji native BiMamba decoder + literal SSD state carry.

The model and carry logic are entirely shared with acm2_sscp_literal. eunji's ACM2 decoder
already branches on use_bimamba_decoder, and Mamba2LiteralSSCPDecoder threads the literal
carry through the forward scan while the backward scan refines the chunk statelessly. So this
policy is just a thin subclass that swaps config_class / name; setting use_bimamba_decoder=True
(the config default here) is all that distinguishes it from acm2_sscp_literal.
"""

from lerobot.policies.acm2_sscp_literal.modeling_acm2_sscp_literal import (
    ACM2SSCPLiteral,
    ACM2SSCPLiteralPolicy,
)

from .configuration_acm2_sscp_literal_bimamba import ACM2SSCPLiteralBiMambaConfig


class ACM2SSCPLiteralBiMambaPolicy(ACM2SSCPLiteralPolicy):
    """cor/lit (literal carry) + eunji native BiMamba (bidirectional chunk decoder)."""

    config_class = ACM2SSCPLiteralBiMambaConfig
    name = "acm2_sscp_literal_bimamba"

    def _build_model(self, config):
        return ACM2SSCPLiteral(config)
