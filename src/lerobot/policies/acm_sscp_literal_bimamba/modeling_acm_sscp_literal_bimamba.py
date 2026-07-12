"""Integrated policy modeling: native BiMamba (Mamba-1) decoder + literal state carry.

The model and carry logic are entirely shared with acm_sscp_literal. The ACM decoder
already branches on use_bimamba_decoder, and Mamba1LiteralSSCPDecoder threads the literal
carry through the forward scan while the backward scan refines the chunk statelessly. So this
policy is just a thin subclass that swaps config_class / name; setting use_bimamba_decoder=True
(the config default here) is all that distinguishes it from acm_sscp_literal.
"""

from lerobot.policies.acm_sscp_literal.modeling_acm_sscp_literal import (
    ACMSSCPLiteral,
    ACMSSCPLiteralPolicy,
)

from .configuration_acm_sscp_literal_bimamba import ACMSSCPLiteralBiMambaConfig


class ACMSSCPLiteralBiMambaPolicy(ACMSSCPLiteralPolicy):
    """cor/lit (literal carry) + native BiMamba (bidirectional chunk decoder), Mamba-1."""

    config_class = ACMSSCPLiteralBiMambaConfig
    name = "acm_sscp_literal_bimamba"

    def _build_model(self, config):
        return ACMSSCPLiteral(config)
