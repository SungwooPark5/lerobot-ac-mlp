"""ACM3 + ICPE + LITERAL SSM State Carryover (literal counterpart of P1cc).

Reuses the verified literal-state-handoff decoder (Mamba3LiteralSSCPDecoder) and the
state-carrying policy wrapper (ACM3SSCPLiteralPolicy) from acm3_sscp_literal; the only
addition is ICPE phase signal on the decoder queries (inherited from ACM3ICPE).
"""

import torch
from torch import Tensor

from lerobot.policies.acm3_icpe.modeling_acm3_icpe import ACM3ICPE, make_icpe_signal
from lerobot.policies.acm3_sscp_literal.modeling_acm3_sscp_literal import (
    ACM3SSCPLiteralPolicy,
    Mamba3LiteralSSCPDecoder,
)
from lerobot.policies.acm3_icpe_sscp_literal.configuration_acm3_icpe_sscp_literal import (
    ACM3ICPESSCPLiteralConfig,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES


class ACM3ICPESSCPLiteral(ACM3ICPE):
    """ACM3ICPE with literal-state-handoff decoder.

    Inherits _encode / ICPE projection / VAE from ACM3ICPE; swaps the decoder and
    returns the per-layer recurrent state for cross-chunk carry.
    """

    def __init__(self, config: ACM3ICPESSCPLiteralConfig):
        super().__init__(config)
        self.decoder = Mamba3LiteralSSCPDecoder(config)

    def forward(self, batch: dict[str, Tensor], carry: list | None = None,
                return_state: bool = False) -> tuple:
        if self.config.use_vae and self.training:
            assert ACTION in batch, "Need action labels for VAE training."
        bs = (batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch
              else batch[OBS_ENV_STATE].shape[0])
        encoder_out, mu, log_sigma_x2 = self._encode(batch, bs)

        K = self.config.chunk_size
        dtype = encoder_out.dtype
        device = encoder_out.device
        pos_emb = self.decoder_pos_embed.weight.unsqueeze(1)
        phase = make_icpe_signal(K, self.config.icpe_mode, device, dtype)
        icpe_emb = self.icpe_proj(phase).unsqueeze(1)
        decoder_in = pos_emb + icpe_emb  # (K, 1, D)

        decoder_out, new_states = self.decoder(
            decoder_in.expand(K, bs, self.config.dim_model), encoder_out, carry=carry)
        actions = self.action_head(decoder_out.transpose(0, 1))
        if return_state:
            return actions, (mu, log_sigma_x2), new_states
        return actions, (mu, log_sigma_x2)


class ACM3ICPESSCPLiteralPolicy(ACM3SSCPLiteralPolicy):
    """ICPE + literal SSCP. Reuses all state-carry logic; only the model differs."""

    config_class = ACM3ICPESSCPLiteralConfig
    name = "acm3_icpe_sscp_literal"

    def _build_model(self, config):
        return ACM3ICPESSCPLiteral(config)
