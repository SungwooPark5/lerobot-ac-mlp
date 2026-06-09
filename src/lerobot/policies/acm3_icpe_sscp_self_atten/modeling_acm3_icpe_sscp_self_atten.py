"""ACM3 + ICPE + SSCP + Post-Mamba Self-Attention on action tokens (C-series + E2)

After Mamba3 decodes the K action tokens, MultiheadAttention lets them attend
to each other across the chunk dimension:

    attn_out = MHA(decoder_out, decoder_out, decoder_out)
    decoder_out = LayerNorm(decoder_out + tanh(γ) * attn_out)

γ is a learned scalar initialised at gamma_init ≈ 1e-4 so:
  tanh(1e-4) ≈ 1e-4 — gate starts near-closed, opening gradually as γ grows.

SSCP carry is extracted from the post-attention representation so it encodes
refined cross-token context rather than raw Mamba output.

All SSCP machinery (carry tracking, chunk-pair training) is inherited unchanged
from ACM3ICPESSCPPolicy.
"""

from itertools import chain

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.acm3_icpe.modeling_acm3_icpe import ACM3ICPE, make_icpe_signal
from lerobot.policies.acm3_icpe_sscp.modeling_acm3_icpe_sscp import ACM3ICPESSCPPolicy
from lerobot.policies.acm3_icpe_sscp_self_atten.configuration_acm3_icpe_sscp_self_atten import (
    ACM3ICPESSCPSelfAttenConfig,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACM3ICPESelfAtten(ACM3ICPE):
    """ACM3ICPE with post-Mamba self-attention on action tokens.

    Inherits all encoder / VAE / ICPE logic.  Overrides forward to insert
    the gated self-attention step after Mamba3 decoding.
    """

    def __init__(self, config: ACM3ICPESSCPSelfAttenConfig):
        super().__init__(config)
        # Gated self-attention on K action tokens (Pre-LN → attn → dropout)
        self.action_self_attn_norm = nn.LayerNorm(config.dim_model)
        self.action_self_attn = nn.MultiheadAttention(
            config.dim_model, config.self_atten_nhead,
            dropout=config.dropout, batch_first=False,
        )
        self.action_self_attn_dropout = nn.Dropout(config.dropout)
        self.gamma = nn.Parameter(torch.full((1,), config.self_atten_gamma_init))

        # Xavier init for the new attention projections
        nn.init.xavier_uniform_(self.action_self_attn.in_proj_weight)
        if self.action_self_attn.out_proj.weight.dim() > 1:
            nn.init.xavier_uniform_(self.action_self_attn.out_proj.weight)

    def forward(
        self,
        batch: dict[str, Tensor],
        carry: Tensor | None = None,
        return_decoder_out: bool = False,
    ) -> tuple:
        if self.config.use_vae and self.training:
            assert ACTION in batch

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        encoder_out, mu, log_sigma_x2 = self._encode(batch, batch_size)

        K = self.config.chunk_size
        dtype = encoder_out.dtype
        device = encoder_out.device

        pos_emb = self.decoder_pos_embed.weight.unsqueeze(1)
        phase = make_icpe_signal(K, self.config.icpe_mode, device, dtype)
        icpe_emb = self.icpe_proj(phase).unsqueeze(1)
        decoder_in = pos_emb + icpe_emb

        # Mamba3 decoder
        decoder_out = self.decoder(
            decoder_in.expand(K, batch_size, self.config.dim_model),
            encoder_out,
            carry=carry,
        )  # (K, B, D)

        # Post-Mamba self-attention: Pre-LN → attn → dropout → tanh-gated residual
        residual = decoder_out
        normed = self.action_self_attn_norm(decoder_out)
        attn_out, _ = self.action_self_attn(normed, normed, normed)
        attn_out = self.action_self_attn_dropout(attn_out)
        decoder_out = residual + torch.tanh(self.gamma) * attn_out

        actions = self.action_head(decoder_out.transpose(0, 1))

        if return_decoder_out:
            return actions, (mu, log_sigma_x2), decoder_out
        return actions, (mu, log_sigma_x2)


class ACM3ICPESSCPSelfAttenPolicy(ACM3ICPESSCPPolicy):
    """SSCP policy with post-Mamba action token self-attention.

    Inherits all carry tracking and chunk-pair training logic from
    ACM3ICPESSCPPolicy.  Only self.model is replaced.
    """

    config_class = ACM3ICPESSCPSelfAttenConfig
    name = "acm3_icpe_sscp_self_atten"

    def __init__(self, config: ACM3ICPESSCPSelfAttenConfig):
        super().__init__(config)
        # Replace model created by parent (ACM3ICPE) with self-attention version
        self.model = ACM3ICPESelfAtten(config)
