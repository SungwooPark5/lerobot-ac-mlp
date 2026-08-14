"""ACM2 + ICPE + SSCP + Action-only BiMamba.

Action-only BiMamba:
  Forward stream:
      [encoder_out, carry?, action_queries]

  Backward stream:
      [encoder_out, carry?, reversed_action_queries]

  After backward Mamba2, only the action part is flipped back.
  Encoder tokens are not reversed.

This mirrors ACM3's bimamba_flip_mode='action_only', but uses Mamba2.
"""

from itertools import chain

import torch
from torch import Tensor, nn

try:
    from mamba_ssm import Mamba2
    HAS_MAMBA2 = True
except ImportError:
    HAS_MAMBA2 = False

from lerobot.policies.acm2_bimamba_action_icpe_sscp.configuration_acm2_bimamba_action_icpe_sscp import (
    ACM2BiMambaActionICPESSCPConfig,
)
from lerobot.policies.acm2_icpe.modeling_acm2_icpe import ACM2ICPE
from lerobot.policies.acm2_icpe_sscp.modeling_acm2_icpe_sscp import ACM2ICPESSCPPolicy


class Mamba2BiMambaActionICPEDecoder(nn.Module):
    """BiMamba decoder with action-only flip and carry support.

    x already contains decoder_pos_embed + ICPE from ACM2ICPE.forward().
    """

    def __init__(self, config: ACM2BiMambaActionICPESSCPConfig):
        super().__init__()

        if not HAS_MAMBA2:
            raise ImportError(
                "mamba-ssm >= 2.0 is required for Mamba2. "
                "Install with: pip install mamba-ssm>=2.0"
            )

        def make_layer(layer_idx: int):
            return Mamba2(
                d_model=config.dim_model,
                d_state=config.mamba2_d_state,
                d_conv=config.mamba2_d_conv,
                expand=config.mamba2_expand,
                headdim=config.mamba2_headdim,
                chunk_size=config.mamba2_chunk_size,
                layer_idx=layer_idx,
            )

        self.forward_layers = nn.ModuleList(
            [make_layer(i) for i in range(config.n_decoder_layers)]
        )
        self.backward_layers = nn.ModuleList(
            [make_layer(i + config.n_decoder_layers) for i in range(config.n_decoder_layers)]
        )

        self.norm = nn.LayerNorm(config.dim_model)
        self.carry_position = getattr(config, "sscp_carry_position", "pre_query")

    def _make_combined(
        self,
        encoder_out: Tensor,  # (B, T, D)
        x: Tensor,            # (B, K, D)
        carry: Tensor | None,
    ) -> Tensor:
        if carry is None:
            return torch.cat([encoder_out, x], dim=1)

        if self.carry_position == "prefix":
            return torch.cat([carry, encoder_out, x], dim=1)

        # default: pre_query
        return torch.cat([encoder_out, carry, x], dim=1)

    def forward(
        self,
        x: Tensor,                       # (K, B, D)
        encoder_out: Tensor,             # (T, B, D)
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
        carry: Tensor | None = None,     # (B, 1, D)
    ) -> Tensor:
        if decoder_pos_embed is not None:
            x = x + decoder_pos_embed
        if encoder_pos_embed is not None:
            encoder_out = encoder_out + encoder_pos_embed

        x = x.transpose(0, 1)                      # (B, K, D)
        encoder_out = encoder_out.transpose(0, 1)  # (B, T, D)

        K = x.shape[1]

        # ── Forward: [encoder, carry?, action] ───────────────────────────────
        fwd = self._make_combined(encoder_out, x, carry)
        for layer in self.forward_layers:
            fwd = layer(fwd)

        # ── Backward action-only:
        # encoder order is kept, only action queries are reversed.
        # [encoder, carry?, flip(action)] ─────────────────────────────────────
        x_rev = x.flip(dims=[1])
        bwd = self._make_combined(encoder_out, x_rev, carry)
        for layer in self.backward_layers:
            bwd = layer(bwd)

        # Restore only the action part.
        fwd_action = fwd[:, -K:, :]
        bwd_action = bwd[:, -K:, :].flip(dims=[1])

        out = 0.5 * (fwd_action + bwd_action)
        out = self.norm(out)

        return out.transpose(0, 1)  # (K, B, D)


class ACM2BiMambaActionICPESSCP(ACM2ICPE):
    """ACM2ICPE with action-only BiMamba decoder."""

    def __init__(self, config: ACM2BiMambaActionICPESSCPConfig):
        super().__init__(config)

        # Replace ICPE unidirectional decoder with action-only BiMamba decoder.
        self.decoder = Mamba2BiMambaActionICPEDecoder(config)

        # Re-init non-Mamba transformer/decoder wrapper params safely.
        self._reset_parameters()

    def _reset_parameters(self):
        if not hasattr(self.decoder, "forward_layers"):
            return

        mamba_ids = (
            {id(p) for p in self.decoder.forward_layers.parameters()}
            | {id(p) for p in self.decoder.backward_layers.parameters()}
        )
        icpe_ids = {id(p) for p in self.icpe_proj.parameters()}

        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1 and id(p) not in mamba_ids and id(p) not in icpe_ids:
                nn.init.xavier_uniform_(p)


class ACM2BiMambaActionICPESSCPPolicy(ACM2ICPESSCPPolicy):
    config_class = ACM2BiMambaActionICPESSCPConfig
    name = "acm2_bimamba_action_icpe_sscp"

    def __init__(self, config: ACM2BiMambaActionICPESSCPConfig):
        super().__init__(config)

        self.model = ACM2BiMambaActionICPESSCP(config)
        self.reset()

        print("========== ACM2 ACTION-ONLY BIMAMBA CHECK ==========")
        print("policy class:", self.__class__)
        print("model class:", self.model.__class__)
        print("decoder class:", self.model.decoder.__class__)
        print("has icpe_proj:", hasattr(self.model, "icpe_proj"))
        print("has forward_layers:", hasattr(self.model.decoder, "forward_layers"))
        print("has backward_layers:", hasattr(self.model.decoder, "backward_layers"))
        print("carry_position:", getattr(self.model.decoder, "carry_position", None))
        print("bimamba_flip_mode:", getattr(self.config, "bimamba_flip_mode", None))
        print("====================================================")