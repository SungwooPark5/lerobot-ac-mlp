"""ACM3 + SSCP + Bidirectional Mamba3 (BiMamba E1), no ICPE.

Integrates two contributions:
  - SSCP carry (ours): a summary token from the previous chunk warms up the SSM.
  - BiMamba (eunji's): the decoder runs a forward AND a backward Mamba3 stack and
    averages them, so each action token sees both past and future chunk context.

The unidirectional Mamba3SSCPDecoder is replaced by Mamba3BiMambaSSCPDecoder:
  1. Forward pass:  process [encoder_out, carry, queries] left-to-right
  2. Backward pass: flip [encoder_out, queries], prepend carry, process, flip back
  3. Average:       0.5 * (forward + backward)

The carry token is included in BOTH directions so both forward and backward SSMs
warm up from previous-chunk context instead of h=0. The last K positions are
extracted after averaging.

All SSCP machinery (carry tracking, chunk-pair training) is inherited from
ACM3SSCPPolicy unchanged — the model interface (forward signature, return shapes)
is identical to the unidirectional version.
"""

from itertools import chain

import torch
from torch import Tensor, nn

try:
    from mamba_ssm import Mamba3
    HAS_MAMBA3 = True
except ImportError:
    HAS_MAMBA3 = False

from lerobot.policies.acm3_sscp.modeling_acm3_sscp import ACM3SSCP, ACM3SSCPPolicy
from lerobot.policies.acm3_sscp_bimamba.configuration_acm3_sscp_bimamba import ACM3SSCPBiMambaConfig


class Mamba3BiMambaSSCPDecoder(nn.Module):
    """BiMamba decoder with carry support — drop-in for Mamba3SSCPDecoder.

    Same forward signature (x, encoder_out, carry) and output shape as the
    unidirectional decoder. The carry token (B, 1, D) is prepended to both the
    forward and backward sequences so both SSMs warm up from previous-chunk
    context rather than h=0.
    """

    def __init__(self, config: ACM3SSCPBiMambaConfig):
        super().__init__()
        if not HAS_MAMBA3:
            raise ImportError(
                "mamba_ssm is required. Install with:\n"
                "  MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall "
                "git+https://github.com/state-spaces/mamba.git --no-build-isolation"
            )

        # offset keeps forward/backward layer_idx disjoint: fwd uses 0..N-1,
        # bwd uses N..2N-1 — avoids overlapping Mamba3 internal cache/RoPE state.
        def _make_layers(offset: int = 0):
            return nn.ModuleList([
                Mamba3(
                    d_model=config.dim_model,
                    d_state=config.mamba3_d_state,
                    expand=config.mamba3_expand,
                    headdim=config.mamba3_headdim,
                    ngroups=config.mamba3_ngroups,
                    rope_fraction=config.mamba3_rope_fraction,
                    is_outproj_norm=config.mamba3_is_outproj_norm,
                    is_mimo=config.mamba3_is_mimo,
                    mimo_rank=config.mamba3_mimo_rank,
                    chunk_size=config.mamba3_chunk_size,
                    layer_idx=i + offset,
                    n_layer=2 * config.n_decoder_layers,
                )
                for i in range(config.n_decoder_layers)
            ])

        self.forward_layers = _make_layers(0)
        self.backward_layers = _make_layers(config.n_decoder_layers)
        self.norm = nn.LayerNorm(config.dim_model)
        # "pre_query": carry adjacent to the queries in the forward pass (recommended);
        # "prefix": carry at the very front. In BOTH cases the backward pass places
        # carry immediately before the (reversed) queries so it warms them up — the
        # backward SSM processes queries first, so there is no "prefix" alternative.
        self.carry_position = getattr(config, "sscp_carry_position", "pre_query")

    def forward(
        self,
        x: Tensor,                    # (K, B, D) — decoder queries with pos_embed
        encoder_out: Tensor,          # (T, B, D) — encoder context
        carry: Tensor | None = None,  # (B, 1, D) — carry from previous chunk
    ) -> Tensor:                      # returns (K, B, D)
        x = x.transpose(0, 1)                       # (B, K, D)
        encoder_out = encoder_out.transpose(0, 1)   # (B, T, D)

        T = encoder_out.shape[1]
        base = torch.cat([encoder_out, x], dim=1)   # (B, T+K, D)

        # Forward direction.
        if carry is not None:
            if self.carry_position == "prefix":
                fwd = torch.cat([carry, base], dim=1)            # [carry, enc, x]
            else:  # pre_query: [enc, carry, x]
                fwd = torch.cat([encoder_out, carry, x], dim=1)
        else:
            fwd = base
        for layer in self.forward_layers:
            fwd = layer(fwd)
        if carry is not None:
            if self.carry_position == "prefix":
                fwd = fwd[:, 1:, :]                               # drop carry@0
            else:
                fwd = torch.cat([fwd[:, :T, :], fwd[:, T + 1:, :]], dim=1)  # drop carry@T

        # Backward direction: flip base, then prepend carry so the backward SSM
        # warms up from carry immediately before the (reversed) queries. Flipping a
        # carry-prefixed sequence would push carry to the end and skip warm-up.
        bwd = base.flip(dims=[1])
        bwd = torch.cat([carry, bwd], dim=1) if carry is not None else bwd
        for layer in self.backward_layers:
            bwd = layer(bwd)
        if carry is not None:
            bwd = bwd[:, 1:, :]  # drop carry slot before flipping back
        bwd = bwd.flip(dims=[1])

        # Average — maintains activation scale vs unidirectional baseline.
        combined = 0.5 * (fwd + bwd)

        K = x.shape[1]
        out = self.norm(combined[:, -K:, :])
        return out.transpose(0, 1)  # (K, B, D)


class ACM3SSCPBiMamba(ACM3SSCP):
    """ACM3SSCP with BiMamba decoder. Inherits _encode(), forward(), etc.

    Only changes:
      - self.decoder is Mamba3BiMambaSSCPDecoder (forward + backward stacks)
      - _reset_parameters excludes both layer stacks from Xavier init
    """

    def __init__(self, config: ACM3SSCPBiMambaConfig):
        super().__init__(config)
        # Replace unidirectional decoder with BiMamba decoder.
        self.decoder = Mamba3BiMambaSSCPDecoder(config)
        self._reset_parameters()

    def _reset_parameters(self):
        # Handle two layer stacks instead of single self.decoder.layers.
        if not hasattr(self.decoder, "forward_layers"):
            # Called during super().__init__() before decoder is replaced — skip.
            return
        mamba_ids = (
            {id(p) for p in self.decoder.forward_layers.parameters()} |
            {id(p) for p in self.decoder.backward_layers.parameters()}
        )
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1 and id(p) not in mamba_ids:
                nn.init.xavier_uniform_(p)


class ACM3SSCPBiMambaPolicy(ACM3SSCPPolicy):
    """SSCP policy with BiMamba decoder (no ICPE).

    Inherits all carry tracking and chunk-pair training logic from
    ACM3SSCPPolicy. Only self.model is replaced.
    """

    config_class = ACM3SSCPBiMambaConfig
    name = "acm3_sscp_bimamba"

    def __init__(self, config: ACM3SSCPBiMambaConfig):
        super().__init__(config)
        # Replace model created by parent (ACM3SSCP) with BiMamba version.
        self.model = ACM3SSCPBiMamba(config)
