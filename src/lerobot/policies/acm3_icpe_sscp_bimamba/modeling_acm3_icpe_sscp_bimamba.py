"""ACM3 + ICPE + SSCP + Bidirectional Mamba3 (C-series + BiMamba E1)

Replaces Mamba3ICPEDecoder with a BiMamba variant that:
  1. Forward pass: process [carry, encoder_out, action_queries] left-to-right
  2. Backward pass: flip same sequence → process → flip back (right-to-left)
  3. Average: 0.5 * (forward + backward)

The carry token is included in BOTH directions so both forward and backward SSMs
warm up from previous-chunk context.  The last K positions are still extracted
after averaging.

All SSCP machinery (carry tracking, chunk-pair training) is inherited from
ACM3ICPESSCPPolicy and works without modification since the model interface
is identical (same forward signature and return shapes).
"""

from itertools import chain

import torch
from torch import Tensor, nn

try:
    from mamba_ssm import Mamba3
    HAS_MAMBA3 = True
except ImportError:
    HAS_MAMBA3 = False

from lerobot.policies.acm3_icpe.modeling_acm3_icpe import ACM3ICPE
from lerobot.policies.acm3_icpe_sscp.modeling_acm3_icpe_sscp import ACM3ICPESSCPPolicy
from lerobot.policies.acm3_icpe_sscp_bimamba.configuration_acm3_icpe_sscp_bimamba import (
    ACM3ICPESSCPBiMambaConfig,
)


class Mamba3BiMambaICPEDecoder(nn.Module):
    """BiMamba decoder with carry support, drop-in for Mamba3ICPEDecoder.

    Accepts the same interface as Mamba3ICPEDecoder (x already has pos+ICPE baked in).
    The carry token (B, 1, D) is prepended to both forward and backward sequences so
    both SSMs warm up from previous-chunk context rather than h=0.
    """

    def __init__(self, config: ACM3ICPESSCPBiMambaConfig):
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

    def forward(
        self,
        x: Tensor,              # (K, B, D) — decoder queries (pos+ICPE already added)
        encoder_out: Tensor,    # (T, B, D) — encoder context (enc_pos already added)
        carry: Tensor | None = None,  # (B, 1, D) — carry from previous chunk
    ) -> Tensor:                # returns (K, B, D)
        x = x.transpose(0, 1)           # (B, K, D)
        encoder_out = encoder_out.transpose(0, 1)  # (B, T, D)

        if carry is not None:
            combined = torch.cat([carry, encoder_out, x], dim=1)  # (B, 1+T+K, D)
        else:
            combined = torch.cat([encoder_out, x], dim=1)          # (B, T+K, D)

        # Forward direction
        fwd = combined
        for layer in self.forward_layers:
            fwd = layer(fwd)

        # Backward direction: same combined (with carry) flipped
        bwd = combined.flip(dims=[1])
        for layer in self.backward_layers:
            bwd = layer(bwd)
        bwd = bwd.flip(dims=[1])

        # Average — maintains activation scale vs unidirectional baseline
        combined = 0.5 * (fwd + bwd)

        K = x.shape[1]
        out = combined[:, -K:, :]
        out = self.norm(out)
        return out.transpose(0, 1)  # (K, B, D)


class ACM3ICPESSCPBiMamba(ACM3ICPE):
    """ACM3ICPE with BiMamba decoder.  Inherits _encode(), forward(), etc.

    Only changes:
      - self.decoder is Mamba3BiMambaICPEDecoder (forward+backward stacks)
      - _reset_parameters excludes both layer stacks from Xavier init
    """

    def __init__(self, config: ACM3ICPESSCPBiMambaConfig):
        super().__init__(config)
        # Replace unidirectional decoder with BiMamba decoder
        self.decoder = Mamba3BiMambaICPEDecoder(config)
        self._reset_parameters()

    def _reset_parameters(self):
        # Handle two layer stacks instead of single self.decoder.layers
        if not hasattr(self.decoder, "forward_layers"):
            # Called during super().__init__() before decoder is replaced — skip
            return
        mamba_ids = (
            {id(p) for p in self.decoder.forward_layers.parameters()} |
            {id(p) for p in self.decoder.backward_layers.parameters()}
        )
        icpe_ids = {id(p) for p in self.icpe_proj.parameters()}
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1 and id(p) not in mamba_ids and id(p) not in icpe_ids:
                nn.init.xavier_uniform_(p)


class ACM3ICPESSCPBiMambaPolicy(ACM3ICPESSCPPolicy):
    """SSCP policy with BiMamba decoder.

    Inherits all carry tracking and chunk-pair training logic from
    ACM3ICPESSCPPolicy.  Only self.model is replaced.
    """

    config_class = ACM3ICPESSCPBiMambaConfig
    name = "acm3_icpe_sscp_bimamba"

    def __init__(self, config: ACM3ICPESSCPBiMambaConfig):
        super().__init__(config)
        # Replace model created by parent (ACM3ICPE) with BiMamba version
        self.model = ACM3ICPESSCPBiMamba(config)
