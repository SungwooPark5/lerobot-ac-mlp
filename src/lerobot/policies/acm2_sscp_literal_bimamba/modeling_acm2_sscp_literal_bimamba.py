#!/usr/bin/env python
"""Bidirectional Mamba-2 chunk decoder on top of ACM2 SSCP-literal (cor + BiMamba).

Only the decoder changes: a backward (reversed-sequence) Mamba-2 scan is added and
fused with the existing forward scan. The forward scan still produces the per-layer
recurrent state that is carried across chunk boundaries (PEC carry / cor), so the carry
mechanism is untouched — the backward scan is an intra-chunk, non-causal refinement of
the already-planned action block.

Recipe = Vim-style external bidirectional Mamba (forward + flip-backward + fuse),
re-implemented on Mamba-2 to match the cor backbone. (Recipe origin: the ecd-eunji
`acm_bimamba` policy, which used Mamba-1 inside an ACT-style decoder.)

Setting chunk_decoder="causal" makes this policy byte-identical in behaviour to
acm2_sscp_literal — i.e. it reproduces cor/lit and serves as the clean control.
"""
import torch
from torch import Tensor, nn

from lerobot.policies.acm2_sscp_literal.modeling_acm2_sscp_literal import (
    ACM2SSCPLiteral,
    ACM2SSCPLiteralPolicy,
    Mamba2LiteralSSCPDecoder,
    mamba2_stateful_forward,
)

from .configuration_acm2_sscp_literal_bimamba import ACM2SSCPLiteralBiMambaConfig


class Mamba2LiteralSSCPBiDecoder(Mamba2LiteralSSCPDecoder):
    """Mamba2LiteralSSCPDecoder + optional bidirectional (forward+backward) scan.

    chunk_decoder == "causal": behaves exactly like the parent (forward-only).
    chunk_decoder == "bidir":  out = fuse(forward_scan, flip(backward_scan(flip(.)))).
    Cross-chunk carry (return_state) ALWAYS comes from the forward scan only.
    """

    def __init__(self, config: ACM2SSCPLiteralBiMambaConfig):
        super().__init__(config)
        self.chunk_decoder = getattr(config, "chunk_decoder", "causal")
        self.share_weights = getattr(config, "bimamba_share_weights", True)
        self.fuse = getattr(config, "bimamba_fuse", "avg")
        self.backward_layers = None
        self.fuse_proj = None
        if self.chunk_decoder == "bidir":
            if not self.share_weights:
                # Separate backward stack — distinct layer_idx to avoid inference-cache collisions.
                from lerobot.policies.acm2_sscp_literal.modeling_acm2_sscp_literal import Mamba2
                n = config.n_decoder_layers
                self.backward_layers = nn.ModuleList([
                    Mamba2(
                        d_model=config.dim_model,
                        d_state=config.mamba2_d_state,
                        d_conv=config.mamba2_d_conv,
                        expand=config.mamba2_expand,
                        headdim=config.mamba2_headdim,
                        chunk_size=config.mamba2_chunk_size,
                        layer_idx=n + i,
                    )
                    for i in range(n)
                ])
            if self.fuse == "proj":
                self.fuse_proj = nn.Linear(2 * config.dim_model, config.dim_model)

    def forward(self, x: Tensor, encoder_out: Tensor, carry: list | None = None):
        if self.chunk_decoder != "bidir":
            return super().forward(x, encoder_out, carry=carry)

        x = x.transpose(0, 1)                       # (B, K, D)
        encoder_out = encoder_out.transpose(0, 1)   # (B, T, D)
        h = torch.cat([encoder_out, x], dim=1)      # (B, T+K, D)

        # Forward (causal) scan — carries per-layer recurrent state across chunks (== cor).
        hf = h
        new_states = []
        for i, layer in enumerate(self.layers):
            init = carry[i] if carry is not None else None
            hf, st = mamba2_stateful_forward(layer, hf, initial_state=init, return_state=True)
            new_states.append(st)

        # Backward (non-causal) scan — intra-chunk only; NO cross-chunk carry (initial_state=None).
        back_layers = self.layers if self.share_weights else self.backward_layers
        hb = h.flip([1])
        for layer in back_layers:
            hb, _ = mamba2_stateful_forward(layer, hb, initial_state=None, return_state=True)
        hb = hb.flip([1])

        if self.fuse == "proj":
            h_out = self.fuse_proj(torch.cat([hf, hb], dim=-1))
        else:
            h_out = 0.5 * (hf + hb)                  # scale-preserving fuse (Vim-style)

        K = x.shape[1]
        out = self.norm(h_out[:, -K:, :])
        return out.transpose(0, 1), new_states       # carry = forward states ONLY


class ACM2SSCPLiteralBiMamba(ACM2SSCPLiteral):
    """ACM2SSCPLiteral with the decoder replaced by the bidirectional variant."""

    def __init__(self, config: ACM2SSCPLiteralBiMambaConfig):
        super().__init__(config)
        self.decoder = Mamba2LiteralSSCPBiDecoder(config)


class ACM2SSCPLiteralBiMambaPolicy(ACM2SSCPLiteralPolicy):
    """cor (PEC carry-gate, upstream) + BiMamba (bidirectional chunk decoder, downstream)."""

    config_class = ACM2SSCPLiteralBiMambaConfig
    name = "acm2_sscp_literal_bimamba"

    def _build_model(self, config):
        return ACM2SSCPLiteralBiMamba(config)
