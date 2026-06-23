#!/usr/bin/env python
"""ACM2 SSCP-literal + bidirectional Mamba-2 chunk decoder (cor x BiMamba).

Adds an orthogonal `chunk_decoder` axis to ACM2SSCPLiteral:
  "causal" : forward-only SSD scan == acm2_sscp_literal (the control; reproduces cor/lit).
  "bidir"  : forward + reversed (flipped) Mamba-2 scan over [encoder_out, queries],
             fused (avg). A non-causal refinement of the *already-planned* action block.

The cross-chunk carry (PEC / cor) is taken from the FORWARD scan only, so carry
semantics are unchanged; the backward scan refines the current chunk's output without
participating in the boundary handoff. carry (upstream, causal) x bidir (downstream,
non-causal) are orthogonal axes.
"""
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2_sscp_literal.configuration_acm2_sscp_literal import (
    ACM2SSCPLiteralConfig,
)

CHUNK_DECODER_MODES = ("causal", "bidir")
BIMAMBA_FUSE_MODES = ("avg", "proj")


@PreTrainedConfig.register_subclass("acm2_sscp_literal_bimamba")
@dataclass
class ACM2SSCPLiteralBiMambaConfig(ACM2SSCPLiteralConfig):
    # "causal" == acm2_sscp_literal (control, reproduces cor/lit). "bidir" == BiMamba refinement.
    chunk_decoder: str = "causal"
    # True: reuse the forward Mamba-2 layers for the backward scan (param-matched to causal,
    # Hydra-style weight tying). False: a separate backward stack (~2x decoder params).
    bimamba_share_weights: bool = True
    # Forward/backward fusion: "avg" = 0.5*(f+b) (scale-preserving), "proj" = Linear(concat[f,b]).
    bimamba_fuse: str = "avg"

    def __post_init__(self):
        super().__post_init__()
        if self.chunk_decoder not in CHUNK_DECODER_MODES:
            raise ValueError(
                f"chunk_decoder must be one of {CHUNK_DECODER_MODES}, got '{self.chunk_decoder}'."
            )
        if self.bimamba_fuse not in BIMAMBA_FUSE_MODES:
            raise ValueError(
                f"bimamba_fuse must be one of {BIMAMBA_FUSE_MODES}, got '{self.bimamba_fuse}'."
            )
