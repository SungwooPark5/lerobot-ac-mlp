"""S6 — ACM2 literal carry + linear state-blend handoff (v22 rank 4).

The model scales the carried ssm_state by a blend factor alpha before it enters the decoder:
S_init = alpha * S_carry. alpha = 1 reproduces the literal carry; alpha < 1 interpolates toward
the fresh/memoryless (zero-init) state, softening the boundary shock when the open-loop carried
state disagrees with the current observation. Valid precisely because Mamba's recurrence is
linear. alpha is either a fixed scalar or a per-decoder-layer learnable parameter. Applies at BOTH
training and inference (the carry is routed through model.forward in both paths).

Only the model's forward is wrapped; the rest (encode, decoder, carry_fusion, seam-loss training,
inference) is inherited unchanged from ACM2SSCPLiteral / ACM2SSCPLiteralSmoothPolicy.
"""

import math

import torch
from torch import nn

from lerobot.policies.acm2_sscp_literal.modeling_acm2_sscp_literal import ACM2SSCPLiteral
from lerobot.policies.acm2_sscp_literal_smooth.modeling_acm2_sscp_literal_smooth import (
    ACM2SSCPLiteralSmoothPolicy,
)
from lerobot.policies.acm2_sscp_literal_smooth_blend.configuration_acm2_sscp_literal_smooth_blend import (
    ACM2SSCPLiteralSmoothBlendConfig,
)


class ACM2SSCPLiteralBlend(ACM2SSCPLiteral):
    """ACM2SSCPLiteral whose forward blends the carried state by alpha before decoding."""

    def __init__(self, config: ACM2SSCPLiteralSmoothBlendConfig):
        super().__init__(config)  # runs _reset_parameters BEFORE blend_logit exists -> safe
        self._blend_conv = bool(getattr(config, "sscp_blend_conv", False))
        if getattr(config, "sscp_blend_learnable", False):
            a = float(min(max(config.sscp_blend_alpha, 1e-3), 1.0 - 1e-3))
            self.blend_logit = nn.Parameter(
                torch.full((config.n_decoder_layers,), math.log(a / (1.0 - a)))
            )
            self._blend_fixed = None
        else:
            self.blend_logit = None
            self._blend_fixed = float(config.sscp_blend_alpha)

    def _alpha(self, i: int, dtype):
        if self.blend_logit is not None:
            return torch.sigmoid(self.blend_logit[i]).to(dtype)
        return self._blend_fixed

    def _blend_carry(self, carry):
        if carry is None:
            return None
        out = []
        for i, (conv, ssm) in enumerate(carry):
            a = self._alpha(i, ssm.dtype)
            conv_b = a * conv if self._blend_conv else conv
            out.append((conv_b, a * ssm))
        return out

    def forward(self, batch, carry=None, return_state=False):
        return super().forward(batch, carry=self._blend_carry(carry), return_state=return_state)


class ACM2SSCPLiteralSmoothBlendPolicy(ACM2SSCPLiteralSmoothPolicy):
    """Literal carry + alpha-blended state handoff (+ inherited smooth training / inference)."""

    config_class = ACM2SSCPLiteralSmoothBlendConfig
    name = "acm2_sscp_literal_smooth_blend"

    def _build_model(self, config):
        return ACM2SSCPLiteralBlend(config)
