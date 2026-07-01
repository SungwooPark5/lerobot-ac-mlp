"""S5 — ACM2 literal carry + velocity(delta)-integration head (v22 rank 5).

The model reinterprets the action-head output as per-step increments (velocity) and integrates
them from an anchor:  a[t] = anchor + cumsum(delta)[:t+1].  The anchor is the current
proprioception (OBS_STATE) — natural for ALOHA position control (actions are joint targets ~
current pose) — so a[0] ~= current pose and every chunk starts near-C0 by construction, with
cross-chunk continuity following from continuous proprioception. The carried Mamba state still
shapes the delta trajectory; nothing else changes (VAE latent, seam-loss training, inference all
inherited). If OBS_STATE's dim != action dim (non-ALOHA), the anchor falls back to zero.

Only the model's forward is wrapped. Because the head output is integrated inside forward, both
training (L1 vs absolute GT) and inference see integrated absolute actions transparently.
"""

import torch

from lerobot.policies.acm2_sscp_literal.modeling_acm2_sscp_literal import ACM2SSCPLiteral
from lerobot.policies.acm2_sscp_literal_smooth.modeling_acm2_sscp_literal_smooth import (
    ACM2SSCPLiteralSmoothPolicy,
)
from lerobot.policies.acm2_sscp_literal_smooth_velint.configuration_acm2_sscp_literal_smooth_velint import (
    ACM2SSCPLiteralSmoothVelIntConfig,
)
from lerobot.utils.constants import OBS_STATE


class ACM2SSCPLiteralVelInt(ACM2SSCPLiteral):
    """ACM2SSCPLiteral whose head output is integrated (velocity -> absolute) from an anchor."""

    def _anchor(self, batch, deltas):
        b, _, a = deltas.shape
        if (
            getattr(self.config, "sscp_velint_anchor", "state") == "state"
            and OBS_STATE in batch
            and batch[OBS_STATE].shape[-1] == a
        ):
            return batch[OBS_STATE]  # (B, A)
        return deltas.new_zeros(b, a)

    def forward(self, batch, carry=None, return_state=False):
        res = super().forward(batch, carry=carry, return_state=return_state)
        deltas = res[0]  # head output reinterpreted as per-step increments, (B, K, A)
        actions = self._anchor(batch, deltas).unsqueeze(1) + torch.cumsum(deltas, dim=1)
        return (actions,) + tuple(res[1:])


class ACM2SSCPLiteralSmoothVelIntPolicy(ACM2SSCPLiteralSmoothPolicy):
    """Literal carry + delta-integration head (+ inherited smooth training / inference)."""

    config_class = ACM2SSCPLiteralSmoothVelIntConfig
    name = "acm2_sscp_literal_smooth_velint"

    def _build_model(self, config):
        return ACM2SSCPLiteralVelInt(config)
