"""S6 — Config for ACM2 literal carry + linear state-blend handoff (v22 rank 4).

Structural: instead of hard-injecting the carried ssm_state at each boundary, blend it toward
the fresh/memoryless state (zero-init) by a factor alpha:  S_init = alpha * S_carry.
alpha = 1 reproduces the literal carry; alpha < 1 softens the boundary shock when the open-loop
carried state disagrees with the fresh observation. Because Mamba's recurrence is *linear*, this
state interpolation is physically meaningful (unlike a nonlinear RNN, where blending hidden
states is ill-defined). alpha can be a fixed scalar or a per-layer learnable parameter, letting
the model find the continuity/reactivity sweet spot. Applies at BOTH training and inference
(the carry is routed through the model forward in both).

sscp_blend_alpha = 1.0 (non-learnable)  ->  identical to acm2_sscp_literal_smooth.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2_sscp_literal_smooth.configuration_acm2_sscp_literal_smooth import (
    ACM2SSCPLiteralSmoothConfig,
)


@PreTrainedConfig.register_subclass("acm2_sscp_literal_smooth_blend")
@dataclass
class ACM2SSCPLiteralSmoothBlendConfig(ACM2SSCPLiteralSmoothConfig):
    # Blend factor in [0, 1]. 1 = full literal carry; 0 = memoryless restart (== acm2).
    # When learnable, this is the init value.
    sscp_blend_alpha: float = 1.0

    # If True, alpha is a per-decoder-layer learnable parameter (sigmoid-parameterized),
    # initialized from sscp_blend_alpha; otherwise it is the fixed scalar above.
    sscp_blend_learnable: bool = False

    # If True, also scale the carried depthwise-conv state by alpha (default: ssm_state only).
    sscp_blend_conv: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not (0.0 <= self.sscp_blend_alpha <= 1.0):
            raise ValueError("sscp_blend_alpha must be in [0, 1].")
