from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm3.configuration_acm3 import ACM3Config


@PreTrainedConfig.register_subclass("acm3_sscp")
@dataclass
class ACM3SSCPConfig(ACM3Config):
    """ACM-3 with SSM State Carryover Protocol (SSCP).

    Extends ACM3Config with inference-time action blending at chunk boundaries.
    Reduces boundary jitter by smoothly transitioning between consecutive chunks.

    Stage 1 검증: boundary jitter 40.5%↓, spike ratio 2.10x→1.25x, SR 80% 유지.

    SSCP blending formula:
        a_sscp[0..blend_steps] = (1 - α(t)) * a_raw[t] + α(t) * a_prev_tail
        α(t) = sscp_blend_alpha * (1 - t / sscp_blend_steps)   # linear decay
    """

    # SSCP-specific fields
    sscp_blend_alpha: float = 0.4   # boundary blend 가중치 (α at t=0)
    sscp_blend_steps: int = 5       # α를 0으로 감쇠시키는 step 수
