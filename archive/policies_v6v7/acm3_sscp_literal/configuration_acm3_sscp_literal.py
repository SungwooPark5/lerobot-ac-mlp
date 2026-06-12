"""Configuration for ACM3 + literal SSM state carryover, with optional
boundary-time carry fusion (the v8 carry family).

Unlike acm3_sscp (which carries a *summary token* — the previous chunk's terminal
decoder output — and re-derives state by scanning it), this policy carries Mamba3's
**actual recurrent state tuple** (angle, ssm, k, v) across chunk boundaries via the
kernel's `Input_States` / `return_final_states` path. The scan literally continues
from where the previous chunk ended. At training, the carried state is detached at
the chunk boundary (truncated BPTT) when sscp_detach=True.

`carry_fusion` selects how the carried state is treated at each chunk boundary:

  "none"  — literal handoff, no modification            (m3_lit,  MTIL-style)
  "ema"   — fixed-coefficient EMA across boundaries,
            gradient-free, no observation               (m3_ema,  ReMem-VLA-style)
  "mlp"   — learned projection of the carried state,
            no observation                              (m3_mlp,  AVA-VLA-style)
  "gated" — PEC gate: h' = (1-G(h,e_obs)) ⊙ h + G ⊙ S(e_obs)
            learned, observation-driven correction      (m3_cor,  proposed)

The carry-method spectrum for the paper is completed by acm3 (zero carry, reset
every chunk) and acm3_sscp (summary-token warm-up).
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm3_sscp.configuration_acm3_sscp import ACM3SSCPConfig

CARRY_FUSION_MODES = ("none", "ema", "mlp", "gated")


@PreTrainedConfig.register_subclass("acm3_sscp_literal")
@dataclass
class ACM3SSCPLiteralConfig(ACM3SSCPConfig):
    """ACM3 + literal Mamba3 hidden-state carryover with optional carry fusion.

    Inherits sscp_enabled / sscp_p_carry / sscp_detach. `sscp_carry_position` is
    unused here (literal carry continues the full stream; there is no token to place).
    """

    # ── Carry fusion at chunk boundaries (v8) ──────────────────────────────────
    # One of CARRY_FUSION_MODES. "none" reproduces the original literal policy
    # exactly (no extra parameters, checkpoint-compatible).
    carry_fusion: str = "none"

    # "ema": c_n = beta * h_n + (1 - beta) * c_{n-1}, with c_{-1} = 0.
    # Fixed coefficient, never trained (faithful to ReMem-VLA's gradient-free EMA).
    carry_ema_beta: float = 0.9

    # "gated": hidden width of the gate MLP G([pool(h); e_obs]).
    carry_fusion_hidden: int = 128

    # "gated": initial bias of the gate logits. sigma(-4) ~= 0.018, so training
    # starts as (almost) literal carry and the gate opens only if useful.
    carry_gate_bias_init: float = -4.0

    def __post_init__(self):
        super().__post_init__()
        if self.carry_fusion not in CARRY_FUSION_MODES:
            raise ValueError(
                f"carry_fusion must be one of {CARRY_FUSION_MODES}, got '{self.carry_fusion}'."
            )
