"""Configuration for ACM (Mamba-1) + literal selective-scan state carryover (SSCP),
with optional boundary-time carry fusion (the v8 carry family), ported from
acm2_sscp_literal.

Unlike a summary-token carry (which carries the previous chunk's terminal decoder
output and re-derives state by scanning it), this policy carries Mamba-1's **actual
recurrent state** (the per-layer ``ssm_state``, shape (B, d_inner, N)) plus the
depthwise-conv state across chunk boundaries. Mamba-1's selective-scan kernel does not
expose an ``initial_states`` argument, so the initial state is injected exactly in
closed form: because Mamba-1's A is diagonal, the homogeneous contribution of an
initial state h_0 is ``exp(A * cumsum(delta)) * h_0``, added on top of the zero-init
scan (see ``mamba1_stateful_forward``). The scan literally continues from where the
previous chunk ended. At training the carried state is detached at the chunk boundary
(truncated BPTT) when ``sscp_detach=True``.

``carry_fusion`` selects how the carried state is treated at each chunk boundary:

  "none"  — literal handoff, no modification            (m1_lit,  MTIL-style)
  "ema"   — fixed-coefficient EMA across boundaries,
            gradient-free, no observation               (m1_ema,  ReMem-VLA-style)
  "mlp"   — learned projection of the carried state,
            no observation                              (m1_mlp,  AVA-VLA-style)
  "gated" — PEC gate: h' = (1-G(h,e_obs)) . h + G . S(e_obs)
            learned, observation-driven correction      (m1_cor,  proposed)

Base backbone = this branch's ACM (Transformer encoder -> Mamba-1 decoder, with the
optional native BiMamba / action self-attention). Default parameters keep this policy
byte-identical to plain acm (sscp_enabled=False, carry_fusion="none").
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm.configuration_acm import ACMConfig

CARRY_FUSION_MODES = ("none", "ema", "mlp", "gated")
CARRY_GATE_MODES = ("reset", "replace", "residual")


@PreTrainedConfig.register_subclass("acm_sscp_literal")
@dataclass
class ACMSSCPLiteralConfig(ACMConfig):
    """ACM (Mamba-1) + literal recurrent state carryover with optional carry fusion."""

    # ── SSCP core (not present on the plain ACM base) ───────────────────────────
    # Master switch. When False the carry path is never used and the policy matches
    # plain acm exactly (checkpoint-compatible, no extra behavior).
    sscp_enabled: bool = False
    # Probability of running the chunk-continuation (paired) training path when
    # `action_n1` labels are present. 0.0 disables paired training (single-chunk only).
    sscp_p_carry: float = 0.5
    # Detach the carried state at the chunk boundary (truncated BPTT).
    sscp_detach: bool = True

    # ── Carry fusion at chunk boundaries (v8) ──────────────────────────────────
    # One of CARRY_FUSION_MODES. "none" reproduces the literal policy exactly
    # (no extra parameters, checkpoint-compatible).
    carry_fusion: str = "none"

    # "ema": c_n = beta * h_n + (1 - beta) * c_{n-1}, with c_{-1} = 0.
    # Fixed coefficient, never trained (faithful to ReMem-VLA's gradient-free EMA).
    carry_ema_beta: float = 0.9

    # "gated": hidden width of the gate MLP G([pool(h); e_obs]).
    carry_fusion_hidden: int = 128

    # "gated": initial bias of the gate logits. sigma(-4) ~= 0.018, so training
    # starts as (almost) literal carry and the gate opens only if useful.
    carry_gate_bias_init: float = -4.0

    # ── v9: carry-divergence augmentation (teach the boundary fusion to correct) ──
    # At training, with probability carry_noise_p, perturb the detached carried
    # ssm_state by Gaussian noise of per-sample scale ~ U(0, carry_noise_std) *
    # std(ssm_state). Manufactures the carry<->observation disagreement the obs-driven
    # gate must learn to correct. 0.0 = no augmentation.
    carry_noise_std: float = 0.0
    carry_noise_p: float = 0.5

    # "gated" sub-mode (v10 ablation axis). The decoder ALWAYS scans the fresh
    # observation (encoder_out) with the carry as the initial state, so:
    #   "reset"    h' = (1-G).h              — gate the carry toward 0 (=ACT).
    #   "replace"  h' = (1-G).h + G.S(e_obs) — Kalman-style correct toward an obs target.
    #   "residual" h' = h + G.D(e_obs)       — additive obs correction (never discards h).
    carry_gate_mode: str = "replace"

    def __post_init__(self):
        super().__post_init__()
        if self.carry_fusion not in CARRY_FUSION_MODES:
            raise ValueError(
                f"carry_fusion must be one of {CARRY_FUSION_MODES}, got '{self.carry_fusion}'."
            )
        if self.carry_gate_mode not in CARRY_GATE_MODES:
            raise ValueError(
                f"carry_gate_mode must be one of {CARRY_GATE_MODES}, got '{self.carry_gate_mode}'."
            )
