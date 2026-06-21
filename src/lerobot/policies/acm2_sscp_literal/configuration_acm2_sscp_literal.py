"""Configuration for ACM2 + literal SSD state carryover, with optional
boundary-time carry fusion (the v8 carry family) — Mamba-2 port of
acm3_sscp_literal.

Unlike acm2_sscp (which carries a *summary token* — the previous chunk's terminal
decoder output — and re-derives state by scanning it), this policy carries Mamba-2's
**actual SSD recurrent state** (the per-layer `ssm_state`, shape (B, H, P, N)) across
chunk boundaries via the SSD kernel's `initial_states` / `return_final_states` path.
The scan literally continues from where the previous chunk ended. At training, the
carried state is detached at the chunk boundary (truncated BPTT) when sscp_detach=True.

`carry_fusion` selects how the carried state is treated at each chunk boundary:

  "none"  — literal handoff, no modification            (m2_lit,  MTIL-style)
  "ema"   — fixed-coefficient EMA across boundaries,
            gradient-free, no observation               (m2_ema,  ReMem-VLA-style)
  "mlp"   — learned projection of the carried state,
            no observation                              (m2_mlp,  AVA-VLA-style)
  "gated" — PEC gate: h' = (1-G(h,e_obs)) ⊙ h + G ⊙ S(e_obs)
            learned, observation-driven correction      (m2_cor,  proposed)

The carry-method spectrum for the paper is completed by acm2 (zero carry, reset
every chunk) and acm2_sscp (summary-token warm-up).
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2_sscp.configuration_acm2_sscp import ACM2SSCPConfig

CARRY_FUSION_MODES = ("none", "ema", "mlp", "gated")
CARRY_GATE_MODES = ("reset", "replace", "residual")


@PreTrainedConfig.register_subclass("acm2_sscp_literal")
@dataclass
class ACM2SSCPLiteralConfig(ACM2SSCPConfig):
    """ACM2 + literal Mamba-2 SSD state carryover with optional carry fusion.

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
    # v9 note: with carry-noise augmentation supplying a learning signal, a less
    # negative bias (e.g. -2.0, sigma ~= 0.12) lets the gate actually move.
    carry_gate_bias_init: float = -4.0

    # ── v9: carry-divergence augmentation (teach the boundary fusion to correct) ──
    # The v8 failure: at training the carried state was always *clean* (single
    # teacher-forced boundary) so the obs-driven gate had no reason to open and stayed
    # shut — but at eval the state drifts over many open-loop boundaries, so the shut
    # gate passed a corrupted state through and SR collapsed. Fix: at training, with
    # probability carry_noise_p, perturb the detached carried ssm_state by Gaussian
    # noise of per-sample scale ~ U(0, carry_noise_std) · std(ssm_state). Combined with
    # fresh chunk-n+1 observations (ChunkPairDataset.load_n1_images), this manufactures
    # the carry↔observation disagreement the gate must learn to correct. Applied to the
    # whole literal family so training is identical and only carry_fusion differs (clean
    # controlled comparison). 0.0 = v8 behavior (no augmentation).
    carry_noise_std: float = 0.0
    carry_noise_p: float = 0.5

    # "gated" sub-mode (v10 ablation axis). The decoder ALWAYS scans the fresh observation
    # (encoder_out) with the carry as the initial state, so:
    #   "reset"    h' = (1-G)·h              — gate the carry toward 0 (=ACT): discard a bad
    #                                           carry and let the fresh obs drive. Robust + parity.
    #   "replace"  h' = (1-G)·h + G·S(e_obs)  — Kalman-style correct toward an obs-derived target.
    #   "residual" h' = h + G·Δ(e_obs)        — additive obs correction (never discards h).
    # Default "replace" keeps v9 (gated=replace) checkpoints loadable; v10 main uses "reset".
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
