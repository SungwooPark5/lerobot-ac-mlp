"""Configuration for eunji's ACM2 + literal Mamba-2 SSD state carryover (SSCP),
with optional boundary-time carry fusion (the v8 carry family).

Unlike a summary-token carry (which carries the previous chunk's terminal decoder
output and re-derives state by scanning it), this policy carries Mamba-2's **actual
SSD recurrent state** (the per-layer ``ssm_state``, shape (B, H, P, N)) plus the
depthwise-conv state across chunk boundaries, via the SSD kernel's ``initial_states`` /
``return_final_states`` path. The scan literally continues from where the previous
chunk ended. At training the carried state is detached at the chunk boundary
(truncated BPTT) when ``sscp_detach=True``.

``carry_fusion`` selects how the carried state is treated at each chunk boundary:

  "none"  — literal handoff, no modification            (m2_lit,  MTIL-style)
  "ema"   — fixed-coefficient EMA across boundaries,
            gradient-free, no observation               (m2_ema,  ReMem-VLA-style)
  "mlp"   — learned projection of the carried state,
            no observation                              (m2_mlp,  AVA-VLA-style)
  "gated" — PEC gate: h' = (1-G(h,e_obs)) . h + G . S(e_obs)
            learned, observation-driven correction      (m2_cor,  proposed)

Base backbone = eunji's ACM2 (Transformer encoder -> Mamba-2 SSD decoder, with the
optional native BiMamba / action self-attention). Default parameters keep this policy
byte-identical to plain acm2 (sscp_enabled=False, carry_fusion="none").
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2.configuration_acm2 import ACM2Config

CARRY_FUSION_MODES = ("none", "ema", "mlp", "gated")
CARRY_GATE_MODES = ("reset", "replace", "residual")

# How the two BiMamba stacks are combined. "mean" is what the decoder has always
# done, so it is the default and leaves existing runs untouched.
BIMAMBA_FUSE_MODES = ("mean", "gate", "concat", "sum")

# What order the second stack scans in. "reverse" is the original behaviour.
BIMAMBA_SCAN_MODES = ("reverse", "same", "random", "backward_only")


@PreTrainedConfig.register_subclass("acm2_sscp_literal")
@dataclass
class ACM2SSCPLiteralConfig(ACM2Config):
    """eunji ACM2 + literal Mamba-2 SSD state carryover with optional carry fusion."""

    # ── SSCP core (not present on the plain ACM2 base) ──────────────────────────
    # Master switch. When False the carry path is never used and the policy matches
    # plain acm2 exactly (checkpoint-compatible, no extra behavior).
    sscp_enabled: bool = False
    # Probability of running the chunk-continuation (paired) training path when
    # `action_n1` labels are present. 0.0 disables paired training (single-chunk only).
    sscp_p_carry: float = 0.5
    # Detach the carried state at the chunk boundary (truncated BPTT).
    sscp_detach: bool = True

    # ── BiMamba structure ablation (only read when use_bimamba_decoder=True) ───
    # How the forward and backward stacks are combined.
    #   "mean"    out = 0.5 * (fwd + bwd)          <- original
    #   "gate"    out = g*fwd + (1-g)*bwd, g = sigmoid(scalar) init 0.5, so it starts
    #             exactly at "mean" and learns the balance. g is the readout: if it
    #             drifts to 1 the backward stack is not being used.
    #   "concat"  out = W [fwd ; bwd]              <- learned per-channel mix,
    #             adds a 2D->D Linear; can express asymmetric and cross-channel
    #             combinations that a scalar weighting cannot.
    #   "sum"     out = fwd + bwd. Kept only so the name resolves -- it is NOT a
    #             separate model. A LayerNorm follows the fuse (and action
    #             self-attention is off by default), and LayerNorm cancels scale, so
    #             sum trains to the same function as mean. Do not sweep it.
    # Defaults to "mean", so runs that do not set it are unchanged.
    bimamba_fuse: str = "mean"

    # What order the second stack scans in.
    #   "reverse"        the whole chunk reversed          <- original
    #   "same"           the second stack scans FORWARD too. The capacity control:
    #                    BiMamba runs two Mamba stacks where the unidirectional
    #                    decoder runs one, so "bimamba beats plain" could just be 2x
    #                    the decoder parameters. This cell has the same parameter
    #                    count as "reverse" and differs only in that the second scan
    #                    is not reversed, which is the only comparison that isolates
    #                    direction from capacity.
    #   "backward_only"  no forward stack at all; the reversed scan is the model.
    #                    Requires sscp_enabled=False -- the carry is produced by the
    #                    forward scan, so there is nothing to hand to the next chunk.
    #   "random"         a fixed random permutation instead of the reversal, to
    #                    separate "bidirectional" from "some second ordering".
    #                    The permutation is derived deterministically from
    #                    bimamba_scan_seed and the sequence length, so it is the same
    #                    every forward and at eval, and needs no stored state.
    bimamba_scan: str = "reverse"
    bimamba_scan_seed: int = 0

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

        if self.bimamba_fuse not in BIMAMBA_FUSE_MODES:
            raise ValueError(
                f"bimamba_fuse must be one of {BIMAMBA_FUSE_MODES}, got '{self.bimamba_fuse}'."
            )

        if self.bimamba_scan not in BIMAMBA_SCAN_MODES:
            raise ValueError(
                f"bimamba_scan must be one of {BIMAMBA_SCAN_MODES}, got '{self.bimamba_scan}'."
            )

        # backward_only drops the forward stack, and the carry is exactly that stack's
        # final state. Silently emitting an empty carry would look like a working run
        # whose chunks never actually connect, so refuse the combination outright.
        if self.bimamba_scan == "backward_only" and self.sscp_enabled:
            raise ValueError(
                "bimamba_scan='backward_only' removes the forward stack, which is where "
                "the carried state comes from. Set sscp_enabled=False."
            )
