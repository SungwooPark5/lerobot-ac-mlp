#!/usr/bin/env python
"""ACM2 reactive — receding-horizon state-refined re-planning (cor x BiMamba, v13).

The MODEL (ACT encoder + BiMamba decoder + carry + obs-driven gate) is byte-identical to
acm2_sscp_literal_bimamba — `_build_model` is NOT changed — so v12 BiMamba checkpoints load
with zero key mismatch. v13 adds only:

  (1) an inference-time reactive loop (reset()/select_action), and
  (2) an optional reactive-consistency training term (forward), which trains the gate to be
      invoked every step instead of once per chunk (train==inference). See v13/PLAN_v13.md.

Two-timescale inference (PLAN_v13 §1.2):
  fast channel — re-plan every `refine_period` executed steps from fresh obs (gate re-corrects
                 the carry toward o_t; BiMamba refines the plan). The carry is NOT committed.
  slow channel — commit the carried recurrent state once per `commit_period` (= chunk length)
                 so the recurrent stream advances at the true execution rate.

  refine_period >= chunk length (or reactive_refine=False) reproduces open-loop chunking —
  i.e. it is byte-identical to acm2_sscp_literal_bimamba (the clean control).
"""
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2_sscp_literal_bimamba.configuration_acm2_sscp_literal_bimamba import (
    ACM2SSCPLiteralBiMambaConfig,
)


@PreTrainedConfig.register_subclass("acm2_reactive")
@dataclass
class ACM2ReactiveConfig(ACM2SSCPLiteralBiMambaConfig):
    # ── Reactive inference (receding-horizon) ──────────────────────────────────
    # False → open-loop chunking (== parent bimamba policy, the control).
    reactive_refine: bool = True
    # Re-plan every `refine_period` EXECUTED steps (fast channel). 1 = re-plan every step
    # (max reactivity). >= chunk length → open-loop (re-plan only at chunk boundaries).
    refine_period: int = 1
    # Carry-commit cadence (slow channel), in executed steps. None/0 → n_action_steps
    # (chunk length), which matches chunk-pair training. Smaller = faster persistent memory
    # (run-ahead risk, see PLAN_v13 §1.2) — keep default unless a memory task needs it.
    commit_period: int | None = None

    # ── Reactive-consistency training (Phase R; needs ChunkPairDataset(reactive_offsets>0)) ──
    # When True AND the batch carries mid-chunk offset keys (action_o{δ}, obs_*_o{δ}, ...),
    # forward() adds a term that supervises re-plans decoded from the carried state at those
    # offsets — i.e. the gate is trained for per-step invocation. Falls back to the parent
    # chunk-pair loss (gate-once) when offsets are absent, so it is safe to leave on.
    reactive_train: bool = False
    # Number of mid-chunk offsets supervised per sample (the dataset emits this many).
    reactive_train_offsets: int = 2
    # Weight on the reactive-consistency term relative to the base chunk-pair loss.
    reactive_consistency_weight: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.refine_period < 1:
            raise ValueError(f"refine_period must be >= 1, got {self.refine_period}.")
        if self.commit_period is not None and self.commit_period < 1:
            raise ValueError(f"commit_period must be >= 1 or None, got {self.commit_period}.")
        if self.reactive_train_offsets < 0:
            raise ValueError(f"reactive_train_offsets must be >= 0, got {self.reactive_train_offsets}.")
