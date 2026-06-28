#!/usr/bin/env python
"""ACM2 reactive — receding-horizon state-refined re-planning (cor x BiMamba, v13).

Model == acm2_sscp_literal_bimamba (ACT encoder + BiMamba decoder + carry + obs-driven gate);
`_build_model` is unchanged so the architecture is identical. v13 changes ONLY the control
flow:

  inference (reset/select_action): receding-horizon reactive loop — re-plan every
      `refine_period` executed steps from fresh obs (gate re-corrects the carry toward o_t,
      BiMamba refines the plan), commit the carry once per chunk.

  training (forward, reactive_train=True): the SAME loop, teacher-forced over a short segment
      of M consecutive chunks. The carry is committed at chunk boundaries; at sampled mid-chunk
      times we re-decode from that committed carry with the fresh (expert) observation and
      supervise the executed head plan[:exec_len] against the ground-truth actions. This trains
      the gate to be invoked EVERY step (train == inference), which open-loop chunk training
      never does. Uses ReactiveSegmentDataset (NOT ChunkPairDataset).

See v13/PLAN_v13.md. There is NO zero-shot path: reactive only exists as a trained model.
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
    # Re-plan every `refine_period` EXECUTED steps (fast channel). 1 = re-plan every step.
    # >= chunk length → re-plan only at chunk boundaries == open-loop chunking (the control).
    refine_period: int = 1
    # False → open-loop chunking (== parent bimamba policy). Used for the open-loop control.
    reactive_refine: bool = True
    # Carry-commit cadence (slow channel), executed steps. None/0 → n_action_steps (chunk length).
    commit_period: int | None = None

    # ── Reactive-consistency training (Phase R) — uses ReactiveSegmentDataset ────
    reactive_train: bool = False
    # Segment length in consecutive chunks (carry is committed at each chunk boundary).
    # >=2 so the gate sees a non-trivial carried state at chunk 2+ (chunk 1's carry is None).
    reactive_train_chunks: int = 2
    # Mid-chunk re-plan samples PER chunk (besides the boundary). These re-decode from the
    # committed carry with a fresh obs and supervise the executed head — this is what teaches
    # per-step invocation. Total decodes/segment = chunks * (1 + samples).
    reactive_train_samples: int = 3
    # Supervised length of a mid-chunk re-plan head (the part actually executed before the next
    # re-plan). Reflects the executed stride; small (≈ refine_period .. a few steps).
    reactive_exec_len: int = 8
    # Weight on the mid-chunk reactive head loss relative to the boundary (open-loop) chunk loss.
    reactive_consistency_weight: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.refine_period < 1:
            raise ValueError(f"refine_period must be >= 1, got {self.refine_period}.")
        if self.commit_period is not None and self.commit_period < 1:
            raise ValueError(f"commit_period must be >= 1 or None, got {self.commit_period}.")
        if self.reactive_train_chunks < 1:
            raise ValueError(f"reactive_train_chunks must be >= 1, got {self.reactive_train_chunks}.")
        if self.reactive_train_samples < 0:
            raise ValueError(f"reactive_train_samples must be >= 0, got {self.reactive_train_samples}.")
        if self.reactive_exec_len < 1:
            raise ValueError(f"reactive_exec_len must be >= 1, got {self.reactive_exec_len}.")
