#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ACM2-DRO — Disturbance-Robust Observer action chunking.

"Train as a chunker, act as a controller": the Mamba-2 decoder state is used as an
online observer. At inference each control step feeds the newly measured proprioception
into the decoder sequence (interleaved with the action query tokens), so a disturbance
is reflected in the very next action instead of after up to `chunk_size` stale steps.

Layers (each toggleable — the ablation axes of the paper):
  A dro_stream       interleaved proprio streaming (this file's core)
  B dro_innovation   next-proprio prediction head; innovation e = o − ô is fed back as
                     an extra input and ‖e‖ gates a fresh-vision context refresh
  C dro_train_*      disturbance-injection training on the teacher-forced proprio
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2.configuration_acm2 import ACM2Config

DRO_VISION_MODES = ("chunk", "gate", "every")


@PreTrainedConfig.register_subclass("acm2_dro")
@dataclass
class ACM2DROConfig(ACM2Config):
    """ACM2 + DRO streaming observer. With every dro_* flag off this is exactly acm2."""

    # ── Layer A: streaming ──────────────────────────────────────────────────────
    dro_stream: bool = True
    # When vision context is refreshed during a chunk:
    #   chunk  only at chunk boundaries (cheapest; proprio still streams every step)
    #   gate   additionally on innovation-gate triggers (needs dro_innovation)
    #   every  every step (full closed-loop upper bound; backbone cost per step)
    dro_vision_refresh: str = "chunk"

    # ── Layer B: innovation observer ────────────────────────────────────────────
    dro_innovation: bool = False
    dro_obs_loss_weight: float = 0.5   # aux L1 weight for next-proprio prediction
    dro_gate_tau: float = 0.0          # ‖e‖ (normalized units) triggering vision refresh; 0=off

    # ── Layer C: disturbance-injection training (on normalized teacher proprio) ──
    dro_train_state_noise: float = 0.0  # gaussian std added to future proprio inputs
    dro_train_push_prob: float = 0.0    # per-sample prob of a persistent mid-chunk offset
    dro_train_push_mag: float = 0.3     # offset L2 norm (× U[0.5,1.5]), normalized units

    def __post_init__(self):
        super().__post_init__()
        if self.dro_vision_refresh not in DRO_VISION_MODES:
            raise ValueError(f"dro_vision_refresh={self.dro_vision_refresh!r} not in {DRO_VISION_MODES}")
        if self.dro_stream:
            if self.use_bimamba_decoder:
                raise ValueError("dro_stream needs a causal (forward-only) decoder; BiMamba is not.")
            if self.use_action_self_attention:
                raise ValueError("dro_stream is incompatible with whole-chunk action self-attention.")
            if self.temporal_ensemble_coeff is not None:
                raise ValueError("dro_stream replaces temporal ensembling; disable it.")
        if self.dro_vision_refresh == "gate" and not (self.dro_innovation and self.dro_gate_tau > 0):
            raise ValueError("dro_vision_refresh='gate' needs dro_innovation=true and dro_gate_tau>0.")
        if self.dro_gate_tau > 0 and not self.dro_innovation:
            raise ValueError("dro_gate_tau>0 needs dro_innovation=true (gate reads the innovation).")

    @property
    def state_delta_indices(self) -> list | None:
        """Future proprio frames for teacher-forced streaming (OBS_STATE only — images
        stay single-frame; see resolve_delta_timestamps)."""
        return list(range(self.chunk_size)) if self.dro_stream else None
