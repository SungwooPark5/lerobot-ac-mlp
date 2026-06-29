#!/usr/bin/env python
"""ACM2 smooth — Smooth Chunking via BiMamba + Boundary State Carry (v14).

Base = `acm2` (ACT Transformer encoder + Mamba-2 decoder, ZERO carry — clean).
v14 adds two ORTHOGONAL smoothing axes, with NO disturbance machinery (no obs-driven
carry gate, no carry noise, no carry-fusion modes — those were v12/v13 reactive/외란 기제):

  ① BiMamba (chunk_decoder=bidir): bidirectional Mamba scan over the chunk → smooth + accurate
     chunk *interior*.  [은지님]
  ② Pure boundary state carry: hand off the Mamba forward-scan recurrent state across chunk
     boundaries (detach only — no gate, no noise) + boundary-continuity (C1) + jerk training
     loss → smooth chunk *boundaries*.  [나]

carry is purely "stitch the boundary smoothly", not "correct for disturbance".
"""
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.acm2.configuration_acm2 import ACM2Config

CHUNK_DECODER_MODES = ("causal", "bidir")
BIMAMBA_FUSE_MODES = ("avg", "proj")
SMOOTH_JERK_MODES = ("l1", "l2")


@PreTrainedConfig.register_subclass("acm2_smooth")
@dataclass
class ACM2SmoothConfig(ACM2Config):
    # ── ① BiMamba (청크 내부, 은지님) ───────────────────────────────────────────
    chunk_decoder: str = "bidir"          # "causal" == acm2 디코더 / "bidir" == BiMamba
    bimamba_fuse: str = "avg"             # forward/backward 융합: avg | proj(학습)
    bimamba_share_weights: bool = True    # True: backward가 forward layer 재사용(param-matched)

    # ── ② 순수 경계 carry (청크 경계, 나) — 게이트/노이즈 없음 ──────────────────
    carry_enabled: bool = True            # False → acm2 zero-carry (carry ablation)
    carry_p: float = 0.5                  # chunk-pair 학습 확률 (boundary loss용 연속 청크쌍)
    carry_detach: bool = True             # 경계에서 carry detach (truncated BPTT)
    # 학습 carry 깊이. 2 → chunk-pair(=carry 1 hop). 추론은 한 에피소드에서 carry를
    # 에피소드 길이/K 만큼(≈4) 누적하므로, 2로 학습하면 청크 3+가 분포 밖(OOD)으로 drift.
    # M>2 → M개 연속 청크를 hop마다 detach-carry로 롤아웃해 추론의 carry 깊이 분포를 학습.
    # (M=ep_len/K 권장; ALOHA ~400스텝/K100 → 4. K가 커서 M개 청크가 에피소드를 넘으면
    #  ReactiveSegmentDataset가 유효 세그먼트를 못 만들 수 있으니 데이터 길이에 맞게.)
    carry_train_chunks: int = 2

    # ── smoothness 손실 (boundary-continuity + jerk) ─────────────────────────────
    smooth_boundary_weight: float = 0.1   # 경계 위치(+속도) 매칭. 0 → off
    smooth_boundary_velocity: bool = True # True: 위치＋속도(C¹) / False: 위치만(C⁰)
    smooth_jerk_weight: float = 0.02      # 청크 내 jerk(2차미분) 억제. 0 → off
    smooth_jerk_mode: str = "l1"          # l1 | l2
    smooth_jerk_excess_only: bool = True  # GT 청크의 jerk 초과분만 (시연보다 과하게 안 함)

    def __post_init__(self):
        super().__post_init__()
        if self.chunk_decoder not in CHUNK_DECODER_MODES:
            raise ValueError(f"chunk_decoder must be one of {CHUNK_DECODER_MODES}, got '{self.chunk_decoder}'.")
        if self.bimamba_fuse not in BIMAMBA_FUSE_MODES:
            raise ValueError(f"bimamba_fuse must be one of {BIMAMBA_FUSE_MODES}, got '{self.bimamba_fuse}'.")
        if self.smooth_jerk_mode not in SMOOTH_JERK_MODES:
            raise ValueError(f"smooth_jerk_mode must be one of {SMOOTH_JERK_MODES}, got '{self.smooth_jerk_mode}'.")
        if self.smooth_boundary_weight < 0 or self.smooth_jerk_weight < 0:
            raise ValueError("smooth_boundary_weight / smooth_jerk_weight must be >= 0.")
        if self.carry_train_chunks < 2:
            raise ValueError("carry_train_chunks must be >= 2 (2 == chunk-pair).")
