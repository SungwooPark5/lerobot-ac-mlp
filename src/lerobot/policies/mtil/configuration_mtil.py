"""Configuration for the MTIL baseline (faithful reproduction).

MTIL = "Encoding Full History with Mamba for Temporal Imitation Learning"
(arXiv:2505.12410, RA-L 2025; code github.com/yulinzhouZYL/MTIL).

This is an EXTERNAL BASELINE reproduced inside our lerobot framework so it shares
the exact dataset, eval harness, tasks and metrics as our ACM3 carry-spectrum
models — the only difference is the temporal model. See PROVENANCE.md for what is
faithful vs adapted.

Faithful to MTIL's method:
  • Mamba-2 (NOT Mamba-3) recurrent history encoder over the observation stream.
  • Per-timestep state advance: at inference the conv/ssm state is carried across
    timesteps for the whole episode (reset only between episodes), query EVERY step.
  • Predicts a `chunk_size`-step action chunk from the current recurrent state.
  • NO correction/gating/fusion of the state with new observations — pure carryover.
  • ACT-style temporal aggregation of overlapping chunk predictions at inference.

Adapted for our framework (documented honestly in PROVENANCE.md):
  • Vision: same ResNet backbone as our ACM3 models (fair comparison) instead of
    MTIL's DINOv2 features — isolates the temporal model as the only difference.
  • Training: bounded observation-history window of `n_obs_steps` frames (MTIL trains
    on full-length sequences). Set n_obs_steps large to reduce the train/inference gap.
  • Deterministic regression head (MTIL has no CVAE), so use_vae is forced False.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("mtil")
@dataclass
class MTILConfig(PreTrainedConfig):
    """MTIL baseline config (Mamba-2 history encoder, no correction)."""

    # Input / output structure.
    # n_obs_steps > 1: the bounded observation-history window fed to the Mamba-2
    # encoder at training time (the recurrent history MTIL relies on).
    n_obs_steps: int = 16
    chunk_size: int = 100
    n_action_steps: int = 1   # MTIL queries every step (+ temporal aggregation)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Vision backbone (shared with ACM3 for a fair comparison).
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False

    # Token / model dimension.
    # Scaled up (768 wide, 9-deep below) so total params ~param-match our 48M ACM3
    # models. MTIL has no Transformer encoder / CVAE, so its Mamba stack must be
    # wider+deeper to reach the same capacity — this removes the "baseline is smaller"
    # confound. (MTIL's own paper uses an even larger d_model=2048 stack.)
    # Smoketest-measured: 9 layers → 46.53M, ~3.4% under the m3 cluster (47.98–48.19M)
    # and the closest integer-layer match (10 layers would overshoot to ~50M).
    # Accepted: the exact capacity control is the internal m3_* spectrum; MTIL is an
    # external baseline reported with its true 46.53M param count.
    dim_model: int = 768

    # Mamba-2 history encoder (mamba_ssm.Mamba2). dim_model*expand must be divisible
    # by headdim: 768*2=1536 = 24*64 ✓.
    mamba2_d_state: int = 128
    mamba2_d_conv: int = 4
    mamba2_expand: int = 2
    mamba2_headdim: int = 64
    n_mamba_layers: int = 9

    # Inference: ACT-style temporal aggregation across overlapping chunk predictions.
    temporal_ensemble_coeff: float | None = 0.01

    # Training / loss.
    dropout: float = 0.1
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"`vision_backbone` must be a ResNet variant. Got {self.vision_backbone}.")
        if self.n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be >= 1. Got {self.n_obs_steps}.")
        # MTIL queries every step; temporal aggregation needs n_action_steps == 1.
        if self.temporal_ensemble_coeff is not None and self.n_action_steps != 1:
            raise ValueError(
                "MTIL uses temporal aggregation (query every step) → n_action_steps must be 1."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})."
            )
        ed = self.dim_model * self.mamba2_expand
        if ed % self.mamba2_headdim != 0:
            raise ValueError(
                f"dim_model*mamba2_expand ({ed}) must be divisible by mamba2_headdim "
                f"({self.mamba2_headdim})."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("MTIL needs at least one image or the environment state.")

    @property
    def observation_delta_indices(self) -> list:
        # Past (n_obs_steps - 1) frames + current frame: indices [-(n_obs_steps-1) .. 0].
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
