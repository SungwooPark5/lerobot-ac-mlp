"""Configuration for the RNN/LSTM history baseline.

A NON-SSM recurrent history-encoder baseline: same role as MTIL (carry a recurrent
state over the observation stream, predict an action chunk, no correction) but with a
classic LSTM/GRU instead of a Mamba SSM. This is the control that answers the reviewer
question "history can be done with an RNN — is the SSM actually needed?" — i.e. it
isolates the *SSM* part of our claim from the *recurrence* part.

Shares the same ResNet backbone, dataset, sim, eval and metrics as our other policies
(fair comparison). Fully isolated: imports no other policy module.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("rnn_hist")
@dataclass
class RNNHistConfig(PreTrainedConfig):
    """LSTM/GRU history-encoder baseline (no state correction)."""

    # Input / output structure.
    n_obs_steps: int = 32          # training observation-history window
    chunk_size: int = 100
    n_action_steps: int = 1        # query every step + temporal aggregation

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Vision backbone (shared with ACM3/MTIL for a fair comparison).
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False

    # Token dim + recurrent stack.
    dim_model: int = 512
    rnn_type: str = "lstm"         # "lstm" | "gru"
    rnn_hidden: int = 1024
    n_rnn_layers: int = 4
    rnn_dropout: float = 0.1

    # Inference history (faithful to a recurrent baseline):
    #   True  → unbounded recurrent state carry across the whole episode (LSTM hidden
    #           (h,c) persists, reset only between episodes).
    #   False → bounded sliding-window re-scan of n_obs_steps frames (fallback).
    unbounded_carry: bool = True

    # Inference: ACT-style temporal aggregation of overlapping chunks.
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
        if self.rnn_type not in ("lstm", "gru"):
            raise ValueError(f"rnn_type must be 'lstm' or 'gru'. Got {self.rnn_type}.")
        if self.n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be >= 1. Got {self.n_obs_steps}.")
        if self.temporal_ensemble_coeff is not None and self.n_action_steps != 1:
            raise ValueError("Temporal aggregation (query every step) needs n_action_steps == 1.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("RNN baseline needs at least one image or the environment state.")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
