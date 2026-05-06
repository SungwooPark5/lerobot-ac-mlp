#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("acm2")
@dataclass
class ACM2Config(PreTrainedConfig):
    """Configuration class for the Action Chunking Mamba-2 (ACM-2) policy.

    ACM-2 replaces the Mamba-1 selective scan decoder with Mamba-2's SSD (State Space Duality) layer,
    which provides improved computational efficiency through its structured matrix decomposition.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
        dim_model: The main hidden dimension.
        n_heads: The number of heads for the transformer encoder and Mamba-2 SSD.
        dim_feedforward: The dimension for the transformer encoder's feed-forward layers.
        n_encoder_layers: The number of transformer layers for the encoder.
        n_decoder_layers: The number of Mamba-2 SSD layers for the decoder.
        mamba2_d_state: SSM state expansion factor for Mamba-2.
        mamba2_d_conv: Local convolution width for Mamba-2.
        mamba2_expand: Block expansion factor for Mamba-2.
        mamba2_headdim: Head dimension for the SSD multi-head structure. dim_model * expand must be
            divisible by headdim.
        use_temporal_weighting: Whether to apply temporal weighting to the loss.
        temporal_execution_weight: Weight mass allocated to the execution region of the chunk.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer encoder layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1

    # Mamba-2 (SSD) decoder configuration.
    mamba2_d_state: int = 64
    mamba2_d_conv: int = 4
    mamba2_expand: int = 2
    mamba2_headdim: int = 64

    # Configuration for temporal weighting.
    use_temporal_weighting: bool = False
    temporal_execution_weight: float = 0.9

    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 3e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        # Validate Mamba-2 dimension constraints.
        expanded_dim = self.dim_model * self.mamba2_expand
        if expanded_dim % self.mamba2_headdim != 0:
            raise ValueError(
                f"dim_model * mamba2_expand ({expanded_dim}) must be divisible by "
                f"mamba2_headdim ({self.mamba2_headdim}). Got remainder {expanded_dim % self.mamba2_headdim}."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
