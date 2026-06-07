"""ChunkPairDataset — consecutive chunk pair wrapper for SSCP chunk-continuation training.

For each valid frame index i, returns:
  - all keys from chunk_n  (dataset[i])     ← standard batch
  - action_n1              (dataset[i + K]) ← chunk n+1 target (normalized)
  - action_is_pad_n1
  - obs_env_state_n1       (if env_state present, normalized)
  - obs_state_n1           (if robot state present, normalized)

Image keys with _n1 suffix are NOT included; the policy _forward_chunk_pair
falls back to chunk_n images automatically.

Validity: frame i is valid iff i + chunk_size < episode_end_index,
ensuring chunk_n+1's first frame is within the same episode.

Normalization: _n1 keys are normalized using the same MEAN_STD stats as their
corresponding base keys so that the policy loss is computed in the same space
as the standard normalized batch.
"""

import logging
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ChunkPairDataset(Dataset):
    """Wraps a LeRobotDataset to return consecutive chunk pairs for SSCP CC training.

    Args:
        base_dataset:  A LeRobotDataset instance.
        chunk_size:    Action chunk size K (must match policy.chunk_size).
        dataset_stats: dict mapping feature name → {"mean": Tensor, "std": Tensor}.
                       Used to normalize _n1 keys so they match the preprocessed
                       chunk_n keys.  If None, _n1 keys are returned unnormalized
                       (will produce incorrect loss — only skip for debugging).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        chunk_size: int,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        self.base = base_dataset
        self.chunk_size = chunk_size
        self.dataset_stats = dataset_stats or {}
        self.valid_indices = self._compute_valid_indices()

        logger.info(
            f"ChunkPairDataset: {len(self.valid_indices)} valid pairs "
            f"from {len(self.base)} total frames (chunk_size={chunk_size})"
        )

    # ── Index computation ──────────────────────────────────────────────────────

    def _compute_valid_indices(self) -> list[int]:
        """Identify frame indices where i + chunk_size stays within the same episode.

        Uses episode metadata directly — no data loading required.
        """
        valid: list[int] = []
        for ep in self.base.meta.episodes:
            ep_start: int = int(ep["dataset_from_index"])
            ep_end: int = int(ep["dataset_to_index"])
            valid_end = ep_end - self.chunk_size
            if valid_end > ep_start:
                valid.extend(range(ep_start, valid_end))
        return valid

    # ── Normalization helper ───────────────────────────────────────────────────

    def _normalize(self, tensor: Tensor, feature_key: str) -> Tensor:
        """Apply MEAN_STD normalization using dataset stats for feature_key.

        Falls back to identity if stats are unavailable (warns once).
        """
        stats = self.dataset_stats.get(feature_key)
        if stats is None:
            return tensor
        if "mean" in stats and "std" in stats:
            mean = torch.as_tensor(stats["mean"]).float()
            std = torch.as_tensor(stats["std"]).float()
            return (tensor.float() - mean) / (std + 1e-8)
        if "min" in stats and "max" in stats:
            # MIN_MAX normalization: scale to [-1, 1]
            lo = torch.as_tensor(stats["min"]).float()
            hi = torch.as_tensor(stats["max"]).float()
            return 2.0 * (tensor.float() - lo) / (hi - lo + 1e-8) - 1.0
        logger.warning(
            f"ChunkPairDataset: unknown stats format for '{feature_key}', "
            "returning unnormalized _n1 key."
        )
        return tensor

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        i = self.valid_indices[idx]
        chunk_n = self.base[i]
        chunk_n1 = self.base[i + self.chunk_size]

        result: dict[str, Any] = dict(chunk_n)

        # ── Action ────────────────────────────────────────────────────────────
        result["action_n1"] = self._normalize(chunk_n1["action"], "action")
        result["action_is_pad_n1"] = (
            chunk_n1["action_is_pad"]
            if "action_is_pad" in chunk_n1
            else torch.zeros(self.chunk_size, dtype=torch.bool)
        )

        # ── Observations (state-based) ────────────────────────────────────────
        # Policy _forward_chunk_pair expects keys "obs_env_state_n1" and "obs_state_n1"
        env_state_key = "observation.environment_state"
        state_key = "observation.state"

        if env_state_key in chunk_n1:
            result["obs_env_state_n1"] = self._normalize(
                chunk_n1[env_state_key], env_state_key
            )
        if state_key in chunk_n1:
            result["obs_state_n1"] = self._normalize(chunk_n1[state_key], state_key)

        # Images are NOT included for chunk_n1:
        # The policy _forward_chunk_pair falls back to chunk_n's images automatically,
        # which is an acceptable approximation for CC training.

        return result

    # ── Passthrough properties (needed by training script & DataLoader) ────────

    @property
    def meta(self):
        return self.base.meta

    @property
    def num_frames(self) -> int:
        return self.base.num_frames

    @property
    def num_episodes(self) -> int:
        return self.base.num_episodes

    @property
    def episodes(self):
        return getattr(self.base, "episodes", None)
