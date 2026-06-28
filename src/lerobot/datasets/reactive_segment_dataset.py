"""ReactiveSegmentDataset — consecutive-chunk segments for v13 reactive-consistency training.

The reactive policy executes a receding-horizon loop: each step it re-decodes a chunk from the
carried recurrent state + the fresh observation and executes the head. To train that loop
(train == inference), we need, for a short segment of M consecutive chunks:

  • each chunk BOUNDARY observation (carry is committed here), and
  • a few MID-CHUNK observations per chunk (re-plans from the committed carry + fresh obs),

each paired with its ground-truth action chunk (base[t] already returns obs_t + action[t:t+K]).

This replaces ChunkPairDataset for reactive training. (ChunkPairDataset only transfers a single
carry n→n+1, which cannot represent the per-step re-planning the reactive loop performs.)

For each valid start frame i, emits J = M·(1+S) decode points in time order, keyed
`step{j}_<feature>` (j = 0..J-1):
  step{j}_action, step{j}_action_is_pad, step{j}_observation.state[/.environment_state],
  step{j}_observation.images.*  (all normalized to match the preprocessed base keys).

The boundary/chunk layout is fixed (j = c·(1+S) + k; k=0 boundary, k≥1 mid-chunk sample), so
the policy.forward reconstructs it from config (reactive_train_chunks / reactive_train_samples)
— the dataset only ships the per-point tensors.
"""
import logging
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def fixed_mid_offsets(chunk_size: int, n: int) -> list[int]:
    """`n` deterministic mid-chunk offsets evenly spaced in [1, chunk_size-1] (dedup, sorted)."""
    if n <= 0 or chunk_size < 2:
        return []
    offs: list[int] = []
    for k in range(n):
        d = round((k + 1) * chunk_size / (n + 1))
        d = max(1, min(chunk_size - 1, d))
        if d not in offs:
            offs.append(d)
    return sorted(offs)


class ReactiveSegmentDataset(Dataset):
    """Wraps a LeRobotDataset to emit M-chunk reactive-training segments.

    Args:
        base_dataset:  A LeRobotDataset (base[t] returns obs_t + action chunk t:t+K).
        chunk_size:    Action chunk size K (== policy.chunk_size).
        n_chunks:      M consecutive chunks per segment (carry committed at each boundary).
        n_samples:     Mid-chunk re-plan samples S per chunk (besides the boundary).
        dataset_stats: feature → {"mean","std"} (or min/max) to normalize the emitted keys.
    """

    def __init__(self, base_dataset: Dataset, chunk_size: int, n_chunks: int = 2,
                 n_samples: int = 3, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        self.base = base_dataset
        self.K = chunk_size
        self.M = n_chunks
        self.dataset_stats = dataset_stats or {}
        self.mid = fixed_mid_offsets(chunk_size, n_samples)        # [o1, o2, ...]
        self.in_chunk = [0] + self.mid                             # boundary + mids, len 1+S
        self.n_points = self.M * len(self.in_chunk)
        self.valid_indices = self._compute_valid_indices()
        logger.info(
            f"ReactiveSegmentDataset: {len(self.valid_indices)} segments "
            f"(K={chunk_size}, chunks={n_chunks}, samples={n_samples}, points/seg={self.n_points})"
        )

    def _compute_valid_indices(self) -> list[int]:
        """Start frames i whose LAST decode point obs is still inside the episode.

        Last point = i + (M-1)·K + max(in_chunk). Its action chunk may run past the episode
        end, but that is handled by action_is_pad (the base dataset pads + flags it).
        """
        margin = (self.M - 1) * self.K + max(self.in_chunk)
        valid: list[int] = []
        for ep in self.base.meta.episodes:
            ep_start, ep_end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
            valid_end = ep_end - margin
            if valid_end > ep_start:
                valid.extend(range(ep_start, valid_end))
        return valid

    def _normalize(self, tensor: Tensor, key: str) -> Tensor:
        stats = self.dataset_stats.get(key)
        if stats is None:
            return tensor
        if "mean" in stats and "std" in stats:
            mean = torch.as_tensor(stats["mean"]).float()
            std = torch.as_tensor(stats["std"]).float()
            return (tensor.float() - mean) / (std + 1e-8)
        if "min" in stats and "max" in stats:
            lo = torch.as_tensor(stats["min"]).float()
            hi = torch.as_tensor(stats["max"]).float()
            return 2.0 * (tensor.float() - lo) / (hi - lo + 1e-8) - 1.0
        return tensor

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        i = self.valid_indices[idx]
        result: dict[str, Any] = {}
        j = 0
        for c in range(self.M):
            for o in self.in_chunk:
                frame = self.base[i + c * self.K + o]
                result[f"step{j}_action"] = self._normalize(frame["action"], "action")
                result[f"step{j}_action_is_pad"] = (
                    frame["action_is_pad"] if "action_is_pad" in frame
                    else torch.zeros(self.K, dtype=torch.bool)
                )
                for key in ("observation.state", "observation.environment_state"):
                    if key in frame:
                        result[f"step{j}_{key}"] = self._normalize(frame[key], key)
                for key in frame:
                    if key.startswith("observation.images"):
                        result[f"step{j}_{key}"] = self._normalize(frame[key], key)
                j += 1
        return result

    # passthrough (training script / DataLoader)
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
