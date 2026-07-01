"""S7 — ACM2 literal carry + overlap-add crossfade at inference (v22 rank 3).

Thin subclass of ACM2SSCPLiteralSmoothPolicy. Training is inherited unchanged (any inherited seam
loss still applies). Only select_action / reset are overridden to run an overlap-add rollout:
chunks are re-predicted every hop = K - overlap steps, and the `overlap` steps that straddle a
boundary are crossfaded (rising window) between the previous chunk's tail and the new chunk's
head. Because the literal (conv, ssm) state is carried, the two overlapping predictions agree, so
the blend is clean rather than a smeared average — the boundary discontinuity is removed.

sscp_overlap = 0  ->  inference identical to acm2_sscp_literal_smooth (hard chunk switch).
Disabled when temporal_ensemble_coeff is set (the ensembler owns the action queue).
"""

from collections import deque

import torch
from torch import Tensor

from lerobot.policies.acm2_sscp_literal_smooth.modeling_acm2_sscp_literal_smooth import (
    ACM2SSCPLiteralSmoothPolicy,
)
from lerobot.policies.acm2_sscp_literal_smooth_overlap.configuration_acm2_sscp_literal_smooth_overlap import (
    ACM2SSCPLiteralSmoothOverlapConfig,
)


class ACM2SSCPLiteralSmoothOverlapPolicy(ACM2SSCPLiteralSmoothPolicy):
    """Literal carry (+ inherited smooth training) with overlap-add crossfade inference."""

    config_class = ACM2SSCPLiteralSmoothOverlapConfig
    name = "acm2_sscp_literal_smooth_overlap"

    def reset(self):
        super().reset()
        self._ola_queue: deque = deque()
        self._ola_tail: Tensor | None = None  # (B, overlap, A) held-back tail of the last chunk

    def _ola_ramp(self, ov: int, device, dtype) -> Tensor:
        """Rising crossfade weight w[i] in (0, 1), shape (1, ov, 1). w=0 -> old tail, w=1 -> new."""
        i = torch.arange(1, ov + 1, device=device, dtype=dtype)
        if getattr(self.config, "sscp_overlap_window", "linear") == "hann":
            w = 0.5 * (1.0 - torch.cos(torch.pi * i / (ov + 1)))
        else:
            w = i / (ov + 1)
        return w.view(1, ov, 1)

    def _ola_emit(self, chunk: Tensor) -> None:
        """Crossfade the new chunk's head into the previous tail; enqueue `hop` executable steps."""
        _, k, _ = chunk.shape
        ov = self.config.sscp_overlap
        hop = k - ov
        if self._ola_tail is None:
            emit = chunk[:, :hop, :]
        else:
            w = self._ola_ramp(ov, chunk.device, chunk.dtype)
            head = (1.0 - w) * self._ola_tail + w * chunk[:, :ov, :]  # (B, ov, A)
            mid = chunk[:, ov:hop, :]                                 # (B, hop-ov, A)
            emit = torch.cat([head, mid], dim=1)                      # (B, hop, A)
        self._ola_tail = chunk[:, hop:, :].clone()                    # (B, ov, A)
        for t in range(emit.shape[1]):
            self._ola_queue.append(emit[:, t, :])

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.temporal_ensemble_coeff is not None or self.config.sscp_overlap <= 0:
            return super().select_action(batch)
        if not self._ola_queue:
            chunk = self._predict_with_carry(batch)  # (B, K, A) — advances self._carry
            self._ola_emit(chunk)
        return self._ola_queue.popleft()
