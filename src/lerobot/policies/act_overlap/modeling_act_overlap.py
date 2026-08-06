"""ACT + overlap-add crossfade (inference-only). See configuration_act_overlap for rationale.

Subclass of ACTPolicy. Adds only the inference-time overlap-add rollout: chunks are re-predicted
every hop = chunk_size - overlap steps, and the `overlap` steps straddling a boundary are
crossfaded (rising window) between the previous chunk's tail and the new chunk's head. No state is
carried (ACT is stateless), so the two overlapping predictions are independent and the blend is a
smeared average — the "overlap without carry" ablation for MOSAIC.

The overlap-add machinery (_ola_ramp / _ola_emit) is the same as the ACM2 MOSAIC policy; the only
difference is that the chunk comes from ACT's stateless predict_action_chunk rather than a
carry-advancing predictor.
"""

from collections import deque

import torch
from torch import Tensor

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act_overlap.configuration_act_overlap import ACTOverlapConfig


class ACTOverlapPolicy(ACTPolicy):
    """ACT + overlap-add crossfade at inference (stateless — no carry)."""

    config_class = ACTOverlapConfig
    name = "act_overlap"

    # ── Inference: overlap-add crossfade ─────────────────────────────────────────
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
            chunk = self.predict_action_chunk(batch)  # (B, K, A) — no carry (ACT is stateless)
            self._ola_emit(chunk)
        return self._ola_queue.popleft()
