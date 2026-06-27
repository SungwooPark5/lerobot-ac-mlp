#!/usr/bin/env python
"""ACM2 reactive — receding-horizon state-refined re-planning (cor x BiMamba, v13).

Model == acm2_sscp_literal_bimamba: `_build_model` returns the identical ACM2SSCPLiteralBiMamba,
so v12 BiMamba checkpoints load with zero key mismatch. Only the inference procedure changes
(reset/select_action), plus an optional reactive-consistency training term (forward).

Two-timescale inference (PLAN_v13 §1.2):
  fast channel (re-plan, period r = refine_period): each r executed steps, decode a fresh
      K-chunk from the *committed* carry + fresh obs and execute its head. The gate re-corrects
      the carry toward o_t and BiMamba's backward scan refines the plan — but the carry is NOT
      committed (so mid-chunk re-plans reuse the same carry_in).
  slow channel (carry, period c = commit_period = chunk length): commit the boundary decode's
      forward-scan state once per chunk so the recurrent stream advances at the true execution
      rate (no K-fold run-ahead).

reactive_refine=False (or refine_period >= chunk length) ⇒ re-plan only at chunk boundaries
⇒ byte-identical to the parent open-loop policy (the clean control; verified by smoketest).
"""
import re

import torch
from torch import Tensor

from lerobot.policies.acm2_sscp_literal.modeling_acm2_sscp_literal import (
    add_carry_noise,
    detach_states,
    ema_states,
)
from lerobot.policies.acm2_sscp_literal_bimamba.modeling_acm2_sscp_literal_bimamba import (
    ACM2SSCPLiteralBiMamba,
    ACM2SSCPLiteralBiMambaPolicy,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from .configuration_acm2_reactive import ACM2ReactiveConfig

_OFFSET_RE = re.compile(r"action_o(\d+)$")


class ACM2ReactivePolicy(ACM2SSCPLiteralBiMambaPolicy):
    """cor x BiMamba with a receding-horizon, state-refined reactive inference loop."""

    config_class = ACM2ReactiveConfig
    name = "acm2_reactive"

    def _build_model(self, config):
        # Identical model to the parent → v12 bimamba weights load unchanged.
        return ACM2SSCPLiteralBiMamba(config)

    # ── Inference state ──────────────────────────────────────────────────────────
    def reset(self):
        super().reset()                       # _action_queue / temporal_ensembler + self._carry = None
        self._plan: Tensor | None = None      # (B, K, action_dim) most recent decoded chunk
        self._carry_next: list | None = None  # carry_out stashed at the boundary → next chunk's carry_in
        self._t: int = 0                      # executed-step index within the chunk [0, commit_period)
        self._j: int = 0                      # index into the current plan (resets to 0 on each re-plan)

    @property
    def _commit_period(self) -> int:
        return self.config.commit_period or self.config.n_action_steps

    @property
    def _eff_refine_period(self) -> int:
        # reactive off, or r >= chunk length → re-plan only at the boundary (= open-loop).
        if not self.config.reactive_refine:
            return self._commit_period
        return max(1, min(self.config.refine_period, self._commit_period))

    # ── Reactive select_action (receding-horizon) ────────────────────────────────
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        # Temporal ensembling is an alternative (output-space) reactive scheme and is mutually
        # exclusive with explicit re-planning — defer to the parent so TE configs still run as
        # a baseline.
        if self.config.temporal_ensemble_coeff is not None:
            return super().select_action(batch)

        r = self._eff_refine_period
        at_boundary = (self._plan is None) or (self._t == 0)
        if at_boundary:
            # slow channel: decode with the committed carry_in; stash carry_out for the next chunk.
            self._plan, self._carry_next = self._decode(batch, self._carry)
            self._j = 0
        elif self._t % r == 0:
            # fast channel: re-plan from the SAME committed carry_in (do NOT advance the carry).
            self._plan, _ = self._decode(batch, self._carry)
            self._j = 0

        action = self._plan[:, self._j]
        self._j += 1
        self._t += 1
        if self._t >= self._commit_period:    # crossed a chunk boundary → commit the slow channel
            self._carry = self._carry_next
            self._t = 0
        return action

    @torch.no_grad()
    def _decode(self, batch: dict[str, Tensor], carry: list | None):
        """One decoder forward from a fixed carry_in, WITHOUT mutating self._carry.

        Returns (actions (B, K, action_dim), carry_out | None). Mirrors the parent's
        _predict_with_carry — including the ema fusion of the returned state — but leaves
        self._carry untouched so mid-chunk re-plans keep reusing the committed carry_in.
        """
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        carry_in = carry if self.config.sscp_enabled else None
        actions, _, new_states = self.model(batch, carry=carry_in, return_state=True)
        carry_out = None
        if self.config.sscp_enabled:
            carry_out = detach_states(new_states)
            if self.config.carry_fusion == "ema":
                carry_out = ema_states(carry_out, self._carry, self.config.carry_ema_beta)
        return actions, carry_out

    # ── Training: chunk-pair (gate) + optional reactive-consistency (Phase R) ─────
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        # Base = the parent chunk-pair loss (trains the boundary gate; the warm-start base).
        base_loss, loss_dict = super().forward(batch)
        if not self.config.reactive_train:
            return base_loss, loss_dict

        offsets = sorted({int(m.group(1)) for k in batch if (m := _OFFSET_RE.search(k))})
        if not offsets:
            # Data not wired (ChunkPairDataset(reactive_offsets=0)) → safe fall back.
            return base_loss, loss_dict

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # carry_out of chunk n == the committed carry_in the reactive loop reuses across chunk n+1.
        batch_n = {k: v for k, v in batch.items()
                   if not k.endswith("_n1") and not _has_offset_suffix(k)}
        _, _, states_n = self._forward_single(batch_n, carry=None)
        carry = detach_states(states_n) if self.config.sscp_detach else states_n
        if getattr(self.config, "carry_noise_std", 0.0) > 0.0:
            carry = add_carry_noise(carry, self.config.carry_noise_std,
                                    getattr(self.config, "carry_noise_p", 0.5))
        if self.config.carry_fusion == "ema":
            carry = ema_states(carry, None, self.config.carry_ema_beta)

        # Supervise re-plans at each mid-chunk offset from the SAME (fixed) carry_in — this is
        # exactly the reactive fast channel, so the gate learns per-step invocation.
        rc = batch_n.get(ACTION).new_zeros(()) if ACTION in batch_n else torch.zeros((), device=base_loss.device)
        for d in offsets:
            loss_o, _, _ = self._forward_single(self._offset_batch(batch, d), carry=carry)
            rc = rc + loss_o
        rc = rc / len(offsets)

        total = base_loss + self.config.reactive_consistency_weight * rc
        loss_dict["reactive_consistency_loss"] = float(rc.detach())
        return total, loss_dict

    def _offset_batch(self, batch: dict[str, Tensor], d: int) -> dict[str, Tensor]:
        """Build a single-chunk training batch from the offset-`d` keys (mirrors _forward_chunk_pair)."""
        cand = {
            OBS_STATE:     batch.get(f"obs_state_o{d}",     batch.get(OBS_STATE)),
            OBS_ENV_STATE: batch.get(f"obs_env_state_o{d}", batch.get(OBS_ENV_STATE)),
            ACTION:        batch[f"action_o{d}"],
            "action_is_pad": batch[f"action_is_pad_o{d}"],
        }
        out = {k: v for k, v in cand.items() if v is not None}
        if OBS_IMAGES in batch:
            imgs = [batch[f"{k}_o{d}"] for k in self.config.image_features if f"{k}_o{d}" in batch]
            out[OBS_IMAGES] = imgs or batch[OBS_IMAGES]
        return out


def _has_offset_suffix(key: str) -> bool:
    return re.search(r"_o\d+$", key) is not None
