"""S7 — ACM2 literal carry + overlap-add crossfade (inference) + optional overlap-consistency (train).

Subclass of ACM2SSCPLiteralSmoothStatePolicy, so it inherits the full State training objective
(base l1 + optional GT-matched seam loss + optional carry-vs-fresh state loss) AND can be combined
with BiMamba (use_bimamba_decoder). Two things are added:

  Inference — overlap-add rollout: chunks are re-predicted every hop = K - overlap steps, and the
  `overlap` steps that straddle a boundary are crossfaded (rising window) between the previous
  chunk's tail and the new chunk's carried head. Because the literal (conv, ssm) state is carried,
  the two overlapping predictions agree, so the blend is a clean continuation, not a smeared
  average — the boundary discontinuity is removed.

  Training (optional, sscp_overlap_train_weight > 0) — overlap-consistency: with the chunk pair
  offset by hop = K - overlap (set by the train pipeline), chunk n's tail and chunk n+1's carried
  head predict the SAME wall-clock steps. We penalize (i) their disagreement and (ii) the
  crossfaded blend vs GT, so the model is trained for the very blend it executes at inference.

Flag summary (all default off -> plain literal-carry train/infer):
  sscp_overlap>0            : overlap-add at inference
  sscp_overlap_train_weight : train the overlap blend (s7-trained)
  sscp_state_weight         : carry-vs-fresh state-continuity loss (s3)
  sscp_smooth_weight        : GT-matched finite-difference seam loss (C1/C2)
  use_bimamba_decoder       : BiMamba decoder
"""

from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.acm2_sscp_literal.modeling_acm2_sscp_literal import (
    add_carry_noise,
    detach_states,
    ema_states,
)
from lerobot.policies.acm2_sscp_literal_smooth_overlap.configuration_acm2_sscp_literal_smooth_overlap import (
    ACM2SSCPLiteralSmoothOverlapConfig,
)
from lerobot.policies.acm2_sscp_literal_smooth_state.modeling_acm2_sscp_literal_smooth_state import (
    ACM2SSCPLiteralSmoothStatePolicy,
)
from lerobot.utils.constants import ACTION


class ACM2SSCPLiteralSmoothOverlapPolicy(ACM2SSCPLiteralSmoothStatePolicy):
    """Literal carry + overlap-add crossfade inference (+ inherited State/seam training,
    + optional overlap-consistency training, + optional BiMamba)."""

    config_class = ACM2SSCPLiteralSmoothOverlapConfig
    name = "acm2_sscp_literal_smooth_overlap"

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
            chunk = self._predict_with_carry(batch)  # (B, K, A) — advances self._carry
            self._ola_emit(chunk)
        return self._ola_queue.popleft()

    # ── Training: State objective + optional overlap-consistency ─────────────────
    def _forward_chunk_pair(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        otw = self._ramp_weight(getattr(self.config, "sscp_overlap_train_weight", 0.0))
        if otw <= 0.0:
            # No overlap-consistency term: delegate to the State objective, which itself falls
            # back to plain literal-carry training when its own weights are 0.
            return super()._forward_chunk_pair(batch)

        if self.training and hasattr(self, "_smooth_step"):
            self._smooth_step += 1
        lam = self._smooth_lambda()
        sw = self._ramp_weight(getattr(self.config, "sscp_state_weight", 0.0))

        # ── chunk n (carry source) -> carry -> chunk n+1 (single pair forward) ───
        batch_n = {k: v for k, v in batch.items() if not k.endswith("_n1")}
        loss_n, ld_n, states_n, act_n = self._forward_single(batch_n, carry=None)

        carry = detach_states(states_n) if self.config.sscp_detach else states_n
        if getattr(self.config, "carry_noise_std", 0.0) > 0.0:
            carry = add_carry_noise(
                carry, self.config.carry_noise_std, getattr(self.config, "carry_noise_p", 0.5)
            )
        if self.config.carry_fusion == "ema":
            carry = ema_states(carry, None, self.config.carry_ema_beta)

        batch_n1 = self._build_n1_batch(batch)
        loss_n1, ld_n1, _, act_n1 = self._forward_single(batch_n1, carry=carry)

        total_loss = loss_n + loss_n1
        combined = {
            "l1_loss": (ld_n["l1_loss"] + ld_n1["l1_loss"]) / 2,
            "l1_loss_n": ld_n["l1_loss"],
            "l1_loss_n1": ld_n1["l1_loss"],
        }
        if "kld_loss" in ld_n:
            combined["kld_loss"] = (ld_n["kld_loss"] + ld_n1.get("kld_loss", 0.0)) / 2

        # optional carry-vs-fresh state-continuity term (s3). NOTE: intended for adjacent pairs;
        # keep sscp_state_weight=0 when training overlap (pair offset = hop, seam is not adjacent).
        if sw > 0.0:
            with torch.no_grad():
                _, _, _, act_fresh = self._forward_single(batch_n1, carry=None)
            gap_carry = self._seam_gap(act_n, act_n1)
            gap_gt = self._seam_gap(batch_n[ACTION], batch_n1[ACTION])
            gap_fresh = self._seam_gap(act_n, act_fresh).detach()
            l_state = (gap_carry - gap_gt).abs() + torch.relu(gap_carry - gap_fresh)
            total_loss = total_loss + sw * l_state
            combined["state_loss"] = l_state.item()

        # optional GT-matched finite-difference seam loss (C1/C2)
        if lam > 0.0:
            a_n, a_n1 = act_n, act_n1
            if getattr(self.config, "sscp_smooth_free_latent", False):
                a_n, a_n1 = self._free_latent_seam_actions(batch_n, batch_n1)
            l_smooth = self._seam_loss(a_n, a_n1, batch_n, batch_n1)
            total_loss = total_loss + lam * l_smooth
            combined["smooth_loss"] = l_smooth.item()

        # ── overlap-consistency term (the s7-trained objective) ──────────────────
        # Requires pair_offset = K - overlap so act_n[hop:K] and act_n1[:ov] (carried head)
        # predict the same wall-clock steps GT[i+hop : i+K] = batch_n1[ACTION][:, :ov].
        ov = int(self.config.sscp_overlap)
        K = act_n.shape[1]
        hop = K - ov
        if ov > 0 and hop > 0:
            pn = act_n[:, hop:K, :]              # (B, ov, A) chunk n tail
            hn1 = act_n1[:, :ov, :]              # (B, ov, A) chunk n+1 carried head
            gt_ov = batch_n1[ACTION][:, :ov, :]  # (B, ov, A) GT for the overlap window
            pad = batch_n1["action_is_pad"][:, :ov].unsqueeze(-1)  # True = padded
            m = (~pad).to(act_n.dtype)
            denom = m.sum().clamp_min(1.0)
            w = self._ola_ramp(ov, act_n.device, act_n.dtype)
            blend = (1.0 - w) * pn + w * hn1
            consist = (F.l1_loss(pn, hn1, reduction="none") * m).sum() / denom
            accur = (F.l1_loss(blend, gt_ov, reduction="none") * m).sum() / denom
            l_ola = consist + accur
            total_loss = total_loss + otw * l_ola
            combined["ola_loss"] = l_ola.item()
            combined["ola_consist"] = consist.item()
            combined["ola_weight"] = otw

        return total_loss, combined
