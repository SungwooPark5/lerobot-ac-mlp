"""ACM3 + SSCP + explicit chunk-boundary continuity loss (no architecture change).

acm3_sscp gives chunk n+1 the *information* about chunk n via the carry token, but
its loss (per-chunk L1 + KLD) never looks at the seam, so continuity is only
encouraged indirectly. This policy adds a GT-matched boundary-continuity loss to
the chunk-continuation (CC) training.

Why GT-matched (not jitter→0):
  The two CC chunks are consecutive real chunks, so the GT transition
  a_n[-1] → a_{n+1}[0] is already smooth. We match the *predicted* seam finite-
  differences (velocity + acceleration) to the *GT* ones, instead of pushing the
  predicted jump to zero. So the objective is aligned with the task (won't make
  the motion sluggish / hurt SR), while still flattening the boundary spike.

Everything else — model, carry tracking, inference (select_action) — is inherited
from ACM3SSCPPolicy unchanged. sscp_smooth_weight = 0 reproduces acm3_sscp exactly.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.acm3_sscp.modeling_acm3_sscp import ACM3SSCPPolicy
from lerobot.policies.acm3_sscp_smooth.configuration_acm3_sscp_smooth import ACM3SSCPSmoothConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACM3SSCPSmoothPolicy(ACM3SSCPPolicy):
    """ACM3+SSCP with a chunk-boundary continuity loss added to CC training.

    Same model and inference as ACM3SSCPPolicy; only `_forward_chunk_pair` is
    overridden to add the seam-smoothness term.
    """

    config_class = ACM3SSCPSmoothConfig
    name = "acm3_sscp_smooth"

    # ── Boundary continuity loss ────────────────────────────────────────────────
    def _seam_loss(
        self,
        act_n: Tensor,        # (B, K, A) predicted actions, chunk n
        act_n1: Tensor,       # (B, K, A) predicted actions, chunk n+1
        batch_n: dict[str, Tensor],
        batch_n1: dict[str, Tensor],
    ) -> Tensor:
        """Match predicted seam velocity/acceleration to GT across the boundary.

        Window straddling the seam: [a_n[-2], a_n[-1] | a_{n+1}[0], a_{n+1}[1]].
        Penalise |Δ(pred) - Δ(gt)| for the 1st (velocity) and 2nd (acceleration)
        finite differences, masking out padded steps.
        """
        pred = torch.cat([act_n[:, -2:, :], act_n1[:, :2, :]], dim=1)            # (B, 4, A)
        gt = torch.cat([batch_n[ACTION][:, -2:, :], batch_n1[ACTION][:, :2, :]], dim=1)

        pad_n = batch_n.get("action_is_pad")
        pad_n1 = batch_n1.get("action_is_pad")
        if pad_n is not None and pad_n1 is not None:
            valid = torch.cat([~pad_n[:, -2:], ~pad_n1[:, :2]], dim=1)           # (B, 4) bool
        else:
            valid = torch.ones(pred.shape[0], 4, dtype=torch.bool, device=pred.device)

        def masked_fd_l1(p: Tensor, g: Tensor, vm: Tensor, order: int) -> Tensor:
            for _ in range(order):
                p = p[:, 1:, :] - p[:, :-1, :]
                g = g[:, 1:, :] - g[:, :-1, :]
                vm = vm[:, 1:] & vm[:, :-1]                 # diff valid iff both endpoints valid
            err = (p - g).abs().mean(-1)                    # (B, n) mean over action dims
            m = vm.float()
            return (err * m).sum() / m.sum().clamp_min(1.0)

        return masked_fd_l1(pred, gt, valid, 1) + masked_fd_l1(pred, gt, valid, 2)

    # ── CC training with the extra seam term ────────────────────────────────────
    def _forward_chunk_pair(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        # weight 0 → behave exactly like acm3_sscp (no seam computation).
        if getattr(self.config, "sscp_smooth_weight", 0.0) <= 0.0:
            return super()._forward_chunk_pair(batch)

        # ── Chunk n ───────────────────────────────────────────────────────────
        batch_n = {k: v for k, v in batch.items() if not k.endswith("_n1")}
        loss_n, loss_dict_n, dec_out_n = self._forward_single(batch_n, carry=None, return_decoder_out=True)

        carry = dec_out_n[-1:, :, :].transpose(0, 1)  # (B, 1, D)
        if self.config.sscp_detach:
            carry = carry.detach()

        # ── Chunk n+1 ─────────────────────────────────────────────────────────
        _n1_candidates = {
            OBS_STATE:       batch.get("obs_state_n1",     batch.get(OBS_STATE)),
            OBS_ENV_STATE:   batch.get("obs_env_state_n1", batch.get(OBS_ENV_STATE)),
            ACTION:          batch["action_n1"],
            "action_is_pad": batch["action_is_pad_n1"],
        }
        batch_n1 = {k: v for k, v in _n1_candidates.items() if v is not None}
        if OBS_IMAGES in batch:
            batch_n1[OBS_IMAGES] = [
                batch[k + "_n1"] for k in self.config.image_features
                if k + "_n1" in batch
            ] or batch[OBS_IMAGES]

        loss_n1, loss_dict_n1, dec_out_n1 = self._forward_single(batch_n1, carry=carry, return_decoder_out=True)

        # ── Predicted actions (reuse action_head; no extra model forward) ──────
        act_n = self.model.action_head(dec_out_n.transpose(0, 1))     # (B, K, A)
        act_n1 = self.model.action_head(dec_out_n1.transpose(0, 1))   # (B, K, A)

        l_smooth = self._seam_loss(act_n, act_n1, batch_n, batch_n1)

        # ── Combined loss ─────────────────────────────────────────────────────
        total_loss = loss_n + loss_n1 + self.config.sscp_smooth_weight * l_smooth
        combined = {
            "l1_loss":     (loss_dict_n["l1_loss"] + loss_dict_n1["l1_loss"]) / 2,
            "l1_loss_n":   loss_dict_n["l1_loss"],
            "l1_loss_n1":  loss_dict_n1["l1_loss"],
            "smooth_loss": l_smooth.item(),
        }
        if "kld_loss" in loss_dict_n:
            combined["kld_loss"] = (loss_dict_n["kld_loss"] + loss_dict_n1.get("kld_loss", 0.0)) / 2

        return total_loss, combined
