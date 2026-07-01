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
    overridden to add the seam-smoothness term. Supports λ warmup, C1/C2 order,
    and an inference-mode (free-latent) variant of the seam loss.
    """

    config_class = ACM3SSCPSmoothConfig
    name = "acm3_sscp_smooth"

    def __init__(self, config: ACM3SSCPSmoothConfig):
        super().__init__(config)
        # training-step counter for λ warmup (persists across resume).
        self.register_buffer("_smooth_step", torch.zeros((), dtype=torch.long), persistent=True)

    # ── λ warmup schedule ───────────────────────────────────────────────────────
    def _smooth_lambda(self) -> float:
        """Current λ: 0 before warmup_start, ramps to sscp_smooth_weight over warmup_steps."""
        w = getattr(self.config, "sscp_smooth_weight", 0.0)
        if w <= 0.0:
            return 0.0
        start = getattr(self.config, "sscp_smooth_warmup_start", 0)
        length = getattr(self.config, "sscp_smooth_warmup_steps", 0)
        step = int(self._smooth_step.item()) if hasattr(self, "_smooth_step") else 0
        if length <= 0:
            return w if step >= start else 0.0
        frac = (step - start) / float(length)
        return w * min(1.0, max(0.0, frac))

    # ── Boundary continuity loss ────────────────────────────────────────────────
    def _seam_loss(
        self,
        act_n: Tensor,        # (B, K, A) predicted actions, chunk n
        act_n1: Tensor,       # (B, K, A) predicted actions, chunk n+1
        batch_n: dict[str, Tensor],
        batch_n1: dict[str, Tensor],
    ) -> Tensor:
        """Match predicted seam finite-differences to GT across the boundary.

        Window straddling the seam: [a_n[-2], a_n[-1] | a_{n+1}[0], a_{n+1}[1]].
        order=1 → C1 (velocity) only; order>=2 → + C2 (acceleration). Masks padding.
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

        order = getattr(self.config, "sscp_smooth_order", 2)
        loss = masked_fd_l1(pred, gt, valid, 1)             # C1 (velocity) always
        if order >= 2:
            loss = loss + masked_fd_l1(pred, gt, valid, 2)  # C2 (acceleration)
        return loss

    def _free_latent_seam_actions(self, batch_n, batch_n1):
        """Inference-mode (VAE latent=0) seam predictions via an eval-mode forward.

        Setting the model to eval makes _encode take the latent=0 branch (no VAE
        teacher forcing) and skips the training assert — matching the actual
        inference boundary. Gradients still flow (only dropout/BN behaviour changes).
        """
        was = self.model.training
        self.model.eval()
        try:
            a_n, _, do_n = self.model(batch_n, carry=None, return_decoder_out=True)
            carry_f = do_n[-1:, :, :].transpose(0, 1).detach()
            a_n1, _, _ = self.model(batch_n1, carry=carry_f, return_decoder_out=True)
        finally:
            self.model.train(was)
        return a_n, a_n1

    # ── CC training with the extra seam term ────────────────────────────────────
    def _forward_chunk_pair(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.training and hasattr(self, "_smooth_step"):
            self._smooth_step += 1

        lam = self._smooth_lambda()
        # λ<=0 (weight 0 OR warmup not started) → identical to acm3_sscp (carry only).
        if lam <= 0.0:
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

        # ── Seam predictions: teacher-forced (default) or inference-mode (free_latent) ──
        if getattr(self.config, "sscp_smooth_free_latent", False):
            act_n, act_n1 = self._free_latent_seam_actions(batch_n, batch_n1)
        else:
            act_n = self.model.action_head(dec_out_n.transpose(0, 1))     # (B, K, A)
            act_n1 = self.model.action_head(dec_out_n1.transpose(0, 1))   # (B, K, A)

        l_smooth = self._seam_loss(act_n, act_n1, batch_n, batch_n1)

        # ── Combined loss (λ from warmup schedule) ────────────────────────────
        total_loss = loss_n + loss_n1 + lam * l_smooth
        combined = {
            "l1_loss":       (loss_dict_n["l1_loss"] + loss_dict_n1["l1_loss"]) / 2,
            "l1_loss_n":     loss_dict_n["l1_loss"],
            "l1_loss_n1":    loss_dict_n1["l1_loss"],
            "smooth_loss":   l_smooth.item(),
            "smooth_lambda": lam,
        }
        if "kld_loss" in loss_dict_n:
            combined["kld_loss"] = (loss_dict_n["kld_loss"] + loss_dict_n1.get("kld_loss", 0.0)) / 2

        return total_loss, combined
