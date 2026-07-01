"""S2 — ACM2 literal carry + spectral (frequency-domain) seam loss (v22 rank 2).

Thin subclass of ACM2SSCPLiteralSmoothPolicy. Same model and inference; the chunk-pair objective
gains a spectral seam term. A short window straddling the boundary (last `halfwin` of chunk n +
first `halfwin` of chunk n+1) is rFFT'd; we penalize only the predicted high-frequency magnitude
that EXCEEDS the ground-truth magnitude in the same band (one-sided, GT-matched). Tremor is
high-frequency, so this hits it directly, while low-frequency task motion is left untouched.

sscp_spectral_weight = 0 AND sscp_smooth_weight = 0  ->  falls back to plain literal-carry training.
The base GT-matched finite-difference seam loss (sscp_smooth_weight) can run alongside.
"""

import torch
from torch import Tensor

from lerobot.policies.acm2_sscp_literal.modeling_acm2_sscp_literal import (
    ACM2SSCPLiteralPolicy,
    add_carry_noise,
    detach_states,
    ema_states,
)
from lerobot.policies.acm2_sscp_literal_smooth.modeling_acm2_sscp_literal_smooth import (
    ACM2SSCPLiteralSmoothPolicy,
)
from lerobot.policies.acm2_sscp_literal_smooth_spectral.configuration_acm2_sscp_literal_smooth_spectral import (
    ACM2SSCPLiteralSmoothSpectralConfig,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACM2SSCPLiteralSmoothSpectralPolicy(ACM2SSCPLiteralSmoothPolicy):
    """Literal carry + optional GT-matched seam loss + spectral high-band excess penalty."""

    config_class = ACM2SSCPLiteralSmoothSpectralConfig
    name = "acm2_sscp_literal_smooth_spectral"

    def _build_n1_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        cand = {
            OBS_STATE: batch.get("obs_state_n1", batch.get(OBS_STATE)),
            OBS_ENV_STATE: batch.get("obs_env_state_n1", batch.get(OBS_ENV_STATE)),
            ACTION: batch["action_n1"],
            "action_is_pad": batch["action_is_pad_n1"],
        }
        b1 = {k: v for k, v in cand.items() if v is not None}
        if OBS_IMAGES in batch:
            b1[OBS_IMAGES] = [
                batch[k + "_n1"] for k in self.config.image_features if k + "_n1" in batch
            ] or batch[OBS_IMAGES]
        return b1

    def _spectral_loss(self, act_n, act_n1, batch_n, batch_n1) -> Tensor:
        """One-sided GT-matched high-frequency energy penalty on a seam-straddling window."""
        hw = self.config.sscp_spectral_halfwin
        pred = torch.cat([act_n[:, -hw:, :], act_n1[:, :hw, :]], dim=1)              # (B, 2hw, A)
        gt = torch.cat([batch_n[ACTION][:, -hw:, :], batch_n1[ACTION][:, :hw, :]], dim=1)

        pad_n = batch_n.get("action_is_pad")
        pad_n1 = batch_n1.get("action_is_pad")
        if pad_n is not None and pad_n1 is not None:
            valid = torch.cat([~pad_n[:, -hw:], ~pad_n1[:, :hw]], dim=1).all(dim=1)  # (B,) full-window valid
        else:
            valid = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
        if int(valid.sum()) == 0:
            return pred.new_zeros(())

        pred, gt = pred[valid], gt[valid]                                            # (Bv, W, A)
        pm = torch.fft.rfft(pred, dim=1).abs()                                       # (Bv, nbins, A)
        gm = torch.fft.rfft(gt, dim=1).abs()
        nbins = pm.shape[1]
        norm_idx = torch.arange(nbins, device=pred.device).float() / max(nbins - 1, 1)
        band = (norm_idx >= self.config.sscp_spectral_high_frac).float().view(1, -1, 1)  # high band
        excess = torch.relu(pm - gm) * band                                         # only excess energy
        denom = band.sum().clamp_min(1.0) * pred.shape[0] * pred.shape[2]
        return excess.sum() / denom

    def _forward_chunk_pair(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.training and hasattr(self, "_smooth_step"):
            self._smooth_step += 1

        lam = self._smooth_lambda()
        spw = self._ramp_weight(getattr(self.config, "sscp_spectral_weight", 0.0))
        if lam <= 0.0 and spw <= 0.0:
            return ACM2SSCPLiteralPolicy._forward_chunk_pair(self, batch)

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

        a_n, a_n1 = act_n, act_n1
        if (lam > 0.0 or spw > 0.0) and getattr(self.config, "sscp_smooth_free_latent", False):
            a_n, a_n1 = self._free_latent_seam_actions(batch_n, batch_n1)

        if lam > 0.0:
            l_smooth = self._seam_loss(a_n, a_n1, batch_n, batch_n1)
            total_loss = total_loss + lam * l_smooth
            combined["smooth_loss"] = l_smooth.item()
            combined["smooth_lambda"] = lam

        if spw > 0.0:
            l_spec = self._spectral_loss(a_n, a_n1, batch_n, batch_n1)
            total_loss = total_loss + spw * l_spec
            combined["spectral_loss"] = l_spec.item()
            combined["spectral_weight"] = spw

        return total_loss, combined

    def _ramp_weight(self, target: float) -> float:
        if target <= 0.0:
            return 0.0
        start = getattr(self.config, "sscp_smooth_warmup_start", 0)
        length = getattr(self.config, "sscp_smooth_warmup_steps", 0)
        step = int(self._smooth_step.item()) if hasattr(self, "_smooth_step") else 0
        if length <= 0:
            return target if step >= start else 0.0
        frac = (step - start) / float(length)
        return target * min(1.0, max(0.0, frac))
