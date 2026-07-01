"""S3 — ACM2 literal carry + SSM state-continuity smoothness (v22 rank 1).

Thin subclass of ACM2SSCPLiteralSmoothPolicy. Same model and inference; only the chunk-pair
training objective gains a *state-continuity* term. The literal carried (conv, ssm) state is the
only thing linking chunk n -> n+1, so we require the CARRIED opening to bridge the seam:

  1. run chunk n+1 WITH the carried state (grad)      -> act_n1     (seam gap = gap_carry)
  2. run chunk n+1 fresh (carry=None, no grad, target) -> act_fresh (seam gap = gap_fresh)
  3. gap_gt = GT seam gap of the two real consecutive chunks (already small).

  L_state = |gap_carry - gap_gt|  +  relu(gap_carry - gap_fresh)

The first term matches the carried seam to the GT seam (won't over-smooth). The hinge forces the
carried opening to be at least as continuous as a memoryless restart — i.e. the *state* must
supply the continuity. This term is undefined without a carried state, so it is the family's
mechanism-novelty axis. The base GT-matched finite-difference seam loss (sscp_smooth_weight) is
kept and runs alongside.

sscp_state_weight = 0 AND sscp_smooth_weight = 0  ->  falls back to plain literal-carry training.
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
from lerobot.policies.acm2_sscp_literal_smooth_state.configuration_acm2_sscp_literal_smooth_state import (
    ACM2SSCPLiteralSmoothStateConfig,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACM2SSCPLiteralSmoothStatePolicy(ACM2SSCPLiteralSmoothPolicy):
    """Literal carry + GT-matched seam loss + carry-vs-fresh state-continuity term."""

    config_class = ACM2SSCPLiteralSmoothStateConfig
    name = "acm2_sscp_literal_smooth_state"

    @staticmethod
    def _seam_gap(act_prev: Tensor, act_next: Tensor) -> Tensor:
        """C0 seam jump: mean_|.|(act_next[:, 0] - act_prev[:, -1]) over batch & action dims."""
        return (act_next[:, 0, :] - act_prev[:, -1, :]).abs().mean()

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

    def _forward_chunk_pair(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.training and hasattr(self, "_smooth_step"):
            self._smooth_step += 1

        lam = self._smooth_lambda()
        sw = self._ramp_weight(getattr(self.config, "sscp_state_weight", 0.0))
        if lam <= 0.0 and sw <= 0.0:
            # both terms inactive -> plain literal-carry chunk-pair training (grandparent).
            return ACM2SSCPLiteralPolicy._forward_chunk_pair(self, batch)

        # ── Chunk n (literal carry source) ────────────────────────────────────────
        batch_n = {k: v for k, v in batch.items() if not k.endswith("_n1")}
        loss_n, ld_n, states_n, act_n = self._forward_single(batch_n, carry=None)

        carry = detach_states(states_n) if self.config.sscp_detach else states_n
        if getattr(self.config, "carry_noise_std", 0.0) > 0.0:
            carry = add_carry_noise(
                carry, self.config.carry_noise_std, getattr(self.config, "carry_noise_p", 0.5)
            )
        if self.config.carry_fusion == "ema":
            carry = ema_states(carry, None, self.config.carry_ema_beta)

        # ── Chunk n+1 (warmed by the carried state) ───────────────────────────────
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

        # ── State-continuity term (teacher-forced predictions) ────────────────────
        if sw > 0.0:
            with torch.no_grad():
                _, _, _, act_fresh = self._forward_single(batch_n1, carry=None)
            gap_carry = self._seam_gap(act_n, act_n1)
            gap_gt = self._seam_gap(batch_n[ACTION], batch_n1[ACTION])
            gap_fresh = self._seam_gap(act_n, act_fresh).detach()
            l_state = (gap_carry - gap_gt).abs() + torch.relu(gap_carry - gap_fresh)
            total_loss = total_loss + sw * l_state
            combined["state_loss"] = l_state.item()
            combined["state_weight"] = sw
            combined["state_gap_carry"] = gap_carry.item()
            combined["state_gap_fresh"] = float(gap_fresh.item())

        # ── Base GT-matched finite-difference seam loss (optional) ────────────────
        if lam > 0.0:
            a_n, a_n1 = act_n, act_n1
            if getattr(self.config, "sscp_smooth_free_latent", False):
                a_n, a_n1 = self._free_latent_seam_actions(batch_n, batch_n1)
            l_smooth = self._seam_loss(a_n, a_n1, batch_n, batch_n1)
            total_loss = total_loss + lam * l_smooth
            combined["smooth_loss"] = l_smooth.item()
            combined["smooth_lambda"] = lam

        return total_loss, combined

    def _ramp_weight(self, target: float) -> float:
        """Warmup-ramp an arbitrary target weight on the shared smooth schedule."""
        if target <= 0.0:
            return 0.0
        start = getattr(self.config, "sscp_smooth_warmup_start", 0)
        length = getattr(self.config, "sscp_smooth_warmup_steps", 0)
        step = int(self._smooth_step.item()) if hasattr(self, "_smooth_step") else 0
        if length <= 0:
            return target if step >= start else 0.0
        frac = (step - start) / float(length)
        return target * min(1.0, max(0.0, frac))
