"""ACM2 literal SSD state carry + chunk-boundary smoothness (v20).

Base = `acm2_sscp_literal` (v10 `m2_lit`: literal Mamba-2 SSD recurrent-state handoff across
chunk boundaries — the carry that DID train in v10). v20 keeps that carry AND its inference
path (`select_action` / `_predict_with_carry`, full-episode carry accumulation) completely
unchanged, and only adds a chunk-boundary smoothness objective to TRAINING:

    L = L1 + λ_b · boundary_continuity(C0[+C1]) + λ_j · interior_jerk

The smoothness loss helpers are reused verbatim from `acm2_smooth` (the C0/C1 boundary match
and the masked 2nd-difference jerk), so the metric definitions stay identical across the two
policies. NO new disturbance machinery: with carry_fusion="none" and carry_noise_std=0
(defaults) the carry is pure literal handoff. The inherited gate/noise/ema branches are kept
only so they remain available for ablation; they are inert at the v20 defaults.

Why a separate policy from acm2_smooth: acm2_smooth rebuilt carry on the clean acm2 base and
is not training reliably; this policy bolts the same smoothness losses onto the literal-carry
policy that trained well in v10, via its existing chunk-pair training path (ChunkPairDataset
ships the `*_n1` keys, so chunk n and chunk n+1 are consecutive demo chunks → the boundary
loss is exactly the chunk seam).

Reduction property: smooth_boundary_weight = smooth_jerk_weight = 0 → identical to
acm2_sscp_literal (same loss, same gradients).
"""

import torch  # noqa: F401  (kept for parity / future use)

from lerobot.policies.acm2_smooth.modeling_acm2_smooth import (
    _boundary_continuity,
    _jerk,
    _masked_l1,
)
from lerobot.policies.acm2_sscp_literal.modeling_acm2_sscp_literal import (
    ACM2SSCPLiteralPolicy,
    add_carry_noise,
    detach_states,
    ema_states,
)
from lerobot.policies.acm2_sscp_literal_smooth.configuration_acm2_sscp_literal_smooth import (
    ACM2SSCPLiteralSmoothConfig,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACM2SSCPLiteralSmoothPolicy(ACM2SSCPLiteralPolicy):
    """Literal-carry policy (v10 m2_lit) + chunk-boundary smoothness training losses.

    Inference is inherited UNCHANGED from ACM2SSCPLiteralPolicy: full-episode carry
    accumulation, pure literal handoff. Only the training `forward` is overridden to add the
    boundary-continuity and jerk terms.
    """

    config_class = ACM2SSCPLiteralSmoothConfig
    name = "acm2_sscp_literal_smooth"

    # ── smoothness helpers (same definitions as acm2_smooth) ─────────────────────
    def _smooth_dims(self, x):
        """Drop action dims excluded from the smoothness losses (slices the action axis)."""
        ex = self.config.smooth_exclude_dims
        if not ex:
            return x
        keep = [d for d in range(x.shape[-1]) if d not in ex]
        return x[..., keep]

    def _jerk_term(self, actions, gt, pad):
        cfg = self.config
        a, g = self._smooth_dims(actions), self._smooth_dims(gt)
        j = _jerk(a, pad, cfg.smooth_jerk_mode)
        if cfg.smooth_jerk_excess_only:
            j = (j - _jerk(g, pad, cfg.smooth_jerk_mode).detach()).clamp_min(0.0)
        return j

    def _losses(self, batch, carry):
        """One chunk decode → (l1, jerk, kld, actions, new_states)."""
        cfg = self.config
        actions, (mu, log_sigma_x2), new_states = self.model(batch, carry=carry, return_state=True)
        pad = batch["action_is_pad"]
        l1 = _masked_l1(batch[ACTION], actions, pad)
        jerk = self._jerk_term(actions, batch[ACTION], pad)
        kld = actions.new_zeros(())
        if cfg.use_vae:
            kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
        return l1, jerk, kld, actions, new_states

    # ── Training ─────────────────────────────────────────────────────────────────
    def forward(self, batch):
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        has_pairs = "action_n1" in batch
        if has_pairs and self.config.sscp_p_carry > 0.0:
            return self._forward_pair_smooth(batch)
        return self._forward_single_smooth(batch, carry=None)[:2]

    def _forward_single_smooth(self, batch, carry):
        cfg = self.config
        l1, jerk, kld, actions, new_states = self._losses(batch, carry)
        loss = l1 + cfg.smooth_jerk_weight * jerk + (kld * cfg.kl_weight if cfg.use_vae else 0.0)
        ld = {"l1_loss": float(l1.detach()), "jerk_loss": float(jerk.detach())}
        if cfg.use_vae:
            ld["kld_loss"] = float(kld.detach())
        ld["loss"] = float(loss.detach())
        return loss, ld, actions, new_states

    def _forward_pair_smooth(self, batch):
        """Chunk-continuation (ChunkPairDataset): chunk n → literal carry → chunk n+1,
        plus the boundary-continuity loss across the seam. Mirrors the lit pair path
        (incl. the inert noise/ema branches) and adds the smoothness terms."""
        cfg = self.config
        batch_n = {k: v for k, v in batch.items() if not k.endswith("_n1")}
        l1_n, jerk_n, kld_n, actions_n, states_n = self._losses(batch_n, carry=None)

        carry = detach_states(states_n) if cfg.sscp_detach else states_n
        # Inert at v20 defaults (carry_noise_std=0, carry_fusion="none"); kept for ablation.
        if getattr(cfg, "carry_noise_std", 0.0) > 0.0:
            carry = add_carry_noise(carry, cfg.carry_noise_std, getattr(cfg, "carry_noise_p", 0.5))
        if cfg.carry_fusion == "ema":
            carry = ema_states(carry, None, cfg.carry_ema_beta)

        _n1 = {
            OBS_STATE:       batch.get("obs_state_n1",     batch.get(OBS_STATE)),
            OBS_ENV_STATE:   batch.get("obs_env_state_n1", batch.get(OBS_ENV_STATE)),
            ACTION:          batch["action_n1"],
            "action_is_pad": batch["action_is_pad_n1"],
        }
        batch_n1 = {k: v for k, v in _n1.items() if v is not None}
        if OBS_IMAGES in batch:
            batch_n1[OBS_IMAGES] = [
                batch[k + "_n1"] for k in self.config.image_features if k + "_n1" in batch
            ] or batch[OBS_IMAGES]

        l1_n1, jerk_n1, kld_n1, actions_n1, _ = self._losses(batch_n1, carry=carry)
        bc = _boundary_continuity(
            self._smooth_dims(actions_n), self._smooth_dims(actions_n1),
            batch_n1["action_is_pad"], cfg.smooth_boundary_velocity,
        )

        l1, jerk = l1_n + l1_n1, jerk_n + jerk_n1
        loss = l1 + cfg.smooth_boundary_weight * bc + cfg.smooth_jerk_weight * jerk
        ld = {
            "l1_loss": float((l1 / 2).detach()), "l1_loss_n": float(l1_n.detach()),
            "l1_loss_n1": float(l1_n1.detach()), "boundary_loss": float(bc.detach()),
            "jerk_loss": float((jerk / 2).detach()),
        }
        if cfg.use_vae:
            kld = (kld_n + kld_n1) / 2
            loss = loss + kld * cfg.kl_weight
            ld["kld_loss"] = float(kld.detach())
        ld["loss"] = float(loss.detach())
        return loss, ld
