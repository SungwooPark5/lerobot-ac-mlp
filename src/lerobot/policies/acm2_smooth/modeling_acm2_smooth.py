#!/usr/bin/env python
"""ACM2 smooth — State-Carried BiMamba decoder + boundary-continuity & jerk training loss (v14).

Architecture == acm2_sscp_literal_bimamba (`_build_model` unchanged → v12 checkpoints load).
Inference (reset/select_action) == parent (open-loop chunk execution with state carry).
Only forward() is overridden to add the v14 smoothness losses. See configuration / PLAN_v14.

Losses (chunk-pair path):
  L = L1(chunk n) + L1(chunk n+1)
      + λ_b · boundary_continuity(chunk n end ↔ chunk n+1 start)   # position (+ velocity = C1)
      + λ_j · jerk_penalty(chunk n) + λ_j · jerk_penalty(chunk n+1) # intra-chunk smoothness
      + kld
Single-chunk path (no pairs): L = L1 + λ_j · jerk_penalty + kld (no boundary term available).
"""
import torch
import torch.nn.functional as F  # noqa: N812
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

from .configuration_acm2_smooth import ACM2SmoothConfig


# ── Smoothness loss helpers ───────────────────────────────────────────────────

def _masked_l1(target: Tensor, pred: Tensor, pad: Tensor, weights: Tensor | None = None) -> Tensor:
    """Mean L1 over non-pad (B,K,A) actions, optional per-step weights (K,)."""
    err = F.l1_loss(target, pred, reduction="none") * ~pad.unsqueeze(-1)
    if weights is not None:
        err = err * weights.view(1, -1, 1)
    return err.mean()


def _jerk(actions: Tensor, pad: Tensor, mode: str) -> Tensor:
    """Jerk proxy = 2nd difference a[t+1]-2a[t]+a[t-1] over a chunk (B,K,A). Masked to valid
    interior triples (all three of t-1,t,t+1 non-pad). Returns scalar (||·||1 or ||·||2)."""
    if actions.shape[1] < 3:
        return actions.new_zeros(())
    accel = actions[:, 2:] - 2.0 * actions[:, 1:-1] + actions[:, :-2]          # (B, K-2, A)
    valid = (~pad[:, 2:]) & (~pad[:, 1:-1]) & (~pad[:, :-2])                    # (B, K-2)
    m = valid.unsqueeze(-1).to(accel.dtype)
    denom = m.sum().clamp_min(1.0)
    if mode == "l2":
        return ((accel.pow(2) * m).sum() / denom)
    return ((accel.abs() * m).sum() / denom)


def _boundary_continuity(actions_n: Tensor, actions_n1: Tensor, pad_n1: Tensor,
                         use_velocity: bool) -> Tensor:
    """C0 (position) [+ C1 (velocity)] continuity across the chunk n → n+1 boundary.

    boundary = (last step of chunk n) ↔ (first step of chunk n+1), which are CONSECUTIVE
    timesteps in the demo (ChunkPairDataset: chunk_n=action[i:i+K], n1=action[i+K:i+2K]).
      position:  pred_n1[0]  ≈ pred_n[-1]
      velocity:  (pred_n1[1] - pred_n1[0]) ≈ (pred_n[-1] - pred_n[-2])
    Masked out where the n+1 boundary steps are padded.
    """
    valid = (~pad_n1[:, 0]).unsqueeze(-1).to(actions_n.dtype)                  # (B,1)
    denom = valid.sum().clamp_min(1.0)
    pos = (F.l1_loss(actions_n1[:, 0], actions_n[:, -1], reduction="none") * valid).sum() / denom
    if not use_velocity:
        return pos
    vel_n = actions_n[:, -1] - actions_n[:, -2]
    vel_n1 = actions_n1[:, 1] - actions_n1[:, 0]
    valid_v = (valid.squeeze(-1) * (~pad_n1[:, 1]).to(actions_n.dtype)).unsqueeze(-1)
    denom_v = valid_v.sum().clamp_min(1.0)
    vel = (F.l1_loss(vel_n1, vel_n, reduction="none") * valid_v).sum() / denom_v
    return pos + vel


class ACM2SmoothPolicy(ACM2SSCPLiteralBiMambaPolicy):
    """ACT encoder × state-carried BiMamba decoder, trained with boundary-continuity + jerk loss."""

    config_class = ACM2SmoothConfig
    name = "acm2_smooth"

    def _build_model(self, config):
        return ACM2SSCPLiteralBiMamba(config)   # identical architecture → v12 weights load

    # inference (reset / select_action) inherited from parent (open-loop chunk + carry).

    # ── Training ─────────────────────────────────────────────────────────────────
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        has_pairs = "action_n1" in batch
        if has_pairs and self.config.sscp_p_carry > 0.0:
            return self._forward_smooth_pair(batch)
        return self._forward_smooth_single(batch)

    def _chunk_losses(self, batch, carry):
        """One chunk: returns (l1, jerk, kld, actions, new_states). actions exposed for boundary."""
        cfg = self.config
        actions, (mu, log_sigma_x2), new_states = self.model(batch, carry=carry, return_state=True)
        pad = batch["action_is_pad"]
        K, n_exec = actions.shape[1], cfg.n_action_steps
        weights = torch.ones(K, device=actions.device)
        if getattr(cfg, "use_temporal_weighting", False) and n_exec < K:
            ew = getattr(cfg, "temporal_execution_weight", 0.9)
            weights[:n_exec] = (K * ew) / n_exec
            weights[n_exec:] = (K * (1 - ew)) / (K - n_exec)
        l1 = _masked_l1(batch[ACTION], actions, pad, weights)
        jerk = self._jerk_term(actions, batch[ACTION], pad)
        kld = actions.new_zeros(())
        if cfg.use_vae:
            kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
        return l1, jerk, kld, actions, new_states

    def _jerk_term(self, actions: Tensor, gt: Tensor, pad: Tensor) -> Tensor:
        """Predicted-action jerk; if excess_only, subtract the GT chunk's jerk (clamped ≥0)."""
        cfg = self.config
        j = _jerk(actions, pad, cfg.smooth_jerk_mode)
        if cfg.smooth_jerk_excess_only:
            j_gt = _jerk(gt, pad, cfg.smooth_jerk_mode).detach()
            j = (j - j_gt).clamp_min(0.0)
        return j

    def _forward_smooth_single(self, batch):
        cfg = self.config
        l1, jerk, kld, _, _ = self._chunk_losses(batch, carry=None)
        loss = l1 + cfg.smooth_jerk_weight * jerk + (kld * cfg.kl_weight if cfg.use_vae else 0.0)
        ld = {"l1_loss": float(l1.detach()), "jerk_loss": float(jerk.detach())}
        if cfg.use_vae:
            ld["kld_loss"] = float(kld.detach())
        ld["loss"] = float(loss.detach())
        return loss, ld

    def _forward_smooth_pair(self, batch):
        cfg = self.config
        batch_n = {k: v for k, v in batch.items() if not k.endswith("_n1")}
        l1_n, jerk_n, kld_n, actions_n, states_n = self._chunk_losses(batch_n, carry=None)

        carry = detach_states(states_n) if cfg.sscp_detach else states_n
        if getattr(cfg, "carry_noise_std", 0.0) > 0.0:
            carry = add_carry_noise(carry, cfg.carry_noise_std, getattr(cfg, "carry_noise_p", 0.5))
        if cfg.carry_fusion == "ema":
            carry = ema_states(carry, None, cfg.carry_ema_beta)

        _cand = {
            OBS_STATE:       batch.get("obs_state_n1",     batch.get(OBS_STATE)),
            OBS_ENV_STATE:   batch.get("obs_env_state_n1", batch.get(OBS_ENV_STATE)),
            ACTION:          batch["action_n1"],
            "action_is_pad": batch["action_is_pad_n1"],
        }
        batch_n1 = {k: v for k, v in _cand.items() if v is not None}
        if OBS_IMAGES in batch:
            batch_n1[OBS_IMAGES] = [batch[k + "_n1"] for k in cfg.image_features
                                    if k + "_n1" in batch] or batch[OBS_IMAGES]

        l1_n1, jerk_n1, kld_n1, actions_n1, _ = self._chunk_losses(batch_n1, carry=carry)

        # ── boundary continuity (the v14 headline term) ──
        bc = _boundary_continuity(actions_n, actions_n1, batch_n1["action_is_pad"],
                                  cfg.smooth_boundary_velocity)

        l1 = l1_n + l1_n1
        jerk = jerk_n + jerk_n1
        loss = l1 + cfg.smooth_boundary_weight * bc + cfg.smooth_jerk_weight * jerk
        ld = {"l1_loss": float((l1 / 2).detach()), "l1_loss_n": float(l1_n.detach()),
              "l1_loss_n1": float(l1_n1.detach()),
              "boundary_loss": float(bc.detach()), "jerk_loss": float((jerk / 2).detach())}
        if cfg.use_vae:
            kld = (kld_n + kld_n1) / 2
            loss = loss + kld * cfg.kl_weight
            ld["kld_loss"] = float(kld.detach())
        ld["loss"] = float(loss.detach())
        return loss, ld
