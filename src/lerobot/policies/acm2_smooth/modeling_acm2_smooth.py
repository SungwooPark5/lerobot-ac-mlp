#!/usr/bin/env python
"""ACM2 smooth — Smooth Chunking via BiMamba + Boundary State Carry (v14).

Base = `acm2` (ACT encoder + Mamba-2 decoder, zero-carry). v14 adds two orthogonal axes on top
of the CLEAN acm2, with NO disturbance machinery:
  ① BiMamba decoder (bidir scan + fuse)            → chunk *interior* smoothness  [은지님]
  ② pure boundary state carry + boundary/jerk loss → chunk *boundary*  smoothness [나]

The only thing reused from the v12 file is the PURE utility `mamba2_stateful_forward`
(replicates Mamba2.forward exposing the recurrent state — needed for carry/BiMamba). The
disturbance machinery (CarryStateFusion gate, add_carry_noise, ema_states) is NOT imported.
"""
from itertools import chain

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.acm2.modeling_acm2 import ACM2, ACM2Policy, Mamba2ACMDecoder
# PURE utilities only (state-exposing Mamba2 forward + detach). NO gate/noise/fusion.
from lerobot.policies.acm2_sscp_literal.modeling_acm2_sscp_literal import (
    detach_states,
    mamba2_stateful_forward,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from .configuration_acm2_smooth import ACM2SmoothConfig

try:
    from mamba_ssm import Mamba2
    HAS_MAMBA2 = True
except ImportError:
    HAS_MAMBA2 = False


# ── Smoothness loss helpers ───────────────────────────────────────────────────

def _masked_l1(target: Tensor, pred: Tensor, pad: Tensor, weights: Tensor | None = None) -> Tensor:
    err = F.l1_loss(target, pred, reduction="none") * ~pad.unsqueeze(-1)
    if weights is not None:
        err = err * weights.view(1, -1, 1)
    return err.mean()


def _jerk(actions: Tensor, pad: Tensor, mode: str) -> Tensor:
    """Jerk proxy = 2nd difference over a chunk (B,K,A), masked to valid interior triples."""
    if actions.shape[1] < 3:
        return actions.new_zeros(())
    accel = actions[:, 2:] - 2.0 * actions[:, 1:-1] + actions[:, :-2]
    valid = (~pad[:, 2:]) & (~pad[:, 1:-1]) & (~pad[:, :-2])
    m = valid.unsqueeze(-1).to(accel.dtype)
    denom = m.sum().clamp_min(1.0)
    if mode == "l2":
        return (accel.pow(2) * m).sum() / denom
    return (accel.abs() * m).sum() / denom


def _boundary_continuity(actions_n: Tensor, actions_n1: Tensor, pad_n1: Tensor,
                         use_velocity: bool) -> Tensor:
    """C0(position)[+C1(velocity)] continuity across chunk n→n+1 (consecutive demo timesteps)."""
    valid = (~pad_n1[:, 0]).unsqueeze(-1).to(actions_n.dtype)
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


# ── Decoder: acm2's Mamba-2 decoder + BiMamba + boundary state carry ──────────

class Mamba2SmoothDecoder(Mamba2ACMDecoder):
    """acm2's Mamba2ACMDecoder, but state-exposing (for carry) and optionally bidirectional.

    chunk_decoder=="causal": forward-only scan (== acm2 decoder behaviour, + carry).
    chunk_decoder=="bidir":  fuse(forward, flip(backward(flip))).  Cross-chunk carry ALWAYS
                             comes from the forward scan only (backward = intra-chunk only).
    """

    def __init__(self, config: ACM2SmoothConfig):
        super().__init__(config)                       # self.layers (Mamba2), self.norm
        self.chunk_decoder = config.chunk_decoder
        self.share = config.bimamba_share_weights
        self.fuse = config.bimamba_fuse
        self.backward_layers = None
        self.fuse_proj = None
        if self.chunk_decoder == "bidir":
            if not self.share:
                n = config.n_decoder_layers
                self.backward_layers = nn.ModuleList([
                    Mamba2(d_model=config.dim_model, d_state=config.mamba2_d_state,
                           d_conv=config.mamba2_d_conv, expand=config.mamba2_expand,
                           headdim=config.mamba2_headdim, chunk_size=config.mamba2_chunk_size,
                           layer_idx=n + i)
                    for i in range(n)
                ])
            if self.fuse == "proj":
                self.fuse_proj = nn.Linear(2 * config.dim_model, config.dim_model)

    def forward(self, x: Tensor, encoder_out: Tensor, decoder_pos_embed: Tensor | None = None,
                encoder_pos_embed: Tensor | None = None, carry: list | None = None):
        if decoder_pos_embed is not None:
            x = x + decoder_pos_embed
        if encoder_pos_embed is not None:
            encoder_out = encoder_out + encoder_pos_embed
        x = x.transpose(0, 1)                          # (B, K, D)
        encoder_out = encoder_out.transpose(0, 1)      # (B, T, D)
        h = torch.cat([encoder_out, x], dim=1)         # (B, T+K, D)

        # Forward (causal) scan — carries per-layer recurrent state across chunks.
        hf = h
        new_states = []
        for i, layer in enumerate(self.layers):
            init = carry[i] if carry is not None else None
            hf, st = mamba2_stateful_forward(layer, hf, initial_state=init, return_state=True)
            new_states.append(st)

        if self.chunk_decoder == "bidir":
            back = self.layers if self.share else self.backward_layers
            hb = h.flip([1])
            for layer in back:
                hb, _ = mamba2_stateful_forward(layer, hb, initial_state=None, return_state=True)
            hb = hb.flip([1])
            h_out = (self.fuse_proj(torch.cat([hf, hb], dim=-1)) if self.fuse == "proj"
                     else 0.5 * (hf + hb))
        else:
            h_out = hf

        K = x.shape[1]
        out = self.norm(h_out[:, -K:, :])
        return out.transpose(0, 1), new_states          # carry = forward states only


# ── Model: acm2 with the smooth (carry+BiMamba) decoder ───────────────────────

class ACM2Smooth(ACM2):
    """acm2 with the decoder replaced by Mamba2SmoothDecoder; forward exposes carry/state."""

    def __init__(self, config: ACM2SmoothConfig):
        super().__init__(config)
        self.decoder = Mamba2SmoothDecoder(config)
        self._reset_parameters()                        # init the new decoder's non-mamba params

    def _reset_parameters(self):
        mamba_ids = {id(p) for p in self.decoder.layers.parameters()}
        if getattr(self.decoder, "backward_layers", None) is not None:
            mamba_ids |= {id(p) for p in self.decoder.backward_layers.parameters()}
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1 and id(p) not in mamba_ids:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor], carry: list | None = None,
                return_state: bool = False) -> tuple:
        # Mirrors ACM2.forward (acm2), but threads carry into the decoder and returns state.
        if self.config.use_vae and self.training:
            assert ACTION in batch, "actions required for the VAE objective in training."
        import einops
        bs = (batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0])

        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=bs)
            vae_in = [cls_embed]
            if self.config.robot_state_feature:
                vae_in.append(self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1))
            vae_in.append(self.vae_encoder_action_input_proj(batch[ACTION]))
            vae_in = torch.cat(vae_in, axis=1)
            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            _ref = batch.get(OBS_ENV_STATE, batch.get(OBS_STATE))
            cls_joint_is_pad = torch.full((bs, 2 if self.config.robot_state_feature else 1), False,
                                          device=_ref.device)
            kpm = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)
            cls_out = self.vae_encoder(vae_in.permute(1, 0, 2), pos_embed=pos_embed.permute(1, 0, 2),
                                       key_padding_mask=kpm)[0]
            lp = self.vae_encoder_latent_output_proj(cls_out)
            mu, log_sigma_x2 = lp[:, : self.config.latent_dim], lp[:, self.config.latent_dim:]
            latent = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent = torch.zeros([bs, self.config.latent_dim], dtype=torch.float32,
                                 device=batch[OBS_STATE].device)

        enc_tokens = [self.encoder_latent_input_proj(latent)]
        enc_pos = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            enc_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            enc_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam = self.backbone(img)["feature_map"]
                cam_pos = self.encoder_cam_feat_pos_embed(cam).to(dtype=cam.dtype)
                cam = self.encoder_img_feat_input_proj(cam)
                enc_tokens.extend(list(einops.rearrange(cam, "b c h w -> (h w) b c")))
                enc_pos.extend(list(einops.rearrange(cam_pos, "b c h w -> (h w) b c")))
        enc_tokens = torch.stack(enc_tokens, axis=0)
        enc_pos = torch.stack(enc_pos, axis=0)

        encoder_out = self.encoder(enc_tokens, pos_embed=enc_pos)
        decoder_in = torch.zeros((self.config.chunk_size, bs, self.config.dim_model),
                                 dtype=enc_pos.dtype, device=enc_pos.device)
        decoder_out, new_states = self.decoder(
            decoder_in, encoder_out, encoder_pos_embed=enc_pos,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1), carry=carry)
        actions = self.action_head(decoder_out.transpose(0, 1))

        if return_state:
            return actions, (mu, log_sigma_x2), new_states
        return actions, (mu, log_sigma_x2)


# ── Policy: acm2 open-loop chunk inference + boundary carry + smoothness loss ──

class ACM2SmoothPolicy(ACM2Policy):
    """Smooth chunking: ACT encoder × BiMamba decoder × pure boundary state carry + smooth loss."""

    config_class = ACM2SmoothConfig
    name = "acm2_smooth"

    def __init__(self, config: ACM2SmoothConfig):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.model = ACM2Smooth(config)
        if config.temporal_ensemble_coeff is not None:
            from lerobot.policies.acm2.modeling_acm2 import ACTTemporalEnsembler
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)
        self.reset()

    # ── Inference ───────────────────────────────────────────────────────────────
    def reset(self):
        super().reset()                                  # _action_queue / temporal_ensembler
        self._carry: list | None = None                  # full-mode episode carry
        self._depth1_carry: list | None = None           # depth1-mode: fresh state of prev chunk

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.temporal_ensemble_coeff is not None:
            actions = self._predict_with_carry(batch)
            return self.temporal_ensembler.update(actions)
        if len(self._action_queue) == 0:
            actions = self._predict_with_carry(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        return self._predict_with_carry(batch)

    def _predict_with_carry(self, batch: dict[str, Tensor]) -> Tensor:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        mode = self.config.carry_infer_mode if self.config.carry_enabled else "off"

        if mode == "off":                                # acm2 zero-carry
            actions, _, _ = self.model(batch, carry=None, return_state=True)
            return actions

        if mode == "depth1":
            # v17: carry = state from a FRESH (carry=None) decode of the PREVIOUS chunk's obs.
            # This is exactly the single-boundary chunk-pair training condition
            # (chunk_n: carry=None → state_n; chunk_{n+1}: carry=state_n) applied at EVERY
            # boundary. The carried state is always depth-1 (never compounds) → cannot diverge
            # across the episode, so base SR is preserved while the seam is still stitched.
            # Pair with carry_train_chunks=2 (single-hop training).
            carry = self._depth1_carry
            actions, _, states = self.model(batch, carry=carry, return_state=True)
            if carry is None:
                self._depth1_carry = detach_states(states)   # 1st chunk: actual decode IS fresh
            else:
                _, _, fresh = self.model(batch, carry=None, return_state=True)
                self._depth1_carry = detach_states(fresh)
            return actions

        # mode == "full" (default): accumulate the carry across the whole episode. Matches
        # training only when trained with the carry_train_chunks=M rollout (v16).
        actions, _, new_states = self.model(batch, carry=self._carry, return_state=True)
        self._carry = detach_states(new_states)          # pure handoff (no gate/noise)
        return actions

    # ── Training ─────────────────────────────────────────────────────────────────
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        # Multi-chunk carry rollout (train == inference carry depth). Triggered when the
        # ReactiveSegmentDataset(n_samples=0) ships M consecutive chunks under step{j}_ keys.
        n_seq = int(getattr(self.config, "carry_train_chunks", 2) or 2)
        use_seq = self.config.carry_enabled and n_seq > 2 and "step0_action" in batch
        if self.config.image_features and not use_seq:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        if use_seq:
            return self._forward_smooth_seq(batch, n_seq)
        has_pairs = "action_n1" in batch
        if self.config.carry_enabled and self.config.carry_p > 0.0 and has_pairs:
            return self._forward_smooth_pair(batch)
        return self._forward_smooth_single(batch)

    def _chunk_losses(self, batch, carry):
        cfg = self.config
        actions, (mu, log_sigma_x2), new_states = self.model(batch, carry=carry, return_state=True)
        pad = batch["action_is_pad"]
        l1 = _masked_l1(batch[ACTION], actions, pad)
        jerk = self._jerk_term(actions, batch[ACTION], pad)
        kld = actions.new_zeros(())
        if cfg.use_vae:
            kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
        return l1, jerk, kld, actions, new_states

    def _smooth_dims(self, x: Tensor) -> Tensor:
        """A-fix: drop action dims excluded from the smoothness losses (e.g. gripper dims that
        must move sharply to grasp). Slices the last (action) axis only. No-op when empty."""
        ex = self.config.smooth_exclude_dims
        if not ex:
            return x
        keep = [d for d in range(x.shape[-1]) if d not in ex]
        return x[..., keep]

    def _vae_free_states(self, batch: dict, carry: list | None) -> list:
        """B-fix: produce the carry the way inference does — decode with latent=0 (no VAE) by
        dropping the ACTION key — so the carried-state distribution matches eval. Detached
        (no grad through the carry source), matching the truncated-BPTT handoff."""
        b = {k: v for k, v in batch.items() if k != ACTION}
        with torch.no_grad():
            _, _, st = self.model(b, carry=carry, return_state=True)
        return st

    def _jerk_term(self, actions: Tensor, gt: Tensor, pad: Tensor) -> Tensor:
        cfg = self.config
        a, g = self._smooth_dims(actions), self._smooth_dims(gt)
        j = _jerk(a, pad, cfg.smooth_jerk_mode)
        if cfg.smooth_jerk_excess_only:
            j = (j - _jerk(g, pad, cfg.smooth_jerk_mode).detach()).clamp_min(0.0)
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
        if cfg.carry_train_vae_free:                                         # B-fix: VAE-free carry
            states_n = self._vae_free_states(batch_n, carry=None)
        carry = detach_states(states_n) if cfg.carry_detach else states_n   # pure handoff

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
        bc = _boundary_continuity(self._smooth_dims(actions_n), self._smooth_dims(actions_n1),
                                  batch_n1["action_is_pad"], cfg.smooth_boundary_velocity)

        l1, jerk = l1_n + l1_n1, jerk_n + jerk_n1
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

    # ── Multi-chunk carry rollout (train == inference) ────────────────────────────
    def _build_step_batch(self, batch: dict, j: int) -> dict | None:
        """Reconstruct chunk j's standard batch from the dataset's `step{j}_` keys.

        ReactiveSegmentDataset(n_samples=0) ships, per consecutive chunk j, the GT action
        chunk + that boundary's observation (all normalized to the preprocessed space).
        """
        cfg = self.config
        if f"step{j}_action" not in batch:
            return None
        out: dict = {ACTION: batch[f"step{j}_action"],
                     "action_is_pad": batch[f"step{j}_action_is_pad"]}
        sk = batch.get(f"step{j}_{OBS_STATE}")
        if sk is not None:
            out[OBS_STATE] = sk
        ek = batch.get(f"step{j}_{OBS_ENV_STATE}")
        if ek is not None:
            out[OBS_ENV_STATE] = ek
        if cfg.image_features:
            imgs = [batch[f"step{j}_{key}"] for key in cfg.image_features
                    if f"step{j}_{key}" in batch]
            if imgs:
                out[OBS_IMAGES] = imgs
        return out

    def _forward_smooth_seq(self, batch: dict, n_chunks: int):
        """Roll the recurrent carry through M consecutive chunks, detaching at each boundary.

        This reproduces inference exactly: chunk 0 starts from a zero carry, chunk j>0 starts
        from the (detached) forward-scan state committed by chunk j-1 — so the model is trained
        on the SAME accumulated-carry distribution (depth 0..M-1) it meets at eval time, instead
        of only the single n→n+1 hop that ChunkPairDataset exposes.
        """
        cfg = self.config
        carry = None
        prev_actions = None
        l1s, jerks, klds, bcs = [], [], [], []
        for j in range(n_chunks):
            batch_j = self._build_step_batch(batch, j)
            if batch_j is None:
                break
            l1_j, jerk_j, kld_j, actions_j, states_j = self._chunk_losses(batch_j, carry=carry)
            l1s.append(l1_j)
            jerks.append(jerk_j)
            klds.append(kld_j)
            if prev_actions is not None and cfg.smooth_boundary_weight > 0:
                bcs.append(_boundary_continuity(self._smooth_dims(prev_actions), self._smooth_dims(actions_j),
                                                batch_j["action_is_pad"], cfg.smooth_boundary_velocity))
            # Pure boundary handoff: detach so each chunk's graph stays shallow (truncated BPTT).
            nxt = self._vae_free_states(batch_j, carry) if cfg.carry_train_vae_free else states_j  # B-fix
            carry = detach_states(nxt) if cfg.carry_detach else nxt
            prev_actions = actions_j

        l1 = torch.stack(l1s).mean()
        jerk = torch.stack(jerks).mean()
        loss = l1 + cfg.smooth_jerk_weight * jerk
        ld = {"l1_loss": float(l1.detach()), "jerk_loss": float(jerk.detach()),
              "n_chunks": float(len(l1s))}
        if bcs:
            bc = torch.stack(bcs).mean()
            loss = loss + cfg.smooth_boundary_weight * bc
            ld["boundary_loss"] = float(bc.detach())
        if cfg.use_vae:
            kld = torch.stack(klds).mean()
            loss = loss + kld * cfg.kl_weight
            ld["kld_loss"] = float(kld.detach())
        ld["loss"] = float(loss.detach())
        return loss, ld
