#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ACM2-DRO Policy — "train as a chunker, act as a controller".

The Mamba-2 decoder scans [encoder context tokens | interleaved (query, proprio) tokens]:

    enc(vision+state @ chunk start), q_0, p_1, q_1, p_2, q_2, ..., q_{K-1}

Action a_k is read at q_k; p_k is the proprioception MEASURED after executing a_{k-1}
(teacher-forced from the dataset during training, live from the robot at inference).
Because the decoder is a causal SSM, feeding the fresh measurement is one recurrent
update — a disturbance shows up in the state, and therefore in the next action,
one control step later instead of up to `chunk_size` steps later (open-loop ACT/acm2).

Layer B (dro_innovation) adds an observer head: at q_k the model predicts the next
proprio ô_{k+1}; the innovation e_{k+1} = o_{k+1} − ô_{k+1} is fed back with p_{k+1}
(Luenberger/Kalman innovation form) and ‖e‖ can gate a fresh-vision context refresh.

Inference recomputes the full decoder scan each step (1 Mamba layer × ~E+2K tokens —
cheap; the vision backbone runs only at chunk starts / gate triggers). An incremental
`inference_params` path is a future optimization, not needed for the benchmark.
"""

import os
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.acm2.modeling_acm2 import ACM2, ACM2Policy
from lerobot.policies.acm2_dro.configuration_acm2_dro import ACM2DROConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

OBS_STATE_IS_PAD = f"{OBS_STATE}_is_pad"


class ACM2DROPolicy(PreTrainedPolicy):
    config_class = ACM2DROConfig
    name = "acm2_dro"

    def __init__(self, config: ACM2DROConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = DROACM2(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return ACM2Policy.get_optim_params(self)

    def reset(self):
        """Called on env reset. Clears the stream (and dumps the innovation trace if asked)."""
        self._maybe_dump_innov_trace()
        self._k = 0
        self._enc_ctx = None
        self._dec_tokens = None
        self._prev_obs_pred = None
        self._innov_trace = []       # per-step (B,) innovation norms, for the gate figure
        self._gate_trace = []        # per-step (B,) bool triggers
        if not self.config.dro_stream:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    # ------------------------------------------------------------------ train
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat), obs_pred = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none")
            * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()
        loss_dict = {"l1_loss": l1_loss.item()}
        loss = l1_loss

        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = loss + mean_kld * self.config.kl_weight

        if obs_pred is not None:
            # ô at q_k targets o_{k+1}: compare obs_pred[:, :-1] with states[:, 1:].
            target = batch[OBS_STATE][:, 1:]
            pred = obs_pred[:, :-1]
            obs_l1 = F.l1_loss(target, pred, reduction="none")
            if OBS_STATE_IS_PAD in batch:
                obs_l1 = obs_l1 * ~batch[OBS_STATE_IS_PAD][:, 1:].unsqueeze(-1)
            obs_l1 = obs_l1.mean()
            loss_dict["obs_l1"] = obs_l1.item()
            loss = loss + self.config.dro_obs_loss_weight * obs_l1

        return loss, loss_dict

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Full-chunk prediction (API compat / non-stream mode). In stream mode the
        chunk is imagined by self-feeding the observer's own next-proprio predictions."""
        self.eval()
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        if not self.config.dro_stream:
            return self.model(batch)[0]
        return self.model.imagine_chunk(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if not self.config.dro_stream:
            if len(self._action_queue) == 0:
                actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
                self._action_queue.extend(actions.transpose(0, 1))
            return self._action_queue.popleft()
        return self._stream_step(batch)

    def _stream_step(self, batch: dict[str, Tensor]) -> Tensor:
        cfg = self.config
        if cfg.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in cfg.image_features]
        o = batch[OBS_STATE]  # (B, Ds) normalized
        pos = self.model.decoder_pos_embed.weight  # (K, D)
        B = o.shape[0]

        if self._k == 0 or self._enc_ctx is None:
            # Chunk start: fresh vision context; q_0 emits a_0 (o_0 is inside the context).
            self._enc_ctx = self.model.encoder_context_inference(batch)
            self._dec_tokens = pos[0].expand(B, 1, -1).to(o.dtype)
            self._prev_obs_pred = None
            self._innov_trace.append(torch.zeros(B, device=o.device))
            self._gate_trace.append(torch.zeros(B, dtype=torch.bool, device=o.device))
        else:
            e = None
            if cfg.dro_innovation and self._prev_obs_pred is not None:
                e = o - self._prev_obs_pred
                self._innov_trace.append(e.norm(dim=-1))
            else:
                self._innov_trace.append(torch.zeros(B, device=o.device))

            trig = torch.zeros(B, dtype=torch.bool, device=o.device)
            if cfg.dro_vision_refresh == "every":
                self._enc_ctx = self.model.encoder_context_inference(batch)
                trig = torch.ones_like(trig)
            elif cfg.dro_vision_refresh == "gate" and e is not None:
                trig = e.norm(dim=-1) > cfg.dro_gate_tau
                if trig.any():
                    fresh = self.model.encoder_context_inference(batch)
                    self._enc_ctx = torch.where(trig.view(B, 1, 1), fresh, self._enc_ctx)
            self._gate_trace.append(trig)

            p_k = self.model.proprio_token(o, self._k, e)          # (B, 1, D)
            q_k = pos[self._k].expand(B, 1, -1).to(o.dtype)
            self._dec_tokens = torch.cat([self._dec_tokens, p_k, q_k], dim=1)

        last = self.model.scan_last(self._enc_ctx, self._dec_tokens)  # (B, D)
        action = self.model.action_head(last)
        if cfg.dro_innovation:
            self._prev_obs_pred = self.model.obs_head(last)

        self._k += 1
        if self._k >= cfg.chunk_size:
            self._k = 0
        return action

    def _maybe_dump_innov_trace(self):
        """DRO_INNOV_DIR가 설정되면 에피소드별 innovation/gate trace를 저장 (killer figure용)."""
        out = os.environ.get("DRO_INNOV_DIR")
        if not out or not getattr(self, "_innov_trace", None):
            return
        try:
            path = Path(out)
            path.mkdir(parents=True, exist_ok=True)
            idx = len(list(path.glob(f"innov_{os.getpid()}_*.pt")))
            torch.save(
                {
                    "innovation": torch.stack(self._innov_trace).cpu(),   # (T, B)
                    "gate": torch.stack(self._gate_trace).cpu(),          # (T, B)
                },
                path / f"innov_{os.getpid()}_{idx:04d}.pt",
            )
        except Exception:  # noqa: BLE001 — logging must never kill an eval
            pass


class DROACM2(ACM2):
    """ACM2 with interleaved proprio streaming + optional innovation observer."""

    def __init__(self, config: ACM2DROConfig):
        super().__init__(config)
        if config.dro_stream:
            if not config.robot_state_feature:
                raise ValueError("dro_stream needs a robot state feature (proprioception).")
            state_dim = config.robot_state_feature.shape[0]
            self.proprio_proj = nn.Linear(state_dim, config.dim_model)
            self.proprio_type_embed = nn.Parameter(torch.zeros(config.dim_model))
            if config.dro_innovation:
                self.obs_head = nn.Linear(config.dim_model, state_dim)
                self.innov_proj = nn.Linear(state_dim, config.dim_model)
                # Zero-init: innovation feedback starts as a no-op and is learned.
                nn.init.zeros_(self.innov_proj.weight)
                nn.init.zeros_(self.innov_proj.bias)

    # -------------------------------------------------------------- components
    def _vae_latent(self, batch: dict[str, Tensor], current_state: Tensor):
        """VAE latent from (current state, action chunk) — base logic with explicit state."""
        batch_size = current_state.shape[0]
        cls_embed = self.vae_encoder_cls_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        robot_state_embed = self.vae_encoder_robot_state_input_proj(current_state).unsqueeze(1)
        action_embed = self.vae_encoder_action_input_proj(batch[ACTION])
        vae_encoder_input = torch.cat([cls_embed, robot_state_embed, action_embed], axis=1)
        pos_embed = self.vae_encoder_pos_enc.clone().detach()
        cls_joint_is_pad = torch.full((batch_size, 2), False, device=current_state.device)
        key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)
        cls_token_out = self.vae_encoder(
            vae_encoder_input.permute(1, 0, 2),
            pos_embed=pos_embed.permute(1, 0, 2),
            key_padding_mask=key_padding_mask,
        )[0]
        latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
        mu = latent_pdf_params[:, : self.config.latent_dim]
        log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
        latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        return latent_sample, mu, log_sigma_x2

    def _encoder_context(self, batch: dict[str, Tensor], latent_sample: Tensor,
                         current_state: Tensor) -> Tensor:
        """Transformer-encoded context with pos embeds folded in. Returns (B, E, D)."""
        import einops

        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(current_state))
        if self.config.env_state_feature:
            from lerobot.utils.constants import OBS_ENV_STATE

            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # Fold pos embeds into the context, as Mamba2ACMDecoder.forward does.
        return (encoder_out + encoder_in_pos_embed).transpose(0, 1)  # (B, E, D)

    @torch.no_grad()
    def encoder_context_inference(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch[OBS_STATE]
        if state.ndim == 3:
            state = state[:, 0]
        latent = torch.zeros(
            [state.shape[0], self.config.latent_dim], dtype=torch.float32, device=state.device
        )
        return self._encoder_context(batch, latent, state)

    def proprio_token(self, state: Tensor, k, innov: Tensor | None = None) -> Tensor:
        """Measured-proprio token(s) at chunk position(s) k. state (B,Ds)→(B,1,D) or
        (B,K',Ds)→(B,K',D) with k an index tensor of shape (K',)."""
        pos = self.decoder_pos_embed.weight[k]  # (D,) or (K', D)
        tok = self.proprio_proj(state) + pos + self.proprio_type_embed
        if innov is not None:
            tok = tok + self.innov_proj(innov)
        if tok.ndim == 2:
            tok = tok.unsqueeze(1)
        return tok

    def _run_decoder(self, seq: Tensor) -> Tensor:
        for layer in self.decoder.layers:
            seq = layer(seq)
        return seq

    def scan_last(self, enc_ctx: Tensor, dec_tokens: Tensor) -> Tensor:
        """Full causal scan over [context | interleaved tokens]; return last position (B, D)."""
        seq = self._run_decoder(torch.cat([enc_ctx, dec_tokens], dim=1))
        return self.decoder.norm(seq[:, -1])

    def _decode_stream(self, enc_ctx: Tensor, proprio: Tensor, innov: Tensor | None):
        """Teacher-forced interleaved decode. proprio (B,K,Ds) → query outs (B,K,D), ô (B,K,Ds)|None."""
        B, K, _ = proprio.shape
        D = self.config.dim_model
        pos = self.decoder_pos_embed.weight  # (K, D)
        q = pos.unsqueeze(0).expand(B, K, D)
        ks = torch.arange(K, device=proprio.device)
        p = self.proprio_token(proprio, ks, innov)  # (B, K, D); index 0 unused
        dec = q.new_zeros(B, 2 * K - 1, D)
        dec[:, 0::2] = q
        dec[:, 1::2] = p[:, 1:]
        seq = self._run_decoder(torch.cat([enc_ctx, dec], dim=1))
        out = self.decoder.norm(seq[:, enc_ctx.shape[1] :][:, 0::2])  # (B, K, D) at q_k
        obs_pred = self.obs_head(out) if self.config.dro_innovation else None
        return out, obs_pred

    def _augment_teacher(self, states: Tensor) -> Tensor:
        """Layer C: disturbance-injection on teacher-forced proprio (normalized space).
        Index 0 (encoder input) is never touched."""
        cfg = self.config
        if not (cfg.dro_train_state_noise > 0 or cfg.dro_train_push_prob > 0):
            return states
        states = states.clone()
        B, K, Ds = states.shape
        if cfg.dro_train_state_noise > 0:
            states[:, 1:] += cfg.dro_train_state_noise * torch.randn_like(states[:, 1:])
        if cfg.dro_train_push_prob > 0 and K > 2:
            hit = torch.rand(B, device=states.device) < cfg.dro_train_push_prob
            if hit.any():
                k0 = torch.randint(1, K - 1, (B,), device=states.device)
                d = torch.randn(B, Ds, device=states.device)
                d = d / (d.norm(dim=-1, keepdim=True) + 1e-9)
                scale = cfg.dro_train_push_mag * (0.5 + torch.rand(B, 1, device=states.device))
                offset = d * scale * hit.view(B, 1).float()
                after = torch.arange(K, device=states.device).view(1, K) >= k0.view(B, 1)
                states = states + offset.unsqueeze(1) * after.unsqueeze(-1).float()
        return states

    # ------------------------------------------------------------------ forward
    def forward(self, batch: dict[str, Tensor]):
        if not self.config.dro_stream:
            actions, latents = super().forward(batch)
            return actions, latents, None

        states = batch[OBS_STATE]
        if states.ndim != 3:
            raise ValueError(
                f"dro_stream training expects stacked proprio (B,K,Ds) via state_delta_indices; "
                f"got shape {tuple(states.shape)}."
            )
        current = states[:, 0]

        if self.config.use_vae and ACTION in batch and self.training:
            latent_sample, mu, log_sigma_x2 = self._vae_latent(batch, current)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros(
                [current.shape[0], self.config.latent_dim], dtype=torch.float32, device=current.device
            )

        enc_ctx = self._encoder_context(batch, latent_sample, current)

        teacher = self._augment_teacher(states) if self.training else states

        out, obs_pred = self._decode_stream(enc_ctx, teacher, innov=None)
        if self.config.dro_innovation:
            # Two-pass: innovation e_k = o_k − ô_k, where ô_k was predicted at q_{k-1}.
            e = torch.zeros_like(teacher)
            e[:, 1:] = teacher[:, 1:] - obs_pred[:, :-1].detach()
            out, obs_pred = self._decode_stream(enc_ctx, teacher, innov=e)

        actions = self.action_head(out)
        return actions, (mu, log_sigma_x2), obs_pred

    # ------------------------------------------------------- imagination (compat)
    @torch.no_grad()
    def imagine_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Open-loop chunk for API compat: self-feed ô (observer world-model rollout) if
        available, else hold the current proprio. NOT the DRO inference path."""
        enc_ctx = self.encoder_context_inference(batch)
        state = batch[OBS_STATE]
        if state.ndim == 3:
            state = state[:, 0]
        B = state.shape[0]
        pos = self.decoder_pos_embed.weight
        dec = pos[0].expand(B, 1, -1).to(state.dtype)
        actions, prev_pred = [], None
        for k in range(self.config.chunk_size):
            if k > 0:
                p_state = prev_pred if prev_pred is not None else state
                dec = torch.cat([dec, self.proprio_token(p_state, k), pos[k].expand(B, 1, -1)], dim=1)
            last = self.scan_last(enc_ctx, dec)
            actions.append(self.action_head(last))
            if self.config.dro_innovation:
                prev_pred = self.obs_head(last)
        return torch.stack(actions, dim=1)
