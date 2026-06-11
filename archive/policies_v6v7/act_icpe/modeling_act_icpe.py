"""ACT + Intra-Chunk Phase Embedding (ICPE) — control experiment.

Applies the identical ICPE phase signal φ(i, K) to ACT's Transformer decoder
input queries. Structurally mirrors the change made in ACM3ICPE.

Expected: no meaningful improvement over vanilla ACT, because:
  - ACT's Transformer decoder computes self-attention over ALL K positions at once.
  - Every query already attends globally → position is implicitly available.
  - The learnable decoder_pos_embed already covers positional information.
  - ICPE is therefore redundant for parallel generation.

Contrast with ACM3ICPE where ICPE yields +22pp: there, Mamba3 processes positions
sequentially (h_{i} depends on h_{i-1}), so explicit phase signals meaningfully
calibrate per-position generation.

This file exists solely to provide the ablation/control evidence that ICPE's
effectiveness is SSM-specific.
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act_icpe.configuration_act_icpe import ACTICPEConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

# Re-use ACT utilities
from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
    get_activation_fn,
)
# Re-use ICPE signal builder
from lerobot.policies.acm3_icpe.modeling_acm3_icpe import make_icpe_signal


class ACTICPEPolicy(PreTrainedPolicy):
    """ACT with ICPE decoder query augmentation (control experiment)."""

    config_class = ACTICPEConfig
    name = "act_icpe"

    def __init__(self, config: ACTICPEConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = ACTICPE(config)
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)
        self.reset()

    def get_optim_params(self):
        return [
            {"params": [p for n, p in self.named_parameters()
                        if not n.startswith("model.backbone") and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters()
                        if n.startswith("model.backbone") and p.requires_grad],
             "lr": self.config.optimizer_lr_backbone},
        ]

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.temporal_ensemble_coeff is not None:
            return self.temporal_ensembler.update(self.predict_action_chunk(batch))
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return self.model(batch)[0]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu, log_sigma_x2) = self.model(batch)

        K      = actions_hat.shape[1]
        n_exec = self.config.n_action_steps
        device = actions_hat.device
        weights = torch.ones(K, device=device)

        if getattr(self.config, "use_temporal_weighting", False) and n_exec < K:
            exec_mass = getattr(self.config, "temporal_execution_weight", 0.9)
            weights[:n_exec] = (K * exec_mass) / n_exec
            weights[n_exec:] = (K * (1 - exec_mass)) / (K - n_exec)

        l1 = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none")
            * ~batch["action_is_pad"].unsqueeze(-1)
            * weights.view(1, -1, 1)
        ).mean()

        loss_dict = {"l1_loss": l1.item()}
        if self.config.use_vae:
            kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
            loss_dict["kld_loss"] = kld.item()
            return l1 + kld * self.config.kl_weight, loss_dict
        return l1, loss_dict


class ACTICPE(nn.Module):
    """ACT with ICPE phase signal added to Transformer decoder queries.

    Identical to ACT except decoder_in[i] += icpe_proj(φ(i, K)).
    """

    def __init__(self, config: ACTICPEConfig):
        super().__init__()
        self.config = config

        # ── VAE encoder ───────────────────────────────────────────────────────
        if config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                config.action_feature.shape[0], config.dim_model
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            n_vae_tokens = 1 + config.chunk_size + (1 if config.robot_state_feature else 0)
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(n_vae_tokens, config.dim_model).unsqueeze(0),
            )

        # ── Vision backbone ───────────────────────────────────────────────────
        if config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # ── Transformer encoder + decoder ─────────────────────────────────────
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # ── Encoder projections ───────────────────────────────────────────────
        if config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.robot_state_feature.shape[0], config.dim_model
            )
        if config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)

        n_1d = 1 + (1 if config.robot_state_feature else 0) + (1 if config.env_state_feature else 0)
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d, config.dim_model)

        # ── Decoder positional embedding + ICPE projection ────────────────────
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.icpe_proj = nn.Linear(config.icpe_dim, config.dim_model, bias=False)
        nn.init.normal_(self.icpe_proj.weight, std=config.icpe_scale_init / math.sqrt(config.dim_model))

        # ── Action head ───────────────────────────────────────────────────────
        self.action_head = nn.Linear(config.dim_model, config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        icpe_ids = {id(p) for p in self.icpe_proj.parameters()}
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1 and id(p) not in icpe_ids:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple]:
        if self.config.use_vae and self.training:
            assert ACTION in batch, "Need action labels for VAE training."

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # ── VAE latent ────────────────────────────────────────────────────────
        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            if self.config.robot_state_feature:
                rs = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)
            act = self.vae_encoder_action_input_proj(batch[ACTION])

            vae_in = [cls_embed] + ([rs] if self.config.robot_state_feature else []) + [act]
            vae_in = torch.cat(vae_in, dim=1)

            pos = self.vae_encoder_pos_enc.clone().detach()
            n_prefix = 2 if self.config.robot_state_feature else 1
            is_pad_pfx = torch.zeros(batch_size, n_prefix, dtype=torch.bool, device=vae_in.device)
            key_pad = torch.cat([is_pad_pfx, batch["action_is_pad"]], dim=1)

            cls_out = self.vae_encoder(vae_in.permute(1, 0, 2),
                                       pos_embed=pos.permute(1, 0, 2),
                                       key_padding_mask=key_pad)[0]
            params = self.vae_encoder_latent_output_proj(cls_out)
            mu, log_sigma_x2 = params[:, :self.config.latent_dim], params[:, self.config.latent_dim:]
            latent = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            if OBS_ENV_STATE in batch:
                _ref = batch[OBS_ENV_STATE]
            elif OBS_STATE in batch:
                _ref = batch[OBS_STATE]
            else:
                _ref = next(v for v in batch.values() if v is not None and hasattr(v, 'device'))
            latent = torch.zeros(batch_size, self.config.latent_dim,
                                 dtype=torch.float32, device=_ref.device)

        # ── Encoder ───────────────────────────────────────────────────────────
        enc_tokens = [self.encoder_latent_input_proj(latent)]
        enc_pos    = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.config.robot_state_feature:
            enc_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            enc_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                feat = self.backbone(img)["feature_map"]
                cam_pos = self.encoder_cam_feat_pos_embed(feat).to(dtype=feat.dtype)
                feat    = self.encoder_img_feat_input_proj(feat)
                feat    = einops.rearrange(feat,    "b c h w -> (h w) b c")
                cam_pos = einops.rearrange(cam_pos, "b c h w -> (h w) b c")
                enc_tokens.extend(list(feat))
                enc_pos.extend(list(cam_pos))

        enc_tokens = torch.stack(enc_tokens, dim=0)
        enc_pos    = torch.stack(enc_pos,    dim=0)
        encoder_out = self.encoder(enc_tokens, pos_embed=enc_pos)

        # ── Decoder with ICPE ─────────────────────────────────────────────────
        K      = self.config.chunk_size
        dtype  = enc_pos.dtype
        device = enc_pos.device

        decoder_in = torch.zeros((K, batch_size, self.config.dim_model), dtype=dtype, device=device)

        # ICPE: same signal as in ACM3ICPE (control: this should NOT help ACT)
        phase    = make_icpe_signal(K, self.config.icpe_mode, device, dtype)  # (K, icpe_dim)
        icpe_emb = self.icpe_proj(phase).unsqueeze(1)                          # (K, 1, D)
        decoder_in = decoder_in + icpe_emb                                     # (K, B, D) via broadcast

        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=enc_pos,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        decoder_out = decoder_out.transpose(0, 1)    # (B, K, D)
        actions = self.action_head(decoder_out)       # (B, K, action_dim)
        return actions, (mu, log_sigma_x2)
