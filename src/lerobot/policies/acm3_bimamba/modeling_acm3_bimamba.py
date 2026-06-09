"""ACM3 + Bidirectional Mamba3 Decoder

Replaces the unidirectional Mamba3ACMDecoder with a Vim-style external BiMamba:
  1. Forward pass  : process combined_seq left-to-right
  2. Backward pass : flip → process → flip back (right-to-left)
  3. Average       : 0.5 * (forward + backward)  — keeps activation scale stable

Two independent Mamba3 layer stacks share no weights.  Decoder parameter count is 2×
compared to ACM3.  All other components (encoder, VAE, action head) are identical.
"""

from collections import deque
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

try:
    from mamba_ssm import Mamba3
    HAS_MAMBA3 = True
except ImportError:
    HAS_MAMBA3 = False

from lerobot.policies.acm3_bimamba.configuration_acm3_bimamba import ACM3BiMambaConfig
from lerobot.policies.acm3.modeling_acm3 import (
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class Mamba3BiMambaDecoder(nn.Module):
    """Bidirectional Mamba3 decoder (Vim-style external BiMamba).

    Processes the combined [encoder_out | action_queries] sequence in both directions
    and averages the results.  The 0.5 scale keeps the output magnitude comparable to
    the unidirectional baseline, making it easy to swap without retuning LR.
    """

    def __init__(self, config: ACM3BiMambaConfig):
        super().__init__()
        if not HAS_MAMBA3:
            raise ImportError(
                "Mamba3 support is required. Install mamba-ssm from source:\n"
                "  MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall "
                "git+https://github.com/state-spaces/mamba.git --no-build-isolation"
            )

        def _make_layers():
            return nn.ModuleList([
                Mamba3(
                    d_model=config.dim_model,
                    d_state=config.mamba3_d_state,
                    expand=config.mamba3_expand,
                    headdim=config.mamba3_headdim,
                    ngroups=config.mamba3_ngroups,
                    rope_fraction=config.mamba3_rope_fraction,
                    is_outproj_norm=config.mamba3_is_outproj_norm,
                    is_mimo=config.mamba3_is_mimo,
                    mimo_rank=config.mamba3_mimo_rank,
                    chunk_size=config.mamba3_chunk_size,
                    layer_idx=i,
                    n_layer=config.n_decoder_layers,
                )
                for i in range(config.n_decoder_layers)
            ])

        self.forward_layers = _make_layers()
        self.backward_layers = _make_layers()
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,            # (K, B, D) action queries
        encoder_out: Tensor,  # (T, B, D) encoder context
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:              # returns (K, B, D)
        if decoder_pos_embed is not None:
            x = x + decoder_pos_embed
        if encoder_pos_embed is not None:
            encoder_out = encoder_out + encoder_pos_embed

        x = x.transpose(0, 1)            # (B, K, D)
        encoder_out = encoder_out.transpose(0, 1)  # (B, T, D)
        combined = torch.cat([encoder_out, x], dim=1)  # (B, T+K, D)

        # Forward direction
        fwd = combined
        for layer in self.forward_layers:
            fwd = layer(fwd)

        # Backward direction: flip sequence, process, flip back
        bwd = combined.flip(dims=[1])
        for layer in self.backward_layers:
            bwd = layer(bwd)
        bwd = bwd.flip(dims=[1])

        # Average to maintain activation scale
        combined = 0.5 * (fwd + bwd)

        chunk_size = x.shape[1]
        out = combined[:, -chunk_size:, :]
        out = self.norm(out)
        return out.transpose(0, 1)  # (K, B, D)


class ACM3BiMamba(nn.Module):
    """ACM3 with BiMamba decoder. Identical to ACM3 except the decoder."""

    def __init__(self, config: ACM3BiMambaConfig):
        super().__init__()
        self.config = config

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
            num_input_token_encoder = 1 + config.chunk_size
            if config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        if config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        self.encoder = ACTEncoder(config)
        self.decoder = Mamba3BiMambaDecoder(config)

        if config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.robot_state_feature.shape[0], config.dim_model
            )
        if config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        n_1d_tokens = 1
        if config.robot_state_feature:
            n_1d_tokens += 1
        if config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, config.action_feature.shape[0])
        self._reset_parameters()

    def _reset_parameters(self):
        # Exclude both BiMamba layer stacks from Xavier init
        mamba_ids = (
            {id(p) for p in self.decoder.forward_layers.parameters()} |
            {id(p) for p in self.decoder.backward_layers.parameters()}
        )
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1 and id(p) not in mamba_ids:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        if self.config.use_vae and self.training:
            assert ACTION in batch

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])
            vae_in = [cls_embed] + ([robot_state_embed] if self.config.robot_state_feature else []) + [action_embed]
            vae_encoder_input = torch.cat(vae_in, axis=1)
            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            if OBS_ENV_STATE in batch:
                _ref = batch[OBS_ENV_STATE]
            elif OBS_STATE in batch:
                _ref = batch[OBS_STATE]
            else:
                _ref = next(v for v in batch.values() if v is not None and hasattr(v, 'device'))
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1), False, device=_ref.device
            )
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
        else:
            mu = log_sigma_x2 = None
            if OBS_ENV_STATE in batch:
                _ref = batch[OBS_ENV_STATE]
            elif OBS_STATE in batch:
                _ref = batch[OBS_STATE]
            else:
                _ref = next(v for v in batch.values() if v is not None and hasattr(v, "device"))
            latent_sample = torch.zeros(
                [batch_size, self.config.latent_dim], dtype=torch.float32, device=_ref.device
            )

        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
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
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        actions = self.action_head(decoder_out.transpose(0, 1))
        return actions, (mu, log_sigma_x2)


class ACM3BiMambaPolicy(PreTrainedPolicy):
    config_class = ACM3BiMambaConfig
    name = "acm3_bimamba"

    def __init__(self, config: ACM3BiMambaConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = ACM3BiMamba(config)
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

        chunk_size = actions_hat.shape[1]
        n_steps = self.config.n_action_steps
        device = actions_hat.device
        weights = torch.ones(chunk_size, device=device)

        if getattr(self.config, "use_temporal_weighting", False) and n_steps < chunk_size:
            exec_mass = getattr(self.config, "temporal_execution_weight", 0.9)
            weights[:n_steps] = (chunk_size * exec_mass) / n_steps
            weights[n_steps:] = (chunk_size * (1 - exec_mass)) / (chunk_size - n_steps)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none")
            * ~batch["action_is_pad"].unsqueeze(-1)
            * weights.view(1, -1, 1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
            loss_dict["kld_loss"] = kld.item()
            return l1_loss + kld * self.config.kl_weight, loss_dict
        return l1_loss, loss_dict
