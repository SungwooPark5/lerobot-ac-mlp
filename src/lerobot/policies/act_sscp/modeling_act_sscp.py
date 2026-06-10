"""ACT + SSCP carry — control experiment for C2 (SSM-specificity of SSCP).

Mechanism (mirrors acm3_sscp, but on a Transformer decoder):
  - After chunk n, take the terminal decoder output token carry_n = decoder_out[-1:].
  - For chunk n+1, append carry_n to the encoder memory so the decoder queries can
    cross-attend to it.

Why this is the right control:
  - In acm3_sscp, carry_n is *prepended into the SSM scan*, warming up the recurrent
    hidden state h that then propagates through all subsequent positions.
  - In ACT, the Transformer decoder is stateless: carry_n is simply one more
    attention key/value.  There is no hidden state to warm up.  If SSCP helps ACM3
    but NOT ACT (this model), the benefit is SSM-specific.

The carry-tracking + chunk-continuation training logic is identical to
ACM3SSCPPolicy — only the underlying model (ACT vs Mamba3) differs.
"""

from collections import deque
from itertools import chain

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
)
from lerobot.policies.act_sscp.configuration_act_sscp import ACTSSCPConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# ── Neural network module ──────────────────────────────────────────────────────

class ACTSSCP(nn.Module):
    """ACT whose decoder can attend to an optional carry token from the previous chunk."""

    def __init__(self, config: ACTSSCPConfig):
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

        # ── Decoder positional embedding + action head ────────────────────────
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        batch: dict[str, Tensor],
        carry: Tensor | None = None,        # (B, 1, D) carry from previous chunk
        return_decoder_out: bool = False,
    ) -> tuple:
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
                _ref = next(v for v in batch.values() if v is not None and hasattr(v, "device"))
            latent = torch.zeros(batch_size, self.config.latent_dim,
                                 dtype=torch.float32, device=_ref.device)

        # ── Encoder ───────────────────────────────────────────────────────────
        enc_tokens = [self.encoder_latent_input_proj(latent)]
        enc_pos = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            enc_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            enc_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                feat = self.backbone(img)["feature_map"]
                cam_pos = self.encoder_cam_feat_pos_embed(feat).to(dtype=feat.dtype)
                feat = self.encoder_img_feat_input_proj(feat)
                feat = einops.rearrange(feat, "b c h w -> (h w) b c")
                cam_pos = einops.rearrange(cam_pos, "b c h w -> (h w) b c")
                enc_tokens.extend(list(feat))
                enc_pos.extend(list(cam_pos))

        enc_tokens = torch.stack(enc_tokens, dim=0)     # (S, B, D)
        enc_pos = torch.stack(enc_pos, dim=0)           # (S, 1, D)
        encoder_out = self.encoder(enc_tokens, pos_embed=enc_pos)  # (S, B, D)

        # ── SSCP: append carry as an extra cross-attention memory token ──────────
        # (Transformer analog: an extra key/value, NOT a recurrent-state warm-up.)
        if carry is not None:
            carry_mem = carry.transpose(0, 1)                       # (1, B, D)
            encoder_out = torch.cat([encoder_out, carry_mem], dim=0)  # (S+1, B, D)
            zero_pos = torch.zeros_like(enc_pos[:1])                 # (1, 1, D)
            enc_pos = torch.cat([enc_pos, zero_pos], dim=0)          # (S+1, 1, D)

        # ── Decoder ───────────────────────────────────────────────────────────
        K = self.config.chunk_size
        decoder_in = torch.zeros((K, batch_size, self.config.dim_model),
                                 dtype=enc_pos.dtype, device=enc_pos.device)
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=enc_pos,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )  # (K, B, D)

        actions = self.action_head(decoder_out.transpose(0, 1))  # (B, K, action_dim)
        if return_decoder_out:
            return actions, (mu, log_sigma_x2), decoder_out
        return actions, (mu, log_sigma_x2)


# ── Policy wrapper (carry tracking + CC training — mirrors ACM3SSCPPolicy) ──────

class ACTSSCPPolicy(PreTrainedPolicy):
    """ACT + SSCP carry (control experiment for C2)."""

    config_class = ACTSSCPConfig
    name = "act_sscp"

    def __init__(self, config: ACTSSCPConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = ACTSSCP(config)
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
        self._carry: Tensor | None = None

    # ── Inference ──────────────────────────────────────────────────────────────

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
        carry = self._carry if self.config.sscp_enabled else None
        actions, _, decoder_out = self.model(batch, carry=carry, return_decoder_out=True)
        if self.config.sscp_enabled:
            self._carry = decoder_out[-1:, :, :].transpose(0, 1).detach()  # (B, 1, D)
        return actions

    # ── Training ───────────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        has_pairs = "action_n1" in batch
        if has_pairs and self.config.sscp_p_carry > 0.0:
            return self._forward_chunk_pair(batch)
        return self._forward_single(batch, carry=None)

    def _forward_single(
        self,
        batch: dict[str, Tensor],
        carry: Tensor | None,
        return_decoder_out: bool = False,
    ) -> tuple:
        if return_decoder_out:
            actions_hat, (mu, log_sigma_x2), decoder_out = self.model(
                batch, carry=carry, return_decoder_out=True)
        else:
            actions_hat, (mu, log_sigma_x2) = self.model(batch, carry=carry)
            decoder_out = None

        K = actions_hat.shape[1]
        n_exec = self.config.n_action_steps
        weights = torch.ones(K, device=actions_hat.device)
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
            loss = l1 + kld * self.config.kl_weight
        else:
            loss = l1
        if return_decoder_out:
            return loss, loss_dict, decoder_out
        return loss, loss_dict

    def _forward_chunk_pair(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Chunk-continuation training: chunk n → extract carry → chunk n+1."""
        batch_n = {k: v for k, v in batch.items() if not k.endswith("_n1")}
        loss_n, loss_dict_n, dec_out_n = self._forward_single(batch_n, carry=None, return_decoder_out=True)

        carry = dec_out_n[-1:, :, :].transpose(0, 1)  # (B, 1, D)
        if self.config.sscp_detach:
            carry = carry.detach()

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

        loss_n1, loss_dict_n1 = self._forward_single(batch_n1, carry=carry)

        total_loss = loss_n + loss_n1
        combined = {
            "l1_loss":    (loss_dict_n["l1_loss"] + loss_dict_n1["l1_loss"]) / 2,
            "l1_loss_n":  loss_dict_n["l1_loss"],
            "l1_loss_n1": loss_dict_n1["l1_loss"],
        }
        if "kld_loss" in loss_dict_n:
            combined["kld_loss"] = (loss_dict_n["kld_loss"] + loss_dict_n1.get("kld_loss", 0.0)) / 2
        return total_loss, combined
