import math

import torch
from torch import Tensor, nn

from lerobot.policies.acm2.modeling_acm2 import (
    ACM2,
    ACM2Policy,
    Mamba2ACMDecoder,
)
from lerobot.policies.acm2_icpe.configuration_acm2_icpe import ACM2ICPEConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


def make_icpe_signal(K: int, mode: str, device, dtype) -> Tensor:
    i = torch.arange(K, dtype=torch.float32, device=device)
    angle = 2.0 * math.pi * i / K

    if mode == "sincos":
        sig = torch.stack([angle.sin(), angle.cos()], dim=-1)
    elif mode == "linear":
        sig = torch.stack([i / K, (K - i) / K], dim=-1)
    else:
        sig = torch.stack([angle.sin(), angle.cos(), i / K, (K - i) / K], dim=-1)

    return sig.to(dtype=dtype)


class Mamba2ICPEDecoder(Mamba2ACMDecoder):
    """ACM2 decoder with optional carry support.

    ICPE itself is already baked into x before this decoder is called.
    """

    def __init__(self, config: ACM2ICPEConfig):
        super().__init__(config)
        self.carry_position = getattr(config, "sscp_carry_position", "pre_query")

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
        carry: Tensor | None = None,
    ) -> Tensor:
        if decoder_pos_embed is not None:
            x = x + decoder_pos_embed
        if encoder_pos_embed is not None:
            encoder_out = encoder_out + encoder_pos_embed

        x = x.transpose(0, 1)                      # (B, K, D)
        encoder_out = encoder_out.transpose(0, 1)  # (B, T, D)

        if carry is not None:
            if self.carry_position == "prefix":
                combined_seq = torch.cat([carry, encoder_out, x], dim=1)
            else:
                combined_seq = torch.cat([encoder_out, carry, x], dim=1)
        else:
            combined_seq = torch.cat([encoder_out, x], dim=1)

        for layer in self.layers:
            combined_seq = layer(combined_seq)

        K = x.shape[1]
        out = combined_seq[:, -K:, :]

        if self.norm is not None:
            out = self.norm(out)

        return out.transpose(0, 1)  # (K, B, D)
    
class ACM2ICPE(ACM2):
    """ACM2 + ICPE.

    Inherits most modules from ACM2, but replaces:
      1. decoder with Mamba2ICPEDecoder
      2. decoder query construction with pos_embed + ICPE
      3. optional carry support for SSCP
    """

    def __init__(self, config: ACM2ICPEConfig):
        super().__init__(config)
        self.config = config

        # Replace decoder with carry-capable ICPE decoder.
        self.decoder = Mamba2ICPEDecoder(config)

        # Add ICPE projection.
        self.icpe_proj = nn.Linear(config.icpe_dim, config.dim_model, bias=False)
        nn.init.normal_(
            self.icpe_proj.weight,
            std=config.icpe_scale_init / math.sqrt(config.dim_model),
        )

    def forward(
        self,
        batch: dict[str, Tensor],
        carry: Tensor | None = None,
        return_decoder_out: bool = False,
    ) -> tuple:
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
        elif OBS_ENV_STATE in batch:
            batch_size = batch[OBS_ENV_STATE].shape[0]
        elif OBS_STATE in batch:
            batch_size = batch[OBS_STATE].shape[0]
        else:
            _ref = next(v for v in batch.values() if v is not None and hasattr(v, "shape"))
            batch_size = _ref.shape[0]

        # -------------------------
        # VAE latent
        # -------------------------
        if self.config.use_vae and ACTION in batch and self.training:
            import einops

            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )

            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)

            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]

            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()

            if OBS_ENV_STATE in batch:
                _ref = batch[OBS_ENV_STATE]
            elif OBS_STATE in batch:
                _ref = batch[OBS_STATE]
            else:
                _ref = next(v for v in batch.values() if v is not None and hasattr(v, "device"))

            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=_ref.device,
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
                [batch_size, self.config.latent_dim],
                dtype=torch.float32,
                device=_ref.device,
            )

        # -------------------------
        # Encoder tokens
        # -------------------------
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))

        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            import einops

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

        # -------------------------
        # ICPE decoder queries
        # -------------------------
        K = self.config.chunk_size
        dtype = encoder_out.dtype
        device = encoder_out.device

        pos_emb = self.decoder_pos_embed.weight.unsqueeze(1)  # (K, 1, D)

        phase = make_icpe_signal(K, self.config.icpe_mode, device, dtype)
        icpe_emb = self.icpe_proj(phase).unsqueeze(1)  # (K, 1, D)

        decoder_in = pos_emb + icpe_emb
        decoder_in = decoder_in.expand(K, batch_size, self.config.dim_model)

        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=None,
            carry=carry,
        )  # (K, B, D)

        actions = self.action_head(decoder_out.transpose(0, 1))  # (B, K, action_dim)

        if return_decoder_out:
            return actions, (mu, log_sigma_x2), decoder_out

        return actions, (mu, log_sigma_x2)
    
class ACM2ICPEPolicy(ACM2Policy):
    config_class = ACM2ICPEConfig
    name = "acm2_icpe"

    def __init__(self, config: ACM2ICPEConfig):
        super().__init__(config)
        self.config = config
        self.model = ACM2ICPE(config)
        self.reset()