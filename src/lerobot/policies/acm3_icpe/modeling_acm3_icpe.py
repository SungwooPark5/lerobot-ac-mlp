"""ACM3 + Intra-Chunk Phase Embedding (ICPE)

Key change vs ACM3:
  The decoder query tokens now carry an explicit temporal position signal
  φ(i, K) ∈ R^{icpe_dim} projected to R^{dim_model} via a learned linear layer.

  decoder_in[i] = decoder_pos_embed[i] + icpe_proj(φ(i, K))

  For Mamba3, this matters because position i's hidden state h_i is the SSM
  accumulation of h_0 .. h_{i-1}. Telling the model "you are at position i/K"
  calibrates uncertainty that otherwise peaks at the chunk END (7.02× baseline).

  For ACT (Transformer), the same signal has no effect because all positions
  attend globally — confirmed by the ACT+ICPE control experiment.

Also introduces Mamba3ICPEDecoder that accepts an optional `carry` tensor
(B, 1, D). When provided, it is prepended to the combined sequence so the SSM
warms up from a non-zero context — the basis of True SSCP.
"""

import math
from collections import deque
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

try:
    from mamba_ssm import Mamba3
    HAS_MAMBA3 = True
except ImportError:
    HAS_MAMBA3 = False

from lerobot.policies.acm3_icpe.configuration_acm3_icpe import ACM3ICPEConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.acm3.modeling_acm3 import (
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
)


# ── ICPE signal builder ────────────────────────────────────────────────────────

def make_icpe_signal(K: int, mode: str, device, dtype) -> Tensor:
    """Build phase signal matrix (K, icpe_dim).

    φ(i, K):
        "full"   4D: [sin(2πi/K), cos(2πi/K), i/K, (K-i)/K]
        "sincos" 2D: [sin(2πi/K), cos(2πi/K)]
        "linear" 2D: [i/K, (K-i)/K]
    """
    i = torch.arange(K, dtype=torch.float32, device=device)
    angle = 2.0 * math.pi * i / K
    if mode == "sincos":
        sig = torch.stack([angle.sin(), angle.cos()], dim=-1)
    elif mode == "linear":
        sig = torch.stack([i / K, (K - i) / K], dim=-1)
    else:  # "full"
        sig = torch.stack([angle.sin(), angle.cos(), i / K, (K - i) / K], dim=-1)
    return sig.to(dtype=dtype)


# ── Mamba3 decoder with ICPE-aware carry support ───────────────────────────────

class Mamba3ICPEDecoder(nn.Module):
    """Mamba3 decoder that optionally accepts a carry token from the previous chunk.

    Interface identical to Mamba3ACMDecoder, plus:
        carry (B, 1, D): if provided, prepended before encoder_out in the combined
                         sequence so the SSM initialises from previous chunk context
                         instead of h=0.  This is the mechanism for True SSCP.

    Note: decoder_pos_embed is NOT applied here — it is already added to decoder_in
    by the parent ACM3ICPE.forward() before calling this decoder.
    """

    def __init__(self, config: ACM3ICPEConfig):
        super().__init__()
        if not HAS_MAMBA3:
            raise ImportError(
                "mamba_ssm is required. Install with:\n"
                "  MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall "
                "git+https://github.com/state-spaces/mamba.git --no-build-isolation"
            )

        self.layers = nn.ModuleList([
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
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,           # (K, B, D) — decoder queries (already has pos_embed + ICPE baked in)
        encoder_out: Tensor, # (T, B, D) — encoder context (already has encoder_pos_embed baked in)
        carry: Tensor | None = None,  # (B, 1, D) — optional carry from previous chunk
    ) -> Tensor:            # returns (K, B, D)
        # Transpose to (B, S, D) for Mamba3
        x = x.transpose(0, 1)           # (B, K, D)
        encoder_out = encoder_out.transpose(0, 1)  # (B, T, D)

        # Prepend carry token if provided
        if carry is not None:
            combined = torch.cat([carry, encoder_out, x], dim=1)  # (B, 1+T+K, D)
        else:
            combined = torch.cat([encoder_out, x], dim=1)         # (B, T+K, D)

        for layer in self.layers:
            combined = layer(combined)

        # Extract last K tokens (action predictions)
        K = x.shape[1]
        out = combined[:, -K:, :]
        out = self.norm(out)
        return out.transpose(0, 1)  # (K, B, D)


# ── Neural network module ──────────────────────────────────────────────────────

class ACM3ICPE(nn.Module):
    """ACM3 with Intra-Chunk Phase Embedding and optional carry token support."""

    def __init__(self, config: ACM3ICPEConfig):
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

        # ── Transformer encoder ───────────────────────────────────────────────
        self.encoder = ACTEncoder(config)

        # ── Encoder 1D feature projections + positional embeddings ───────────
        if config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.robot_state_feature.shape[0], config.dim_model
            )
        if config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)

        n_1d_tokens = 1 + (1 if config.robot_state_feature else 0) + (1 if config.env_state_feature else 0)
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)

        # ── Decoder positional embedding + ICPE projection ────────────────────
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.icpe_proj = nn.Linear(config.icpe_dim, config.dim_model, bias=False)
        # Small-scale init: ICPE starts as a minor perturbation, not noise
        nn.init.normal_(self.icpe_proj.weight, std=config.icpe_scale_init / math.sqrt(config.dim_model))

        # ── Mamba3 ICPE decoder ───────────────────────────────────────────────
        self.decoder = Mamba3ICPEDecoder(config)

        # ── Action head ───────────────────────────────────────────────────────
        self.action_head = nn.Linear(config.dim_model, config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        mamba_ids = {id(p) for p in self.decoder.layers.parameters()}
        icpe_ids  = {id(p) for p in self.icpe_proj.parameters()}
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1 and id(p) not in mamba_ids and id(p) not in icpe_ids:
                nn.init.xavier_uniform_(p)

    def _encode(self, batch: dict[str, Tensor], batch_size: int):
        """VAE + Transformer encoder forward. Returns (encoder_out, enc_pos_embed, mu, log_sigma_x2)."""
        config = self.config

        # VAE latent
        if config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            if config.robot_state_feature:
                rs_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)
            act_embed = self.vae_encoder_action_input_proj(batch[ACTION])

            vae_in = [cls_embed] + ([rs_embed] if config.robot_state_feature else []) + [act_embed]
            vae_in = torch.cat(vae_in, dim=1)

            pos = self.vae_encoder_pos_enc.clone().detach()
            n_prefix = 2 if config.robot_state_feature else 1
            is_pad_prefix = torch.zeros(batch_size, n_prefix, dtype=torch.bool, device=vae_in.device)
            key_pad = torch.cat([is_pad_prefix, batch["action_is_pad"]], dim=1)

            cls_out = self.vae_encoder(vae_in.permute(1, 0, 2),
                                       pos_embed=pos.permute(1, 0, 2),
                                       key_padding_mask=key_pad)[0]
            params = self.vae_encoder_latent_output_proj(cls_out)
            mu, log_sigma_x2 = params[:, :config.latent_dim], params[:, config.latent_dim:]
            latent = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            _ref = batch.get(OBS_ENV_STATE) or batch.get(OBS_STATE) or next(
                v for v in batch.values() if v is not None and hasattr(v, 'device')
            )
            latent = torch.zeros(batch_size, config.latent_dim,
                                 dtype=torch.float32, device=_ref.device)

        # Encoder tokens
        tokens = [self.encoder_latent_input_proj(latent)]  # (B, D) each
        pos_list = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))  # (D,) → (1, D)

        if config.robot_state_feature:
            tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if config.env_state_feature:
            tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
        if config.image_features:
            for img in batch[OBS_IMAGES]:
                feat = self.backbone(img)["feature_map"]
                cam_pos = self.encoder_cam_feat_pos_embed(feat).to(dtype=feat.dtype)
                feat = self.encoder_img_feat_input_proj(feat)
                feat    = einops.rearrange(feat,    "b c h w -> (h w) b c")
                cam_pos = einops.rearrange(cam_pos, "b c h w -> (h w) b c")
                tokens.extend(list(feat))
                pos_list.extend(list(cam_pos))

        enc_tokens  = torch.stack(tokens,    dim=0)  # (T, B, D)
        enc_pos     = torch.stack(pos_list,  dim=0)  # (T, 1, D)
        encoder_out = self.encoder(enc_tokens, pos_embed=enc_pos)  # (T, B, D)

        # Bake encoder pos embed into encoder_out so the decoder doesn't need it separately
        encoder_out = encoder_out + enc_pos

        return encoder_out, mu, log_sigma_x2

    def forward(
        self,
        batch: dict[str, Tensor],
        carry: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor | None, Tensor | None]]:
        """
        Args:
            batch: observation dict.
            carry: (B, 1, D) carry token from previous chunk (for True SSCP).
        Returns:
            actions (B, K, action_dim), (mu, log_sigma_x2).
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, "Need action labels for VAE training."

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        encoder_out, mu, log_sigma_x2 = self._encode(batch, batch_size)

        # ── Build decoder queries with pos embed + ICPE ───────────────────────
        K      = self.config.chunk_size
        dtype  = encoder_out.dtype
        device = encoder_out.device

        # Learnable position embedding (K, D) → (K, 1, D) broadcast over batch
        pos_emb  = self.decoder_pos_embed.weight.unsqueeze(1)

        # ICPE phase signal (K, icpe_dim) → projected to (K, 1, D)
        phase    = make_icpe_signal(K, self.config.icpe_mode, device, dtype)
        icpe_emb = self.icpe_proj(phase).unsqueeze(1)

        # Combine: zeros + pos_embed + ICPE  → (K, B, D) via broadcast
        decoder_in = pos_emb + icpe_emb  # (K, 1, D) → broadcast to (K, B, D) inside decoder

        # ── Mamba3 decoder ────────────────────────────────────────────────────
        decoder_out = self.decoder(
            decoder_in.expand(K, batch_size, self.config.dim_model),
            encoder_out,
            carry=carry,
        )  # (K, B, D)

        actions = self.action_head(decoder_out.transpose(0, 1))  # (B, K, action_dim)
        return actions, (mu, log_sigma_x2)


# ── Policy wrapper ─────────────────────────────────────────────────────────────

class ACM3ICPEPolicy(PreTrainedPolicy):
    config_class = ACM3ICPEConfig
    name = "acm3_icpe"

    def __init__(self, config: ACM3ICPEConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = ACM3ICPE(config)
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
            w_e = (K * exec_mass) / n_exec
            w_f = (K * (1 - exec_mass)) / (K - n_exec)
            weights[:n_exec] = w_e
            weights[n_exec:] = w_f

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
