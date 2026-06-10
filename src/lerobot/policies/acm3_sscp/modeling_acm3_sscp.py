"""ACM3 + SSM State Carryover Protocol (SSCP)

Terminology note (important for the paper): SSCP does NOT copy Mamba3's literal
recurrent state tensor across the chunk boundary. Instead it carries a single
*summary token* (the previous chunk's terminal decoder output) and lets the SSM
*re-derive* a non-zero hidden state by scanning that token before the queries.
Describe it as "summary-token state warm-up", not "literal hidden-state transfer".

How SSCP works (inference):
  1. After chunk n, extract the Mamba3 decoder's LAST OUTPUT TOKEN:
       carry_n = decoder_out[:, -1:, :]   # (B, 1, D)
     This dense vector summarises the accumulated SSM context at the end of chunk n.

  2. Before chunk n+1's action queries, insert carry_n into the combined sequence
     inside Mamba3SSCPDecoder. Default placement is "pre_query":
       combined = cat([encoder_out_{n+1}, carry_n, decoder_queries_{n+1}], dim=1)
     so the SSM scans carry_n immediately before the queries — warming up its
     hidden state from previous-chunk context instead of h=0, without that state
     being washed out across the long encoder token stream. (The original "prefix"
     placement cat([carry_n, encoder_out, queries]) is kept as a config ablation.)

  3. The K action tokens are still extracted from the LAST K positions of combined_out.

Why this is SSM-specific (not applicable to ACT Transformer):
  - Mamba3 processes carry_n sequentially, so carry_n's hidden state propagates
    into all subsequent positions.
  - In ACT (Transformer), adding carry_n as an extra attention key does not warm
    up any hidden state; it is merely another token in global attention.

Training strategy (Chunk-Continuation):
  With probability sscp_p_carry, the batch provides consecutive chunk pairs.
  chunk n → compute carry (detach) → use as prefix for chunk n+1.
  Loss = L1(chunk_n) + L1(chunk_n+1).
  With probability (1 - sscp_p_carry), standard training without carry.

No ICPE:
  This policy has no Intra-Chunk Phase Embedding. Decoder queries use only the
  learned positional embedding (no phase signal injection). Compare with
  acm3_icpe_sscp which adds both ICPE and SSCP.
"""

from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from mamba_ssm import Mamba3
    HAS_MAMBA3 = True
except ImportError:
    HAS_MAMBA3 = False

from lerobot.policies.acm3_sscp.configuration_acm3_sscp import ACM3SSCPConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.acm3.modeling_acm3 import (
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
)

import einops
import torchvision
from itertools import chain
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d


# ── Mamba3 decoder with carry support (no ICPE) ───────────────────────────────

class Mamba3SSCPDecoder(nn.Module):
    """Mamba3 decoder that optionally accepts a carry token from the previous chunk.

    Identical interface to Mamba3ICPEDecoder but without any ICPE dependency.
    The carry mechanism is the sole extension over the base ACM3 decoder.

    carry (B, 1, D): if provided, prepended before encoder_out so the SSM
                     initialises from the previous chunk's hidden state context.
    """

    def __init__(self, config: ACM3SSCPConfig):
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
        # "pre_query" places carry right before the queries (recommended); "prefix"
        # places it before the encoder stream (original, diluted) — see config.
        self.carry_position = getattr(config, "sscp_carry_position", "pre_query")

    def forward(
        self,
        x: Tensor,                        # (K, B, D) — decoder queries with pos_embed
        encoder_out: Tensor,              # (T, B, D) — encoder context
        carry: Tensor | None = None,      # (B, 1, D) — optional carry from previous chunk
    ) -> Tensor:                          # returns (K, B, D)
        x           = x.transpose(0, 1)            # (B, K, D)
        encoder_out = encoder_out.transpose(0, 1)  # (B, T, D)

        if carry is not None:
            if self.carry_position == "prefix":
                combined = torch.cat([carry, encoder_out, x], dim=1)  # (B, 1+T+K, D)
            else:  # "pre_query": carry adjacent to the action queries
                combined = torch.cat([encoder_out, carry, x], dim=1)  # (B, T+1+K, D)
        else:
            combined = torch.cat([encoder_out, x], dim=1)         # (B, T+K, D)

        for layer in self.layers:
            combined = layer(combined)

        K   = x.shape[1]
        out = combined[:, -K:, :]
        out = self.norm(out)
        return out.transpose(0, 1)  # (K, B, D)


# ── Neural network module ──────────────────────────────────────────────────────

class ACM3SSCP(nn.Module):
    """ACM3 with SSCP carry support (no ICPE)."""

    def __init__(self, config: ACM3SSCPConfig):
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

        # ── Decoder positional embedding (no ICPE projection) ─────────────────
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # ── Mamba3 SSCP decoder ───────────────────────────────────────────────
        self.decoder = Mamba3SSCPDecoder(config)

        # ── Action head ───────────────────────────────────────────────────────
        self.action_head = nn.Linear(config.dim_model, config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        mamba_ids = {id(p) for p in self.decoder.layers.parameters()}
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1 and id(p) not in mamba_ids:
                nn.init.xavier_uniform_(p)

    def _encode(self, batch: dict[str, Tensor], batch_size: int):
        """VAE + Transformer encoder forward. Returns (encoder_out, mu, log_sigma_x2)."""
        config = self.config

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
            if OBS_ENV_STATE in batch:
                _ref = batch[OBS_ENV_STATE]
            elif OBS_STATE in batch:
                _ref = batch[OBS_STATE]
            else:
                _ref = next(v for v in batch.values() if v is not None and hasattr(v, 'device'))
            latent = torch.zeros(batch_size, config.latent_dim,
                                 dtype=torch.float32, device=_ref.device)

        tokens   = [self.encoder_latent_input_proj(latent)]
        pos_list = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if config.robot_state_feature:
            tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if config.env_state_feature:
            tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
        if config.image_features:
            for img in batch[OBS_IMAGES]:
                feat    = self.backbone(img)["feature_map"]
                cam_pos = self.encoder_cam_feat_pos_embed(feat).to(dtype=feat.dtype)
                feat    = self.encoder_img_feat_input_proj(feat)
                feat    = einops.rearrange(feat,    "b c h w -> (h w) b c")
                cam_pos = einops.rearrange(cam_pos, "b c h w -> (h w) b c")
                tokens.extend(list(feat))
                pos_list.extend(list(cam_pos))

        enc_tokens  = torch.stack(tokens,   dim=0)
        enc_pos     = torch.stack(pos_list, dim=0)
        encoder_out = self.encoder(enc_tokens, pos_embed=enc_pos)
        encoder_out = encoder_out + enc_pos

        return encoder_out, mu, log_sigma_x2

    def forward(
        self,
        batch: dict[str, Tensor],
        carry: Tensor | None = None,
        return_decoder_out: bool = False,
    ) -> tuple:
        if self.config.use_vae and self.training:
            assert ACTION in batch, "Need action labels for VAE training."

        batch_size = (
            batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch
            else batch[OBS_ENV_STATE].shape[0]
        )

        encoder_out, mu, log_sigma_x2 = self._encode(batch, batch_size)

        K   = self.config.chunk_size
        # Decoder queries: positional embedding only (no ICPE)
        decoder_in = self.decoder_pos_embed.weight.unsqueeze(1)  # (K, 1, D)

        decoder_out = self.decoder(
            decoder_in.expand(K, batch_size, self.config.dim_model),
            encoder_out,
            carry=carry,
        )  # (K, B, D)

        actions = self.action_head(decoder_out.transpose(0, 1))  # (B, K, action_dim)
        if return_decoder_out:
            return actions, (mu, log_sigma_x2), decoder_out
        return actions, (mu, log_sigma_x2)


# ── Policy wrapper ─────────────────────────────────────────────────────────────

class ACM3SSCPPolicy(PreTrainedPolicy):
    """ACM3 + SSM State Carryover Protocol (no ICPE).

    At inference: carry the terminal decoder output token across chunk boundaries.
    At training:  chunk-continuation pairs teach the model to handle non-zero carry.
    """

    config_class = ACM3SSCPConfig
    name = "acm3_sscp"

    def __init__(self, config: ACM3SSCPConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = ACM3SSCP(config)
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
        else:
            return self._forward_single(batch, carry=None)

    def _forward_single(
        self,
        batch: dict[str, Tensor],
        carry: Tensor | None,
        return_decoder_out: bool = False,
    ) -> tuple:
        if return_decoder_out:
            actions_hat, (mu, log_sigma_x2), decoder_out = self.model(batch, carry=carry, return_decoder_out=True)
        else:
            actions_hat, (mu, log_sigma_x2) = self.model(batch, carry=carry)
            decoder_out = None

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
            loss = l1 + kld * self.config.kl_weight
        else:
            loss = l1
        if return_decoder_out:
            return loss, loss_dict, decoder_out
        return loss, loss_dict

    def _forward_chunk_pair(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Chunk-continuation training: chunk n → extract carry → chunk n+1."""
        # ── Chunk n ───────────────────────────────────────────────────────────
        batch_n = {k: v for k, v in batch.items() if not k.endswith("_n1")}
        loss_n, loss_dict_n, dec_out_n = self._forward_single(batch_n, carry=None, return_decoder_out=True)

        # ── Extract carry from chunk n's decoder output ───────────────────────
        carry = dec_out_n[-1:, :, :].transpose(0, 1)  # (B, 1, D)
        if self.config.sscp_detach:
            carry = carry.detach()

        # ── Chunk n+1 ─────────────────────────────────────────────────────────
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

        # ── Combined loss ─────────────────────────────────────────────────────
        total_loss = loss_n + loss_n1
        combined = {
            "l1_loss":    (loss_dict_n["l1_loss"] + loss_dict_n1["l1_loss"]) / 2,
            "l1_loss_n":  loss_dict_n["l1_loss"],
            "l1_loss_n1": loss_dict_n1["l1_loss"],
        }
        if "kld_loss" in loss_dict_n:
            combined["kld_loss"] = (loss_dict_n["kld_loss"] + loss_dict_n1.get("kld_loss", 0.0)) / 2

        return total_loss, combined
