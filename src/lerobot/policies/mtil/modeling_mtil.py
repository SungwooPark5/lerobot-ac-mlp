"""MTIL baseline — faithful reproduction inside lerobot (external baseline).

MTIL: "Encoding Full History with Mamba for Temporal Imitation Learning"
(arXiv:2505.12410, RA-L 2025; github.com/yulinzhouZYL/MTIL). See PROVENANCE.md.

Architecture (faithful):
  per-timestep token  =  Linear(state) + Σ_cam Linear(pool(ResNet(image)))
  Mamba-2 stack       =  n_mamba_layers × mamba_ssm.Mamba2  (history encoder)
  chunk head          =  Linear(D, chunk_size · action_dim)
  → predict a chunk_size action chunk from the recurrent state; NO correction/gating
    of the state with new observations (pure carryover — the property our paper
    contrasts against). ACT-style temporal aggregation at inference.

This module is fully self-contained: it imports NO other policy module, so it
cannot be affected by (or affect) the ACM3 / carry-fusion code.
"""

from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

try:
    from mamba_ssm import Mamba2
    HAS_MAMBA2 = True
except ImportError:
    HAS_MAMBA2 = False

from lerobot.policies.mtil.configuration_mtil import MTILConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# ── Temporal aggregation (self-contained; ACT-style EWMA over overlapping chunks) ──

class _TemporalEnsembler:
    """Exponential temporal aggregation of overlapping action chunks (ACT/MTIL)."""

    def __init__(self, coeff: float, chunk_size: int):
        self.coeff = coeff
        self.chunk_size = chunk_size
        self.reset()

    def reset(self):
        self._buf = None        # (B, T_accum, K, A) padded online buffer
        self._t = 0

    def update(self, actions: Tensor) -> Tensor:
        """actions: (B, K, A). Returns the aggregated action for the current step."""
        b, k, a = actions.shape
        dev = actions.device
        if self._buf is None:
            self._buf = torch.zeros(b, 0, a, device=dev)
            self._count = torch.zeros(b, 0, device=dev)
        # Extend buffers to cover the horizon of this chunk.
        need = self._t + k - self._buf.shape[1]
        if need > 0:
            self._buf = torch.cat([self._buf, torch.zeros(b, need, a, device=dev)], dim=1)
            self._count = torch.cat([self._count, torch.zeros(b, need, device=dev)], dim=1)
        # Exponentially weight older predictions less.
        idx = torch.arange(self._t, self._t + k, device=dev)
        w = torch.exp(-self.coeff * (self._count[:, idx]))   # (B, K)
        self._buf[:, idx] = self._buf[:, idx] * (1 - w).unsqueeze(-1) + actions * w.unsqueeze(-1)
        # ^ simple online blend; weight grows with #contributions
        self._count[:, idx] += 1
        out = self._buf[:, self._t].clone()                  # (B, A)
        self._t += 1
        return out


# ── Per-timestep observation encoder ──────────────────────────────────────────

class MTILObsEncoder(nn.Module):
    """Encode one observation (state + images) into a single D-dim token."""

    def __init__(self, config: MTILConfig):
        super().__init__()
        self.config = config
        D = config.dim_model
        if config.robot_state_feature:
            self.state_proj = nn.Linear(config.robot_state_feature.shape[0], D)
        if config.env_state_feature:
            self.env_proj = nn.Linear(config.env_state_feature.shape[0], D)
        if config.image_features:
            bb = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(bb, return_layers={"layer4": "feature_map"})
            self.img_proj = nn.Linear(bb.fc.in_features, D)
        self.norm = nn.LayerNorm(D)

    def forward(self, state: Tensor | None, env_state: Tensor | None,
                images: list[Tensor] | None) -> Tensor:
        """state/env_state: (N, d). images: list of (N, C, H, W). Returns (N, D)."""
        tok = 0.0
        if state is not None and self.config.robot_state_feature:
            tok = tok + self.state_proj(state)
        if env_state is not None and self.config.env_state_feature:
            tok = tok + self.env_proj(env_state)
        if images is not None and self.config.image_features:
            for img in images:
                fm = self.backbone(img)["feature_map"]            # (N, c, h, w)
                pooled = fm.mean(dim=(2, 3))                       # (N, c) global avg pool
                tok = tok + self.img_proj(pooled)
        return self.norm(tok)


# ── Model ───────────────────────────────────────────────────────────────────────

class MTILModel(nn.Module):
    def __init__(self, config: MTILConfig):
        super().__init__()
        if not HAS_MAMBA2:
            raise ImportError(
                "mamba_ssm with Mamba2 is required for the MTIL baseline. Install with:\n"
                "  MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall "
                "git+https://github.com/state-spaces/mamba.git --no-build-isolation"
            )
        self.config = config
        D = config.dim_model
        self.obs_encoder = MTILObsEncoder(config)
        self.layers = nn.ModuleList([
            Mamba2(
                d_model=D,
                d_state=config.mamba2_d_state,
                d_conv=config.mamba2_d_conv,
                expand=config.mamba2_expand,
                headdim=config.mamba2_headdim,
                layer_idx=i,
            )
            for i in range(config.n_mamba_layers)
        ])
        self.norm = nn.LayerNorm(D)
        self.action_dim = config.action_feature.shape[0]
        self.head = nn.Linear(D, config.chunk_size * self.action_dim)

    def encode_window(self, batch: dict[str, Tensor]) -> Tensor:
        """Build the (B, T, D) per-timestep token sequence from a windowed batch.

        Observation tensors are expected with a time axis at dim=1 (lerobot loads
        `n_obs_steps` frames via observation_delta_indices). A missing time axis
        (single frame) is treated as T=1.
        """
        def split(x, is_img):
            # Returns (B, T, ...) → flatten to (B*T, ...) plus (B, T).
            if x is None:
                return None, None, None
            if is_img:
                if x.dim() == 5:      # (B, T, C, H, W)
                    b, t = x.shape[:2]
                    return x.reshape(b * t, *x.shape[2:]), b, t
                else:                 # (B, C, H, W) → T=1
                    return x, x.shape[0], 1
            else:
                if x.dim() == 3:      # (B, T, d)
                    b, t = x.shape[:2]
                    return x.reshape(b * t, x.shape[2]), b, t
                else:                 # (B, d) → T=1
                    return x, x.shape[0], 1

        state = batch.get(OBS_STATE)
        env_state = batch.get(OBS_ENV_STATE)
        imgs = batch.get(OBS_IMAGES)

        B = T = None
        state_f, B, T = split(state, False) if state is not None else (None, None, None)
        env_f, b2, t2 = split(env_state, False) if env_state is not None else (None, None, None)
        if B is None:
            B, T = b2, t2
        imgs_f = None
        if imgs is not None:
            imgs_f = []
            for im in imgs:
                fim, b3, t3 = split(im, True)
                imgs_f.append(fim)
                if B is None:
                    B, T = b3, t3
        tokens = self.obs_encoder(state_f, env_f, imgs_f)        # (B*T, D)
        return tokens.reshape(B, T, -1)                           # (B, T, D)

    def predict_chunk_from_seq(self, seq: Tensor) -> Tensor:
        """seq: (B, T, D) → run Mamba-2 stack, predict chunk from the LAST step."""
        h = seq
        for layer in self.layers:
            h = layer(h)
        h_last = self.norm(h[:, -1])                              # (B, D)
        flat = self.head(h_last)                                  # (B, K*A)
        return flat.view(h.shape[0], self.config.chunk_size, self.action_dim)


# ── Policy wrapper ───────────────────────────────────────────────────────────────

class MTILPolicy(PreTrainedPolicy):
    """MTIL baseline policy (Mamba-2 history encoder, no state correction)."""

    config_class = MTILConfig
    name = "mtil"

    def __init__(self, config: MTILConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = MTILModel(config)
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = _TemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)
        self.reset()

    def get_optim_params(self):
        return [
            {"params": [p for n, p in self.named_parameters()
                        if not n.startswith("model.obs_encoder.backbone") and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters()
                        if n.startswith("model.obs_encoder.backbone") and p.requires_grad],
             "lr": self.config.optimizer_lr_backbone},
        ]

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)
        # Sliding observation-token window (bounded history of length n_obs_steps).
        self._obs_window: deque | None = deque([], maxlen=self.config.n_obs_steps)

    # ── Inference: maintain a bounded obs-token window, query every step ──────────

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.temporal_ensemble_coeff is not None:
            actions = self._predict(batch)
            return self.temporal_ensembler.update(actions)
        if len(self._action_queue) == 0:
            actions = self._predict(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        return self._predict(batch)

    def _current_frame(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Extract the current (most recent) frame, dropping any time axis."""
        out = {}
        if self.config.image_features:
            out[OBS_IMAGES] = []
            for key in self.config.image_features:
                x = batch[key]
                out[OBS_IMAGES].append(x[:, -1] if x.dim() == 5 else x)
        if OBS_STATE in batch:
            x = batch[OBS_STATE]; out[OBS_STATE] = x[:, -1] if x.dim() == 3 else x
        if OBS_ENV_STATE in batch:
            x = batch[OBS_ENV_STATE]; out[OBS_ENV_STATE] = x[:, -1] if x.dim() == 3 else x
        return out

    def _predict(self, batch: dict[str, Tensor]) -> Tensor:
        cur = self._current_frame(batch)
        # Encode current frame to a token and push into the sliding window.
        tok = self.model.obs_encoder(
            cur.get(OBS_STATE), cur.get(OBS_ENV_STATE), cur.get(OBS_IMAGES))  # (B, D)
        self._obs_window.append(tok)
        seq = torch.stack(list(self._obs_window), dim=1)         # (B, t<=n_obs_steps, D)
        return self.model.predict_chunk_from_seq(seq)            # (B, K, A)

    # ── Training: windowed history → predict chunk at the last step ──────────────

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        seq = self.model.encode_window(batch)                    # (B, T, D)
        actions_hat = self.model.predict_chunk_from_seq(seq)     # (B, K, A)

        target = batch[ACTION]                                   # (B, K, A)
        l1 = F.l1_loss(target, actions_hat, reduction="none")
        if "action_is_pad" in batch:
            l1 = l1 * ~batch["action_is_pad"].unsqueeze(-1)
        loss = l1.mean()
        return loss, {"l1_loss": loss.item()}
