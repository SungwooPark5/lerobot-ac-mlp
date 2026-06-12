"""RNN/LSTM history baseline — non-SSM recurrent history encoder.

Same role as MTIL (recurrent history encoder over the obs stream → predict an action
chunk, no correction) but with a classic LSTM/GRU instead of a Mamba SSM. Isolates the
"SSM-specific" part of our claim from generic recurrence.

Fully self-contained: imports NO other policy module.
"""

from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.rnn_hist.configuration_rnn_hist import RNNHistConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# ── Temporal aggregation (self-contained; ACT-style) ──────────────────────────

class _TemporalEnsembler:
    def __init__(self, coeff: float, chunk_size: int):
        self.coeff = coeff
        self.chunk_size = chunk_size
        self.reset()

    def reset(self):
        self._buf = None
        self._count = None
        self._t = 0

    def update(self, actions: Tensor) -> Tensor:
        b, k, a = actions.shape
        dev = actions.device
        if self._buf is None:
            self._buf = torch.zeros(b, 0, a, device=dev)
            self._count = torch.zeros(b, 0, device=dev)
        need = self._t + k - self._buf.shape[1]
        if need > 0:
            self._buf = torch.cat([self._buf, torch.zeros(b, need, a, device=dev)], dim=1)
            self._count = torch.cat([self._count, torch.zeros(b, need, device=dev)], dim=1)
        idx = torch.arange(self._t, self._t + k, device=dev)
        w = torch.exp(-self.coeff * self._count[:, idx])
        self._buf[:, idx] = self._buf[:, idx] * (1 - w).unsqueeze(-1) + actions * w.unsqueeze(-1)
        self._count[:, idx] += 1
        out = self._buf[:, self._t].clone()
        self._t += 1
        return out


# ── Per-timestep observation encoder (same design as MTIL, copied for isolation) ──

class RNNObsEncoder(nn.Module):
    def __init__(self, config: RNNHistConfig):
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

    def forward(self, state, env_state, images) -> Tensor:
        tok = 0.0
        if state is not None and self.config.robot_state_feature:
            tok = tok + self.state_proj(state)
        if env_state is not None and self.config.env_state_feature:
            tok = tok + self.env_proj(env_state)
        if images is not None and self.config.image_features:
            for img in images:
                fm = self.backbone(img)["feature_map"]
                tok = tok + self.img_proj(fm.mean(dim=(2, 3)))
        return self.norm(tok)


# ── Model ───────────────────────────────────────────────────────────────────────

class RNNHistModel(nn.Module):
    def __init__(self, config: RNNHistConfig):
        super().__init__()
        self.config = config
        D, H = config.dim_model, config.rnn_hidden
        self.obs_encoder = RNNObsEncoder(config)
        rnn_cls = nn.LSTM if config.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=D, hidden_size=H, num_layers=config.n_rnn_layers,
            batch_first=True, dropout=config.rnn_dropout if config.n_rnn_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(H)
        self.action_dim = config.action_feature.shape[0]
        self.head = nn.Linear(H, config.chunk_size * self.action_dim)

    def encode_window(self, batch: dict[str, Tensor]) -> Tensor:
        """Windowed obs batch → (B, T, D) per-timestep token sequence."""
        def split(x, is_img):
            if x is None:
                return None, None, None
            if is_img:
                if x.dim() == 5:
                    b, t = x.shape[:2]
                    return x.reshape(b * t, *x.shape[2:]), b, t
                return x, x.shape[0], 1
            if x.dim() == 3:
                b, t = x.shape[:2]
                return x.reshape(b * t, x.shape[2]), b, t
            return x, x.shape[0], 1

        state, env_state, imgs = batch.get(OBS_STATE), batch.get(OBS_ENV_STATE), batch.get(OBS_IMAGES)
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
        out, _ = self.rnn(seq)                                    # (B, T, H)
        h_last = self.norm(out[:, -1])
        flat = self.head(h_last)
        return flat.view(seq.shape[0], self.config.chunk_size, self.action_dim)

    def step_one(self, tok: Tensor, hidden):
        """Advance the RNN state by one token (B, D), carrying `hidden` across calls."""
        out, hidden = self.rnn(tok.unsqueeze(1), hidden)          # (B, 1, H)
        flat = self.head(self.norm(out[:, -1]))
        chunk = flat.view(tok.shape[0], self.config.chunk_size, self.action_dim)
        return chunk, hidden


# ── Policy wrapper ───────────────────────────────────────────────────────────────

class RNNHistPolicy(PreTrainedPolicy):
    config_class = RNNHistConfig
    name = "rnn_hist"

    def __init__(self, config: RNNHistConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = RNNHistModel(config)
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

    def state_dict(self, *args, **kwargs):
        # nn.LSTM/GRU pack weight_ih_l*/weight_hh_l*/bias_* into a single flattened
        # buffer (_flat_weights), so those tensors SHARE one storage. safetensors
        # (save_model, used by save_pretrained → checkpointing) refuses to serialize
        # storage-sharing tensors ("no suitable name to keep ... None covers the entire
        # storage"). Clone each tensor so it owns its storage. Save-time only; loading
        # (load_state_dict) is unaffected. Without this, rnn_hist crashes at the first
        # checkpoint save during training.
        sd = super().state_dict(*args, **kwargs)
        for k, v in list(sd.items()):
            if isinstance(v, torch.Tensor):
                sd[k] = v.clone()
        return sd

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._obs_window = deque([], maxlen=self.config.n_obs_steps)
        self._hidden = None   # unbounded RNN state carry (h,c) / h, reset per episode

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.temporal_ensemble_coeff is not None:
            return self.temporal_ensembler.update(self._predict(batch))
        if len(self._action_queue) == 0:
            actions = self._predict(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        return self._predict(batch)

    def _current_frame(self, batch):
        out = {}
        if self.config.image_features:
            out[OBS_IMAGES] = [(batch[k][:, -1] if batch[k].dim() == 5 else batch[k])
                               for k in self.config.image_features]
        if OBS_STATE in batch:
            x = batch[OBS_STATE]; out[OBS_STATE] = x[:, -1] if x.dim() == 3 else x
        if OBS_ENV_STATE in batch:
            x = batch[OBS_ENV_STATE]; out[OBS_ENV_STATE] = x[:, -1] if x.dim() == 3 else x
        return out

    def _predict(self, batch: dict[str, Tensor]) -> Tensor:
        cur = self._current_frame(batch)
        tok = self.model.obs_encoder(cur.get(OBS_STATE), cur.get(OBS_ENV_STATE), cur.get(OBS_IMAGES))
        if self.config.unbounded_carry:
            chunk, self._hidden = self.model.step_one(tok, self._hidden)
            return chunk
        self._obs_window.append(tok)
        seq = torch.stack(list(self._obs_window), dim=1)
        return self.model.predict_chunk_from_seq(seq)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        seq = self.model.encode_window(batch)                     # (B, T, D)
        actions_hat = self.model.predict_chunk_from_seq(seq)      # (B, K, A)
        l1 = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        if "action_is_pad" in batch:
            l1 = l1 * ~batch["action_is_pad"].unsqueeze(-1)
        loss = l1.mean()
        return loss, {"l1_loss": loss.item()}
