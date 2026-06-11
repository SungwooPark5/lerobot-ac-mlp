"""ACM3 + literal SSM state carryover, with optional boundary-time carry fusion.

Carries Mamba3's actual recurrent state (angle, ssm, k, v) across chunk boundaries
using the SISO kernel's `Input_States` / `return_final_states` path — verified
numerically equivalent to a single continuous scan (see v7/00_mamba3_stateful_probe).

Design (Option A — continuous stream):
  Each chunk the decoder scans [encoder_out, queries]. The per-layer final state is
  saved and fed as the initial state of the next chunk's scan. The SSM therefore
  never resets: the effective stream is [enc_0, q_0, enc_1, q_1, ...]. At training,
  the carried state is detached at the boundary (truncated BPTT) when sscp_detach.

Carry fusion (v8): the carried state is a *prediction* of where the stream should
be after open-loop execution; reality may have diverged (perturbations, tracking
error). `config.carry_fusion` selects how the carry is treated at each boundary:

  "none"  — literal handoff (m3_lit, MTIL-style; original behavior)
  "ema"   — fixed EMA across boundaries, gradient-free (m3_ema, ReMem-VLA-style)
  "mlp"   — learned projection, no observation (m3_mlp, AVA-VLA-style)
  "gated" — PEC gate, learned observation-driven correction (m3_cor, proposed):
              h' = (1 - G) ⊙ h + G ⊙ S(e_obs)
              G  = σ( g([pool(h); e_obs]) + b₀ ),  b₀ < 0
            G → 0 recovers literal carry (MTIL limit); G → 1 with S → 0 recovers a
            zero-state reset (ACT limit) — the gate spans both extremes.

Fusion acts on the ssm_state component only; angle (positional continuity) and the
k/v trapezoid-rule correction terms pass through unchanged.

Assumes SISO mode (is_mimo=False, is_outproj_norm=False) — the canonical config.
"""

from collections import deque
from itertools import chain

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn

try:
    from mamba_ssm import Mamba3
    from mamba_ssm.modules.mamba3 import mamba3_siso_combined
    HAS_MAMBA3 = True
except ImportError:
    HAS_MAMBA3 = False

from lerobot.policies.acm3_sscp.modeling_acm3_sscp import ACM3SSCP
from lerobot.policies.acm3_sscp_literal.configuration_acm3_sscp_literal import ACM3SSCPLiteralConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.acm3.modeling_acm3 import ACTTemporalEnsembler
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

# State tuple = (angle_state, ssm_state, k_state, v_state) as returned by the kernel.
StateTuple = "tuple[Tensor, Tensor, Tensor, Tensor]"


# ── Stateful Mamba3 SISO forward (verified in v7/00_mamba3_stateful_probe.ipynb) ──

def mamba3_stateful_forward(m, u: Tensor, initial_state=None, return_state: bool = False):
    """Replicates Mamba3.forward (SISO path) but exposes Input_States + final states.

    u: (B, L, d_model). initial_state: (angle, ssm, k, v) tuple or None.
    Returns out (B, L, d_model), or (out, final_state_tuple) if return_state.
    """
    assert not m.is_mimo and not m.is_outproj_norm, (
        "acm3_sscp_literal requires SISO mode (is_mimo=False, is_outproj_norm=False)."
    )
    zxBCdtAtrap = m.in_proj(u)
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
        zxBCdtAtrap,
        [m.d_inner, m.d_inner,
         m.d_state * m.num_bc_heads * m.mimo_rank,
         m.d_state * m.num_bc_heads * m.mimo_rank,
         m.nheads, m.nheads, m.nheads, m.num_rope_angles],
        dim=-1)
    z = rearrange(z, "b l (h p) -> b l h p", p=m.headdim)
    x = rearrange(x, "b l (h p) -> b l h p", p=m.headdim)
    B = rearrange(B, "b l (r g n) -> b l r g n", r=m.mimo_rank, g=m.num_bc_heads)
    C = rearrange(C, "b l (r g n) -> b l r g n", r=m.mimo_rank, g=m.num_bc_heads)
    trap = rearrange(trap, "b l h -> b h l")
    _A = -F.softplus(dd_A.to(torch.float32))
    _A = torch.clamp(_A, max=-m.A_floor)
    DT = F.softplus(dd_dt + m.dt_bias)
    ADT = _A * DT
    DT = rearrange(DT, "b l n -> b n l")
    ADT = rearrange(ADT, "b l n -> b n l")
    angles = angles.unsqueeze(-2).expand(-1, -1, m.nheads, -1).to(torch.float32)
    B = m.B_norm(B)
    C = m.C_norm(C)
    ret = mamba3_siso_combined(
        Q=C.squeeze(2), K=B.squeeze(2), V=x, ADT=ADT, DT=DT, Trap=trap,
        Q_bias=m.C_bias.squeeze(1), K_bias=m.B_bias.squeeze(1), Angles=angles,
        D=m.D, Z=z, chunk_size=m.chunk_size,
        Input_States=initial_state,
        return_final_states=return_state,
        cu_seqlens=None,
    )
    if return_state:
        y = ret[0]
        final = tuple(ret[1:5])  # (angle, ssm, k, v)
    else:
        y = ret[0] if isinstance(ret, (tuple, list)) else ret
        final = None
    y = rearrange(y, "b l h p -> b l (h p)")
    out = m.out_proj(y.to(x.dtype))
    return (out, final) if return_state else out


def detach_states(states):
    """Detach every tensor in a per-layer list of state tuples."""
    if states is None:
        return None
    return [tuple(t.detach() if torch.is_tensor(t) else t for t in st) for st in states]


def ema_states(new_states, prev_states, beta: float):
    """ReMem-VLA-style fixed EMA on the ssm_state across boundaries (gradient-free).

    c_n = beta * h_n + (1 - beta) * c_{n-1}, with c_{-1} = 0 (so the first boundary
    yields beta * h_0). angle/k/v are taken from new_states unchanged.
    """
    if new_states is None:
        return None
    fused = []
    for i, st in enumerate(new_states):
        angle, ssm, k, v = st
        prev_ssm = prev_states[i][1] if prev_states is not None else None
        ssm = beta * ssm + (1.0 - beta) * prev_ssm if prev_ssm is not None else beta * ssm
        fused.append((angle, ssm, k, v))
    return fused


class CarryStateFusion(nn.Module):
    """Boundary-time fusion of the carried per-layer ssm_state ("mlp" / "gated").

    The ssm_state has shape (B, H, P, N) — H heads, P headdim, N d_state. Gates and
    observation targets are computed per (head, state-channel) and broadcast over P.

    mlp   — h' = h + MLP(h), residual with zero-init output layer (starts as
            literal carry). No observation input: faithful adaptation of
            AVA-VLA's plain recurrent-state projection r = B(h).
    gated — h' = (1 - G) ⊙ h + G ⊙ S(e_obs), the proposed PEC correction.
            G = σ(g([mean_P(h); e_obs]) + b₀), b₀ = carry_gate_bias_init < 0,
            S = zero-init linear map of e_obs into state space.
    """

    def __init__(self, config):
        super().__init__()
        self.mode = config.carry_fusion
        d_inner = config.dim_model * config.mamba3_expand
        self.nheads = d_inner // config.mamba3_headdim
        self.d_state = config.mamba3_d_state
        hn = self.nheads * self.d_state
        n_layers = config.n_decoder_layers

        # Interpretability hook: when `record_gate` is True, forward() stashes the
        # per-layer mean gate G in `last_gate` (list[float], len n_layers). Used by
        # the gate-analysis notebook to show "G opens after perturbation, closes
        # while a cue must be remembered". No cost when disabled.
        self.record_gate: bool = False
        self.last_gate: list[float] | None = None

        if self.mode == "mlp":
            def make_mlp():
                lin2 = nn.Linear(self.d_state, self.d_state)
                nn.init.zeros_(lin2.weight)
                nn.init.zeros_(lin2.bias)
                return nn.Sequential(nn.Linear(self.d_state, self.d_state), nn.Tanh(), lin2)
            self.proj = nn.ModuleList([make_mlp() for _ in range(n_layers)])
        elif self.mode == "gated":
            hidden = config.carry_fusion_hidden
            def make_gate():
                out = nn.Linear(hidden, hn)
                nn.init.zeros_(out.weight)
                nn.init.constant_(out.bias, config.carry_gate_bias_init)
                return nn.Sequential(nn.Linear(hn + config.dim_model, hidden), nn.SiLU(), out)
            def make_target():
                s = nn.Linear(config.dim_model, hn)
                nn.init.zeros_(s.weight)
                nn.init.zeros_(s.bias)
                return s
            self.gate = nn.ModuleList([make_gate() for _ in range(n_layers)])
            self.target = nn.ModuleList([make_target() for _ in range(n_layers)])
        else:
            raise ValueError(f"CarryStateFusion supports 'mlp'/'gated', got '{self.mode}'.")

    def forward(self, states, e_obs: Tensor):
        """states: per-layer list of (angle, ssm, k, v). e_obs: (B, D) pooled obs."""
        fused = []
        gate_log: list[float] = []
        for i, st in enumerate(states):
            angle, ssm, k, v = st
            b, h, p, n = ssm.shape
            assert h == self.nheads and n == self.d_state, (
                f"ssm_state shape {tuple(ssm.shape)} != configured (H={self.nheads}, N={self.d_state})"
            )
            ssm_in = ssm.to(e_obs.dtype)
            if self.mode == "mlp":
                ssm_out = ssm_in + self.proj[i](ssm_in)
            else:  # gated
                pooled = ssm_in.mean(dim=2).reshape(b, h * n)            # (B, H*N)
                gate = torch.sigmoid(self.gate[i](torch.cat([pooled, e_obs], dim=-1)))
                gate = gate.view(b, h, 1, n)                              # broadcast over P
                target = self.target[i](e_obs).view(b, h, 1, n)
                ssm_out = (1.0 - gate) * ssm_in + gate * target
                if self.record_gate:
                    gate_log.append(float(gate.mean().item()))
            fused.append((angle, ssm_out.to(ssm.dtype), k, v))
        if self.record_gate and gate_log:
            self.last_gate = gate_log
        return fused


# ── Decoder with literal per-layer state handoff ──────────────────────────────

class Mamba3LiteralSSCPDecoder(nn.Module):
    """Mamba3 decoder that carries each layer's recurrent state across chunks."""

    def __init__(self, config: ACM3SSCPLiteralConfig):
        super().__init__()
        if not HAS_MAMBA3:
            raise ImportError(
                "mamba_ssm with Mamba3 is required. Install with:\n"
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
        x: Tensor,                  # (K, B, D) — decoder queries with pos_embed
        encoder_out: Tensor,        # (T, B, D) — encoder context
        carry: list | None = None,  # per-layer list of state tuples, or None
    ):
        x = x.transpose(0, 1)               # (B, K, D)
        encoder_out = encoder_out.transpose(0, 1)  # (B, T, D)
        h = torch.cat([encoder_out, x], dim=1)     # (B, T+K, D)

        new_states = []
        for i, layer in enumerate(self.layers):
            init = carry[i] if carry is not None else None
            h, st = mamba3_stateful_forward(layer, h, initial_state=init, return_state=True)
            new_states.append(st)

        K = x.shape[1]
        out = self.norm(h[:, -K:, :])
        return out.transpose(0, 1), new_states  # (K, B, D), per-layer states


# ── Model ──────────────────────────────────────────────────────────────────────

class ACM3SSCPLiteral(ACM3SSCP):
    """ACM3SSCP with the decoder replaced by a literal-state-handoff decoder.

    With carry_fusion in ("mlp", "gated") the incoming carry is fused at the start
    of the chunk — for "gated" using the *current* chunk's pooled encoder output as
    e_obs, so the correction is driven by the fresh observation.
    """

    def __init__(self, config: ACM3SSCPLiteralConfig):
        super().__init__(config)
        self.decoder = Mamba3LiteralSSCPDecoder(config)
        self.carry_fusion = (
            CarryStateFusion(config) if config.carry_fusion in ("mlp", "gated") else None
        )

    def forward(self, batch: dict[str, Tensor], carry: list | None = None,
                return_state: bool = False) -> tuple:
        if self.config.use_vae and self.training:
            assert ACTION in batch, "Need action labels for VAE training."
        bs = (batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch
              else batch[OBS_ENV_STATE].shape[0])
        encoder_out, mu, log_sigma_x2 = self._encode(batch, bs)
        if carry is not None and self.carry_fusion is not None:
            e_obs = encoder_out.mean(dim=0)  # (B, D) pooled fresh observation
            carry = self.carry_fusion(carry, e_obs)
        K = self.config.chunk_size
        decoder_in = self.decoder_pos_embed.weight.unsqueeze(1)  # (K, 1, D)
        decoder_out, new_states = self.decoder(
            decoder_in.expand(K, bs, self.config.dim_model), encoder_out, carry=carry)
        actions = self.action_head(decoder_out.transpose(0, 1))  # (B, K, action_dim)
        if return_state:
            return actions, (mu, log_sigma_x2), new_states
        return actions, (mu, log_sigma_x2)


# ── Policy wrapper (carries per-layer STATE instead of a token) ────────────────

class ACM3SSCPLiteralPolicy(PreTrainedPolicy):
    """ACM3 + literal SSM state carryover."""

    config_class = ACM3SSCPLiteralConfig
    name = "acm3_sscp_literal"

    def __init__(self, config: ACM3SSCPLiteralConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = self._build_model(config)
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)
        self.reset()

    def _build_model(self, config):
        return ACM3SSCPLiteral(config)

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
        self._carry: list | None = None  # per-layer state tuples

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
        actions, _, new_states = self.model(batch, carry=carry, return_state=True)
        if self.config.sscp_enabled:
            new_states = detach_states(new_states)
            if self.config.carry_fusion == "ema":
                new_states = ema_states(new_states, self._carry, self.config.carry_ema_beta)
            self._carry = new_states
        return actions

    # ── Training ───────────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        has_pairs = "action_n1" in batch
        if has_pairs and self.config.sscp_p_carry > 0.0:
            return self._forward_chunk_pair(batch)
        return self._forward_single(batch, carry=None)[:2]

    def _forward_single(self, batch: dict[str, Tensor], carry: list | None) -> tuple:
        actions_hat, (mu, log_sigma_x2), new_states = self.model(batch, carry=carry, return_state=True)

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
        return loss, loss_dict, new_states

    def _forward_chunk_pair(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Chunk-continuation: chunk n → carry literal state → chunk n+1."""
        batch_n = {k: v for k, v in batch.items() if not k.endswith("_n1")}
        loss_n, loss_dict_n, states_n = self._forward_single(batch_n, carry=None)

        carry = detach_states(states_n) if self.config.sscp_detach else states_n
        if self.config.carry_fusion == "ema":
            # First boundary of the pair: c = beta * h (c_{-1} = 0), matching inference.
            carry = ema_states(carry, None, self.config.carry_ema_beta)

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

        loss_n1, loss_dict_n1, _ = self._forward_single(batch_n1, carry=carry)

        total_loss = loss_n + loss_n1
        combined = {
            "l1_loss":    (loss_dict_n["l1_loss"] + loss_dict_n1["l1_loss"]) / 2,
            "l1_loss_n":  loss_dict_n["l1_loss"],
            "l1_loss_n1": loss_dict_n1["l1_loss"],
        }
        if "kld_loss" in loss_dict_n:
            combined["kld_loss"] = (loss_dict_n["kld_loss"] + loss_dict_n1.get("kld_loss", 0.0)) / 2
        return total_loss, combined
