"""eunji ACM2 + literal Mamba-2 SSD state carryover (SSCP), with optional
boundary-time carry fusion.

Backbone = eunji's ACM2 (Transformer encoder -> Mamba-2 SSD decoder, with the optional
native BiMamba / action self-attention). This policy carries Mamba-2's actual recurrent
state across chunk boundaries using the SSD kernel's `initial_states` / `return_final_states`
path, plus the depthwise-conv state, so the scan continues exactly from where the previous
chunk ended.

Per-layer state = (conv_state, ssm_state):
  conv_state : (B, conv_dim, W-1)              last (W-1) conv inputs (causal tail)
  ssm_state  : (B, nheads, headdim, d_state)   SSD recurrent state

Design (Option A — continuous stream): each chunk the decoder scans [encoder_out, queries].
The per-layer final state is saved and fed as the initial state of the next chunk's scan,
so the SSM never resets. At training the carried state is detached at the boundary
(truncated BPTT) when sscp_detach.

BiMamba x carry are orthogonal: when use_bimamba_decoder=True the *forward* scan carries
the cross-chunk state (initial_states / return_final_states); the *backward* (flipped) scan
is a stateless, intra-chunk refinement of the already-planned action block (no cross-chunk
carry, since a reverse-time state across boundaries is undefined).

Carry fusion (v8): the carried state is a prediction of where the stream should be after
open-loop execution; reality may have diverged. `config.carry_fusion` selects how the carry
is treated at each boundary — see configuration_acm2_sscp_literal for the "none"/"ema"/"mlp"/
"gated" spectrum. Fusion acts on the ssm_state only; conv_state passes through unchanged.
"""

from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn

try:
    from mamba_ssm import Mamba2

    HAS_MAMBA2 = True
except ImportError:
    HAS_MAMBA2 = False

# SSD kernel that exposes initial_states / return_final_states.
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    HAS_SSD_COMBINED = True
except ImportError:  # pragma: no cover - older/newer layouts
    try:
        from mamba_ssm.ops.ssd_combined import mamba_chunk_scan_combined

        HAS_SSD_COMBINED = True
    except ImportError:
        mamba_chunk_scan_combined = None
        HAS_SSD_COMBINED = False

from lerobot.policies.acm2.modeling_acm2 import ACM2, ACTTemporalEnsembler
from lerobot.policies.acm2_sscp_literal.configuration_acm2_sscp_literal import ACM2SSCPLiteralConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# ── Depthwise causal conv with carried state (version-independent, F.conv1d based) ──

def _causal_conv_step(conv1d: nn.Conv1d, x: Tensor, conv_state_in: Tensor | None, return_state: bool):
    """Apply the module's depthwise causal conv to x with an optional initial state.

    x: (B, C, L). Returns (out (B, C, L), final_conv_state (B, C, W-1) or None).
    Reproduces Conv1d(C, C, W, groups=C, padding=W-1)(x)[..., :L] when conv_state_in
    is None (zeros), and continues the causal stream when conv_state_in is given.
    """
    b, c, l = x.shape
    w = conv1d.kernel_size[0]
    if w <= 1:
        out = F.conv1d(x, conv1d.weight, conv1d.bias, groups=c)
        return out, (x.new_zeros(b, c, 0) if return_state else None)
    if conv_state_in is None:
        conv_state_in = x.new_zeros(b, c, w - 1)
    x_in = torch.cat([conv_state_in, x], dim=2)  # (B, C, (W-1)+L)
    out = F.conv1d(x_in, conv1d.weight, conv1d.bias, groups=c)  # valid conv -> (B, C, L)
    final = x_in[..., -(w - 1):] if return_state else None
    return out, final


# ── Stateful Mamba-2 SSD forward (exposes conv/ssm state) ─────────────────────────

def mamba2_stateful_forward(m, u: Tensor, initial_state=None, return_state: bool = False):
    """Replicate Mamba2.forward (non-fused path) exposing initial/return states.

    u: (B, L, d_model). initial_state: (conv_state, ssm_state) tuple or None.
    Returns out (B, L, d_model), or (out, (conv_state, ssm_state)) if return_state.
    """
    if not HAS_SSD_COMBINED:
        raise ImportError(
            "mamba_chunk_scan_combined unavailable. Install mamba-ssm>=2.0 (Triton SSD kernel)."
        )
    conv_state_in, ssm_state_in = (initial_state if initial_state is not None else (None, None))

    zxbcdt = m.in_proj(u)  # (B, L, d_in_proj)
    d_mlp = (zxbcdt.shape[-1] - 2 * m.d_inner - 2 * m.ngroups * m.d_state - m.nheads) // 2
    z0, x0, z, xBC, dt = torch.split(
        zxbcdt,
        [d_mlp, d_mlp, m.d_inner, m.d_inner + 2 * m.ngroups * m.d_state, m.nheads],
        dim=-1,
    )

    # Depthwise causal conv with carried conv state.
    xBC_t = xBC.transpose(1, 2)  # (B, conv_dim, L)
    xBC_t, conv_state_final = _causal_conv_step(m.conv1d, xBC_t, conv_state_in, return_state)
    xBC = m.act(xBC_t.transpose(1, 2))  # SiLU, (B, L, conv_dim)

    x, Bmat, Cmat = torch.split(
        xBC, [m.d_inner, m.ngroups * m.d_state, m.ngroups * m.d_state], dim=-1
    )
    A = -torch.exp(m.A_log.float())  # (nheads,)
    D_has_hdim = getattr(m, "D_has_hdim", False)

    ret = mamba_chunk_scan_combined(
        rearrange(x, "b l (h p) -> b l h p", p=m.headdim),
        dt,
        A,
        rearrange(Bmat, "b l (g n) -> b l g n", g=m.ngroups),
        rearrange(Cmat, "b l (g n) -> b l g n", g=m.ngroups),
        chunk_size=m.chunk_size,
        D=rearrange(m.D, "(h p) -> h p", p=m.headdim) if D_has_hdim else m.D,
        z=rearrange(z, "b l (h p) -> b l h p", p=m.headdim) if not m.rmsnorm else None,
        dt_bias=m.dt_bias,
        dt_softplus=True,
        initial_states=ssm_state_in,
        return_final_states=return_state,
    )
    if return_state:
        y, ssm_state_final = ret
    else:
        y, ssm_state_final = ret, None

    y = rearrange(y, "b l h p -> b l (h p)")  # (B, L, d_inner)
    if m.rmsnorm:
        y = m.norm(y, z)
    if d_mlp > 0:
        y = torch.cat([F.silu(z0) * x0, y], dim=-1)
    out = m.out_proj(y)
    if return_state:
        return out, (conv_state_final, ssm_state_final)
    return out


def detach_states(states):
    """Detach every tensor in a per-layer list of (conv, ssm) state tuples."""
    if states is None:
        return None
    return [tuple(t.detach() if torch.is_tensor(t) else t for t in st) for st in states]


def ema_states(new_states, prev_states, beta: float):
    """ReMem-VLA-style fixed EMA on the ssm_state across boundaries (gradient-free).

    c_n = beta * h_n + (1 - beta) * c_{n-1}, with c_{-1} = 0. conv_state is taken from
    new_states unchanged.
    """
    if new_states is None:
        return None
    fused = []
    for i, (conv, ssm) in enumerate(new_states):
        prev_ssm = prev_states[i][1] if prev_states is not None else None
        ssm = beta * ssm + (1.0 - beta) * prev_ssm if prev_ssm is not None else beta * ssm
        fused.append((conv, ssm))
    return fused


def add_carry_noise(states, std: float, p: float = 0.5):
    """v9: perturb the carried ssm_state to simulate open-loop divergence (train only).

    For each sample independently, with probability p, add Gaussian noise of per-sample
    scale s ~ U(0, std) * std(ssm_state) to that sample's ssm_state. conv_state is left
    untouched. No-op when std <= 0.
    """
    if states is None or std <= 0.0:
        return states
    out = []
    for conv, ssm in states:
        b = ssm.shape[0]
        apply = (torch.rand(b, device=ssm.device) < p).to(ssm.dtype)  # (B,)
        scale = torch.rand(b, device=ssm.device).to(ssm.dtype) * std * apply  # (B,)
        ref = ssm.reshape(b, -1).std(dim=1).clamp_min(1e-6).to(ssm.dtype)  # (B,)
        coeff = (scale * ref).view(b, *([1] * (ssm.dim() - 1)))
        out.append((conv, ssm + torch.randn_like(ssm) * coeff))
    return out


class CarryStateFusion(nn.Module):
    """Boundary-time fusion of the carried per-layer ssm_state ("mlp" / "gated").

    The ssm_state has shape (B, H, P, N) — H heads, P headdim, N d_state. Both variants act
    at the d_state granularity, SHARED across heads (H) and headdim (P), so the gate is
    per-state-channel and the parameter count is matched to m2_mlp (does not scale with nheads).

    mlp   — h' = h + MLP(h), residual with zero-init output layer (starts as literal carry).
    gated — observation-driven boundary correction. G = sigma(g([mean_{H,P}(h); e_obs]) + b0),
            broadcast over H,P. `carry_gate_mode` selects the correction:
              reset    h' = (1-G)*h
              replace  h' = (1-G)*h + G*S(e_obs)
              residual h' = h + G*D(e_obs)
    """

    def __init__(self, config: ACM2SSCPLiteralConfig):
        super().__init__()
        self.mode = config.carry_fusion
        self.gate_mode = getattr(config, "carry_gate_mode", "replace")  # reset|replace|residual
        d_inner = config.dim_model * config.mamba2_expand
        self.nheads = d_inner // config.mamba2_headdim
        self.d_state = config.mamba2_d_state
        n_layers = config.n_decoder_layers
        N, D = self.d_state, config.dim_model

        # Interpretability hook: when `record_gate` is True, forward() stashes the per-layer
        # mean gate G in `last_gate` (list[float]). No cost when disabled.
        self.record_gate: bool = False
        self.last_gate: list[float] | None = None

        if self.mode == "mlp":
            def make_mlp():
                lin2 = nn.Linear(N, N)
                nn.init.zeros_(lin2.weight)
                nn.init.zeros_(lin2.bias)
                return nn.Sequential(nn.Linear(N, N), nn.Tanh(), lin2)

            self.proj = nn.ModuleList([make_mlp() for _ in range(n_layers)])
        elif self.mode == "gated":
            hidden = config.carry_fusion_hidden

            def make_gate():
                out = nn.Linear(hidden, N)  # per-state-channel gate, broadcast over H,P
                nn.init.zeros_(out.weight)
                nn.init.constant_(out.bias, config.carry_gate_bias_init)
                return nn.Sequential(nn.Linear(N + D, hidden), nn.SiLU(), out)

            self.gate = nn.ModuleList([make_gate() for _ in range(n_layers)])
            if self.gate_mode in ("replace", "residual"):
                def make_target():
                    s = nn.Linear(D, N)  # obs -> per-channel target/delta, broadcast over H,P
                    nn.init.zeros_(s.weight)
                    nn.init.zeros_(s.bias)
                    return s

                self.target = nn.ModuleList([make_target() for _ in range(n_layers)])
            # "reset" needs no target module (h' = (1-G)*h)
        else:
            raise ValueError(f"CarryStateFusion supports 'mlp'/'gated', got '{self.mode}'.")

    def forward(self, states, e_obs: Tensor):
        """states: per-layer list of (conv, ssm). e_obs: (B, D) pooled obs."""
        fused = []
        gate_log: list[float] = []
        for i, (conv, ssm) in enumerate(states):
            b, h, p, n = ssm.shape
            assert h == self.nheads and n == self.d_state, (
                f"ssm_state shape {tuple(ssm.shape)} != configured (H={self.nheads}, N={self.d_state})"
            )
            ssm_in = ssm.to(e_obs.dtype)
            if self.mode == "mlp":
                ssm_out = ssm_in + self.proj[i](ssm_in)
            else:  # gated — per-state-channel gate, broadcast over heads (H) and headdim (P)
                pooled = ssm_in.mean(dim=(1, 2))  # (B, N)
                gate = torch.sigmoid(self.gate[i](torch.cat([pooled, e_obs], dim=-1)))
                gate = gate.view(b, 1, 1, n)  # broadcast over H, P
                if self.gate_mode == "reset":
                    ssm_out = (1.0 - gate) * ssm_in  # gate toward 0 (= ACT)
                elif self.gate_mode == "residual":
                    delta = self.target[i](e_obs).view(b, 1, 1, n)
                    ssm_out = ssm_in + gate * delta  # additive obs correction
                else:  # replace
                    target = self.target[i](e_obs).view(b, 1, 1, n)
                    ssm_out = (1.0 - gate) * ssm_in + gate * target  # Kalman-style replace
                if self.record_gate:
                    gate_log.append(float(gate.mean().item()))
            fused.append((conv, ssm_out.to(ssm.dtype)))
        if self.record_gate and gate_log:
            self.last_gate = gate_log
        return fused


# ── Decoder with literal per-layer state handoff (eunji-compatible: plain / BiMamba) ──

class Mamba2LiteralSSCPDecoder(nn.Module):
    """eunji's Mamba2ACMDecoder made stateful: each layer's recurrent (conv, ssm) state is
    carried across chunks via the forward scan. Mirrors eunji's decoder options.

    use_bimamba_decoder == False: single forward stack, carries state.
    use_bimamba_decoder == True:  forward stack carries state; backward (flipped) stack is
                                  stateless intra-chunk refinement; out = 0.5*(fwd + bwd).
    Optional action self-attention is applied to the last-K tokens, exactly as in eunji.
    Cross-chunk carry (new_states) ALWAYS comes from the forward scan only.
    """

    def __init__(self, config: ACM2SSCPLiteralConfig):
        super().__init__()
        if not HAS_MAMBA2:
            raise ImportError(
                "mamba-ssm >= 2.0 is required for Mamba2. Install with: pip install mamba-ssm>=2.0"
            )

        self.use_bimamba_decoder = getattr(config, "use_bimamba_decoder", False)

        def make_mamba2_layer(layer_idx: int):
            return Mamba2(
                d_model=config.dim_model,
                d_state=config.mamba2_d_state,
                d_conv=config.mamba2_d_conv,
                expand=config.mamba2_expand,
                headdim=config.mamba2_headdim,
                chunk_size=config.mamba2_chunk_size,
                layer_idx=layer_idx,
            )

        if self.use_bimamba_decoder:
            self.forward_layers = nn.ModuleList(
                [make_mamba2_layer(i) for i in range(config.n_decoder_layers)]
            )
            self.backward_layers = nn.ModuleList(
                [make_mamba2_layer(i + config.n_decoder_layers) for i in range(config.n_decoder_layers)]
            )
            self.layers = None
        else:
            self.layers = nn.ModuleList(
                [make_mamba2_layer(i) for i in range(config.n_decoder_layers)]
            )
            self.forward_layers = None
            self.backward_layers = None

        self.use_action_self_attention = getattr(config, "use_action_self_attention", False)
        self.action_self_attention_use_gate = getattr(config, "action_self_attention_use_gate", True)

        if self.use_action_self_attention:
            self.action_self_attn_norm = nn.LayerNorm(config.dim_model)
            self.action_self_attn = nn.MultiheadAttention(
                config.dim_model, config.n_heads, dropout=config.dropout
            )
            self.action_self_attn_dropout = nn.Dropout(config.dropout)
            gamma_init = getattr(config, "action_self_attention_gamma_init", 1e-4)
            self.action_self_attn_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        else:
            self.action_self_attn_norm = None
            self.action_self_attn = None
            self.action_self_attn_dropout = None
            self.action_self_attn_gamma = None

        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,  # (K, B, D) — decoder queries
        encoder_out: Tensor,  # (T, B, D) — encoder context
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
        carry: list | None = None,  # per-layer list of (conv, ssm) tuples (forward stack), or None
    ):
        if decoder_pos_embed is not None:
            x = x + decoder_pos_embed
        if encoder_pos_embed is not None:
            encoder_out = encoder_out + encoder_pos_embed

        x = x.transpose(0, 1)  # (B, K, D)
        encoder_out = encoder_out.transpose(0, 1)  # (B, T, D)
        combined_seq = torch.cat([encoder_out, x], dim=1)  # (B, T+K, D)

        new_states = []
        if self.use_bimamba_decoder:
            forward_seq = combined_seq
            for i, layer in enumerate(self.forward_layers):
                init = carry[i] if carry is not None else None
                forward_seq, st = mamba2_stateful_forward(
                    layer, forward_seq, initial_state=init, return_state=True
                )
                new_states.append(st)

            backward_seq = combined_seq.flip([1])
            for layer in self.backward_layers:
                backward_seq, _ = mamba2_stateful_forward(
                    layer, backward_seq, initial_state=None, return_state=True
                )
            backward_seq = backward_seq.flip([1])

            combined_seq = 0.5 * (forward_seq + backward_seq)
        else:
            for i, layer in enumerate(self.layers):
                init = carry[i] if carry is not None else None
                combined_seq, st = mamba2_stateful_forward(
                    layer, combined_seq, initial_state=init, return_state=True
                )
                new_states.append(st)

        chunk_size = x.shape[1]
        out = combined_seq[:, -chunk_size:, :]

        if self.use_action_self_attention:
            out_t = out.transpose(0, 1)  # (K, B, D)
            residual = out_t
            out_norm = self.action_self_attn_norm(out_t)
            attn_delta = self.action_self_attn(
                out_norm, out_norm, value=out_norm, need_weights=False
            )[0]
            attn_delta = self.action_self_attn_dropout(attn_delta)
            if self.action_self_attention_use_gate:
                scale = torch.tanh(self.action_self_attn_gamma)
                out_t = residual + scale * attn_delta
            else:
                out_t = residual + attn_delta
            out = out_t.transpose(0, 1)  # (B, K, D)

        out = self.norm(out)
        return out.transpose(0, 1), new_states  # (K, B, D), per-layer forward states


# ── Model (subclasses eunji ACM2; swaps decoder for the stateful one) ──────────────

class ACM2SSCPLiteral(ACM2):
    """eunji ACM2 with the decoder replaced by a literal-state-handoff decoder.

    With carry_fusion in ("mlp", "gated") the incoming carry is fused at the start of the
    chunk — for "gated" using the current chunk's pooled encoder output as e_obs, so the
    correction is driven by the fresh observation.
    """

    def __init__(self, config: ACM2SSCPLiteralConfig):
        super().__init__(config)
        self.decoder = Mamba2LiteralSSCPDecoder(config)
        self.carry_fusion = (
            CarryStateFusion(config) if config.carry_fusion in ("mlp", "gated") else None
        )
        # Re-init the freshly built stateful decoder (encoder re-init is harmless at init
        # time). Mamba2 params are protected by id in ACM2._reset_parameters; carry_fusion is
        # not under encoder/decoder so its custom zero-init is preserved.
        self._reset_parameters()

    def _encode(self, batch: dict[str, Tensor], batch_size: int):
        """eunji ACM2.forward encode block: returns encoder_out (S,B,D), pos_embed (S,B,D),
        mu, log_sigma_x2."""
        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = self.vae_encoder_cls_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()

            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
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
            latent_sample = torch.zeros(
                [batch_size, self.config.latent_dim], dtype=torch.float32
            ).to(batch[OBS_STATE].device)

        # Prepare transformer encoder inputs.
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
                cam_features = rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        return encoder_out, encoder_in_pos_embed, mu, log_sigma_x2

    def forward(self, batch: dict[str, Tensor], carry: list | None = None, return_state: bool = False):
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )
        batch_size = (
            batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]
        )
        encoder_out, encoder_in_pos_embed, mu, log_sigma_x2 = self._encode(batch, batch_size)

        if carry is not None and self.carry_fusion is not None:
            e_obs = encoder_out.mean(dim=0)  # (B, D) pooled fresh observation (encoder_out is (S,B,D))
            carry = self.carry_fusion(carry, e_obs)

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out, new_states = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
            carry=carry,
        )
        decoder_out = decoder_out.transpose(0, 1)  # (B, S, C)
        actions = self.action_head(decoder_out)
        if return_state:
            return actions, (mu, log_sigma_x2), new_states
        return actions, (mu, log_sigma_x2)


# ── Policy wrapper (carries per-layer STATE across chunk boundaries) ───────────────

class ACM2SSCPLiteralPolicy(PreTrainedPolicy):
    """eunji ACM2 + literal Mamba-2 SSD state carryover (with optional carry fusion)."""

    config_class = ACM2SSCPLiteralConfig
    name = "acm2_sscp_literal"

    def __init__(self, config: ACM2SSCPLiteralConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = self._build_model(config)
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)
        self.reset()

    def _build_model(self, config):
        return ACM2SSCPLiteral(config)

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
        self._carry: list | None = None  # per-layer (conv, ssm) state tuples

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
        if has_pairs and self.config.sscp_enabled and self.config.sscp_p_carry > 0.0:
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
        """Chunk-continuation: chunk n -> carry literal state -> chunk n+1."""
        batch_n = {k: v for k, v in batch.items() if not k.endswith("_n1")}
        loss_n, loss_dict_n, states_n = self._forward_single(batch_n, carry=None)

        carry = detach_states(states_n) if self.config.sscp_detach else states_n
        # v9: inject carry-divergence noise so the boundary fusion (esp. the obs-driven gate)
        # learns to correct a carried state that disagrees with the fresh observation. Applied
        # before fusion so every mode sees the same dirtied carry. No-op when std == 0.
        if getattr(self.config, "carry_noise_std", 0.0) > 0.0:
            carry = add_carry_noise(
                carry, self.config.carry_noise_std, getattr(self.config, "carry_noise_p", 0.5)
            )
        if self.config.carry_fusion == "ema":
            # First boundary of the pair: c = beta * h (c_{-1} = 0), matching inference.
            carry = ema_states(carry, None, self.config.carry_ema_beta)

        _n1_candidates = {
            OBS_STATE: batch.get("obs_state_n1", batch.get(OBS_STATE)),
            OBS_ENV_STATE: batch.get("obs_env_state_n1", batch.get(OBS_ENV_STATE)),
            ACTION: batch["action_n1"],
            "action_is_pad": batch["action_is_pad_n1"],
        }
        batch_n1 = {k: v for k, v in _n1_candidates.items() if v is not None}
        if OBS_IMAGES in batch:
            batch_n1[OBS_IMAGES] = [
                batch[k + "_n1"] for k in self.config.image_features if k + "_n1" in batch
            ] or batch[OBS_IMAGES]

        loss_n1, loss_dict_n1, _ = self._forward_single(batch_n1, carry=carry)

        total_loss = loss_n + loss_n1
        combined = {
            "l1_loss": (loss_dict_n["l1_loss"] + loss_dict_n1["l1_loss"]) / 2,
            "l1_loss_n": loss_dict_n["l1_loss"],
            "l1_loss_n1": loss_dict_n1["l1_loss"],
        }
        if "kld_loss" in loss_dict_n:
            combined["kld_loss"] = (loss_dict_n["kld_loss"] + loss_dict_n1.get("kld_loss", 0.0)) / 2
        return total_loss, combined
