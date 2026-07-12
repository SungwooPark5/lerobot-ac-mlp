"""Tests for acm_sscp_literal (Mamba-1 literal carry) — the Mamba-1 port of acm2_sscp_literal.

The critical piece is `mamba1_stateful_forward`: Mamba-1's selective-scan kernel has no
`initial_states` argument, so the carried ssm_state is injected in closed form
(homogeneous solution of the diagonal recurrence, exp(A * cumsum(delta)) * h_0).

  1. test_initial_state_correction_math   — CPU, torch-only: the closed form == a naive
                                            sequential scan. Runs anywhere with torch.
  2. test_stateful_forward_matches_plain  — GPU + mamba_ssm: no-carry stateful forward ==
                                            plain Mamba.forward (fused kernel).
  3. test_split_scan_continuation         — GPU + mamba_ssm: scanning [u1; u2] in one go ==
                                            scanning u1, carrying (conv, ssm), scanning u2.
                                            This is literal carry correctness.
  4. test_policy_carry_smoke              — GPU + mamba_ssm: tiny policy end-to-end
                                            (chunk-pair training, carry inference, BiMamba,
                                            overlap/MOSAIC rollout).

Run:  python tests/test_acm_sscp_literal.py   (also pytest-compatible; GPU tests skip on CPU)
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.acm_sscp_literal.configuration_acm_sscp_literal import (  # noqa: E402
    ACMSSCPLiteralConfig,
)
from lerobot.policies.acm_sscp_literal.modeling_acm_sscp_literal import (  # noqa: E402
    HAS_MAMBA,
    HAS_SELECTIVE_SCAN,
    ACMSSCPLiteralPolicy,
    _initial_state_correction,
    mamba1_stateful_forward,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE  # noqa: E402

try:
    import pytest

    HAS_PYTEST = True
except ImportError:  # allow plain `python tests/test_acm_sscp_literal.py`
    HAS_PYTEST = False

GPU_OK = torch.cuda.is_available() and HAS_MAMBA and HAS_SELECTIVE_SCAN


def _skip_gpu(reason="needs CUDA + mamba_ssm"):
    if HAS_PYTEST:
        return pytest.mark.skipif(not GPU_OK, reason=reason)
    return lambda f: f


# ── 1. closed-form initial-state injection == naive sequential scan (CPU) ──────────


def test_initial_state_correction_math():
    """h_t = exp(dt A) h_{t-1} + dt B x; y_t = C . h_t. Zero-init scan + closed-form
    correction must equal the scan started from h0 (both outputs and final state)."""
    torch.manual_seed(0)
    b, d, n, l = 2, 5, 4, 9
    A = -(torch.rand(d, n) + 0.05)
    delta = torch.rand(b, d, l) * 0.8 + 0.01
    Bm = torch.randn(b, n, l)
    Cm = torch.randn(b, n, l)
    x = torch.randn(b, d, l)
    h0 = torch.randn(b, d, n)

    def naive(h_init):
        h = h_init.clone()
        ys = []
        for t in range(l):
            dA = torch.exp(delta[..., t].unsqueeze(-1) * A)  # (b, d, n)
            h = dA * h + delta[..., t].unsqueeze(-1) * Bm[:, :, t].unsqueeze(1) * x[..., t].unsqueeze(-1)
            ys.append(torch.einsum("bdn,bn->bd", h, Cm[:, :, t]))
        return torch.stack(ys, dim=-1), h  # (b, d, l), (b, d, n)

    y_full, h_full = naive(h0)
    y_zero, h_zero = naive(torch.zeros_like(h0))
    y_extra, h_extra = _initial_state_correction(A, delta, Cm, h0)

    assert (y_zero + y_extra - y_full).abs().max().item() < 1e-5
    assert (h_zero + h_extra - h_full).abs().max().item() < 1e-5


# ── 2/3. kernel-level parity (GPU) ─────────────────────────────────────────────────


def _mamba_layer(d_model=64, d_state=16, d_conv=4, expand=2, seed=0):
    from mamba_ssm import Mamba

    torch.manual_seed(seed)
    return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand).cuda().eval()


@_skip_gpu()
def test_stateful_forward_matches_plain():
    """No initial state: mamba1_stateful_forward must reproduce plain Mamba.forward
    (which uses the fused kernel)."""
    m = _mamba_layer()
    torch.manual_seed(1)
    u = torch.randn(3, 24, 64).cuda()
    with torch.no_grad():
        ref = m(u)
        out, (conv_st, ssm_st) = mamba1_stateful_forward(m, u, initial_state=None, return_state=True)
    err = (ref - out).abs().max().item()
    assert err < 2e-4, f"stateful (no-carry) vs plain Mamba mismatch: max|Δ|={err:.2e}"
    assert conv_st.shape == (3, m.d_inner, m.d_conv - 1)
    assert ssm_st.shape == (3, m.d_inner, m.d_state)


@_skip_gpu()
def test_split_scan_continuation():
    """Literal carry: scan([u1; u2]) == scan(u1) -> carry (conv, ssm) -> scan(u2)."""
    m = _mamba_layer(seed=2)
    torch.manual_seed(3)
    l1, l2 = 17, 13
    u = torch.randn(2, l1 + l2, 64).cuda()
    with torch.no_grad():
        y_full, st_full = mamba1_stateful_forward(m, u, initial_state=None, return_state=True)
        y1, st1 = mamba1_stateful_forward(m, u[:, :l1], initial_state=None, return_state=True)
        y2, st2 = mamba1_stateful_forward(m, u[:, l1:], initial_state=st1, return_state=True)

    err_y1 = (y_full[:, :l1] - y1).abs().max().item()
    err_y2 = (y_full[:, l1:] - y2).abs().max().item()
    err_conv = (st_full[0] - st2[0]).abs().max().item()
    err_ssm = (st_full[1] - st2[1]).abs().max().item()
    assert err_y1 < 1e-4, f"first-half mismatch: {err_y1:.2e}"
    assert err_y2 < 1e-4, f"continuation mismatch (carry broken): {err_y2:.2e}"
    assert err_conv < 1e-4, f"conv state mismatch: {err_conv:.2e}"
    assert err_ssm < 1e-4, f"ssm state mismatch: {err_ssm:.2e}"


# ── 4. policy-level smoke (GPU) ────────────────────────────────────────────────────

B, K, DS, DE, DA = 2, 8, 14, 6, 14


def _cfg(config_cls=ACMSSCPLiteralConfig, **overrides):
    kw = dict(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(DS,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(DE,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(DA,))},
        chunk_size=K,
        n_action_steps=K,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=2,
        n_vae_encoder_layers=1,
        latent_dim=8,
        dropout=0.0,
        sscp_enabled=True,
        sscp_p_carry=1.0,
    )
    kw.update(overrides)
    return config_cls(**kw)


def _pair_batch(seed=0):
    torch.manual_seed(seed)
    return {
        OBS_STATE: torch.randn(B, DS).cuda(),
        OBS_ENV_STATE: torch.randn(B, DE).cuda(),
        ACTION: torch.randn(B, K, DA).cuda(),
        "action_is_pad": torch.zeros(B, K, dtype=torch.bool).cuda(),
        "obs_state_n1": torch.randn(B, DS).cuda(),
        "obs_env_state_n1": torch.randn(B, DE).cuda(),
        "action_n1": torch.randn(B, K, DA).cuda(),
        "action_is_pad_n1": torch.zeros(B, K, dtype=torch.bool).cuda(),
    }


def _obs(seed=0):
    torch.manual_seed(seed)
    return {OBS_STATE: torch.randn(B, DS).cuda(), OBS_ENV_STATE: torch.randn(B, DE).cuda()}


@_skip_gpu()
def test_policy_carry_smoke():
    """Chunk-pair training backward + carry-affects-inference, plain and BiMamba."""
    for bimamba in (False, True):
        torch.manual_seed(0)
        policy = ACMSSCPLiteralPolicy(_cfg(use_bimamba_decoder=bimamba)).cuda()

        # training: chunk-pair path runs and backprops
        policy.train()
        loss, ld = policy.forward(_pair_batch())
        assert torch.isfinite(loss), f"loss not finite (bimamba={bimamba})"
        assert "l1_loss_n1" in ld
        loss.backward()

        # inference: the carried state must change the next chunk's prediction
        policy.eval()
        policy.reset()
        with torch.no_grad():
            c1 = policy.predict_action_chunk(_obs(1))  # fills carry
            assert policy._carry is not None
            c2_carried = policy.predict_action_chunk(_obs(2))
            policy.reset()
            _ = policy.predict_action_chunk(_obs(1))
            policy.reset()  # drop carry -> fresh
            c2_fresh = policy.predict_action_chunk(_obs(2))
        assert c1.shape == (B, K, DA)
        diff = (c2_carried - c2_fresh).abs().max().item()
        assert diff > 1e-6, f"carry has no effect on inference (bimamba={bimamba})"


@_skip_gpu()
def test_overlap_rollout_smoke():
    """MOSAIC overlap-add rollout: emits hop steps per predict, crossfades the head."""
    from lerobot.policies.acm_sscp_literal_smooth_overlap.configuration_acm_sscp_literal_smooth_overlap import (
        ACMSSCPLiteralSmoothOverlapConfig,
    )
    from lerobot.policies.acm_sscp_literal_smooth_overlap.modeling_acm_sscp_literal_smooth_overlap import (
        ACMSSCPLiteralSmoothOverlapPolicy,
    )

    ov = 2
    torch.manual_seed(0)
    policy = ACMSSCPLiteralSmoothOverlapPolicy(
        _cfg(
            config_cls=ACMSSCPLiteralSmoothOverlapConfig,
            use_bimamba_decoder=True,
            sscp_overlap=ov,
            sscp_overlap_window="hann",
        )
    ).cuda().eval()
    policy.reset()

    hop = K - ov
    with torch.no_grad():
        steps = [policy.select_action(_obs(i)) for i in range(2 * hop)]
    assert all(s.shape == (B, DA) for s in steps)
    assert policy._ola_tail is not None and policy._ola_tail.shape == (B, ov, DA)
    assert policy._carry is not None

    # overlap-consistency training path (pair offset handled by the train pipeline; here we
    # just check the extra loss term computes and backprops)
    policy.train()
    cfg = policy.config
    cfg.sscp_overlap_train_weight = 0.1
    loss, ld = policy.forward(_pair_batch(1))
    assert "ola_loss" in ld and torch.isfinite(loss)
    loss.backward()


if __name__ == "__main__":
    test_initial_state_correction_math()
    print("OK: initial-state correction math (CPU)")
    if GPU_OK:
        test_stateful_forward_matches_plain()
        print("OK: stateful forward == plain Mamba")
        test_split_scan_continuation()
        print("OK: split-scan continuation (literal carry)")
        test_policy_carry_smoke()
        print("OK: policy carry smoke (plain + BiMamba)")
        test_overlap_rollout_smoke()
        print("OK: overlap/MOSAIC rollout smoke")
    else:
        print("SKIP GPU tests (needs CUDA + mamba_ssm)")
