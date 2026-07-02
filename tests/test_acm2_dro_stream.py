"""GPU tests for acm2_dro (needs CUDA + mamba_ssm — run on the cluster).

The golden test: with dro_stream and no dropout, the STREAMING inference path
(select_action step by step, fed the same proprio sequence) must reproduce the
TEACHER-FORCED training forward exactly. This pins train==inference consistency,
which was the root cause of the v21 carry gap.

Run:  python tests/test_acm2_dro_stream.py   (also pytest-compatible)
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.acm2_dro.configuration_acm2_dro import ACM2DROConfig  # noqa: E402
from lerobot.policies.acm2_dro.modeling_acm2_dro import ACM2DROPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE  # noqa: E402

B, K, DS, DE, DA = 2, 8, 14, 6, 14


def _cfg(**overrides):
    kw = dict(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(DS,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(DE,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(DA,))},
        chunk_size=K,
        n_action_steps=1,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
        mamba2_d_state=16,
        mamba2_headdim=32,
        dropout=0.0,
    )
    kw.update(overrides)
    return ACM2DROConfig(**kw)


def _policy(**overrides):
    torch.manual_seed(0)
    return ACM2DROPolicy(_cfg(**overrides)).cuda().eval()


def _obs_batch(states_k, env):
    return {OBS_STATE: states_k, OBS_ENV_STATE: env}


def test_stream_equals_teacher_forced():
    """Streaming inference == teacher-forced forward on the same proprio sequence."""
    policy = _policy()
    torch.manual_seed(1)
    states = torch.randn(B, K, DS).cuda()
    env = torch.randn(B, DE).cuda()

    with torch.no_grad():
        acts_tf, _, obs_pred = policy.model(_obs_batch(states, env))
    assert acts_tf.shape == (B, K, DA)
    assert obs_pred is None  # innovation off

    policy.reset()
    stream = []
    for k in range(K):
        stream.append(policy.select_action(_obs_batch(states[:, k], env)))
    stream = torch.stack(stream, dim=1)

    err = (acts_tf - stream).abs().max().item()
    assert err < 1e-4, f"stream vs teacher-forced mismatch: max|Δ|={err:.2e}"


def test_chunk_rollover_reencodes():
    """After chunk_size steps the stream restarts a fresh context without errors."""
    policy = _policy()
    env = torch.randn(B, DE).cuda()
    policy.reset()
    for k in range(2 * K + 3):
        a = policy.select_action(_obs_batch(torch.randn(B, DS).cuda(), env))
        assert a.shape == (B, DA) and torch.isfinite(a).all()
    assert policy._k == 3


def test_innovation_training_and_loss():
    policy = _policy(dro_innovation=True, dro_obs_loss_weight=0.5)
    policy.train()
    torch.manual_seed(2)
    batch = {
        OBS_STATE: torch.randn(B, K, DS).cuda(),
        OBS_ENV_STATE: torch.randn(B, DE).cuda(),
        ACTION: torch.randn(B, K, DA).cuda(),
        "action_is_pad": torch.zeros(B, K, dtype=torch.bool).cuda(),
        f"{OBS_STATE}_is_pad": torch.zeros(B, K, dtype=torch.bool).cuda(),
    }
    loss, loss_dict = policy.forward(batch)
    assert torch.isfinite(loss)
    assert {"l1_loss", "kld_loss", "obs_l1"} <= set(loss_dict)
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


def test_innovation_gate_triggers_refresh():
    policy = _policy(dro_innovation=True, dro_gate_tau=1e-9, dro_vision_refresh="gate")
    env = torch.randn(B, DE).cuda()
    policy.reset()
    for k in range(4):
        policy.select_action(_obs_batch(torch.randn(B, DS).cuda(), env))
    gates = torch.stack(policy._gate_trace)  # (T, B)
    assert not gates[0].any(), "chunk start must not count as a gate trigger"
    assert gates[1:].all(), f"tau≈0 must trigger every post-start step, got {gates}"
    innov = torch.stack(policy._innov_trace)
    assert (innov[1:] > 0).all(), "innovation norms must be recorded"


def test_zero_init_innovation_is_noop():
    """With freshly-initialized (zero) innov_proj, ANY innovation input must leave the
    decode unchanged — the feedback starts as a no-op and is learned."""
    policy = _policy(dro_innovation=True)
    torch.manual_seed(3)
    states = torch.randn(B, K, DS).cuda()
    env = torch.randn(B, DE).cuda()
    model = policy.model
    with torch.no_grad():
        enc = model.encoder_context_inference(_obs_batch(states, env))
        out1, _ = model._decode_stream(enc, states, None)
        out2, _ = model._decode_stream(enc, states, 10.0 * torch.randn_like(states))
    assert (out1 - out2).abs().max().item() < 1e-5


def test_layer_c_augmentation_runs():
    policy = _policy(dro_train_state_noise=0.05, dro_train_push_prob=1.0, dro_train_push_mag=0.3)
    policy.train()
    batch = {
        OBS_STATE: torch.randn(B, K, DS).cuda(),
        OBS_ENV_STATE: torch.randn(B, DE).cuda(),
        ACTION: torch.randn(B, K, DA).cuda(),
        "action_is_pad": torch.zeros(B, K, dtype=torch.bool).cuda(),
    }
    loss, _ = policy.forward(batch)
    assert torch.isfinite(loss)


def test_imagine_chunk_shape():
    policy = _policy(dro_innovation=True)
    with torch.no_grad():
        chunk = policy.predict_action_chunk(
            _obs_batch(torch.randn(B, DS).cuda(), torch.randn(B, DE).cuda())
        )
    assert chunk.shape == (B, K, DA) and torch.isfinite(chunk).all()


def test_state_delta_indices_property():
    assert _cfg().state_delta_indices == list(range(K))
    assert _cfg(dro_stream=False).state_delta_indices is None


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available (mamba_ssm SSD kernel is CUDA-only)")
        sys.exit(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
