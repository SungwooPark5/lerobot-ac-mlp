#!/usr/bin/env python
"""Check that refactoring acm_bimamba into a subclass of acm changed nothing.

Run this on a GPU node -- mamba_ssm imports triton, which needs a live CUDA
driver, so it fails on the login node:

    export PYTHONPATH=/home1/eunji24/lerobot_project/lerobot-bimos/src
    python tests/refactor_equivalence_acm_bimamba.py

Compares the pre-refactor file (_legacy_modeling_acm_bimamba.py, a verbatim copy
of the flat 842-line original) against the new subclass, on three axes:

  1. parameter structure -- identical state_dict keys and shapes
  2. RNG consumption     -- same seed gives bit-identical initial weights,
                            so training from scratch reproduces
  3. computation         -- with the SAME weights loaded into both, the same
                            input produces bit-identical actions and loss

(3) is the one that matters: it isolates "does the network compute the same
function" from "was it initialised from the same random draw". (2) is stricter
than equivalence requires but is what makes past runs reproducible.

Exit code 0 = the refactor is behaviour-preserving.
"""

import sys

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.acm_bimamba._legacy_modeling_acm_bimamba import (
    ACMBimambaPolicy as LegacyPolicy,
)
from lerobot.policies.acm_bimamba.configuration_acm_bimamba import ACMBiMambaConfig
from lerobot.policies.acm_bimamba.modeling_acm_bimamba import (
    ACMBimambaPolicy as NewPolicy,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

SEED = 0
BATCH = 2
CHUNK = 8
STATE_DIM = 14
ACTION_DIM = 14
IMG_HW = (96, 96)


def make_config() -> ACMBiMambaConfig:
    """Small but structurally faithful config: BiMamba on, VAE on, one camera."""
    cfg = ACMBiMambaConfig(
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_mamba=True,
        use_bimamba_decoder=True,
        use_vae=True,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        temporal_ensemble_coeff=None,
    )
    cfg.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        f"{OBS_IMAGES}.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, *IMG_HW)),
    }
    cfg.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }
    cfg.normalization_mapping = {
        "STATE": NormalizationMode.IDENTITY,
        "VISUAL": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    return cfg


def make_batch() -> dict:
    g = torch.Generator().manual_seed(1234)
    return {
        OBS_STATE: torch.randn(BATCH, STATE_DIM, generator=g),
        f"{OBS_IMAGES}.top": torch.rand(BATCH, 3, *IMG_HW, generator=g),
        ACTION: torch.randn(BATCH, CHUNK, ACTION_DIM, generator=g),
        "action_is_pad": torch.zeros(BATCH, CHUNK, dtype=torch.bool),
    }


def build(policy_cls):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return policy_cls(make_config())


def fail(msg: str) -> None:
    print(f"\n  FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    print("building both policies from seed", SEED)
    old = build(LegacyPolicy)
    new = build(NewPolicy)

    # ── 1. parameter structure ────────────────────────────────────────────────
    old_sd, new_sd = old.state_dict(), new.state_dict()
    if old_sd.keys() != new_sd.keys():
        only_old = sorted(set(old_sd) - set(new_sd))
        only_new = sorted(set(new_sd) - set(old_sd))
        fail(f"state_dict keys differ\n  only in legacy: {only_old}\n  only in new: {only_new}")
    bad_shape = [k for k in old_sd if old_sd[k].shape != new_sd[k].shape]
    if bad_shape:
        fail(f"shapes differ for {bad_shape[:5]}")
    n_params = sum(p.numel() for p in old.parameters())
    print(f"  [1/3] structure OK -- {len(old_sd)} tensors, {n_params:,} params")

    # ── 2. RNG consumption ────────────────────────────────────────────────────
    drift = [k for k in old_sd if not torch.equal(old_sd[k], new_sd[k])]
    if drift:
        print(f"  [2/3] WARNING -- {len(drift)} tensors differ at init, e.g. {drift[:3]}")
        print("        Same seed no longer reproduces the old initial weights.")
        print("        Architecture may still be identical; step 3 decides that.")
        rng_ok = False
    else:
        print("  [2/3] init OK -- same seed gives bit-identical weights")
        rng_ok = True

    # ── 3. computation, with weights forced equal ─────────────────────────────
    new.load_state_dict(old_sd)
    batch = make_batch()

    old.eval()
    new.eval()
    with torch.no_grad():
        a_old = old.predict_action_chunk(dict(batch))
        a_new = new.predict_action_chunk(dict(batch))
    if not torch.equal(a_old, a_new):
        fail(f"eval actions differ, max |diff| = {(a_old - a_new).abs().max().item():.3e}")

    torch.manual_seed(7)
    old.train()
    loss_old, dict_old = old.forward(dict(batch))
    torch.manual_seed(7)
    new.train()
    loss_new, dict_new = new.forward(dict(batch))
    if not torch.equal(loss_old, loss_new):
        fail(f"train loss differs: {loss_old.item():.8f} vs {loss_new.item():.8f}")
    if dict_old.keys() != dict_new.keys():
        extra = sorted(set(dict_new) - set(dict_old))
        missing = sorted(set(dict_old) - set(dict_new))
        fail(f"loss_dict keys differ -- extra {extra}, missing {missing}")

    print(f"  [3/3] compute OK -- actions bit-identical, loss {loss_old.item():.8f}")

    print("\nPASS -- the refactor is behaviour-preserving." if rng_ok else
          "\nPASS (compute) -- but initial weights drifted; see [2/3].")
    sys.exit(0 if rng_ok else 2)


if __name__ == "__main__":
    main()
