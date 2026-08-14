#!/usr/bin/env python
"""Check that refactoring a flat policy into a subclass changed nothing.

Workflow for one variant, e.g. acm_bimamba:

    cd src/lerobot/policies/acm_bimamba
    cp modeling_acm_bimamba.py _legacy_modeling_acm_bimamba.py   # freeze the original
    ...rewrite modeling_acm_bimamba.py as a subclass...
    export PYTHONPATH=/home1/eunji24/lerobot_project/lerobot-bimos/src
    python tests/refactor_equivalence.py acm_bimamba             # must PASS
    git rm src/lerobot/policies/acm_bimamba/_legacy_modeling_acm_bimamba.py

Must run on a GPU node: mamba_ssm's causal_conv1d kernel is CUDA-only.

Three checks:

  1. parameter structure -- identical state_dict keys and shapes
  2. RNG consumption     -- same seed gives bit-identical initial weights, so
                            training from scratch still reproduces
  3. computation         -- with the SAME weights loaded into both, the same
                            input gives bit-identical actions and loss

(3) is the real test: it separates "computes the same function" from "was
initialised from the same random draw". (2) is stricter than equivalence needs
but is what keeps old runs reproducible, so a failure there is a warning, not
an error -- the exit code is 2 rather than 0.

Results so far (RTX A6000), all three checks passing:
  acm_bimamba        842 -> 119 lines, 198 tensors, loss 79.96869659
  acm_refiner       1231 -> 219 lines, 191 tensors, loss 79.94760895
  acm_bimamba_gate  1122 -> 157 lines, 204 tensors, loss 79.91233063

acm_bimamba was re-run after BiMambaWholeFlipDecoder gained the _post_scan hook
and returned the same loss to the last digit, which is the hook being an
identity rather than an assertion that it is. acm_bimamba_gate's 204 tensors are
acm_bimamba's 198 plus the gate's six, so the gate really was built and executed
-- with use_bimamba_forget_gate left at its default it would not have been.

Caveat for acm_refiner and, from a quick scan, acm_cross_atten / acm_moe /
acm_self_atten / acm_accel_loss: those configs never declared mamba_d_state,
mamba_d_conv or mamba_expand while their decoders read all three, so the
pre-refactor policy raised AttributeError on construction and could not run at
all. The frozen config gets those three fields back at ACMConfig's values (the
ones the refactored config now inherits) purely so there is something to compare
against; the comparison is then against what the original would have computed
had it been runnable.
"""

import argparse
import dataclasses
import importlib
import sys

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

SEED = 0
BATCH = 2
CHUNK = 8
STATE_DIM = 14
ACTION_DIM = 14
IMG_HW = (96, 96)

# Variants whose file names do not follow modeling_<variant>.py.
MODULE_OVERRIDES = {
    "acm_self_atten": "modeling_acm_sekf_atten",  # sic -- typo in the filename
    "acm_accel_loss": "modeling_acm_accel",
}


def module_names(variant: str) -> tuple[str, str]:
    stem = MODULE_OVERRIDES.get(variant, f"modeling_{variant}")
    pkg = f"lerobot.policies.{variant}"
    return f"{pkg}.{stem}", f"{pkg}._legacy_{stem}"


def find_policy_class(module, variant: str):
    """The policy class this module defines -- the one whose `name` is the variant.

    Matching on `name` rather than the class name because the classes are called
    all sorts of things (ACMBimambaPolicy, ACMPolicy, ...), and because a module
    that imports its parent would otherwise offer two candidates.
    """
    hits = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, PreTrainedPolicy)
        and getattr(obj, "name", None) == variant
    ]
    if not hits:
        raise SystemExit(f"no PreTrainedPolicy with name == '{variant}' in {module.__name__}")
    if len(hits) > 1:
        raise SystemExit(f"ambiguous: {[h.__name__ for h in hits]} in {module.__name__}")
    return hits[0]


WANTED = {
    "chunk_size": CHUNK,
    "n_action_steps": CHUNK,
    "dim_model": 64,
    "n_heads": 4,
    "dim_feedforward": 128,
    "n_encoder_layers": 1,
    "n_decoder_layers": 1,
    "use_mamba": True,
    "use_bimamba_decoder": True,
    "use_bimamba_forget_gate": True,
    "use_vae": True,
    "vision_backbone": "resnet18",
    "pretrained_backbone_weights": None,
    "temporal_ensemble_coeff": None,
}


def shared_kwargs(*config_classes) -> dict:
    """The settings above, restricted to fields every config class declares.

    The frozen configs are older snapshots with fewer fields -- acm_bimamba
    declares use_bimamba_decoder, acm_refiner does not -- and passing an unknown
    one is a TypeError. Intersecting rather than filtering per class matters:
    filtering separately would hand the refactored config a field its frozen
    counterpart never saw, and the two sides would no longer be running the same
    settings. Anything dropped is exercised at its default, which is the value
    the pre-refactor code effectively ran with.
    """
    known = set.intersection(*({f.name for f in dataclasses.fields(c)} for c in config_classes))
    return {k: v for k, v in WANTED.items() if k in known}


def make_config(config_cls, kwargs: dict):
    """Small but structurally faithful: VAE on, one camera, optional blocks on where
    the config supports them -- a flag left at its default would leave the very code
    the variant exists for untested."""
    cfg = config_cls(**kwargs)
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


def make_batch(device) -> dict:
    # Built on CPU from a private generator so the batch is identical regardless of
    # how much CUDA RNG the model construction consumed, then moved to the device.
    g = torch.Generator().manual_seed(1234)
    batch = {
        OBS_STATE: torch.randn(BATCH, STATE_DIM, generator=g),
        f"{OBS_IMAGES}.top": torch.rand(BATCH, 3, *IMG_HW, generator=g),
        ACTION: torch.randn(BATCH, CHUNK, ACTION_DIM, generator=g),
        "action_is_pad": torch.zeros(BATCH, CHUNK, dtype=torch.bool),
    }
    return {k: v.to(device) for k, v in batch.items()}


def seed_all(seed: int) -> None:
    """Dropout draws from the CUDA generator once the model is on GPU, so both
    generators have to be reset before each training forward."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build(policy_cls, kwargs, device):
    seed_all(SEED)
    return policy_cls(make_config(policy_cls.config_class, kwargs)).to(device)


def fail(msg: str) -> None:
    print(f"\n  FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variant", help="policy directory name, e.g. acm_bimamba")
    variant = ap.parse_args().variant

    if not torch.cuda.is_available():
        fail(
            "no CUDA device. mamba_ssm's causal_conv1d kernel is CUDA-only, so this "
            "test must run on a GPU node (it will not work on the login node)."
        )
    device = torch.device("cuda")
    print(f"variant: {variant}   device: {torch.cuda.get_device_name(0)}")

    new_mod_name, legacy_mod_name = module_names(variant)
    try:
        legacy_mod = importlib.import_module(legacy_mod_name)
    except ModuleNotFoundError:
        fail(
            f"{legacy_mod_name} not found. Freeze the pre-refactor file first:\n"
            f"    cp <modeling>.py _legacy_<modeling>.py"
        )
    new_mod = importlib.import_module(new_mod_name)

    LegacyPolicy = find_policy_class(legacy_mod, variant)
    NewPolicy = find_policy_class(new_mod, variant)
    print(f"comparing {LegacyPolicy.__name__} (legacy) vs {NewPolicy.__name__} (new), seed {SEED}")

    kwargs = shared_kwargs(LegacyPolicy.config_class, NewPolicy.config_class)
    print(f"config fields applied: {len(kwargs)}/{len(WANTED)}")
    old = build(LegacyPolicy, kwargs, device)
    new = build(NewPolicy, kwargs, device)

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
    batch = make_batch(device)

    old.eval()
    new.eval()
    with torch.no_grad():
        a_old = old.predict_action_chunk(dict(batch))
        a_new = new.predict_action_chunk(dict(batch))
    if not torch.equal(a_old, a_new):
        fail(f"eval actions differ, max |diff| = {(a_old - a_new).abs().max().item():.3e}")

    seed_all(7)
    old.train()
    loss_old, dict_old = old.forward(dict(batch))
    seed_all(7)
    new.train()
    loss_new, dict_new = new.forward(dict(batch))
    if not torch.equal(loss_old, loss_new):
        fail(f"train loss differs: {loss_old.item():.8f} vs {loss_new.item():.8f}")
    if dict_old.keys() != dict_new.keys():
        extra = sorted(set(dict_new) - set(dict_old))
        missing = sorted(set(dict_old) - set(dict_new))
        fail(f"loss_dict keys differ -- extra {extra}, missing {missing}")

    print(f"  [3/3] compute OK -- actions bit-identical, loss {loss_old.item():.8f}")

    print(
        "\nPASS -- the refactor is behaviour-preserving."
        if rng_ok
        else "\nPASS (compute) -- but initial weights drifted; see [2/3]."
    )
    sys.exit(0 if rng_ok else 2)


if __name__ == "__main__":
    main()
