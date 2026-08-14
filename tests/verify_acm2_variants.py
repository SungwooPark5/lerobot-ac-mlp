#!/usr/bin/env python
"""Check that acm2_bimamba and acm2_self_atten are acm2 plus exactly one flag.

    export PYTHONPATH=/home1/eunji24/lerobot_project/lerobot-bimos/src
    python tests/verify_acm2_variants.py

Needs a GPU node -- mamba_ssm's kernels are CUDA-only.

These two are not refactors of existing files, so refactor_equivalence.py does not
apply: there is no pre-refactor version to compare against. They replace
acm2_bimamba_self_atten, which was a byte-identical copy of acm2 apart from its
`name` string and left both of the flags it was named after at False.

So what is checked instead is the claim each variant actually makes:

    ACM2BiMambaPolicy()                 == ACM2Policy(use_bimamba_decoder=True)
    ACM2SelfAttenPolicy()               == ACM2Policy(use_action_self_attention=True)

and, to show the flag is not inert, that each differs from plain acm2.

Exit code 0 = both variants are what they say they are.
"""

import sys

import torch

from lerobot.policies.acm2.configuration_acm2 import ACM2Config
from lerobot.policies.acm2.modeling_acm2 import ACM2Policy
from lerobot.policies.acm2_bimamba.configuration_acm2_bimamba import ACM2BiMambaConfig
from lerobot.policies.acm2_bimamba.modeling_acm2_bimamba import ACM2BiMambaPolicy
from lerobot.policies.acm2_self_atten.configuration_acm2_self_atten import ACM2SelfAttenConfig
from lerobot.policies.acm2_self_atten.modeling_acm2_self_atten import ACM2SelfAttenPolicy

from refactor_equivalence import SEED, make_batch, make_config, seed_all

BASE = {
    "chunk_size": 8,
    "n_action_steps": 8,
    "dim_model": 64,
    "n_heads": 4,
    "dim_feedforward": 128,
    "n_encoder_layers": 1,
    "n_decoder_layers": 1,
    "use_vae": True,
    "vision_backbone": "resnet18",
    "pretrained_backbone_weights": None,
    "temporal_ensemble_coeff": None,
}


def build(policy_cls, config_cls, extra, device):
    seed_all(SEED)
    return policy_cls(make_config(config_cls, {**BASE, **extra})).to(device)


def fail(msg: str) -> None:
    print(f"\n  FAIL: {msg}")
    sys.exit(1)


def compare(label, reference, variant, device, must_differ_from):
    """reference and variant must agree; both must differ from must_differ_from."""
    ref_sd, var_sd = reference.state_dict(), variant.state_dict()
    if ref_sd.keys() != var_sd.keys():
        only_ref = sorted(set(ref_sd) - set(var_sd))
        only_var = sorted(set(var_sd) - set(ref_sd))
        fail(f"{label}: state_dict keys differ\n  only in acm2: {only_ref}\n  only in variant: {only_var}")

    drift = [k for k in ref_sd if not torch.equal(ref_sd[k], var_sd[k])]
    if drift:
        fail(f"{label}: {len(drift)} tensors differ at init, e.g. {drift[:3]}")

    batch = make_batch(device)
    variant.load_state_dict(ref_sd)

    reference.eval()
    variant.eval()
    with torch.no_grad():
        a_ref = reference.predict_action_chunk(dict(batch))
        a_var = variant.predict_action_chunk(dict(batch))
    if not torch.equal(a_ref, a_var):
        fail(f"{label}: actions differ, max |diff| = {(a_ref - a_var).abs().max().item():.3e}")

    n = sum(p.numel() for p in variant.parameters())
    print(f"  {label}: matches acm2 with the flag set -- {len(var_sd)} tensors, {n:,} params")

    # The flag has to actually do something, or the variant is decoration.
    plain_sd = must_differ_from.state_dict()
    if plain_sd.keys() == var_sd.keys():
        must_differ_from.eval()
        with torch.no_grad():
            a_plain = must_differ_from.predict_action_chunk(dict(batch))
        if torch.equal(a_plain, a_var):
            fail(f"{label}: identical to plain acm2 -- the flag changes nothing")
        print(f"  {label}: differs from plain acm2 in output, as it should")
    else:
        extra = len(var_sd) - len(plain_sd)
        print(f"  {label}: differs from plain acm2 in structure ({extra:+d} tensors)")


def main() -> None:
    if not torch.cuda.is_available():
        fail("no CUDA device. mamba_ssm is CUDA-only, so run this on a GPU node.")
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}\n")

    plain = build(ACM2Policy, ACM2Config, {}, device)

    print("acm2_bimamba")
    compare(
        "bimamba",
        build(ACM2Policy, ACM2Config, {"use_bimamba_decoder": True}, device),
        build(ACM2BiMambaPolicy, ACM2BiMambaConfig, {}, device),
        device,
        plain,
    )

    print("\nacm2_self_atten")
    compare(
        "self_atten",
        build(ACM2Policy, ACM2Config, {"use_action_self_attention": True}, device),
        build(ACM2SelfAttenPolicy, ACM2SelfAttenConfig, {}, device),
        device,
        plain,
    )

    print("\nPASS -- both variants are acm2 plus exactly the flag they name.")


if __name__ == "__main__":
    main()
