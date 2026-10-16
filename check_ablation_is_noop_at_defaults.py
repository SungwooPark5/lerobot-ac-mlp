#!/usr/bin/env python
"""Check the BiMamba ablation flags changed nothing at their defaults.

    source /home1/eunji24/venvs/lerobot_libero/bin/activate
    export PYTHONPATH=/home1/eunji24/lerobot_project/lerobot-final/src:/home1/eunji24/lerobot_project/lerobot-final
    python check_ablation_is_noop_at_defaults.py

Takes about a minute on a GPU node. Run it before trusting the stage-1 numbers.

Why it exists
-------------
bimamba_fuse / bimamba_scan were added to Mamba2LiteralSSCPDecoder while the stage-1
stride evals were already running. Those evals load checkpoints whose saved configs
predate the fields, so they fall back to bimamba_fuse="mean" and bimamba_scan=
"reverse", which are meant to reproduce the old decoder exactly. If that is wrong,
every stage-1 eval that started after the edit is measuring a different model than
the ones that started before it, and the grid is silently inconsistent.

So: build the decoder from the frozen pre-edit file and from the live one, load the
same weights into both, and compare outputs on the same input. Both the BiMamba path
and the unidirectional path are checked, with and without a carry.

Exit 0 means the defaults are a no-op and stage 1 is safe.
"""

import sys

import torch

from lerobot.policies.acm2_sscp_literal._pre_ablation_modeling import (
    Mamba2LiteralSSCPDecoder as OldDecoder,
)
from lerobot.policies.acm2_sscp_literal.configuration_acm2_sscp_literal import (
    ACM2SSCPLiteralConfig,
)
from lerobot.policies.acm2_sscp_literal.modeling_acm2_sscp_literal import (
    Mamba2LiteralSSCPDecoder as NewDecoder,
)

SEED = 0
B, T, K, D = 2, 6, 8, 64


def make_config(bimamba: bool) -> ACM2SSCPLiteralConfig:
    cfg = ACM2SSCPLiteralConfig(
        dim_model=D,
        n_heads=4,
        n_decoder_layers=1,
        use_bimamba_decoder=bimamba,
        chunk_size=K,
        n_action_steps=K,
    )
    # Left at their defaults on purpose -- reproducing the old decoder is exactly the
    # claim being tested.
    assert cfg.bimamba_fuse == "mean", cfg.bimamba_fuse
    assert cfg.bimamba_scan == "reverse", cfg.bimamba_scan
    return cfg


def build(cls, cfg, device):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return cls(cfg).to(device).eval()


def fail(msg: str) -> None:
    print(f"\n  FAIL: {msg}")
    sys.exit(1)


def compare(label: str, bimamba: bool, with_carry: bool, device) -> None:
    cfg = make_config(bimamba)
    old = build(OldDecoder, cfg, device)
    new = build(NewDecoder, cfg, device)

    old_sd, new_sd = old.state_dict(), new.state_dict()
    if old_sd.keys() != new_sd.keys():
        only_old = sorted(set(old_sd) - set(new_sd))
        only_new = sorted(set(new_sd) - set(old_sd))
        fail(f"{label}: parameter sets differ\n    only old: {only_old}\n    only new: {only_new}")
    new.load_state_dict(old_sd)

    g = torch.Generator().manual_seed(1234)
    x = torch.randn(K, B, D, generator=g).to(device)
    enc = torch.randn(T, B, D, generator=g).to(device)

    carry = None
    if with_carry:
        # One warm-up pass produces a real (conv, ssm) state to hand back in, which is
        # the path the deployed policy actually uses.
        with torch.no_grad():
            _, carry = old(x, enc)
        if not carry:
            print(f"  {label}: no carry produced, skipping the carried variant")
            return

    with torch.no_grad():
        out_old, st_old = old(x, enc, carry=carry)
        out_new, st_new = new(x, enc, carry=carry)

    if not torch.equal(out_old, out_new):
        fail(f"{label}: outputs differ, max |diff| = {(out_old - out_new).abs().max().item():.3e}")

    if len(st_old) != len(st_new):
        fail(f"{label}: carry length differs, {len(st_old)} vs {len(st_new)}")
    for i, (a, b) in enumerate(zip(st_old, st_new)):
        for j, (ta, tb) in enumerate(zip(a, b)):
            if not torch.equal(ta, tb):
                fail(f"{label}: carry[{i}][{j}] differs, max |diff| = {(ta - tb).abs().max().item():.3e}")

    n = sum(p.numel() for p in new.parameters())
    print(f"  {label}: identical -- {len(new_sd)} tensors, {n:,} params, carry len {len(st_new)}")


def main() -> None:
    if not torch.cuda.is_available():
        fail("no CUDA device. The Mamba2 SSD kernel is CUDA-only, so run this on a GPU node.")
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print("comparing _pre_ablation_modeling (frozen at HEAD) vs the live decoder\n")

    compare("bimamba, no carry ", bimamba=True, with_carry=False, device=device)
    compare("bimamba, carried  ", bimamba=True, with_carry=True, device=device)
    compare("unidirectional    ", bimamba=False, with_carry=False, device=device)
    compare("unidirectional, c ", bimamba=False, with_carry=True, device=device)

    print("\nPASS -- at their defaults the new flags reproduce the old decoder exactly.")
    print("Stage 1 evals started before and after the edit are measuring the same model.")


if __name__ == "__main__":
    main()
