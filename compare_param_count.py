#!/usr/bin/env python
"""Parameter-count comparison for ACM2 plain (forward-only) / dual_forward (BiMamba,
same-scan) / action_only (BiMamba, fixed action-only reverse scan) at K=100, loaded
straight from their trained checkpoints -- no training, no dataset iteration, just
`policy.named_parameters()`.

Checks:
1. total / trainable / decoder / non-decoder parameter counts per model (table).
2. Whether dual_forward and action_only are byte-for-byte the same parameterization
   (same parameter names AND same shapes) -- they should be: in
   `Mamba2LiteralSSCPDecoder.__init__` (modeling_acm2_sscp_literal.py), the branch that
   builds `forward_layers`/`backward_layers` is `elif self.use_bimamba_decoder:` and
   does not itself branch on `bimamba_scan` -- scan mode only changes _reorder/_unreorder
   (parameter-free flips/permutations) applied inside forward(), not what gets built.
   Only `bimamba_scan == "backward_only"` changes the module structure (single stack).
3. A module-level breakdown of where the plain -> dual_forward/action_only parameter
   increase actually comes from (visual backbone / transformer encoder / vae_encoder /
   decoder forward stack / decoder backward stack / decoder fuse / action_head / other),
   plus a raw dump of every distinct top-level submodule name found, so nothing is
   hidden behind a possibly-wrong guess at the naming scheme.
4. Plain -> dual_forward/action_only growth in total params (M, %) and decoder params
   (M, %).

Usage:
    export PYTHONPATH=/home1/eunji24/lerobot_project/lerobot-final/src:/home1/eunji24/lerobot_project/lerobot-final
    /home1/eunji24/venvs/lerobot_libero/bin/python compare_param_count.py \
        --ckpt act=/home1/eunji24/lerobot_project/outputs/fair_ksweep/libero_10/act_K100/seed0/checkpoints/150000/pretrained_model \
        --ckpt plain=/home1/eunji24/lerobot_project/outputs/fair_ksweep/libero_10/plain_K100/seed0/checkpoints/150000/pretrained_model \
        --ckpt dual_forward=/home1/eunji24/lerobot_project/outputs/nopairs/libero_10/scan_same_K100/seed0/checkpoints/150000/pretrained_model \
        --ckpt action_only=/home1/eunji24/lerobot_project/outputs/nopairs/libero_10/bimamba_actiononly_K100/seed0/checkpoints/150000/pretrained_model

ACT uses a different policy class (`self.decoder = ACTDecoder(...)`, from modeling_act.py)
but also names its module `.decoder.`, so the generic ".decoder." bucketing already
covers it correctly -- no ACT-specific code needed.

CPU is enough -- this never runs a forward pass, just builds the modules and counts.
"""

import argparse
from collections import defaultdict

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy

# Table row order follows whatever --ckpt args are given, in order (see main()).
# "dual_forward"/"action_only" structural-identity check and "plain"-relative growth
# section only run when those specific names are present among the --ckpt args.


def parse_ckpt_arg(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--ckpt must be name=path, got: {spec!r}")
    name, path = spec.split("=", 1)
    return name, path


def load_policy_cpu(ckpt_dir: str, ds_meta):
    """Load just the policy module (config + weights) on CPU -- no dataset iteration,
    no GPU needed for a parameter count."""
    cfg = TrainPipelineConfig.from_pretrained(ckpt_dir)
    cfg.policy.pretrained_path = str(ckpt_dir)
    cfg.policy.device = "cpu"
    policy = make_policy(cfg=cfg.policy, ds_meta=ds_meta)
    return policy


def bucket_name(name: str) -> str:
    if ".decoder.forward_layers" in name:
        return "decoder/forward_stack"
    if ".decoder.backward_layers" in name:
        return "decoder/backward_stack"
    if ".decoder." in name and "fuse" in name:
        return "decoder/fuse"
    if ".decoder." in name:
        return "decoder/other"
    if "vae_encoder" in name:
        return "vae_encoder"
    if "backbone" in name:
        return "visual_backbone"
    if ".encoder." in name or name.split(".")[1:2] == ["encoder"]:
        return "transformer_encoder"
    if "action_head" in name:
        return "action_head"
    return "other"


def counts(policy) -> dict:
    total = 0
    trainable = 0
    decoder = 0
    buckets = defaultdict(int)
    raw_prefix = defaultdict(int)  # first 2 name segments -> numel, as an unfiltered cross-check
    for name, p in policy.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        if ".decoder." in name:
            decoder += n
        buckets[bucket_name(name)] += n
        prefix = ".".join(name.split(".")[:3])
        raw_prefix[prefix] += n
    return {
        "total": total,
        "trainable": trainable,
        "decoder": decoder,
        "non_decoder": total - decoder,
        "buckets": dict(buckets),
        "raw_prefix": dict(raw_prefix),
    }


def fmt_m(n: int) -> str:
    return f"{n / 1e6:.3f}M"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", required=True, type=parse_ckpt_arg, dest="ckpts",
                    metavar="name=path", help="repeatable: name=pretrained_model_dir. Any names -- pass whichever checkpoints you want in the table.")
    args = ap.parse_args()

    ckpts = dict(args.ckpts)
    ORDER = [n for n, _ in args.ckpts]  # preserve the order given on the CLI
    if len(set(ORDER)) != len(ORDER):
        raise SystemExit(f"--ckpt names must be unique, got: {ORDER}")

    base_cfg = TrainPipelineConfig.from_pretrained(ckpts[ORDER[0]])
    print(f"dataset meta: {base_cfg.dataset.repo_id} (chunk_size={base_cfg.policy.chunk_size})")
    ds_meta = LeRobotDatasetMetadata(
        base_cfg.dataset.repo_id, root=base_cfg.dataset.root, revision=base_cfg.dataset.revision
    )

    policies = {}
    stats = {}
    for name in ORDER:
        path = ckpts[name]
        print(f"[{name}] loading {path}")
        policy = load_policy_cpu(path, ds_meta)
        policies[name] = policy
        stats[name] = counts(policy)

    # ---- main table ----
    print("\n" + "=" * 78)
    print(f"{'Model':<16} {'Total Params':>16} {'Trainable':>16} {'Decoder':>14} {'Non-decoder':>14}")
    for name in ORDER:
        s = stats[name]
        print(f"{name:<16} {fmt_m(s['total']):>16} {fmt_m(s['trainable']):>16} "
              f"{fmt_m(s['decoder']):>14} {fmt_m(s['non_decoder']):>14}")
    print("=" * 78)

    # ---- dual_forward vs action_only structural identity (only if both given) ----
    if "dual_forward" in stats and "action_only" in stats:
        print("\n=== dual_forward vs action_only: same parameterization? ===")
        df_named = {n: tuple(p.shape) for n, p in policies["dual_forward"].named_parameters()}
        ao_named = {n: tuple(p.shape) for n, p in policies["action_only"].named_parameters()}
        same_names = set(df_named) == set(ao_named)
        same_shapes = same_names and all(df_named[n] == ao_named[n] for n in df_named)
        print(f"  same parameter names: {same_names}")
        print(f"  same shapes (given same names): {same_shapes}")
        if not same_names:
            only_df = sorted(set(df_named) - set(ao_named))
            only_ao = sorted(set(ao_named) - set(df_named))
            if only_df:
                print(f"  only in dual_forward ({len(only_df)}): {only_df[:20]}{' ...' if len(only_df) > 20 else ''}")
            if only_ao:
                print(f"  only in action_only ({len(only_ao)}): {only_ao[:20]}{' ...' if len(only_ao) > 20 else ''}")
        elif not same_shapes:
            mismatched = [n for n in df_named if df_named[n] != ao_named[n]]
            print(f"  shape-mismatched names ({len(mismatched)}): {mismatched[:20]}{' ...' if len(mismatched) > 20 else ''}")
        print(f"  dual_forward vs action_only total param count equal: "
              f"{stats['dual_forward']['total'] == stats['action_only']['total']} "
              f"({fmt_m(stats['dual_forward']['total'])} vs {fmt_m(stats['action_only']['total'])})")

    # ---- module-level breakdown ----
    print("\n=== module-level breakdown (semantic buckets) ===")
    all_buckets = sorted(set().union(*(s["buckets"].keys() for s in stats.values())))
    header = f"{'bucket':<24}" + "".join(f"{name:>16}" for name in ORDER)
    print(header)
    for b in all_buckets:
        row = f"{b:<24}"
        for name in ORDER:
            row += f"{fmt_m(stats[name]['buckets'].get(b, 0)):>16}"
        print(row)

    print("\n=== raw top-level submodule breakdown (unfiltered cross-check) ===")
    all_prefixes = sorted(set().union(*(s["raw_prefix"].keys() for s in stats.values())))
    print(f"{'module.submodule':<40}" + "".join(f"{name:>16}" for name in ORDER))
    for p in all_prefixes:
        row = f"{p:<40}"
        for name in ORDER:
            row += f"{fmt_m(stats[name]['raw_prefix'].get(p, 0)):>16}"
        print(row)

    # ---- growth vs plain (only if plain given) ----
    if "plain" in stats:
        print("\n=== growth vs plain ===")
        plain_total = stats["plain"]["total"]
        plain_dec = stats["plain"]["decoder"]
        for name in ORDER:
            if name == "plain":
                continue
            s = stats[name]
            dt = s["total"] - plain_total
            dd = s["decoder"] - plain_dec
            print(f"  {name}:")
            print(f"    total   +{fmt_m(dt)}  (+{100 * dt / plain_total:.2f}%)")
            print(f"    decoder +{fmt_m(dd)}  (+{100 * dd / plain_dec:.2f}%)" if plain_dec > 0
                  else f"    decoder +{fmt_m(dd)}  (plain decoder params = 0, %% undefined)")
    else:
        print("\n(no 'plain' checkpoint given -- skipping growth-vs-plain section)")


if __name__ == "__main__":
    main()
