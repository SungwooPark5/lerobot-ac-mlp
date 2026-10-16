#!/usr/bin/env python
"""Position-wise action prediction error within a K=100 chunk, comparing plain (ACM2,
forward-only) / action_only (BiMamba, forward+reverse, fixed scan) / dual_forward
(BiMamba, forward+forward, capacity-matched control) -- and optionally whole_flip
(BiMamba, forward+reverse, pre-fix whole-sequence flip) -- with paired, bootstrapped
statistics, not just a mean-difference eyeball.

Every model predicts the full K-action chunk from observation alone via
`predict_action_chunk()` -- NOT teacher-forced with the ground-truth action: in
`modeling_acm2_sscp_literal.py`, `forward()` builds `decoder_in` as all zeros (never
populated from the batch's action), and the VAE-latent path that *would* read
`batch[ACTION]` is gated on `self.training` (False once `predict_action_chunk` calls
`self.eval()`). So this is purely observation-conditioned prediction, compared
position-by-position against the dataset's ground-truth action chunk (both in
normalized/preprocessor space).

What changed vs the first version of this script
--------------------------------------------------
1. Sample selection: only chunks where ALL K=100 positions are unpadded
   (action_is_pad all-False) are used, so every position has the exact same sample set
   and the exact same n -- no per-position count drift from episode-boundary padding.
2. Paired, per-sample statistics: for each kept sample i and position j we keep
   error_model(i, j) (not just the position mean), so we can form
       d(i, j) = error_plain(i, j) - error_action_only(i, j)
       d(i, j) = error_dual_forward(i, j) - error_action_only(i, j)
   and bootstrap (resampling *samples*, i.e. resampling i with replacement and
   re-averaging over j's own samples, which preserves each sample's own
   across-position correlation) a 95% CI for the position-wise mean difference --
   instead of just reporting two independent means per model.
3. Early/middle/late region rollups (positions 0-9 / 10-49 / 50-99) with the same
   paired-bootstrap treatment, collapsing position within region per sample first.
4. Two figures: the MAE-vs-position curve for all models, and the paired
   difference-vs-position curve with its 95% CI band and a y=0 reference line.

Position labeling note: for every bimamba_scan mode (same / random / action_only /
reverse aka "whole flip"), `_unreorder` exactly inverts whatever `_reorder` did before
the action_head is applied -- each branch is a flip (self-inverse) or an explicit
permutation inverse -- so "position j" means the same thing (the j-th decoder query in
original, un-flipped order) across every checkpoint compared here.

Caveat: HuggingFaceVLA/libero has no held-out validation split, so this necessarily
reuses the training data distribution -- "how well did each model fit its own training
chunks position-by-position", not a held-out generalization test.

No LIBERO rollout / no MUJOCO_GL needed -- just dataset + model inference. A GPU speeds
it up but isn't required for correctness.

Usage:
    export PYTHONPATH=/home1/eunji24/lerobot_project/lerobot-final/src:/home1/eunji24/lerobot_project/lerobot-final
    /home1/eunji24/venvs/lerobot_libero/bin/python compare_position_error.py \
        --ckpt plain=/home1/eunji24/lerobot_project/outputs/fair_ksweep/libero_10/plain_K100/seed0/checkpoints/150000/pretrained_model \
        --ckpt action_only=/home1/eunji24/lerobot_project/outputs/nopairs/libero_10/bimamba_actiononly_K100/seed0/checkpoints/150000/pretrained_model \
        --ckpt dual_forward=/home1/eunji24/lerobot_project/outputs/nopairs/libero_10/scan_same_K100/seed0/checkpoints/150000/pretrained_model \
        --ckpt whole_flip=/home1/eunji24/lerobot_project/outputs/nopairs/libero_10/bimamba_K100/seed0/checkpoints/100000/pretrained_model \
        --n-samples 4000 --batch-size 32 --n-boot 2000 \
        --out position_error_K100.csv --fig-prefix position_error_K100

`--ckpt` names "plain", "action_only", "dual_forward" are required for the paired
comparisons; "whole_flip" is optional (MAE curve only, no CI -- kept only for context,
it's an older, differently-trained checkpoint).
"""

import argparse

import csv

import numpy as np
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION

REQUIRED_NAMES = ("plain", "action_only", "dual_forward")
OPTIONAL_NAMES = ("whole_flip",)
PAIRED_COMPARISONS = [
    ("plain", "action_only"),
    ("dual_forward", "action_only"),
]
REGIONS = [("early", 0, 9), ("middle", 10, 49), ("late", 50, 99)]


def parse_ckpt_arg(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--ckpt must be name=path, got: {spec!r}")
    name, path = spec.split("=", 1)
    return name, path


def load_policy(ckpt_dir: str, dataset, device: str):
    """Load a trained policy + its saved pre-processor from a `pretrained_model` dir,
    forcing `pretrained_path` to this checkpoint dir so `make_policy` loads the trained
    weights (matches how `lerobot_eval.py --policy.path=...` loads a checkpoint)."""
    cfg = TrainPipelineConfig.from_pretrained(ckpt_dir)
    cfg.policy.pretrained_path = str(ckpt_dir)

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.to(device)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=str(ckpt_dir),
    )
    return policy, preprocessor


def select_fully_valid_indices(dataset, n_samples: int, seed: int, scan_batch: int = 256) -> list[int]:
    """Dataset indices whose action chunk has NO padding at any position -- a property
    of the raw dataset sample (action_is_pad), independent of any model, so this is
    computed once and reused for every checkpoint."""
    pad_key = f"{ACTION}_is_pad"
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=g).tolist()

    valid: list[int] = []
    pos = 0
    while len(valid) < n_samples and pos < len(order):
        chunk_idx = order[pos: pos + scan_batch]
        pos += scan_batch
        subset = torch.utils.data.Subset(dataset, chunk_idx)
        loader = torch.utils.data.DataLoader(subset, batch_size=scan_batch, shuffle=False, num_workers=4)
        for batch in loader:
            is_pad = batch[pad_key]  # (B, K) bool, True = padded
            fully_valid = (~is_pad).all(dim=1)
            for local_i, ok in enumerate(fully_valid.tolist()):
                if ok:
                    valid.append(chunk_idx[local_i])
        print(f"  ... scanned {pos}/{len(order)}, found {len(valid)}/{n_samples} fully-valid", end="\r")
        if len(valid) >= n_samples:
            break
    print()
    if len(valid) < n_samples:
        print(f"WARNING: only found {len(valid)} fully-valid (unpadded, all K positions) "
              f"samples out of {pos} scanned (requested {n_samples}). Using what was found.")
    return valid[:n_samples]


@torch.no_grad()
def position_error_matrix(policy, preprocessor, dataset, indices: list[int], device: str, batch_size: int) -> torch.Tensor:
    """(N, K) per-sample, per-position mean-abs-error (over action_dim), in the exact
    order of `indices` (Subset + shuffle=False preserves order, so row i is the same
    underlying sample across every model this is called for)."""
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)

    chunks = []
    seen = 0
    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        batch = preprocessor(batch)
        preds = policy.predict_action_chunk(batch)  # (B, K, A), observation-only
        gt = batch[ACTION]  # (B, K, A)
        abs_err = (preds - gt).abs().mean(dim=-1)  # (B, K)
        chunks.append(abs_err.cpu())
        seen += preds.shape[0]
        print(f"  ... {seen}/{len(indices)} samples", end="\r")
    print()
    return torch.cat(chunks, dim=0)  # (N, K)


def paired_bootstrap(diff: np.ndarray, n_boot: int, alpha: float, seed: int):
    """diff: (N,) or (N, K). Resamples sample index i (with replacement) once per
    bootstrap draw and applies that same resampling across all columns (positions),
    preserving each sample's own cross-position correlation. Returns (mean, lo, hi),
    each shaped like diff.shape[1:] (scalar if diff is 1-D)."""
    N = diff.shape[0]
    rng = np.random.default_rng(seed)
    is_1d = diff.ndim == 1
    K = 1 if is_1d else diff.shape[1]
    boot = np.empty((n_boot, K), dtype=np.float64)
    flat = diff.reshape(N, K)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boot[b] = flat[idx].mean(axis=0)
    mean = flat.mean(axis=0)
    lo = np.percentile(boot, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
    if is_1d:
        return mean[0], lo[0], hi[0]
    return mean, lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", required=True, type=parse_ckpt_arg, dest="ckpts",
                    metavar="name=path",
                    help="repeatable: name=pretrained_model_dir. Needs plain/action_only/dual_forward; whole_flip optional.")
    ap.add_argument("--n-samples", type=int, default=4000, help="target number of FULLY-VALID (unpadded) chunks")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05, help="95% CI by default")
    ap.add_argument("--out", default="position_error_K100.csv")
    ap.add_argument("--fig-prefix", default="position_error_K100")
    args = ap.parse_args()

    ckpts = dict(args.ckpts)
    missing = [n for n in REQUIRED_NAMES if n not in ckpts]
    if missing:
        raise SystemExit(f"--ckpt is missing required name(s): {missing} (have: {list(ckpts)})")
    names = [n for n in REQUIRED_NAMES if n in ckpts] + [n for n in OPTIONAL_NAMES if n in ckpts]
    extra = [n for n in ckpts if n not in REQUIRED_NAMES and n not in OPTIONAL_NAMES]
    if extra:
        print(f"note: ignoring unrecognized --ckpt name(s) for the paired analysis: {extra} "
              f"(only {REQUIRED_NAMES + OPTIONAL_NAMES} are used)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    base_cfg = TrainPipelineConfig.from_pretrained(ckpts[names[0]])
    print(f"loading dataset {base_cfg.dataset.repo_id} (chunk_size={base_cfg.policy.chunk_size}) ...")
    dataset = make_dataset(base_cfg)
    print(f"dataset has {len(dataset)} samples total")

    print(f"selecting {args.n_samples} fully-valid (all {base_cfg.policy.chunk_size} positions unpadded) samples ...")
    indices = select_fully_valid_indices(dataset, args.n_samples, args.seed)
    N = len(indices)
    print(f"using n={N} samples for every model/position")

    err = {}  # name -> (N, K) tensor
    K = None
    for name in names:
        path = ckpts[name]
        print(f"[{name}] loading {path}")
        policy, preprocessor = load_policy(path, dataset, device)
        if K is None:
            K = policy.config.chunk_size
        else:
            assert policy.config.chunk_size == K, f"chunk_size mismatch: {name}={policy.config.chunk_size} vs {K}"
        print(f"[{name}] computing per-sample per-position error over n={N}")
        err[name] = position_error_matrix(policy, preprocessor, dataset, indices, device, args.batch_size)
        del policy, preprocessor
        torch.cuda.empty_cache()

    mae = {name: err[name].mean(dim=0).numpy() for name in names}  # (K,) each

    # ---- paired, bootstrapped position-wise differences ----
    diffs = {}  # (a, b) -> dict(mean, lo, hi) arrays of shape (K,)
    for a, b in PAIRED_COMPARISONS:
        d = (err[a] - err[b]).numpy()  # (N, K)
        mean, lo, hi = paired_bootstrap(d, args.n_boot, args.alpha, args.seed)
        diffs[(a, b)] = {"mean": mean, "lo": lo, "hi": hi}
        print(f"[{a} - {b}] bootstrapped ({args.n_boot} draws)")

    # ---- CSV ----
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        header = ["position"] + [f"{name}_mae" for name in names]
        for a, b in PAIRED_COMPARISONS:
            tag = f"{a}_minus_{b}"
            header += [tag, f"{tag}_ci_low", f"{tag}_ci_high"]
        header.append("n")
        w.writerow(header)
        for j in range(K):
            row = [j] + [mae[name][j] for name in names]
            for a, b in PAIRED_COMPARISONS:
                d = diffs[(a, b)]
                row += [d["mean"][j], d["lo"][j], d["hi"][j]]
            row.append(N)
            w.writerow(row)
    print(f"\nwrote {args.out}")

    # ---- region rollups ----
    print(f"\n=== region rollups (n={N}, {args.n_boot}-draw bootstrap, {int((1 - args.alpha) * 100)}% CI, seed={args.seed}) ===")
    region_rows = []
    region_summary = {}
    for region_name, lo_pos, hi_pos in REGIONS:
        print(f"\n-- {region_name} (position {lo_pos}-{hi_pos}) --")
        region_mae = {name: err[name][:, lo_pos:hi_pos + 1].mean().item() for name in names}
        for name in names:
            print(f"  {name:14s} mean_mae={region_mae[name]:.4f}")
        region_summary[region_name] = {"mae": region_mae}
        for a, b in PAIRED_COMPARISONS:
            per_sample = (err[a][:, lo_pos:hi_pos + 1] - err[b][:, lo_pos:hi_pos + 1]).mean(dim=1).numpy()  # (N,)
            mean, lo, hi = paired_bootstrap(per_sample, args.n_boot, args.alpha, args.seed)
            sig = "significant (CI excludes 0)" if (lo > 0 or hi < 0) else "not significant (CI includes 0)"
            print(f"  {a}-{b} paired diff: mean={mean:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  {sig}")
            region_rows.append([region_name, lo_pos, hi_pos, a, b, mean, lo, hi, N])
            region_summary[region_name][(a, b)] = (mean, lo, hi)

    region_csv = args.out.rsplit(".", 1)[0] + "_regions.csv"
    with open(region_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "pos_lo", "pos_hi", "a", "b", "mean_diff_a_minus_b", "ci_low", "ci_high", "n"])
        w.writerows(region_rows)
    print(f"\nwrote {region_csv}")

    # ---- compact requested table: Region | Plain MAE | Action-only MAE | Dual-forward MAE | Plain-Bi diff [95% CI] | Dual-Bi diff [95% CI] ----
    if {"plain", "action_only", "dual_forward"}.issubset(names):
        def ci_str(mean, lo, hi):
            return f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]"

        print(f"\n=== summary table (n={N}, {args.n_boot}-draw paired bootstrap, "
              f"{int((1 - args.alpha) * 100)}% CI, seed={args.seed}) ===")
        col_w = [10, 11, 15, 15, 26, 26]
        headers = ["Region", "Plain MAE", "Action-only MAE", "Dual-forward MAE",
                   "Plain-Bi diff [95% CI]", "Dual-Bi diff [95% CI]"]
        print("  ".join(h.ljust(w) for h, w in zip(headers, col_w)))
        for region_name, _, _ in REGIONS:
            rs = region_summary[region_name]
            plain_mae = rs["mae"]["plain"]
            ao_mae = rs["mae"]["action_only"]
            df_mae = rs["mae"]["dual_forward"]
            plain_bi = ci_str(*rs[("plain", "action_only")])
            dual_bi = ci_str(*rs[("dual_forward", "action_only")])
            row = [region_name, f"{plain_mae:.4f}", f"{ao_mae:.4f}", f"{df_mae:.4f}", plain_bi, dual_bi]
            print("  ".join(v.ljust(w) for v, w in zip(row, col_w)))

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for name in names:
            ax.plot(range(K), mae[name], label=name)
        ax.set_xlabel("action-chunk position")
        ax.set_ylabel("MAE (normalized action space)")
        ax.set_title(f"Position-wise action MAE (K={K}, n={N})")
        ax.legend()
        fig.tight_layout()
        mae_path = f"{args.fig_prefix}_mae.png"
        fig.savefig(mae_path, dpi=150)
        plt.close(fig)
        print(f"wrote {mae_path}")

        fig, ax = plt.subplots(figsize=(8, 5))
        for a, b in PAIRED_COMPARISONS:
            d = diffs[(a, b)]
            x = range(K)
            ax.plot(x, d["mean"], label=f"{a} - {b}")
            ax.fill_between(x, d["lo"], d["hi"], alpha=0.2)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("action-chunk position")
        ax.set_ylabel("paired MAE difference")
        ax.set_title(f"Paired position-wise MAE difference, {int((1 - args.alpha) * 100)}% bootstrap CI (n={N})")
        ax.legend()
        fig.tight_layout()
        diff_path = f"{args.fig_prefix}_diff.png"
        fig.savefig(diff_path, dpi=150)
        plt.close(fig)
        print(f"wrote {diff_path}")
    except ImportError as e:
        print(f"matplotlib not available ({e}) -- skipping figures. CSVs were still written.")

    print(f"\n{'pos':>5}" + "".join(f"{name:>14}" for name in names))
    step = max(1, K // 20)
    for j in range(0, K, step):
        print(f"{j:>5}" + "".join(f"{mae[name][j]:>14.4f}" for name in names))


if __name__ == "__main__":
    main()
