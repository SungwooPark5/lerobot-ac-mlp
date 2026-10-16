#!/usr/bin/env python
"""val/test eval for acm2_bimamba_self_atten (lerobot-ac-mlp worktree).

Unlike lerobot-final, ac-mlp training does NOT write metrics.json/eval_curve, so this driver runs
validation itself. Per seed:
  1. re-eval each saved checkpoint (100k/130k/150k/180k/200k) with 50 ep, env seed = train seed  (= validation)
  2. pick the checkpoint with the highest validation SR (ties -> earliest step)
  3. run a held-out test on that checkpoint: 500 ep, test seed 100000
  4. aggregate test SR across seeds -> mean +/- std

Both stages call lerobot_eval, which writes <output_dir>/eval/eval_info.json (info["overall"]["pc_success"],
a percentage 0-100). Results are cached: an existing eval_info.json is reused unless --force.

  cd /home1/eunji24/lerobot_project/lerobot-ac-mlp
  python eval_acm2_bimamba_self_atten.py                 # seeds 0 1 2
  python eval_acm2_bimamba_self_atten.py --seeds 3 4
  python eval_acm2_bimamba_self_atten.py --dry-run       # only re-eval val + pick best, skip 500ep test
  python eval_acm2_bimamba_self_atten.py --force         # ignore cache, recompute
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BASE = Path("/home1/eunji24/lerobot_project/outputs/v23/train/transfer/valtest_1e-5/acm2_bimamba_self_atten")
CKPT_STEPS = [100000, 130000, 150000, 180000, 200000]
SEEDS = [0, 1, 2]
VAL_N_EPISODES = 300       # validation, env seed = train seed (re-evaluated here; overridable via --val-episodes)
TEST_SEED = 100000         # held-out; disjoint from validation env seeds (train seeds 0..4 -> 0..49 etc.)
TEST_N_EPISODES = 500
BATCH_SIZE = 50
ENV_ARGS = ["--env.type=aloha", "--env.task=AlohaTransferCube-v0"]
PYTHONPATH = "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src:/home1/eunji24/lerobot_project/lerobot-ac-mlp"


def run_eval(ckpt_dir: Path, out_dir: Path, seed: int, n_episodes: int, force: bool = False) -> float:
    """Run lerobot_eval on a checkpoint; return pc_success (percent). Cached by out_dir."""
    cached = out_dir / "eval" / "eval_info.json"
    if cached.exists() and not force:
        info = json.loads(cached.read_text())
        return info["overall"]["pc_success"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_eval",
        f"--policy.path={ckpt_dir}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        *ENV_ARGS,
        f"--seed={seed}",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.batch_size={BATCH_SIZE}",
        "--eval.use_async_envs=false",
        f"--output_dir={out_dir}",
    ]
    env = dict(os.environ, MUJOCO_GL="egl", PYTHONPATH=PYTHONPATH)
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    info = json.loads(cached.read_text())
    return info["overall"]["pc_success"]


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n if n > 1 else 0.0  # population std (matches np.std default)
    return m, var ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--val-episodes", type=int, default=VAL_N_EPISODES, help="validation episodes per checkpoint")
    ap.add_argument("--dry-run", action="store_true", help="only run validation + pick best; skip 500ep test")
    ap.add_argument("--force", action="store_true", help="ignore cached eval_info.json, recompute")
    args = ap.parse_args()
    val_n = args.val_episodes

    rows = []          # (seed, best_step, best_val_sr, test_sr)
    test_srs = []
    for seed in args.seeds:
        run_dir = BASE / f"seed{seed}"
        if not run_dir.is_dir():
            print(f"[skip] seed{seed}: {run_dir} not found")
            continue

        # ---- validation: re-eval each checkpoint (50 ep, env seed = train seed) ----
        val = {}
        for step in CKPT_STEPS:
            ckpt = run_dir / "checkpoints" / str(step) / "pretrained_model"
            if not ckpt.exists():
                print(f"  [warn] seed{seed} step{step}: checkpoint missing ({ckpt})")
                continue
            out_dir = run_dir / "val_eval" / f"step{step}_seed{seed}_n{val_n}"
            sr = run_eval(ckpt, out_dir, seed=seed, n_episodes=val_n, force=args.force)
            val[step] = sr
            print(f"[seed{seed}] val step{step}: SR={sr:.2f}")
        if not val:
            print(f"[skip] seed{seed}: no validated checkpoints")
            continue

        best_sr = max(val.values())
        best_step = min(s for s, sr in val.items() if sr == best_sr)  # earliest on tie
        print(f"[seed{seed}] best-val step={best_step} (val SR={best_sr:.2f})")

        if args.dry_run:
            rows.append((seed, best_step, best_sr, None))
            continue

        # ---- test: held-out seed, 500 ep ----
        best_ckpt = run_dir / "checkpoints" / str(best_step) / "pretrained_model"
        test_out = run_dir / "test_eval" / f"step{best_step}_seed{TEST_SEED}_n{TEST_N_EPISODES}"
        test_sr = run_eval(best_ckpt, test_out, seed=TEST_SEED, n_episodes=TEST_N_EPISODES, force=args.force)
        print(f"[seed{seed}] test SR (seed={TEST_SEED}, n={TEST_N_EPISODES}) = {test_sr:.2f}")
        rows.append((seed, best_step, best_sr, test_sr))
        test_srs.append(test_sr)

    # ---- summary ----
    print("\n" + "=" * 64)
    print(f"{'seed':<6}{'best_step':<11}{'val_SR':<10}{'test_SR':<10}")
    print("-" * 64)
    for seed, step, val, test in rows:
        ts = f"{test:.2f}" if test is not None else "-"
        print(f"{seed:<6}{step:<11}{val:<10.2f}{ts:<10}")
    if not args.dry_run and test_srs:
        m, s = mean_std(test_srs)
        print("-" * 64)
        print(f"acm2_bimamba_self_atten  test mean={m:.2f}  std={s:.2f}  (n={len(test_srs)} seeds, %)")

    summary = {
        "model": "acm2_bimamba_self_atten",
        "test_seed": TEST_SEED,
        "test_n_episodes": TEST_N_EPISODES,
        "val_n_episodes": val_n,
        "rows": [
            {"seed": sd, "best_step": st, "val_sr": v, "test_sr": t} for (sd, st, v, t) in rows
        ],
        "aggregate": (dict(zip(("mean", "std"), mean_std(test_srs))) | {"n": len(test_srs)})
        if test_srs else None,
    }
    (BASE / "valtest_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {BASE / 'valtest_summary.json'}")


if __name__ == "__main__":
    main()
