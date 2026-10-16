#!/usr/bin/env python
"""Smoothness of the executed trajectories for the LIBERO-10 K sweeps, TE vs no TE.

    source /home1/eunji24/venvs/lerobot_libero/bin/activate   # or any env with torch
    python measure_smoothness_ksweep.py
    python measure_smoothness_ksweep.py --csv smoothness_ksweep.csv --step 150000

No GPU, no rollouts: it reads the action logs the evals already wrote, at
    outputs/<arm>/libero_10/K<k>/seed0/eval/videos_step_<step>/action_logs/episode_*.pt
each a (T, 7) tensor of the actions actually executed.

Arms
----
    carry_te    bimos_carry_te_ksweep     TE=0.01, n_action_steps=1   -> stride 1
    carry_note  bimos_carry_note_ksweep   no TE,   n_action_steps=10  -> stride 10

Both are carry + BiMamba with overlap off, so the pair isolates temporal ensembling
and the execution stride that comes with it. Nothing else differs.

Metrics
-------
Everything is computed on the executed action stream, per episode, then averaged.

  jerk_rms        RMS of the third finite difference, per control step. The plain
                  measure of how ragged the commanded trajectory is.
  boundary_jerk   jerk RMS restricted to samples within +-2 steps of a chunk
                  transition, i.e. where one chunk hands over to the next.
  interior_jerk   jerk RMS everywhere else.
  boundary_ratio  boundary / interior. This is the number that matters for a
                  chunk-continuity claim: >1 means the seams are rougher than the
                  inside of a chunk, which is exactly what carry is supposed to fix.
                  A method can have high overall jerk and still have ratio ~1.
  ldj             log dimensionless jerk, the standard scale-free smoothness score
                  (less negative = smoother). Comparable across episodes of
                  different length and speed, unlike raw jerk.

Transitions are at multiples of the execution stride: 1 for the TE arm, 10 for the
other. At stride 1 every step is a transition, so boundary/interior does not
partition anything -- those cells are reported as n/a rather than as a number that
looks meaningful.

LIBERO runs at 20 Hz (control dt = 0.05 s).
"""

import argparse
import csv
import math
import statistics as st
from pathlib import Path

import torch

ROOT = Path("/home1/eunji24/lerobot_project/outputs")
CONTROL_HZ = 20.0
BOUNDARY_RADIUS = 2

ARMS = {
    # label:            (output dir,                   execution stride)
    "carry_te": ("bimos_carry_te_ksweep", 1),
    "carry_note": ("bimos_carry_note_ksweep", 10),
}
KS = [10, 15, 20, 50, 100, 150]


def third_difference(a: torch.Tensor) -> torch.Tensor:
    """Third finite difference along time: jerk, up to the dt scaling."""
    return a[3:] - 3 * a[2:-1] + 3 * a[1:-2] - a[:-3]


def episode_metrics(actions: torch.Tensor, stride: int) -> dict | None:
    """actions: (T, D) executed action stream."""
    a = actions.to(torch.float64)
    T = a.shape[0]
    if T < 8:
        return None

    dt = 1.0 / CONTROL_HZ
    jerk = third_difference(a) / (dt**3)  # (T-3, D)
    per_step = jerk.norm(dim=-1)  # (T-3,)
    # Sample i of the third difference is centred at t = i + 1.5.
    centres = torch.arange(per_step.shape[0], dtype=torch.float64) + 1.5

    out = {"T": T, "jerk_rms": float(per_step.pow(2).mean().sqrt())}

    if stride > 1:
        transitions = torch.arange(stride, T, stride, dtype=torch.float64)
        if transitions.numel():
            dist = (centres[:, None] - transitions[None, :]).abs().min(dim=1).values
            near = dist <= BOUNDARY_RADIUS
            if near.any() and (~near).any():
                out["boundary_jerk"] = float(per_step[near].pow(2).mean().sqrt())
                out["interior_jerk"] = float(per_step[~near].pow(2).mean().sqrt())

    # Log dimensionless jerk on the action-space "speed" profile: scale-free, so
    # episodes of different length and amplitude can be averaged.
    speed = (a[1:] - a[:-1]).norm(dim=-1) / dt
    peak = float(speed.max())
    duration = (T - 1) * dt
    if peak > 0 and duration > 0:
        integral = float(per_step.pow(2).sum()) * dt
        out["ldj"] = -math.log(max(duration**3 / (peak**2) * integral, 1e-30))
    return out


def collect(arm_dir: str, k: int, step: int, stride: int) -> dict | None:
    d = ROOT / arm_dir / "libero_10" / f"K{k}" / "seed0" / "eval" / f"videos_step_{step}" / "action_logs"
    files = sorted(d.glob("episode_*.pt"))
    if not files:
        return None

    rows = []
    for f in files:
        try:
            a = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not torch.is_tensor(a) or a.ndim != 2:
            continue
        m = episode_metrics(a, stride)
        if m:
            rows.append(m)
    if not rows:
        return None

    def mean(key):
        v = [r[key] for r in rows if key in r]
        return st.mean(v) if v else None

    ratio = None
    b, i = mean("boundary_jerk"), mean("interior_jerk")
    if b is not None and i:
        ratio = b / i

    return {
        "episodes": len(rows),
        "steps": mean("T"),
        "jerk_rms": mean("jerk_rms"),
        "boundary_jerk": b,
        "interior_jerk": i,
        "boundary_ratio": ratio,
        "ldj": mean("ldj"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, default=150000, help="checkpoint (default 150000)")
    ap.add_argument("--csv", type=Path, help="also write the table to this CSV")
    args = ap.parse_args()

    print(f"\n=== executed-trajectory smoothness · LIBERO-10 · checkpoint {args.step} ===")
    print("    lower jerk = smoother;  ldj less negative = smoother;  ratio ~1 = seams")
    print("    no rougher than the interior of a chunk\n")

    head = (
        f"{'arm':<12}{'K':>5}{'stride':>7}{'eps':>5}{'jerk_rms':>11}"
        f"{'boundary':>11}{'interior':>11}{'ratio':>8}{'ldj':>9}"
    )
    print(head)
    print("-" * len(head))

    rows = []
    for label, (arm_dir, stride) in ARMS.items():
        for k in KS:
            r = collect(arm_dir, k, args.step, stride)
            if r is None:
                print(f"{label:<12}{k:>5}{stride:>7}{'--':>5}   (action_logs 없음)")
                continue
            f = lambda x, w, p=1: (f"{x:>{w}.{p}f}" if x is not None else f"{'n/a':>{w}}")  # noqa: E731
            print(
                f"{label:<12}{k:>5}{stride:>7}{r['episodes']:>5}"
                f"{f(r['jerk_rms'], 11)}{f(r['boundary_jerk'], 11)}{f(r['interior_jerk'], 11)}"
                f"{f(r['boundary_ratio'], 8, 3)}{f(r['ldj'], 9, 2)}"
            )
            rows.append({"arm": label, "K": k, "stride": stride, **r})
        print()

    print("  stride 1 (the TE arm) re-plans every step, so every sample sits on a")
    print("  transition and the boundary/interior split does not apply -- shown as n/a.")
    print("  Compare arms at equal K; comparing across K also changes how often the")
    print("  policy re-plans relative to the chunk, which is itself a smoothness factor.")

    if args.csv and rows:
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
