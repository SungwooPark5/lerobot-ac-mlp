#!/usr/bin/env python
"""Aggregate the K=50 carry x BiMamba x stride grid.

    python report_stage1_2x2.py
    python report_stage1_2x2.py --csv stage1.csv

Reads outputs/stage1_2x2_K50/libero_10/<arm>/s<stride>/step<step>/eval_info.json
and prints, for every (arm, stride) cell, the success rate at each checkpoint plus
the mean and sd across them.

What to look for
----------------
The hypothesis is that BiMamba is invisible at short stride and pays off as the
stride approaches K, because at stride s only the first s actions of the chunk are
executed and a backward scan mostly improves the rest. That predicts the two
BiMamba rows separating from the other two on the right of the grid and not on the
left. The `bimamba - plain` and `carry - plain` deltas at the bottom are the
numbers that answer it.

Read the sd column before believing any of them. On this task checkpoint sd runs
0.6-7.9pp, re-evaluating one checkpoint moves ~1.8pp, and the binomial se at 1000
episodes is 1.6pp. A gap under ~5pp from a single seed is not a result.

Caveat carried over from training: all four arms trained with use_chunk_pairs=true,
which drops the last chunk_size frames of every episode, so each saw the same
188,815 of 273,465 frames. The square is internally fair. It is not comparable to
the ACT runs, which trained on all of them.
"""

import argparse
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path("/home1/eunji24/lerobot_project/outputs/stage1_2x2_K50/libero_10")

ARMS = [
    ("plain", "carry OFF  bimamba OFF"),
    ("bimamba", "carry OFF  bimamba ON"),
    ("carry", "carry ON   bimamba OFF"),
    ("carry_bimamba", "carry ON   bimamba ON"),
    # Baseline column, filled by stage1_act_baseline.sh. Rows stay blank until it runs.
    ("act", "ACT baseline"),
    ("act_te", "ACT + TE 0.01"),
]
STRIDES = [1, 10, 25, 50]


def read_eval(path: Path) -> tuple[int, int] | None:
    """Return (successes, episodes), or None if the file has no per-task data."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    per_task = data.get("per_task")
    if per_task:
        ok = sum(sum(t["metrics"]["successes"]) for t in per_task)
        n = sum(len(t["metrics"]["successes"]) for t in per_task)
        return (ok, n) if n else None

    # lerobot_eval's own format: aggregated pc_success over eval.n_episodes.
    agg = data.get("aggregated") or {}
    pc = agg.get("pc_success")
    n = agg.get("eval_n_episodes") or data.get("n_episodes")
    if pc is not None and n:
        return round(pc / 100 * n), int(n)
    return None


def collect() -> dict[tuple[str, int], dict[int, tuple[int, int]]]:
    cells: dict[tuple[str, int], dict[int, tuple[int, int]]] = {}
    for arm, _ in ARMS:
        for s in STRIDES:
            per_step: dict[int, tuple[int, int]] = {}
            for d in sorted((ROOT / arm / f"s{s}").glob("step*")):
                got = read_eval(d / "eval_info.json")
                if got:
                    per_step[int(d.name.removeprefix("step"))] = got
            cells[(arm, s)] = per_step
    return cells


def rate(per_step, step) -> float | None:
    got = per_step.get(step)
    return 100 * got[0] / got[1] if got else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, help="also write the table to this CSV")
    args = ap.parse_args()

    cells = collect()
    steps = sorted({s for per in cells.values() for s in per})
    if not steps:
        print(f"\n  (no evals yet under {ROOT})")
        return

    print(f"\n=== K=50 carry x BiMamba, by execution stride  ({ROOT}) ===")
    print("    all arms: 150k steps, use_chunk_pairs=true, same 188,815 training frames\n")

    head = "arm                       stride" + "".join(f"{s // 1000:>8}k" for s in steps)
    print(head + f"{'mean':>9}{'sd':>7}{'n':>7}")
    print("-" * len(head + f"{'mean':>9}{'sd':>7}{'n':>7}"))

    rows = []
    means: dict[tuple[str, int], float] = {}
    for arm, label in ARMS:
        for s in STRIDES:
            per = cells[(arm, s)]
            vals = [rate(per, step) for step in steps]
            present = [v for v in vals if v is not None]
            n_tot = sum(per[step][1] for step in per)
            cols = "".join(f"{v:>9.1f}" if v is not None else f"{'--':>9}" for v in vals)
            if present:
                m = sum(present) / len(present)
                means[(arm, s)] = m
                sd = st.stdev(present) if len(present) > 1 else float("nan")
                print(f"{label:<26}{s:>6}{cols}{m:>9.2f}{sd:>7.2f}{n_tot:>7}")
            else:
                print(f"{label:<26}{s:>6}{cols}{'--':>9}{'--':>7}{0:>7}")
            rows.append(
                {
                    "arm": arm,
                    "stride": s,
                    **{f"step_{step}": (f"{v:.1f}" if v is not None else "") for step, v in zip(steps, vals)},
                    "mean": f"{sum(present) / len(present):.2f}" if present else "",
                    "n": n_tot,
                }
            )
        print()

    # The comparison the grid exists for.
    print("deltas vs plain (pp), by stride:")
    print(f"    {'stride':>6}{'bimamba':>10}{'carry':>10}{'both':>10}")
    for s in STRIDES:
        base = means.get(("plain", s))
        if base is None:
            continue
        out = f"    {s:>6}"
        for arm in ("bimamba", "carry", "carry_bimamba"):
            m = means.get((arm, s))
            out += f"{m - base:>+10.1f}" if m is not None else f"{'--':>10}"
        print(out)

    # The comparison that decides the final model.
    if any(("act", s) in means for s in STRIDES):
        print("\ndeltas vs the ACT baseline (pp), by stride:")
        print(f"    {'stride':>6}{'plain':>10}{'bimamba':>10}{'carry':>10}{'both':>10}")
        for s in STRIDES:
            base = means.get(("act", s))
            if base is None:
                continue
            out = f"    {s:>6}"
            for arm in ("plain", "bimamba", "carry", "carry_bimamba"):
                m = means.get((arm, s))
                out += f"{m - base:>+10.1f}" if m is not None else f"{'--':>10}"
            print(out)
        print(
            "\n  ACT trained on all 273,465 frames; the BiMOS arms on 188,815, because\n"
            "  use_chunk_pairs drops the last chunk_size frames of every episode. ACT has\n"
            "  ~45% more data here, so a BiMOS win is conservative and an ACT win is\n"
            "  uninterpretable. Train one ACT K=50 with use_chunk_pairs=true to close it."
        )

    print(
        "\n  Checkpoint sd on this task is 0.6-7.9pp and re-evaluating one checkpoint\n"
        "  moves ~1.8pp, so treat anything under ~5pp as unresolved at one seed."
    )

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
