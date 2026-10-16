#!/usr/bin/env python
"""Aggregate Carry + Temporal Ensemble K-sweep results on LIBERO-10.

Usage:
    python report_bimam_carry_te_ksweep.py                      # carry + TE (default)
    python report_bimam_carry_te_ksweep.py --arm carry_note     # carry, no TE
    python report_bimam_carry_te_ksweep.py --arm nocarry_note   # no carry, no TE
    python report_bimam_carry_te_ksweep.py --arm plain_note     # neither, no TE
    python report_bimam_carry_te_ksweep.py --csv carry_te_ksweep.csv

Expected training setup (see ARMS for what varies per arm):
    sscp_overlap = 0
    sscp_enabled / use_bimamba_decoder / temporal_ensemble_coeff per arm

Expected cells:
    K10, K15, K20, K50, K100, K150

For each K, this script reads the per-episode `successes` lists from:
    <ROOT>/<TAG>/seed*/eval/eval_step_*.json

and reports:
    - success rate at each checkpoint
    - mean across checkpoints
    - standard deviation across checkpoints
    - best checkpoint success rate
    - best checkpoint step
    - pooled binomial standard error
    - total evaluated episodes

The checkpoint mean/sd are useful because performance can fluctuate noticeably
between checkpoints even for one fixed training configuration.
"""

import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path


OUTPUTS = Path("/home1/eunji24/lerobot_project/outputs")

# The K-sweep arms that share this layout (<ROOT>/K*/seed*/eval/eval_step_*.json).
# All of them use sscp_overlap=0, so the only execution-side difference is the
# TE/stride pair in the label -- keep that visible, the header used to hardcode
# "TE coeff=0.01" and silently mislabelled every non-TE arm.
ARMS = {
    "carry_te": (
        OUTPUTS / "bimos_carry_te_ksweep/libero_10",
        "Carry=ON | BiMamba=ON | Overlap=OFF | TE coeff=0.01 | n_action_steps=1",
    ),
    "carry_note": (
        OUTPUTS / "bimos_carry_note_ksweep/libero_10",
        "Carry=ON | BiMamba=ON | Overlap=OFF | TE=off | n_action_steps=10",
    ),
    "nocarry_note": (
        OUTPUTS / "bimos_nocarry_note_ksweep/libero_10",
        "Carry=OFF | BiMamba=ON | Overlap=OFF | TE=off | n_action_steps=10",
    ),
    "plain_note": (
        OUTPUTS / "bimos_plain_note_ksweep/libero_10",
        "Carry=OFF | BiMamba=OFF | Overlap=OFF | TE=off | n_action_steps=10",
    ),
}

ROOT, LABEL = ARMS["carry_te"]

EXPECTED_TAGS = ("K10", "K15", "K20", "K50", "K100", "K150")


def read_eval(path: Path) -> tuple[int, int] | None:
    """Return (successes, episodes) from an eval json."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    per_task = data.get("per_task")
    if not per_task:
        return None

    ok = 0
    n = 0

    for task in per_task:
        metrics = task.get("metrics", {})
        successes = metrics.get("successes")
        if successes is None:
            continue

        ok += sum(successes)
        n += len(successes)

    return (ok, n) if n else None


def k_sort_key(tag: str) -> int:
    """Sort K10, K15, ..., K150 numerically."""
    m = re.fullmatch(r"K(\d+)", tag)
    return int(m.group(1)) if m else 10**9


def collect() -> dict[str, dict[int, tuple[int, int]]]:
    """Return {K-tag: {step: (successes, episodes)}}."""
    cells: dict[str, dict[int, tuple[int, int]]] = {}

    # Keep expected cells visible even if some have not finished yet.
    for tag in EXPECTED_TAGS:
        cells[tag] = {}

    for f in sorted(ROOT.glob("K*/seed*/eval/eval_step_*.json")):
        tag = f.parents[2].name

        # Only collect the intended K sweep cells.
        if not re.fullmatch(r"K\d+", tag):
            continue

        m = re.search(r"eval_step_(\d+)\.json$", f.name)
        if not m:
            continue

        got = read_eval(f)
        if got:
            cells.setdefault(tag, {})[int(m.group(1))] = got

    return cells


def print_table(
    cells: dict[str, dict[int, tuple[int, int]]]
) -> list[dict[str, str]]:
    """Print table and return rows for optional CSV export."""
    print(f"\n=== chunk-size sweep  ({ROOT}) ===")
    print(f"    {LABEL}")

    available = {k: v for k, v in cells.items() if v}
    if not available:
        print("\n  (no results yet)")
        return []

    steps = sorted(
        {step for per_step in available.values() for step in per_step}
    )

    head = f"{'cell':<10}"
    head += "".join(f"{step // 1000:>9}k" for step in steps)
    head += (
        f"{'mean':>10}"
        f"{'sd':>8}"
        f"{'best':>9}"
        f"{'best@':>9}"
        f"{'binom':>9}"
        f"{'n':>8}"
    )

    print()
    print(head)
    print("-" * len(head))

    csv_rows: list[dict[str, str]] = []

    for cell in sorted(cells, key=k_sort_key):
        per_step = cells[cell]

        if not per_step:
            line = f"{cell:<10}"
            line += "".join(f"{'-':>10}" for _ in steps)
            line += (
                f"{'-':>10}"
                f"{'-':>8}"
                f"{'-':>9}"
                f"{'-':>9}"
                f"{'-':>9}"
                f"{'-':>8}"
            )
            print(line)
            continue

        rates_by_step = {
            step: 100 * ok / n
            for step, (ok, n) in per_step.items()
        }

        rates = list(rates_by_step.values())

        tot_ok = sum(ok for ok, _ in per_step.values())
        tot_n = sum(n for _, n in per_step.values())

        mean = st.mean(rates)
        sd = st.stdev(rates) if len(rates) > 1 else float("nan")

        best_step = max(rates_by_step, key=rates_by_step.get)
        best_rate = rates_by_step[best_step]

        p = tot_ok / tot_n
        binom = 100 * (p * (1 - p) / tot_n) ** 0.5

        line = f"{cell:<10}"

        for step in steps:
            if step in per_step:
                line += f"{rates_by_step[step]:>10.1f}"
            else:
                line += f"{'-':>10}"

        sd_text = f"{sd:.2f}" if len(rates) > 1 else "-"

        line += (
            f"{mean:>10.2f}"
            f"{sd_text:>8}"
            f"{best_rate:>9.1f}"
            f"{best_step // 1000:>8}k"
            f"{binom:>9.2f}"
            f"{tot_n:>8}"
        )
        print(line)

        row = {
            "cell": cell,
            "K": str(k_sort_key(cell)),
            "mean": f"{mean:.2f}",
            "sd": f"{sd:.2f}" if len(rates) > 1 else "",
            "best": f"{best_rate:.2f}",
            "best_step": str(best_step),
            "binom_se": f"{binom:.2f}",
            "episodes": str(tot_n),
        }

        for step in steps:
            row[f"step_{step}"] = (
                f"{rates_by_step[step]:.2f}"
                if step in rates_by_step
                else ""
            )

        csv_rows.append(row)

    best_mean_cell = max(
        available,
        key=lambda c: st.mean(
            [100 * ok / n for ok, n in available[c].values()]
        ),
    )

    best_single_cell = max(
        available,
        key=lambda c: max(
            100 * ok / n for ok, n in available[c].values()
        ),
    )

    best_single_step = max(
        available[best_single_cell],
        key=lambda step: (
            100
            * available[best_single_cell][step][0]
            / available[best_single_cell][step][1]
        ),
    )

    best_single_rate = (
        100
        * available[best_single_cell][best_single_step][0]
        / available[best_single_cell][best_single_step][1]
    )

    print(f"\n  highest checkpoint-mean: {best_mean_cell}")
    print(
        f"  highest single checkpoint: "
        f"{best_single_cell} @ {best_single_step // 1000}k "
        f"= {best_single_rate:.1f}%"
    )
    print(
        "  Use the checkpoint mean/sd for K-sweep interpretation; "
        "the single best checkpoint is shown only as a reference."
    )

    return csv_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        print("\nCSV not written because no results were found.")
        return

    checkpoint_cols = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("step_")
        },
        key=lambda x: int(x.split("_")[1]),
    )

    fieldnames = [
        "cell",
        "K",
        *checkpoint_cols,
        "mean",
        "sd",
        "best",
        "best_step",
        "binom_se",
        "episodes",
    ]

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--arm",
        choices=sorted(ARMS),
        default="carry_te",
        help="which K-sweep arm to report (default: carry_te)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="also write the table to this CSV",
    )
    args = parser.parse_args()

    global ROOT, LABEL
    ROOT, LABEL = ARMS[args.arm]

    cells = collect()
    rows = print_table(cells)

    if args.csv:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()