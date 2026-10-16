#!/usr/bin/env python
"""Aggregate BiMOS sweep results on LIBERO-10 into per-cell tables.

    python report_bimos.py            # chunk-size (K) sweep
    python report_bimos.py --stride   # execution-stride sweep
    python report_bimos.py --layers   # BiMamba decoder-depth sweep, one table per K
    python report_bimos.py --all      # all three
    python report_bimos.py --csv out.csv

Reads the per-episode `successes` lists out of each eval json and reports, for every
cell, the success rate at each checkpoint plus the mean and sd across checkpoints.

Why the mean across checkpoints rather than the last checkpoint: in the Aloha repeat
runs of one identical config the checkpoint means were 56.55 (120k), 64.55 (140k) and
60.20 (150k) -- an 8pp swing with nothing changed, and the final checkpoint was 4.4pp
below the 140k one. A single checkpoint carries sd ~2.7pp, which is the same order as
any plausible K or stride effect. `sd` below is that noise, measured per cell.

`binom` is the binomial standard error of the pooled rate, sqrt(p(1-p)/n) -- the floor
from episode sampling alone. It does NOT include checkpoint or run-to-run variation, so
`sd` is the number to compare cells against, not `binom`.
"""

import argparse
import json
import re
import statistics as st
from pathlib import Path

KSWEEP_ROOT = Path("/home1/eunji24/lerobot_project/outputs/bimos_ksweep/libero_10")
STRIDE_ROOT = Path("/home1/eunji24/lerobot_project/outputs/bimos_stride/libero_10")
LAYERS_ROOT = Path("/home1/eunji24/lerobot_project/outputs/bimos_layers/libero_10")
ABLATE_ROOT = Path("/home1/eunji24/lerobot_project/outputs/bimos_ablate/libero_10")
PLAIN_ROOT = Path("/home1/eunji24/lerobot_project/outputs/bimos_plain/libero_10")

# The K=20 carry x BiMamba 2x2. The (on, on) corner is the K sweep's own K020 cell rather
# than a fourth run -- sweep_K020.sh already trains it -- so it comes from KSWEEP_ROOT.
ABLATE_CELLS = {
    "K020":            "K020 carry+bimamba",   # under KSWEEP_ROOT
    "K020_nobimamba":  "K020 carry only",
    "K020_nocarry":    "K020 bimamba only",
    "K020_plain":      "K020 plain (neither)",
    # K=50. Its (carry ON, BiMamba ON) reference is not a training run here -- it is the
    # eval-only cell bimos_stride/.../K050/hard_h018 (70.0), which `--stride` already prints.
    "K050_nobimamba":  "K050 carry only",
}

# K-sweep cells that double as the L=1 baseline of the depth sweep. sweep_K020.sh and
# sweep_K100.sh already train exactly that config (n_decoder_layers=1, everything else
# identical to the matching sweep_K<k>_L<l>.sh), so no L=1 run is repeated under
# LAYERS_ROOT and the baseline has to be pulled from KSWEEP_ROOT instead.
LAYER_BASELINE_TAGS = ("K020", "K100")


def read_eval(path: Path) -> tuple[int, int] | None:
    """Return (successes, episodes) from an eval json, or None if it has no per-task data."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    per_task = data.get("per_task")
    if not per_task:
        return None
    ok = sum(sum(t["metrics"]["successes"]) for t in per_task)
    n = sum(len(t["metrics"]["successes"]) for t in per_task)
    return (ok, n) if n else None


def collect_ksweep() -> dict[str, dict[int, tuple[int, int]]]:
    """{cell: {step: (successes, episodes)}} for the K sweep (in-training evals)."""
    cells: dict[str, dict[int, tuple[int, int]]] = {}
    for f in sorted(KSWEEP_ROOT.glob("*/seed*/eval/eval_step_*.json")):
        m = re.search(r"eval_step_(\d+)\.json$", f.name)
        if not m:
            continue
        got = read_eval(f)
        if got:
            cells.setdefault(f.parents[2].name, {})[int(m.group(1))] = got
    return cells


def collect_stride() -> dict[str, dict[int, tuple[int, int]]]:
    """{"<TAG>/<cell>": {step: (successes, episodes)}} for the stride sweep (eval-only)."""
    cells: dict[str, dict[int, tuple[int, int]]] = {}
    for f in sorted(STRIDE_ROOT.glob("*/*/step*/eval_info.json")):
        m = re.match(r"step(\d+)$", f.parent.name)
        if not m:
            continue
        got = read_eval(f)
        if got:
            cells.setdefault(f"{f.parents[2].name}/{f.parents[1].name}", {})[int(m.group(1))] = got
    return cells


def collect_layers() -> dict[str, dict[str, dict[int, tuple[int, int]]]]:
    """{"K020": {"K020 L1": {step: (successes, episodes)}, ...}, ...} for the depth sweep.

    Grouped by K rather than flattened into one table: K020 and K100 sit ~53pp apart, so a
    single table's "highest mean" line would only ever re-state the K result. One table per
    K makes that line answer the question the sweep is actually asking -- which depth wins
    at this chunk size.

    L=1 rows come from KSWEEP_ROOT (see LAYER_BASELINE_TAGS), L>=2 from LAYERS_ROOT. Both
    layouts are in-training evals, so the same seed*/eval/eval_step_*.json glob works.
    """
    groups: dict[str, dict[str, dict[int, tuple[int, int]]]] = {}

    def add(k: str, layers: int, run_dir: Path) -> None:
        for f in sorted(run_dir.glob("seed*/eval/eval_step_*.json")):
            m = re.search(r"eval_step_(\d+)\.json$", f.name)
            if not m:
                continue
            got = read_eval(f)
            if got:
                cell = f"{k} L{layers}"
                groups.setdefault(k, {}).setdefault(cell, {})[int(m.group(1))] = got

    for tag in LAYER_BASELINE_TAGS:
        add(tag, 1, KSWEEP_ROOT / tag)

    for run_dir in sorted(LAYERS_ROOT.glob("*")):
        m = re.match(r"(K\d+)_L(\d+)$", run_dir.name)
        if m:
            add(m.group(1), int(m.group(2)), run_dir)

    return groups


def collect_plain() -> dict[str, dict[int, tuple[int, int]]]:
    """{"K0NN plain" / "K0NN BiMOS(h18)": {step: (successes, episodes)}} at execution stride 18.

    Two layouts in one table on purpose. The plain cells are training runs under PLAIN_ROOT
    whose in-training evals already sit at stride 18 (the scripts pin n_action_steps=18), so
    they carry three checkpoints. The BiMOS references are the eval-only hard_h018 cells under
    STRIDE_ROOT, which exist at 150k only -- hence the "-" in the earlier columns. Same
    checkpoint step, same stride, same seed, so the 150k column is the comparison.
    """
    cells: dict[str, dict[int, tuple[int, int]]] = {}

    for run_dir in sorted(PLAIN_ROOT.glob("K*_plain")):
        k = run_dir.name.split("_")[0]
        for f in sorted(run_dir.glob("seed*/eval/eval_step_*.json")):
            m = re.search(r"eval_step_(\d+)\.json$", f.name)
            if m and (got := read_eval(f)):
                cells.setdefault(f"{k} plain", {})[int(m.group(1))] = got

    for f in sorted(STRIDE_ROOT.glob("K*/hard_h018/step*/eval_info.json")):
        tag = f.parents[2].name
        if "_" in tag:  # K100_L4 etc. belong to the depth sweep, not the plain comparison
            continue
        m = re.match(r"step(\d+)$", f.parent.name)
        if m and (got := read_eval(f)):
            cells.setdefault(f"{tag} BiMOS(h18)", {})[int(m.group(1))] = got

    return cells


def collect_ablate() -> dict[str, dict[int, tuple[int, int]]]:
    """{"K020 <corner>": {step: (successes, episodes)}} for the K=20 carry x BiMamba 2x2.

    The (carry ON, BiMamba ON) corner lives under KSWEEP_ROOT (it is sweep_K020.sh's own
    run); the other three are under ABLATE_ROOT. Both are in-training eval layouts, so the
    same seed*/eval/eval_step_*.json glob works for either.
    """
    cells: dict[str, dict[int, tuple[int, int]]] = {}
    for tag, label in ABLATE_CELLS.items():
        run_dir = (KSWEEP_ROOT if tag == "K020" else ABLATE_ROOT) / tag
        for f in sorted(run_dir.glob("seed*/eval/eval_step_*.json")):
            m = re.search(r"eval_step_(\d+)\.json$", f.name)
            if not m:
                continue
            got = read_eval(f)
            if got:
                cells.setdefault(label, {})[int(m.group(1))] = got
    return cells


def table(title: str, cells: dict[str, dict[int, tuple[int, int]]], rows: list[list[str]]) -> None:
    print(f"\n=== {title} ===")
    if not cells:
        print("  (no results yet)")
        return

    steps = sorted({s for per_step in cells.values() for s in per_step})
    head = f"{'cell':<22}" + "".join(f"{s // 1000:>8}k" for s in steps)
    head += f"{'mean':>9}{'sd':>7}{'binom':>8}{'n':>7}"
    print(head)
    print("-" * len(head))

    for cell in sorted(cells):
        per_step = cells[cell]
        line = f"{cell:<22}"
        for s in steps:
            if s in per_step:
                ok, n = per_step[s]
                line += f"{100 * ok / n:>9.1f}"
            else:
                line += f"{'-':>9}"
        rates = [100 * ok / n for ok, n in per_step.values()]
        tot_ok = sum(ok for ok, _ in per_step.values())
        tot_n = sum(n for _, n in per_step.values())
        p = tot_ok / tot_n
        binom = 100 * (p * (1 - p) / tot_n) ** 0.5
        sd = st.stdev(rates) if len(rates) > 1 else float("nan")
        line += f"{st.mean(rates):>9.2f}{sd:>7.2f}{binom:>8.2f}{tot_n:>7}"
        print(line)
        rows.append([cell, f"{st.mean(rates):.2f}", f"{sd:.2f}", f"{binom:.2f}", str(tot_n)]
                    + [f"{100 * per_step[s][0] / per_step[s][1]:.1f}" if s in per_step else ""
                       for s in steps])

    best = max(cells, key=lambda c: st.mean([100 * ok / n for ok, n in cells[c].values()]))
    print(f"\n  highest mean: {best}")
    print("  Treat that as a screen, not a result: picking the max of several noisy cells is")
    print("  biased upward by roughly 1 sd. Re-evaluate the selected cell before quoting it.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stride", action="store_true", help="stride sweep instead of the K sweep")
    ap.add_argument("--layers", action="store_true", help="decoder-depth sweep instead of the K sweep")
    ap.add_argument("--plain", action="store_true", help="plain ACM2 vs BiMOS at stride 18, per K")
    ap.add_argument("--ablate", action="store_true", help="K=20 carry x BiMamba 2x2 instead of the K sweep")
    ap.add_argument("--all", action="store_true", help="all tables")
    ap.add_argument("--csv", type=Path, help="also write the rows to this CSV")
    args = ap.parse_args()

    rows: list[list[str]] = []
    if args.all or not (args.stride or args.layers or args.ablate or args.plain):
        table(f"chunk-size sweep  ({KSWEEP_ROOT})", collect_ksweep(), rows)
    if args.all or args.stride:
        table(f"execution-stride sweep  ({STRIDE_ROOT})", collect_stride(), rows)
    if args.all or args.layers:
        groups = collect_layers()
        if not groups:
            table(f"decoder-depth sweep  ({LAYERS_ROOT})", {}, rows)
        for k, cells in sorted(groups.items()):
            table(f"decoder-depth sweep, K={int(k[1:])}  ({LAYERS_ROOT})", cells, rows)
    if args.all or args.ablate:
        table(f"K=20 carry x BiMamba ablation  ({ABLATE_ROOT})", collect_ablate(), rows)
    if args.all or args.plain:
        table(f"plain ACM2 vs BiMOS at stride 18  ({PLAIN_ROOT})", collect_plain(), rows)

    if args.csv:
        import csv

        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cell", "mean", "sd", "binom_se", "episodes", "..."])
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
