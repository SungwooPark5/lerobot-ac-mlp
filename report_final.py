#!/usr/bin/env python
"""Every LIBERO and Aloha cell in one place, with the invalid ones marked as invalid.

    python report_final.py
    python report_final.py --csv final.csv

Reads results from wherever they happen to live -- the training-time evals under each run's
eval/, the stride sweeps under stage1b/stage1c/stage1_2x2_K50, and the overnight gate runs
under overnight/eval/ -- and prints one table per chunk size plus the Aloha sweep.

Why some cells say INVALID
--------------------------
The carry-OFF arms in the original K sweeps (bimos_plain_note_ksweep,
bimos_nocarry_note_ksweep) trained with use_chunk_pairs=true. ChunkPairDataset restricts the
sampled index to [ep_start, ep_end - K) (chunk_pair_dataset.py:76-84). A carry arm loses
nothing to that, because the pair's second half is supervised at observation i+K; but
forward() discards the entire _n1 half when sscp_enabled is false
(modeling_acm2_sscp_literal.py:710-713), so those arms were never asked to act from an
observation in the last K frames of an episode.

LIBERO-10 episodes average 162 frames, so the fraction of query positions those arms ever
saw is (L-K)/L:

    K=10  93.8%    K=50  69.0%    K=100  38.8%    K=150  18.6%

which is why plain reads 2.6 at K=100 and 0.0 at K=150 while carry reads 61.4 and 36.6. That
gap is a training-coverage artifact, not evidence about carry. K<=20 is unaffected and those
cells are reported as-is; K>=50 is replaced by the nopairs/ retrains where they exist.

Comparisons against ACT were never affected -- ACT trained with use_chunk_pairs=false.
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path

ROOT = Path("/home1/eunji24/lerobot_project/outputs")

# Fraction of episode start positions the pairs-trained carry-OFF arms ever saw.
COVERAGE = {10: 93.8, 15: 90.7, 20: 87.6, 50: 69.0, 100: 38.8, 150: 18.6}
# Below this the arm is reported, above it the cell is INVALID unless a retrain exists.
COVERAGE_OK = 85.0


def _rate(path: Path) -> float | None:
    """Success rate in percent from an eval_info.json, whichever schema it uses."""
    for f in (path / "eval_info.json", path / "eval" / "eval_info.json"):
        if not f.exists():
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if pt := d.get("per_task"):
            ok = sum(sum(t["metrics"]["successes"]) for t in pt)
            n = sum(len(t["metrics"]["successes"]) for t in pt)
            if n:
                return 100 * ok / n
        for block in ("overall", "aggregated"):
            o = d.get(block) or {}
            for k in ("pc_success", "success_rate"):
                if k in o:
                    v = o[k]
                    return v * 100 if v <= 1 else v
    return None


def training_eval(run: str, K: int) -> float | None:
    """Mean over the checkpoints a training run evaluated itself at."""
    vals = []
    for f in sorted(glob.glob(str(ROOT / run / "libero_10" / f"K{K}" / "seed0" / "eval" / "eval_step_*.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        if pt := d.get("per_task"):
            ok = sum(sum(t["metrics"]["successes"]) for t in pt)
            n = sum(len(t["metrics"]["successes"]) for t in pt)
            if n:
                vals.append(100 * ok / n)
    return sum(vals) / len(vals) if vals else None


def sweep(base: str, arm: str, stride: int, steps=(100000, 125000, 150000)) -> float | None:
    vals = [v for v in (_rate(ROOT / base / "libero_10" / arm / f"s{stride}" / f"step{s}") for s in steps) if v is not None]
    return sum(vals) / len(vals) if vals else None


def overnight(label: str, stride: int, step: int = 100000) -> float | None:
    return _rate(ROOT / "overnight" / "eval" / label / f"s{stride}" / f"step{step}")


# arm label -> (original ksweep dir, stride-sweep dir name, retrained-arm prefix or None)
ARMS = [
    ("plain",         "bimos_plain_note_ksweep",   "plain",         "plain"),
    ("BiMamba only",  "bimos_nocarry_note_ksweep", "bimamba",       "bimamba"),
    ("carry+BiMamba", "bimos_carry_note_ksweep",   "carry_bimamba", None),
    ("ACT",           "act_ksweep",                "act",           None),
    ("ACT+TE",        "act_te_ksweep",             None,            None),
]

# K -> (stride-sweep root, strides shown)
GRIDS = {
    50:  ("stage1_2x2_K50",  [1, 10, 25, 50]),
    100: ("stage1b_K100",    [10, 25, 50, 75, 100]),
    150: ("stage1c_K150",    [10, 50, 100, 150]),
}


def cell(label: str, ksdir: str, swname: str | None, retrain: str | None, K: int, stride: int):
    """(value, note). note is 'retrain' / 'invalid' / '' so the caller can mark the table."""
    contaminated = retrain is not None and COVERAGE.get(K, 100) < COVERAGE_OK

    if retrain is not None:
        v = overnight(f"{retrain}_K{K}", stride)
        if v is not None:
            return v, "retrain"

    v = None
    # gate5 / extra / claim write carry and ACT cells under overnight/eval too, and those are
    # the newest measurements -- prefer them over the older stride-sweep directories.
    for lbl, step in ((f"carry_bimamba_K{K}", 150000), (f"carry_K{K}", 150000), (f"act_K{K}", 150000)):
        if label == "carry+BiMamba" and not lbl.startswith("carry"):
            continue
        if label == "ACT" and not lbl.startswith("act"):
            continue
        if label not in ("carry+BiMamba", "ACT"):
            break
        v = overnight(lbl, stride, step)
        if v is not None:
            break
    if v is None and swname:
        v = sweep(GRIDS[K][0], swname, stride)
    if v is None and stride == 10 and label != "ACT+TE":
        v = training_eval(ksdir, K)
    if v is None and stride == 1 and label == "ACT+TE":
        v = training_eval(ksdir, K)

    if v is not None and contaminated:
        return v, "invalid"
    return v, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    rows = []
    for K, (_, strides) in GRIDS.items():
        cov = COVERAGE.get(K)
        print(f"\n{'=' * (22 + 9 * len(strides))}")
        print(f"LIBERO-10  ·  chunk {K}    (pairs 학습 arm의 시작위치 커버리지 {cov}%)")
        print("=" * (22 + 9 * len(strides)))
        print(f"{'arm':<22}" + "".join(f"{'s' + str(s):>9}" for s in strides))
        print("-" * (22 + 9 * len(strides)))
        for label, ksdir, swname, retrain in ARMS:
            line = f"{label:<22}"
            for s in strides:
                v, note = cell(label, ksdir, swname, retrain, K, s)
                if v is None:
                    line += f"{'·':>9}"
                elif note == "invalid":
                    line += f"{'INVALID':>9}"
                else:
                    star = "*" if note == "retrain" else " "
                    line += f"{v:>8.1f}{star}"
                rows.append({"benchmark": "libero_10", "K": K, "stride": s, "arm": label,
                             "value": v, "note": note})
            print(line)
    print("\n  * = use_chunk_pairs=false 로 재학습한 모델 (100k step)")
    print("  INVALID = pairs 학습 + 커버리지 부족.  재학습 결과가 나오면 자동으로 대체됩니다.")
    print("  · = 미측정")

    # ── Aloha ──────────────────────────────────────────────────────────────────
    A = [("ACT", "act"), ("unidirectional", "A_unidir"),
         ("BiMamba only", "A_bimamba"), ("carry+BiMamba", "A_carry_bimamba")]
    strides = [1, 10, 25, 50, 75, 100]
    print(f"\n{'=' * (22 + 9 * len(strides))}")
    print("Aloha AlohaTransferCube  ·  chunk 100  ·  100k ckpt  ·  n=200  ·  seed 100000")
    print("=" * (22 + 9 * len(strides)))
    print(f"{'arm':<22}" + "".join(f"{'s' + str(s):>9}" for s in strides))
    print("-" * (22 + 9 * len(strides)))
    for label, d in A:
        line = f"{label:<22}"
        for s in strides:
            v = _rate(ROOT / "aloha_stride_bimamba" / d / f"s{s}" / "step100000")
            line += f"{v:>9.1f}" if v is not None else f"{'·':>9}"
            rows.append({"benchmark": "aloha", "K": 100, "stride": s, "arm": label,
                         "value": v, "note": ""})
        print(line)
    print("\n  Aloha 는 pair 문제 없음: carry-OFF arm 들이 ACM2 계열이라 use_chunk_pairs 를 쓰지 않습니다.")

    if args.csv:
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(r for r in rows if r["value"] is not None)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
