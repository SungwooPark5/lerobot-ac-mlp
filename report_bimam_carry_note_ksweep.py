#!/usr/bin/env python
"""Aggregate Carry + no-TE K-sweep results on LIBERO-10.

Usage:
    python report_bimam_carry_note_ksweep.py
    python report_bimam_carry_note_ksweep.py --csv carry_note_ksweep.csv

Training setup (BiMOS_noTE_K*.sh):
    sscp_enabled = true
    use_bimamba_decoder = true
    sscp_overlap = 0
    temporal_ensemble_coeff = unset (off)
    n_action_steps = 10          <- open-loop execution stride

This is the no-TE partner of report_bimam_carry_te_ksweep.py. The two differ on
BOTH axes at once: the TE arm ensembles at n_action_steps=1, this one runs
open-loop at 10. So the gap between the tables is TE+stride together, not TE
alone -- do not quote it as a temporal-ensemble ablation.

The eval budgets also differ: this arm evaluated 100 episodes/task per
checkpoint, the TE arm 50, so `binom` is smaller here for the same rate.

Cells: K10, K15, K20, K50, K100, K150 under
    /home1/eunji24/lerobot_project/outputs/bimos_carry_note_ksweep/libero_10

Everything below delegates to report_bimam_carry_te_ksweep, which owns the
aggregation and the arm table (ARMS["carry_note"]).
"""

import argparse
from pathlib import Path

import report_bimam_carry_te_ksweep as base

ARM = "carry_note"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="also write the table to this CSV",
    )
    args = parser.parse_args()

    base.ROOT, base.LABEL = base.ARMS[ARM]

    rows = base.print_table(base.collect())

    if args.csv:
        base.write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
