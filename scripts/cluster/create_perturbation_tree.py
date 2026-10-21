#!/usr/bin/env python3
"""create_perturbation_tree.py -- build ready-to-go perturbed ZFS run folders.

Thin CLI wrapper; the logic lives in spinlabs.vasp (see its docstrings).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spinlabs.vasp import build_perturbation_tree, default_phonon
from spinlabs.logging_utils import configure_logging


def main() -> int:
    configure_logging(verbosity=1)
    ap = argparse.ArgumentParser(
        description="Build ready-to-go perturbed ZFS run folders (first_order/second_order x basis x perturbation)"
    )
    ap.add_argument("output", nargs="?", type=Path, default=None, help="defect folder (defect name = its basename)")
    ap.add_argument(
        "--pert",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2],
        help="perturbation amplitudes Q (default: 0.05 0.1 0.2)",
    )
    ap.add_argument(
        "--phonon",
        type=Path,
        default=None,
        help="phonon data file (default: <out>/data/phonon_data.npz; "
        "the runs always use a symmetrised file in <out>/data)",
    )
    ap.add_argument("--vasp-binary", type=Path, default=None, help="VASP binary baked into the sbatch scripts")
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing perturbation folders instead of skipping them; requires typing APPROVE to confirm",
    )
    ap.add_argument(
        "--pair-mode",
        choices=("diag", "all"),
        default="diag",
        help="second-order pairs: diag = only (i, i) terms (the "
        "original behaviour), all = every pair with i <= j "
        "(default: diag)",
    )
    ap.add_argument(
        "--array-jobs", type=int, default=10, help="number of SLURM array jobs to split the sweep into (default: 10)"
    )
    ap.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="cap on the per-job time limit in hours; the array "
        "job count is raised until the limit fits (e.g. "
        "--max-hours 8)",
    )
    ap.add_argument(
        "--no-mass-weighted",
        dest="mass_weighted",
        action="store_false",
        help="displace by Q*e instead of Q*e/sqrt(m) (MASS_WEIGHTED=0 in the job scripts), to stay consistent "
        "with trees made that way, e.g. NV_64",
    )
    args = ap.parse_args()

    if args.force:
        try:
            answer = input("Overwrite existing perturbation folders? Type APPROVE to continue: ")
        except EOFError:
            answer = ""
        if answer.strip() != "APPROVE":
            sys.exit("Aborted: overwrite not confirmed")

    if args.output is None:
        sys.exit("Error: --output is required (the defect name is read from it)")
    out = args.output.resolve()
    phonon = args.phonon.resolve() if args.phonon else default_phonon(out)
    vasp_binary = args.vasp_binary.resolve() if args.vasp_binary else None
    try:
        failures = build_perturbation_tree(
            out,
            args.pert,
            phonon,
            pair_mode=args.pair_mode,
            array_jobs=args.array_jobs,
            max_hours=args.max_hours,
            vasp_binary=vasp_binary,
            force=args.force,
            mass_weighted=args.mass_weighted,
        )
    except FileNotFoundError as e:
        sys.exit(f"Error: {e}")

    if failures:
        print(f"\n{len(failures)} problem(s) found -- inspect before launching")
        return 1
    print("\nDone. All folders ready for sbatch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
