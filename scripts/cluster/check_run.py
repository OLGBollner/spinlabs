#!/usr/bin/env python3
"""check_run.py -- sanity-check finished perturbation runs.

For each run folder (runs/<i> or runs/<i>_<j>; the modes come from the name): did VASP
finish, is the OUTCAR clean, does VASP's symmetry handling fit the modes (at most 6 ops
for A1/A2, 2 for E), and does the ZFS change have the shape the modes' symmetry allows
(A1/A2: only the A1 part; E: mainly the E part). The ground-state ZFS comes from the
tree's ZFS_hyp/ZFS_occup OUTCAR unless --reference is given.

    python check_run.py runs/*/ --phonons ../../../data/phonon_data.npz [--warn-only]

The job runner calls it on each finished run with --write-check: a suspicious run gets
a CHECK file in its folder. Exit 1 when any run is suspicious.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spinlabs.parsers import parse_phonon_data
from spinlabs.vasp.run_checks import check_run
from spinlabs.vasp.structures import run_modes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", type=Path, nargs="+", help="run folder(s), e.g. runs/*/")
    ap.add_argument("--phonons", type=Path, required=True, help="phonon file the runs were made from")
    ap.add_argument("--reference", type=Path, help="ground-state OUTCAR (default: the tree's ZFS_hyp/ZFS_occup)")
    ap.add_argument("--write-check", action="store_true", help="write a CHECK file into suspicious run folders")
    ap.add_argument("--warn-only", action="store_true", help="print only suspicious runs")
    args = ap.parse_args()

    spectrum = parse_phonon_data(args.phonons)
    n_bad = 0
    for run in args.runs:
        modes = run_modes(run)
        if modes is None:
            print(f"{run}: not a runs/<i> or runs/<i>_<j> folder, skipped")
            continue
        try:
            check = check_run(run, spectrum, modes, args.reference)
        except ValueError as error:  # e.g. a mode outside the phonon file
            print(f"{run}: cannot check ({error})")
            n_bad += 1
            continue
        if check.status == "suspicious":
            n_bad += 1
            if args.write_check:
                (run / "CHECK").write_text(check.report() + "\n")
        if check.status == "suspicious" or not args.warn_only:
            print(f"{run}: " + check.report())
    print(f"checked {len(args.runs)} run(s): {n_bad} suspicious")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
