#!/usr/bin/env python3
"""add_convergence_tree.py -- add step-size convergence folders to an existing defect folder.

Creates <defect>/convergence_test_all/ and convergence_test_occup/ (input/ from ZFS_hyp and
ZFS_occup, and run_convergence.sh) next to the perturbation trees, as create_perturbation_tree.py
does for a new defect. The displacement defaults to the one the defect's trees use, so the
convergence runs match them, and the VASP binary to the one their job scripts name.

Thin CLI wrapper; the logic lives in spinlabs.vasp.trees (build_convergence_tree).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spinlabs.logging_utils import configure_logging
from spinlabs.vasp import default_phonon
from spinlabs.vasp.structures import add_displacement_args, displacement_from_args
from spinlabs.vasp.trees import (
    build_convergence_tree,
    convergence_steps,
    defect_displacement,
    defect_vasp_binary,
    resolve_sym_phonon,
)


def main() -> int:
    configure_logging(verbosity=1)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("output", type=Path, help="defect folder (holds ZFS_hyp/, ZFS_occup/ and its trees)")
    ap.add_argument("--modes", type=int, nargs="+", required=True, help="1-based modes to run, e.g. 13 62 231 378")
    steps = ap.add_mutually_exclusive_group(required=True)
    steps.add_argument("--steps", type=float, nargs="+", help="signed steps, e.g. -0.05 -0.025 0.025 0.05")
    steps.add_argument("--pert", type=float, nargs="+", help="tree step(s) p: runs +-p/2, +-p and +-2p")
    add_displacement_args(ap, default=None)
    ap.add_argument("--phonon", type=Path, default=None, help="phonon data file (default: <out>/data/phonon_data.npz)")
    ap.add_argument("--vasp-binary", type=Path, default=None, help="VASP binary (default: the existing trees' one)")
    ap.add_argument("--array-jobs", type=int, default=5, help="SLURM array jobs per convergence folder (default: 5)")
    ap.add_argument("--max-hours", type=float, default=None, help="cap on the per-job time limit in hours")
    ap.add_argument("--force", action="store_true", help="recreate existing convergence folders")
    args = ap.parse_args()

    out = args.output.resolve()
    try:
        if args.displacement or args.no_mass_weighted:
            displacement = displacement_from_args(args, ap)
        else:
            displacement = defect_displacement(out)
            if displacement is None:
                ap.error("no perturbation trees to take the displacement from; pass --displacement")
        print(f"displacement: {displacement}")
        if args.force:
            try:
                answer = input("Overwrite existing convergence folders? Type APPROVE to continue: ")
            except EOFError:
                answer = ""
            if answer.strip() != "APPROVE":
                sys.exit("Aborted: overwrite not confirmed")
        phonon = resolve_sym_phonon(out, args.phonon.resolve() if args.phonon else default_phonon(out))
        failures = build_convergence_tree(
            out,
            phonon,
            args.modes,
            args.steps if args.steps else convergence_steps(args.pert),
            displacement=displacement,
            array_jobs=args.array_jobs,
            max_hours=args.max_hours,
            vasp_binary=args.vasp_binary.resolve() if args.vasp_binary else defect_vasp_binary(out),
            force=args.force,
        )
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")
    if failures:
        print(f"\n{len(failures)} problem(s) found -- inspect before launching")
        return 1
    print("\nDone. Submit from each convergence folder with: sh run_convergence.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
