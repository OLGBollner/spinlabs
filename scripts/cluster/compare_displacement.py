#!/usr/bin/env python3
"""compare_displacement.py -- how large is a perturbed POSCAR, really?

Thin CLI wrapper; the logic lives in spinlabs.vasp (see its docstrings).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spinlabs.parsers import parse_phonon_data
from spinlabs.vasp import compare_displacement_cli
from spinlabs.vasp.structures import add_displacement_args, displacement_from_args
from spinlabs.vasp.trees import tree_displacement


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Report max/rms Cartesian displacement of one or more "
        "perturbed structures relative to the relaxed reference. Each path may be a "
        "POSCAR/CONTCAR, an OUTCAR, or a run directory. An OUTCAR (also the one in a run directory) gives the "
        "geometry VASP actually computed, which POSCAR/CONTCAR may not be after a rerun"
    )
    ap.add_argument("reference", type=Path, help="relaxed reference (e.g. ZFS_hyp/POSCAR, ZFS_hyp/ or ZFS_hyp/OUTCAR)")
    ap.add_argument("perturbed", type=Path, nargs="+", help="perturbed structure(s) to compare, e.g. runs/38_57/OUTCAR")
    ap.add_argument(
        "--phonons",
        type=Path,
        metavar="FILE",
        help="phonon_data.npz/phonopy.yaml the runs were generated from; adds the expected displacement "
        "of each runs/<i> or runs/<i>_<j> folder and the fitted actual/expected scale",
    )
    ap.add_argument(
        "--pert",
        type=float,
        metavar="STEP",
        help="step for the expected values (default: from the pert_<x> folder or <i>_pert_<x> run folder)",
    )
    add_displacement_args(ap, default=None)
    args = ap.parse_args()
    if args.displacement or args.no_mass_weighted:
        displacement = displacement_from_args(args, ap)
    else:  # the convention of the first run's tree (its job script), else mass_weighted
        tree = args.perturbed[0].resolve()
        tree = tree.parent.parent if tree.is_dir() else tree.parent.parent.parent
        displacement = tree_displacement(tree) or "mass_weighted"

    if not args.reference.exists():
        sys.exit(f"Error: reference not found: {args.reference}")
    missing = [p for p in args.perturbed if not p.exists()]
    if missing:
        sys.exit(f"Error: not found: {', '.join(map(str, missing))}")
    if args.phonons is not None and not args.phonons.is_file():
        sys.exit(f"Error: phonon file not found: {args.phonons}")

    compare_displacement_cli(
        args.reference,
        args.perturbed,
        spectrum=parse_phonon_data(args.phonons) if args.phonons else None,
        amplitude=args.pert,
        displacement=displacement,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
