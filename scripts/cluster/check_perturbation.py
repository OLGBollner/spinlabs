#!/usr/bin/env python3
"""check_perturbation.py -- is a perturbed POSCAR displaced as intended?

Compares POSCAR with the reference structure and the displacement the generator applies
(Q * e / sqrt(m) along each mode; --no-mass-weighted for the old Q * e runs). Prints one
line with max/rms displacement and the fitted actual/expected scale; exit 1 when the
scale or the residual is off by more than 1%. The job runner calls it right after
perturbing, before VASP starts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spinlabs.parsers import parse_phonon_data
from spinlabs.vasp.run_checks import check_perturbation


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("poscar", type=Path, help="perturbed POSCAR")
    ap.add_argument("reference", type=Path, help="unperturbed POSCAR (the tree's input/POSCAR)")
    ap.add_argument("phonons", type=Path, help="phonon file the structure was perturbed with")
    ap.add_argument("modes", type=int, nargs="+", help="1-based mode number(s)")
    ap.add_argument("--pert", type=float, required=True, help="amplitude Q in Angstrom*sqrt(amu)")
    ap.add_argument("--no-mass-weighted", dest="mass_weighted", action="store_false")
    args = ap.parse_args()
    try:
        ok, line = check_perturbation(
            args.poscar, args.reference, parse_phonon_data(args.phonons), tuple(args.modes), args.pert,
            mass_weighted=args.mass_weighted,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"perturbation check failed: {error}")
        return 1
    print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
