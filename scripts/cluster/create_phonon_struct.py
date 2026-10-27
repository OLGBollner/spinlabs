#!/usr/bin/env python3
"""create_phonon_struct.py -- perturb a POSCAR along phonon normal modes.

Thin CLI wrapper; the logic lives in spinlabs.vasp (apply_perturbation).
"""

from __future__ import annotations

import argparse
import sys

from spinlabs.logging_utils import configure_logging
from spinlabs.vasp import apply_perturbation, load_phonon_data, load_poscar, write_perturbed_poscar
from spinlabs.vasp.structures import add_displacement_args, displacement_from_args


def parse_index(s: str) -> list[int]:
    if "," in s:
        return [int(x) for x in s.split(",")]
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s)]


def main() -> int:
    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("poscar_file")
    ap.add_argument("phonon_data")
    ap.add_argument("mode_indices", type=parse_index)
    ap.add_argument(
        "amplitude",
        type=float,
        help="step: a Cartesian length in Angstrom (normalized) or Q in Angstrom*sqrt(amu); may be negative",
    )
    ap.add_argument("-o", "--output", default=None)
    add_displacement_args(ap)
    args = ap.parse_args()
    displacement = displacement_from_args(args, ap)

    structure = load_poscar(args.poscar_file)
    data = load_phonon_data(args.phonon_data)

    for idx in args.mode_indices:
        perturbed = apply_perturbation(
            structure,
            data.eigenvectors,
            data.atomic_masses,
            idx,
            args.amplitude,
            original_indices=data.original_indices,
            displacement=displacement,
        )
        out = args.output if args.output else f"POSCAR_pert_{args.amplitude}_mode_{idx}"
        write_perturbed_poscar(
            perturbed, out, f"Mode {idx}, step {args.amplitude} ({displacement})", template_poscar=args.poscar_file
        )
        print(f"Saved: {out}")
    print("\nSuccess!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
