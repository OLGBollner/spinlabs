#!/usr/bin/env python3
"""create_combined_phonon_struct.py -- two-mode combined perturbed POSCAR.

Thin CLI wrapper; the logic lives in spinlabs.vasp (apply_combined_perturbation).
"""

from __future__ import annotations

import argparse
import sys

from spinlabs.vasp import apply_combined_perturbation, load_phonon_data, load_poscar, write_perturbed_poscar
from spinlabs.vasp.structures import add_displacement_args, displacement_from_args


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("poscar_file", help="Path to VASP POSCAR file")
    ap.add_argument("phonon_file", help="Path to phonon data file (.npz or phonopy .yaml)")
    ap.add_argument("mode_i", type=int, help="First phonon mode index (1-based)")
    ap.add_argument("mode_j", type=int, help="Second phonon mode index (1-based)")
    ap.add_argument(
        "amplitude",
        type=float,
        help="step per mode: a Cartesian length in Angstrom (normalized) or Q in Angstrom*sqrt(amu)",
    )
    ap.add_argument("-o", "--output", default=None, help="Output filename")
    add_displacement_args(ap)
    args = ap.parse_args()
    displacement = displacement_from_args(args, ap)

    try:
        structure = load_poscar(args.poscar_file)
        phonon_data = load_phonon_data(args.phonon_file)
        perturbed = apply_combined_perturbation(
            structure,
            phonon_data.eigenvectors,
            args.mode_i,
            args.mode_j,
            phonon_data.atomic_masses,
            args.amplitude,
            original_indices=phonon_data.original_indices,
            displacement=displacement,
        )
        output_file = args.output or f"POSCAR_combined_{args.mode_i}_{args.mode_j}_amp_{args.amplitude}"
        write_perturbed_poscar(
            perturbed,
            output_file,
            f"Combined modes {args.mode_i}+{args.mode_j}, step {args.amplitude} ({displacement})",
            template_poscar=args.poscar_file,
        )
        print("\nSuccess!")
    except Exception as e:
        print(f"Error: {e!s}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
