"""The phonon modes a perturbation tree runs, numbered as its run folders.

``runs/<i>`` and ``runs/<i>_<j>`` are 1-based mode numbers in the full spectrum. A
filtered npz (from sym_analysis.py) keeps its 0-based positions in the full spectrum as
``idx``, so mode numbers are ``idx + 1``; an unfiltered npz runs every mode, 1..n.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def mode_numbers(phonon: Path) -> list[int]:
    """1-based mode numbers of the run folders, in the npz's order."""
    with np.load(phonon, allow_pickle=True) as data:
        if "idx" in data:
            return [int(i) + 1 for i in data["idx"]]
        return list(range(1, int(data["freqs"].shape[0]) + 1))


def mode_table(phonon: Path) -> list[tuple[int, float, str, float, float]]:
    """(mode number, frequency in cm^-1, symmetry label, <r>, P(2 shells)) per mode.

    Labels are classified if the npz has none. <r> (Angstrom) and P(2 shells) say how near the defect
    the mode lives (PhononSpectrum.mean_defect_distance, .defect_participation); NaN when the defect
    centre cannot be found.
    """
    from spinlabs.parsers import parse_phonon_npz
    from spinlabs.symmetry import classify_and_pair

    numbers = mode_numbers(phonon)
    with np.load(phonon, allow_pickle=True) as data:
        freqs = np.asarray(data["freqs"])
    spectrum = parse_phonon_npz(phonon)
    if spectrum.symmetries is None:
        classify_and_pair(spectrum)
    labels = list(spectrum.symmetries)
    if len(labels) != len(numbers):
        raise ValueError(
            f"{len(labels)} symmetry labels for {len(numbers)} modes; re-run sym_analysis.py to regenerate this npz"
        )
    try:
        mean_r, near = spectrum.mean_defect_distance, spectrum.defect_participation
    except (ValueError, TypeError) as exc:  # no or several defects found: the table still lists the modes
        print(f"note: defect localisation not computed ({exc})", file=sys.stderr)
        mean_r = near = np.full(len(numbers), np.nan)
    return [
        (n, float(f), str(label), float(r), float(p))
        for n, f, label, r, p in zip(numbers, freqs, labels, mean_r, near)
    ]


def main(argv: list[str] | None = None) -> int:
    """``get_n_modes.py <phonon.npz> [--sym]``: the mode numbers on one line, or a table with --sym."""
    ap = argparse.ArgumentParser(prog="get_n_modes.py", description=__doc__.splitlines()[0])
    ap.add_argument("phonon", type=Path, help="phonon .npz the tree runs from")
    ap.add_argument(
        "--sym",
        action="store_true",
        help="print mode, frequency, symmetry and how near the defect it lives (<r> in Angstrom and the "
        "fraction of its weight within the first two neighbour shells) per line",
    )
    args = ap.parse_args(argv)
    if args.phonon.suffix != ".npz":
        ap.error(f"not an .npz file: {args.phonon}")
    if not args.sym:
        print(" ".join(str(n) for n in mode_numbers(args.phonon)))
        return 0
    try:
        table = mode_table(args.phonon)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("mode freq[cm-1] sym r_mean[A] P_2shell")
    for n, freq, label, mean_r, near in table:
        print(f"{n} {freq:.2f} {label} {mean_r:.2f} {near:.3f}")
    return 0
