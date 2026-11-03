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


def pair_numbers(phonon: Path) -> list[tuple[int, int]]:
    """Second-order mode pairs (i, j), i <= j, in the order PAIR_MODE=all runs them: nearest in frequency first.

    Energy conservation lets a pair contribute to the two-phonon rate only when its two
    frequencies lie within about the broadening of each other (about 90 % of the rate comes from
    pairs closer than 2 sigma), so the pairs are run by increasing |w_i - w_j|: a tree stopped
    part-way has the pairs that matter most. The diagonal pairs (gap 0) come first, then the
    partners of degenerate doublets; equal gaps keep the index order (i, then j).
    """
    numbers = mode_numbers(phonon)
    with np.load(phonon, allow_pickle=True) as data:
        freqs = np.asarray(data["freqs"], dtype=float)
    if len(freqs) != len(numbers):
        raise ValueError(f"{len(freqs)} frequencies for {len(numbers)} modes in {phonon}")
    order = sorted(range(len(numbers)), key=lambda k: numbers[k])
    pairs = [(a, b) for x, a in enumerate(order) for b in order[x:]]  # i <= j by mode number
    pairs.sort(key=lambda ab: abs(freqs[ab[0]] - freqs[ab[1]]))  # stable: ties stay in (i, j) order
    return [(numbers[a], numbers[b]) for a, b in pairs]


def mode_table(phonon: Path) -> list[tuple[int, float, str, float, float]]:
    """(mode number, frequency in cm^-1, symmetry label, <r>, P_2bond) per mode.

    Labels are classified if the npz has none. <r> (Angstrom, from the defect centre) and P_2bond (the
    weight within two bond lengths of it) say how near the defect the mode lives
    (PhononSpectrum.mean_defect_distance, .defect_participation); NaN when no single defect centre is
    found.
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
    """``get_n_modes.py <phonon.npz> [--sym | --pairs]``: the mode numbers on one line, a table with --sym,
    or the second-order pairs ``<i>_<j>`` one per line, nearest in frequency first, with --pairs."""
    ap = argparse.ArgumentParser(prog="get_n_modes.py", description=__doc__.splitlines()[0])
    ap.add_argument("phonon", type=Path, help="phonon .npz the tree runs from")
    ap.add_argument(
        "--sym",
        action="store_true",
        help="print mode, frequency, symmetry and how near the defect it lives (<r> in Angstrom from the "
        "defect centre, and the fraction of its weight within two bond lengths of it) per line",
    )
    ap.add_argument(
        "--pairs",
        action="store_true",
        help="print the second-order pairs <i>_<j> (i <= j) one per line, sorted by their frequency gap "
        "|w_i - w_j|, the order PAIR_MODE=all runs them",
    )
    args = ap.parse_args(argv)
    if args.phonon.suffix != ".npz":
        ap.error(f"not an .npz file: {args.phonon}")
    if args.sym and args.pairs:
        ap.error("--sym and --pairs are exclusive")
    if args.pairs:
        try:
            pairs = pair_numbers(args.phonon)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("\n".join(f"{i}_{j}" for i, j in pairs))
        return 0
    if not args.sym:
        print(" ".join(str(n) for n in mode_numbers(args.phonon)))
        return 0
    try:
        table = mode_table(args.phonon)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("mode freq[cm-1] sym r_mean[A] P_2bond")
    for n, freq, label, mean_r, near in table:
        print(f"{n} {freq:.2f} {label} {mean_r:.2f} {near:.3f}")
    return 0
