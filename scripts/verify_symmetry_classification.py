"""Verify general symmetry classification against previously labeled npz files.

Loads one or more phonon npz files (raw phonon_data.npz or previously
sym-labeled phonon_data_sym_n*.npz), runs the general classify_modes, and
compares labels against any stored reference labels.

Usage:
    python scripts/verify_symmetry_classification.py <npz file> [...]
    python scripts/verify_symmetry_classification.py   # scans repo for candidates
    python scripts/verify_symmetry_classification.py --legacy <npz> [...]  # legacy analyzer
"""

import argparse
import glob
import os
import sys
import collections
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spinlabs.parsers import parse_phonon_npz  # noqa: E402
from spinlabs.symmetry import classify_modes  # noqa: E402


def find_reference(data) -> list[str] | None:
    """Best available reference labels stored in the file."""
    for key in ("sym", "symmetries"):
        if key in data:
            return [str(s) for s in data[key]]
    return None


def report(path: str, labels: list[str | None], ref: list[str] | None) -> int:
    n = len(labels)
    print(f"\n=== {path} ({n} modes) ===")
    counts = collections.Counter(labels)
    print("  classified:", dict(counts))
    if ref is None:
        print("  no stored reference labels; classification only")
        return 0
    if len(ref) != n:
        print(f"  WARNING: reference length {len(ref)} != {n}; skipping comparison")
        return 1
    match = sum(got == want for got, want in zip(labels, ref))
    print(f"  reference match: {match}/{n} ({100.0 * match / n:.2f}%)")
    mismatches = [(i, got, want) for i, (got, want) in enumerate(zip(labels, ref)) if got != want]
    if mismatches:
        print(f"  mismatches ({len(mismatches)}):")
        for i, got, want in mismatches[:20]:
            print(f"    mode {i}: classified {got!r}, stored {want!r}")
        if len(mismatches) > 20:
            print(f"    ... and {len(mismatches) - 20} more")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="npz files to check; empty scans repo")
    ap.add_argument(
        "--legacy",
        action="store_true",
        help="use the legacy PhononSpectrum.analyze_c3v_symmetry instead of the general classify_modes",
    )
    args = ap.parse_args()
    files = args.files or sorted(glob.glob("*/phonon_data.npz") + glob.glob("*/phonon_data_sym_*.npz"))
    rc = 0
    for path in files:
        try:
            spec = parse_phonon_npz(path)
        except Exception as exc:
            print(f"\n=== {path} ===\n  load error: {exc}")
            rc = 1
            continue
        if args.legacy:
            # Reset stored labels first: analyze_c3v_symmetry early-returns
            # existing symmetries, which would compare the file against itself.
            spec.symmetries = None
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                labels = list(spec.analyze_c3v_symmetry())
        else:
            labels, groups = classify_modes(spec)
        data = np.load(path, allow_pickle=True)
        ref = find_reference(data)
        rc |= report(path, labels, ref)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
