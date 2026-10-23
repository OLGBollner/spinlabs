"""Determine symmetry of phonon modes and save the filtered spectrum.

0-based idx convention
----------------------
Mode indices are 0-based positions in the *full* spectrum. The legacy npz
`idx` arrays are 0-based, while VASP folder numbering (from `phonons.yaml`)
and the `get_n_modes.py` printout (`i+1`) are 1-based. Downstream code that
joins on `idx` must not mix the two conventions.

Enforced here so the convention cannot silently regress if
`spinlabs.symmetry.filter_degenerate_partners` changes:

1. The filter must return an `original_indices` array whose values
   are exactly the 0-based positions of the kept modes in the full spectrum.
2. The filtered spectrum's mode count must match `len(original_indices)`.
"""

from argparse import ArgumentParser as Parser
from pathlib import Path

from spinlabs.parsers import parse_phonon_npz, save_phonon_npz
from spinlabs.symmetry import (
    classify_and_pair,
    filter_degenerate_partners,
    sym_check_summary,
    symmetrize_degenerate_groups,
)


if __name__ == "__main__":
    parser = Parser("Determine symmetry of phonon modes.")
    parser.add_argument("phonon_path", metavar="phonon_path", help="Path to phonon data.")
    parser.add_argument(
        "out_path",
        metavar="out_path",
        nargs="?",
        help="Where to save the filtered spectrum (default: <phonon_path stem>_sym.npz).",
    )

    args = parser.parse_args()

    spectrum = parse_phonon_npz(args.phonon_path)
    classify_and_pair(spectrum)
    symmetrized = symmetrize_degenerate_groups(spectrum)
    filtered = filter_degenerate_partners(symmetrized)
    out_path = args.out_path
    if out_path is None:
        stem = Path(args.phonon_path).stem
        out_path = str(Path(args.phonon_path).with_name(f"{stem}_sym.npz"))
    save_phonon_npz(filtered, out_path)
    check = sym_check_summary(getattr(spectrum, "sym_check", None))
    note = "" if not check else f"; sym_check: {check}"
    print(f"saved filtered spectrum ({filtered.n_modes} of {spectrum.n_modes} modes) to {out_path}{note}")
