"""CLI command ``spinlabs symmetrize``: symmetrize a phonon file and keep one mode per degenerate set.

Same steps as the perturbation-tree setup (``spinlabs.vasp.trees.resolve_sym_phonon``):
classify the modes, re-diagonalize accidentally mixed near-degenerate clusters into
pure irreps, then drop the redundant partners of each degenerate set. A phonopy yaml
is read through ``prepare_phonopy_yaml``, so repeated q-point blocks are dropped and,
when the yaml has no structure, a structure-complete ``<stem>_full.yaml`` is written
from the POSCAR next to it.
"""

import argparse
from pathlib import Path

from spinlabs.cli import add_verbosity_args
from spinlabs.utils import resolve_path

PHONON_SUFFIXES = {".yaml", ".yml", ".npz"}


def build_symmetrize_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("phonon", type=resolve_path, help="phonon file: phonopy .yaml/.yml or .npz")
    parser.add_argument(
        "-o",
        "--out",
        type=resolve_path,
        default=None,
        help="output .npz (default: <phonon stem>_sym<n>.npz next to the input, n = kept mode count)",
    )
    parser.add_argument(
        "--poscar",
        type=resolve_path,
        default=None,
        help="structure for a phonopy yaml without one (default: POSCAR next to the yaml)",
    )
    add_verbosity_args(parser)
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    from spinlabs.parsers import parse_phonon_npz, prepare_phonopy_yaml, save_phonon_npz
    from spinlabs.symmetry import (
        classify_and_pair,
        filter_degenerate_partners,
        SymmetryDetectionError,
        sym_check_summary,
        symmetrize_degenerate_groups,
    )

    path: Path = args.phonon
    suffix = path.suffix.lower()
    if not path.is_file():
        raise SystemExit(f"error: not found: {path}")
    if suffix not in PHONON_SUFFIXES:
        raise SystemExit(f"error: unsupported phonon file type {path.suffix!r} (expected .yaml/.yml or .npz)")
    if args.out is not None and args.out.suffix.lower() != ".npz":
        raise SystemExit(f"error: output must end in .npz, got {args.out}")

    try:
        if suffix == ".npz":
            spectrum = parse_phonon_npz(path)
        else:
            spectrum, _ = prepare_phonopy_yaml(path, poscar_path=args.poscar)
    except Exception as error:  # a clean message, not a traceback
        raise SystemExit(f"error: failed to parse {path}: {error}")

    try:
        classify_and_pair(spectrum)
        symmetrized = symmetrize_degenerate_groups(spectrum)
        filtered = filter_degenerate_partners(symmetrized)
    except SymmetryDetectionError as error:
        hint = " (no structure in the yaml: pass --poscar or put a POSCAR next to it)" if suffix != ".npz" else ""
        raise SystemExit(f"error: {error}{hint}")

    out = args.out or path.with_name(f"{path.stem}_sym{filtered.n_modes}.npz")
    save_phonon_npz(filtered, out)

    report = getattr(symmetrized, "symmetrization_report", None) or {}
    n_sym = len(report.get("symmetrized", []))
    n_skip = len(report.get("skipped", []))
    check = sym_check_summary(getattr(spectrum, "sym_check", None))
    print(
        f"wrote {out}: {filtered.n_modes} of {spectrum.n_modes} modes kept; "
        f"{n_sym} mixed cluster(s) symmetrized, {n_skip} skipped"
        + (f"; sym_check: {check}" if check else "")
    )
