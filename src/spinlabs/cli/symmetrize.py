"""CLI command ``spinlabs symmetrize``: symmetrize a phonon file and keep one mode per degenerate set.

Same steps as the perturbation-tree setup (``spinlabs.vasp.trees.resolve_sym_phonon``):
classify the modes, re-diagonalize accidentally mixed near-degenerate clusters into
pure irreps, then drop the redundant partners of each degenerate set. With ``--align`` every
degenerate E set is first rotated onto one mirror plane (``symmetry.align_degenerate_pairs``) and
the file is named ``<stem>_sym<n>_aligned.npz``; its ``aligned`` key says which kind a file is. The aligned
full spectrum (every mode, same mode indices) is written too, as ``<stem>_aligned.npz``: runs made from
the aligned file must be analysed with it, not with the original spectrum. A phonopy yaml
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
        help="output .npz (default: <phonon stem>_sym<n>.npz next to the input, n = kept mode count; "
        "<phonon stem>_sym<n>_aligned.npz with --align)",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Rotate every degenerate E set onto one mirror plane, so each kept E mode is exactly mirror-odd "
        "(sigma_v2); for new perturbation runs only, never for the phonon file of existing runs (they were "
        "displaced along the old vectors)",
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
    from spinlabs.models import reduced_spectrum_reason
    from spinlabs.parsers import parse_phonon_npz, prepare_phonopy_yaml, save_phonon_npz
    from spinlabs.symmetry import (
        align_degenerate_pairs,
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
    reduced = reduced_spectrum_reason(spectrum)
    if reduced is not None:
        # nothing left to filter or align: it would be written back as its own "full" spectrum
        raise SystemExit(f"error: {path} is already symmetry-reduced ({reduced}); pass the full spectrum")

    try:
        classify_and_pair(spectrum)
        symmetrized = symmetrize_degenerate_groups(spectrum)
        report = getattr(symmetrized, "symmetrization_report", None) or {}
        if args.align:
            symmetrized = align_degenerate_pairs(symmetrized)
        else:
            symmetrized.aligned = False
        filtered = filter_degenerate_partners(symmetrized)
    except (SymmetryDetectionError, NotImplementedError) as error:
        hint = " (no structure in the yaml: pass --poscar or put a POSCAR next to it)" if suffix != ".npz" else ""
        raise SystemExit(f"error: {error}{hint}")

    stub = "_aligned" if args.align else ""
    out = args.out or path.with_name(f"{path.stem}_sym{filtered.n_modes}{stub}.npz")
    save_phonon_npz(filtered, out)
    full_out = None
    if args.align:
        # The analysis needs every mode (ZFSManager rejects a reduced spectrum) along the same vectors
        # the runs were displaced along: the aligned spectrum before the partners were dropped.
        full_out = out.with_name(f"{path.stem}_aligned.npz")
        save_phonon_npz(symmetrized, full_out)

    n_sym = len(report.get("symmetrized", []))
    n_skip = len(report.get("skipped", []))
    check = sym_check_summary(getattr(spectrum, "sym_check", None))
    aligned = ""
    if args.align:
        rep = getattr(symmetrized, "alignment_report", None) or {}
        aligned = (f"; {len(rep.get('aligned', []))} degenerate E set(s) aligned to a mirror, "
                   f"{len(rep.get('skipped', []))} not")
    print(
        f"wrote {out}: {filtered.n_modes} of {spectrum.n_modes} modes kept; "
        f"{n_sym} mixed cluster(s) symmetrized, {n_skip} skipped"
        + aligned
        + (f"; sym_check: {check}" if check else "")
        + f"; aligned: {filtered.aligned}"
    )
    if full_out is not None:
        print(f"wrote {full_out}: the aligned full spectrum ({symmetrized.n_modes} modes); analyse runs made from "
              f"{out.name} with it, not with {path.name}")
