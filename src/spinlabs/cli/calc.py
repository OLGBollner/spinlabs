"""CLI command ``spinlabs calc zfs``: the ZFS of one SCF/ZFS run folder from its OUTCAR(s) (spinlabs.zfs_calc)."""

from __future__ import annotations

import argparse
import json

import numpy as np

from spinlabs.cli import add_verbosity_args
from spinlabs.methods import CLI_CHOICES
from spinlabs.utils import resolve_path

TRACE_TOL_MHZ = 0.05  # VASP prints the tensor to 3 decimals; a traceless one sums to within rounding


def build_calc_parser(subparsers) -> None:
    zfs = subparsers.add_parser(
        "zfs",
        help="The ZFS of one SCF/ZFS run folder from its OUTCAR(s): D, E, principal axes and the tensor",
    )
    zfs.add_argument("path", type=resolve_path, help="the run folder: its OUTCAR (AO, UO) or its ms_*/OUTCAR (decont)")
    zfs.add_argument(
        "--method", choices=CLI_CHOICES, default=None,
        help="all/AO: the OUTCAR tensor as VASP reports it; unpaired/UO: with the defect-band spin correction "
        "(S from NUPDOWN); decont: the spin-decontaminated tensor of the ms_*/ sublevels. Default: from the "
        "folder (ms_*/ folders: decont; ZFS_AO, ZFS_UO, ZFS_full by name)",
    )
    zfs.add_argument("--json", type=resolve_path, default=None, help="also write the numbers to this JSON file")
    add_verbosity_args(zfs)
    zfs.set_defaults(func=_run_zfs)


def _run_zfs(args: argparse.Namespace) -> None:
    from spinlabs.zfs_calc import compute_zfs

    try:
        result = compute_zfs(args.path, args.method)
    except ValueError as error:
        raise SystemExit(f"error: {error}")
    summary = result.summary()
    spin = f", S = {result.spin:g}" if result.spin is not None else ""
    print(f"{result.folder}: {result.method}{spin}")
    for name, part in result.parts.items():
        print(f"  {name:12s} D = {part.D:10.3f} MHz   E = {part.E:8.3f} MHz   trace = {np.trace(part.matrix):8.3f} MHz")
    for note in result.notes:
        print(f"  {note}")
    d_xx, d_yy, d_zz = summary["principal_values_MHz"]
    print(f"ZFS ({result.method}): D = {result.tensor.D:.3f} MHz, E = {result.tensor.E:.3f} MHz, "
          f"trace = {np.trace(result.tensor.matrix):.3f} MHz")
    worst = max((abs(np.trace(t.matrix)) for t in result.parts.values()), default=0.0)
    if worst > TRACE_TOL_MHZ:
        print(f"  warning: an OUTCAR tensor has a trace of {worst:.3f} MHz; VASP's dipolar D is traceless for any "
              "wavefunction, so its D evaluation itself is off (e.g. a gamma-only build's LDMATRIX)")
    print(f"  principal values (MHz): {d_xx:.3f} {d_yy:.3f} {d_zz:.3f}; z axis "
          + " ".join(f"{c:.4f}" for c in summary["principal_z_axis"]))
    print("  tensor (MHz):")
    for row in result.tensor.matrix:
        print("    " + " ".join(f"{v:10.3f}" for v in row))
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"written: {args.json}")
