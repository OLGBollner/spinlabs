"""CLI commands ``spinlabs init tree`` and ``spinlabs init zfs``: set up a new defect (spinlabs.vasp.init)."""

import argparse
from pathlib import Path

from spinlabs.cli import add_verbosity_args
from spinlabs.methods import CLI_CHOICES


def build_init_parser(subparsers) -> None:
    tree = subparsers.add_parser(
        "tree",
        help="Create a defect folder: relax/ from the inputs, data/ (phonon files), ZFS_AO/ and ZFS_UO/",
    )
    tree.add_argument("defect", help="defect folder name, e.g. NV_64 (defect name and cell size)")
    tree.add_argument(
        "inputs", type=Path,
        help="folder with POSCAR (or structure.vasp), INCAR, KPOINTS and POTCAR, searched recursively; its data/ "
        "and phonopy/phonon files go to data/",
    )
    tree.add_argument("-o", "--output", type=Path, default=Path("."), help="where to create <defect>/ (default: .)")
    tree.add_argument("--force", action="store_true", help="replace an existing <defect>/")
    add_verbosity_args(tree)
    tree.set_defaults(func=_run_tree)

    zfs = subparsers.add_parser(
        "zfs",
        help="Fill a ground-state ZFS folder (ZFS_AO, ZFS_UO or ZFS_full) from a finished relaxation",
    )
    zfs.add_argument("relax", type=Path, help="the finished relaxation (CONTCAR, CHGCAR, EIGENVAL, INCAR, ...)")
    zfs.add_argument(
        "--method", choices=CLI_CHOICES, default="all",
        help="all/AO: ZFS_AO with the full occupations; unpaired/UO: ZFS_UO with only the unpaired orbitals "
        "occupied; decont: ZFS_full, the full spin decontamination (SCF + one ALGO=None run per spin "
        "sublevel). Default: all",
    )
    zfs.add_argument("--vasp-binary", type=Path, default=None, help="VASP binary for run_ground_state.sh")
    zfs.add_argument("--max-hours", type=float, default=None, help="cap on the job's time limit in hours")
    zfs.add_argument("--force", action="store_true", help="refill a folder that was already filled")
    add_verbosity_args(zfs)
    zfs.set_defaults(func=_run_zfs)


def _run_tree(args: argparse.Namespace) -> None:
    from spinlabs.vasp.init import init_tree

    try:
        out = init_tree(args.defect, args.inputs, parent=args.output, force=args.force)
    except (FileNotFoundError, FileExistsError) as error:
        raise SystemExit(f"error: {error}")
    print(f"created {out}; relax in {out / 'relax'}, then: spinlabs init zfs {out / 'relax'} --method all|unpaired|decont")


def _run_zfs(args: argparse.Namespace) -> None:
    from spinlabs.vasp.init import init_zfs

    if not args.relax.is_dir():
        raise SystemExit(f"error: not a folder: {args.relax}")
    failures = init_zfs(args.relax, args.method, vasp_binary=args.vasp_binary, max_hours=args.max_hours,
                        force=args.force)
    if failures:
        raise SystemExit("error: " + "; ".join(failures))
