"""CLI commands ``spinlabs init tree``, ``init zfs`` and ``init convergence``: set up a new defect
(spinlabs.vasp.init) and its step-size convergence runs (spinlabs.vasp.mode_selection, spinlabs.vasp.trees)."""

import argparse

from spinlabs.cli import add_verbosity_args
from spinlabs.methods import CLI_CHOICES, METHODS, normalize_calc_method
from spinlabs.utils import resolve_path
from spinlabs.vasp.mode_selection import DEFAULT_N_MODES
from spinlabs.vasp.structures import add_displacement_args, displacement_from_args


def build_init_parser(subparsers) -> None:
    tree = subparsers.add_parser(
        "tree",
        help="Create a defect folder: relax/ from the inputs, data/ (phonon files), ZFS_AO/ and ZFS_UO/",
    )
    tree.add_argument("defect", help="defect folder name, e.g. NV_64 (defect name and cell size)")
    tree.add_argument(
        "inputs", type=resolve_path,
        help="folder with POSCAR (or structure.vasp), INCAR, KPOINTS and POTCAR, searched recursively; its data/ "
        "and phonopy/phonon files go to data/",
    )
    tree.add_argument("-o", "--output", type=resolve_path, default=resolve_path("."), help="where to create <defect>/ (default: .)")
    tree.add_argument("--force", action="store_true", help="replace an existing <defect>/")
    add_verbosity_args(tree)
    tree.set_defaults(func=_run_tree)

    zfs = subparsers.add_parser(
        "zfs",
        help="Fill a ground-state ZFS folder (ZFS_AO, ZFS_UO or ZFS_full) from a finished relaxation",
    )
    zfs.add_argument("relax", type=resolve_path, help="the finished relaxation (CONTCAR, CHGCAR, EIGENVAL, INCAR, ...)")
    zfs.add_argument(
        "--method", choices=CLI_CHOICES, default="all",
        help="all/AO: ZFS_AO with the full occupations; unpaired/UO: ZFS_UO with only the unpaired orbitals "
        "occupied; decont: ZFS_full, the full spin decontamination (SCF + one ALGO=None run per spin "
        "sublevel). Default: all",
    )
    zfs.add_argument("--vasp-binary", type=resolve_path, default=None, help="VASP binary for run_ground_state.sh")
    zfs.add_argument("--max-hours", type=float, default=None, help="cap on the job's time limit in hours")
    zfs.add_argument("--force", action="store_true", help="refill a folder that was already filled")
    add_verbosity_args(zfs)
    zfs.set_defaults(func=_run_zfs)

    conv = subparsers.add_parser(
        "convergence",
        help="Pick a varied set of phonon modes (IPR, irrep, frequency, nearness to the defect), show them and, "
        "once confirmed, create the step-size convergence folders convergence_test_<method>/",
    )
    conv.add_argument("defect", type=resolve_path, help="defect folder (holds ZFS_AO/, ZFS_UO/ and data/)")
    conv.add_argument(
        "--method", choices=CLI_CHOICES, nargs="+", default=None,
        help="all/AO: convergence_test_AO from ZFS_AO; unpaired/UO: convergence_test_UO from ZFS_UO; decont: "
        "convergence_test_decont, the full spin decontamination (plus ZFS_full). Default: all unpaired",
    )
    pick = conv.add_mutually_exclusive_group()
    pick.add_argument("-n", "--n-modes", type=int, default=DEFAULT_N_MODES,
                      help=f"how many modes to pick (default: {DEFAULT_N_MODES})")
    pick.add_argument("--modes", type=int, nargs="+", default=None,
                      help="run these 1-based modes instead of picking (their properties are still shown)")
    steps = conv.add_mutually_exclusive_group()
    steps.add_argument("--pert", type=float, nargs="+", default=None,
                       help="tree step(s) p: runs +-p/2, +-p, +-2p and +-4p (4p is the 2Q helper of 2p). "
                       "Default: the steps of the defect's perturbation trees")
    steps.add_argument("--steps", type=float, nargs="+", default=None,
                       help="step sizes, each run at -s and +s, e.g. 0.025 0.05")
    add_displacement_args(conv, default=None)
    conv.add_argument("--phonon", type=resolve_path, default=None,
                      help="phonon data file (default: <defect>/data/phonon_data.npz); symmetrized into "
                      "data/phonon_data_sym<n>.npz first, as for the trees")
    conv.add_argument("--vasp-binary", type=resolve_path, default=None,
                      help="VASP binary (default: the one the defect's trees name)")
    conv.add_argument("--array-jobs", type=int, default=5, help="SLURM array jobs per convergence folder (default: 5)")
    conv.add_argument("--max-hours", type=float, default=None, help="cap on the per-job time limit in hours")
    conv.add_argument("--force", action="store_true", help="rebuild existing convergence folders")
    conv.add_argument("-y", "--yes", action="store_true", help="create the runs without asking")
    add_verbosity_args(conv)
    conv.set_defaults(func=_run_convergence, _parser=conv)


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


def confirm(question: str) -> bool:
    """Ask a [Y/n] question on the terminal: Enter or y/yes is yes. No answer (no terminal) is no."""
    try:
        answer = input(question)
    except EOFError:
        print("\nno answer (no terminal); pass -y/--yes to go ahead without asking")
        return False
    return answer.strip().lower() in ("", "y", "yes")


def _run_convergence(args: argparse.Namespace) -> None:
    from spinlabs.vasp import default_phonon
    from spinlabs.vasp.mode_selection import (
        format_selection,
        given_modes,
        mode_properties,
        select_convergence_modes,
        spectrum_summary,
    )
    from spinlabs.vasp.trees import (
        build_convergence_tree,
        convergence_steps,
        defect_displacement,
        defect_steps,
        defect_vasp_binary,
        resolve_sym_phonon,
        step_sizes,
    )
    from spinlabs.symmetry import SymmetryDetectionError

    out = args.defect
    if not out.is_dir():
        raise SystemExit(f"error: not a folder: {out}")
    keys = list(dict.fromkeys(normalize_calc_method(m) for m in (args.method or ["all", "unpaired"])))
    tree_methods = [k for k in keys if k != "full_decont"]
    try:
        if args.displacement or args.no_mass_weighted:
            displacement = displacement_from_args(args, args._parser)
        else:
            displacement = defect_displacement(out) or "normalized"
        if args.steps:
            steps = step_sizes(args.steps)
        else:
            pert = args.pert or defect_steps(out)
            if not pert:
                raise SystemExit(f"error: {out.name} has no perturbation trees to take the step from; pass --pert or --steps")
            steps = convergence_steps(pert)
        phonon = resolve_sym_phonon(out, args.phonon or default_phonon(out))
        modes = mode_properties(phonon)
        selection = given_modes(modes, args.modes) if args.modes else select_convergence_modes(modes, args.n_modes)
    except (FileNotFoundError, ValueError, SymmetryDetectionError) as error:
        raise SystemExit(f"error: {error}")

    numbers = [s.mode.number for s in selection]
    folders = [METHODS[k].convergence for k in keys]
    print(f"{out.name}: {phonon.name}: {spectrum_summary(modes)}")
    print(format_selection(selection))
    print(f"modes:        {' '.join(str(k) for k in numbers)}")
    print(f"steps:        {' '.join(f'{s:g}' for s in steps)} (each at -s and +s), {displacement} displacement")
    print(f"folders:      {', '.join(folders)}: {len(numbers) * 2 * len(steps)} runs each")
    question = "create convergence runs of modes? [Y/n]: "
    if args.force:
        question = "create convergence runs of modes (existing folders are rebuilt)? [Y/n]: "
    if not args.yes and not confirm(question):
        print("no runs created")
        return

    vasp_binary = args.vasp_binary or defect_vasp_binary(out)
    common = dict(displacement=displacement, array_jobs=args.array_jobs, max_hours=args.max_hours,
                  vasp_binary=vasp_binary, force=args.force)
    try:
        failures: list[str] = []
        if tree_methods:
            failures += build_convergence_tree(out, phonon, numbers, steps, methods=tree_methods, **common)
        if "full_decont" in keys:
            failures += build_convergence_tree(out, phonon, numbers, steps, decont=True, **common)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}")
    if failures:
        raise SystemExit("error: " + "; ".join(failures))
    print("submit from each convergence folder with: sh run_convergence.sh")
