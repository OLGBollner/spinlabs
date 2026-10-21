"""CLI subcommand for the end-to-end analysis pipeline."""

import argparse
from itertools import product
from pathlib import Path

from spinlabs.cli import add_verbosity_args
from spinlabs.cli.coupling import build_coupling_parser
from spinlabs.cli.sigma_scan import build_sigma_scan_parser
from spinlabs.models import normalize_calc_method
from spinlabs.pipeline import DEFAULT_SIGMA_MEV, run_full_pipeline
from spinlabs.registry import ensure_db, parse_orders
from spinlabs.relaxation_dynamics import INIT_STATES, normalize_init_state

from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)


def build_run_parser(parser: argparse.ArgumentParser) -> None:
    """Configure ``parser`` as the ``run`` subcommand."""

    inputs = parser.add_argument_group("input mode (exactly one)")
    inputs.add_argument("--sim-folder", nargs="+", metavar="DIR", help="Path to VASP simulation folder(s)")
    inputs.add_argument("--raw-zfs-file", nargs="+", metavar="FILE", help="Path to raw ZFS .npz dataset file(s)")
    inputs.add_argument("--raw-zfs-file-1d", metavar="FILE", help="Raw ZFS 1st-order .npz (combined 1d+2d mode)")
    inputs.add_argument("--raw-zfs-file-2d", metavar="FILE", help="Raw ZFS 2nd-order .npz (combined 1d+2d mode)")
    inputs.add_argument("--coupling-file", metavar="FILE", help="Path to pre-computed spin-phonon coupling .npz")
    inputs.add_argument(
        "--query",
        action="store_true",
        help="Run the newest packaged dataset of every configuration in the registry matching the filters "
        "(--defect/--cell-size/--method/--order/--pert-scale). Analyzed data is run again into a new "
        "run folder; plot/table with --latest to use the newest. "
        "All other run options (temperatures, --init-state, --plot, ...) apply to each dataset",
    )

    opt = parser.add_argument_group("options")
    opt.add_argument("-ph", "--phonon-file", metavar="FILE", help="phonopy.yaml or phonon_data.npz")
    opt.add_argument("--two-phonon", metavar="FILE", help="Two-phonon Raman .npz file (optional)")
    opt.add_argument(
        "--method",
        nargs="+",
        choices=["all", "approx"],
        help="Calculation method: 'all' for all_bands (ZFS_hyp), 'approx' for defect_band_approx (ZFS_occup). "
        "Inferred from the input file name when omitted. With --query, both may be given",
    )
    opt.add_argument(
        "--order",
        nargs="+",
        type=int,
        choices=[1, 2, 12],
        help="Perturbation order(s) to analyze, e.g. --order 1 2 (or 1 / 2; 12 is the legacy alias for 1 2)",
    )
    opt.add_argument(
        "--pert-scale",
        nargs="+",
        type=float,
        help="Override perturbation scale (e.g. 0.025). With --query, several may be given (e.g. 0.1 0.025)",
    )
    opt.add_argument(
        "--defect",
        nargs="+",
        help="Override defect name (e.g. NV, ClV). With --query, several may be given (e.g. NV DiV ClV)",
    )
    opt.add_argument(
        "--cell-size",
        nargs="+",
        type=int,
        help="Override supercell size (e.g. 64, 128). With --query, several may be given (e.g. 128 512)",
    )
    opt.add_argument("--t-start", type=float, default=0.0, help="Start temperature in K (default: 0.0)")
    opt.add_argument("--t-end", type=float, default=300.0, help="End temperature in K (default: 300.0)")
    opt.add_argument("--t-step", type=float, default=10.0, help="Temperature step in K (default: 10.0)")
    opt.add_argument(
        "--temperatures", type=float, nargs="+", metavar="T", help="Explicit temperature grid in K (overrides --t-*)"
    )
    opt.add_argument(
        "--init-state",
        nargs="+",
        type=normalize_init_state,
        choices=INIT_STATES,
        default=list(INIT_STATES),
        help="Initial spin state(s) to solve the relaxation dynamics for; all share one run directory "
        "(default: all of ms_1 ms_0 ms_-1; 1 0 -1 also accepted)",
    )
    opt.add_argument(
        "-o", "--output-root", default="runs", help="Root folder for output run directories (default: runs)"
    )
    opt.add_argument(
        "--run-name",
        help="Custom run folder name; an increment index is appended if it exists. "
        "With --query it is rejected when the query matches more than one dataset",
    )
    opt.add_argument("--plot", action="store_true", help="Generate plots inside <run_dir>/figures/")
    opt.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA_MEV,
        metavar="MEV",
        help=f"Gaussian broadening of the spectral functions in meV, standing in for the continuous phonon "
        f"bands of a macroscopic crystal (default: {DEFAULT_SIGMA_MEV:g}; see run sigma-scan)",
    )
    opt.add_argument(
        "--keep-inconsistent-pairs",
        action="store_true",
        help="Keep second-order entries whose ZFS change departs from the first-order response by a multiple of it "
        "(over-displaced or stale runs); they are excluded and listed by default",
    )
    add_verbosity_args(opt)

    parser.set_defaults(func=_run)

    # Optional stage subcommands; plain `spinlabs run ...` is the full pipeline.
    stages = parser.add_subparsers(dest="stage", metavar="STAGE", title="stages (optional)")
    build_coupling_parser(
        stages.add_parser(
            "coupling",
            help="Only compute the spin-phonon couplings (ZFSManager) and print a summary; nothing is saved",
        )
    )
    build_sigma_scan_parser(
        stages.add_parser(
            "sigma-scan",
            help="T1 against the spectral broadening sigma for one dataset; nothing is saved",
        )
    )


_METHOD_MAP = {"all": ("all_bands", "ZFS_hyp"), "approx": ("defect_band_approx", "ZFS_occup")}


def _infer_method_from_file(path: Path) -> str:
    """Infer 'all' or 'approx' from a packaged filename (falls back to 'all')."""
    name = path.name.lower()
    return "approx" if ("approx" in name or "zfs_occup" in name) else "all"


def _infer_method(args: argparse.Namespace) -> str:
    """Return 'all' or 'approx', inferring from input file/folder names when --method is omitted."""
    if args.method:
        return args.method[0]
    names = " ".join(
        str(p)
        for p in (
            *(args.sim_folder or []),
            *(args.raw_zfs_file or []),
            args.raw_zfs_file_1d,
            args.raw_zfs_file_2d,
            args.coupling_file,
        )
        if p
    ).lower()
    has_all = "all_bands" in names or "zfs_hyp" in names
    has_approx = "defect_band_approx" in names or "approx" in names or "zfs_occup" in names
    if has_all == has_approx:
        raise SystemExit(
            "could not infer --method from input names; pass --method all or --method approx"
            if not has_all
            else "input names match both methods; pass --method all or --method approx explicitly"
        )
    return "all" if has_all else "approx"


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Enforce exactly one primary input mode."""
    modes = [
        ("--sim-folder", args.sim_folder),
        ("--raw-zfs-file", args.raw_zfs_file),
        (
            "--raw-zfs-file-1d/2d",
            [args.raw_zfs_file_1d, args.raw_zfs_file_2d] if (args.raw_zfs_file_1d or args.raw_zfs_file_2d) else None,
        ),
        ("--coupling-file", args.coupling_file),
        ("--query", args.query),
    ]
    given = [name for name, val in modes if val]
    if not given:
        parser.error("one of --sim-folder, --raw-zfs-file, --raw-zfs-file-1d/2d, --coupling-file is required")
    if len(given) > 1:
        parser.error(f"input modes are mutually exclusive, got: {', '.join(given)}")
    if (args.raw_zfs_file_1d is None) != (args.raw_zfs_file_2d is None):
        parser.error("--raw-zfs-file-1d and --raw-zfs-file-2d must be given together")
    if not args.query:
        for flag, values in _filter_flags(args):
            if values and len(values) > 1:
                parser.error(f"multiple {flag} values are only supported with --query")


def _filter_flags(args: argparse.Namespace) -> list[tuple[str, list | None]]:
    """The multi-valued registry filter flags, as (flag, values) pairs."""
    return [
        ("--defect", args.defect),
        ("--cell-size", args.cell_size),
        ("--method", args.method),
        ("--pert-scale", args.pert_scale),
    ]


def _first(values: list | None):
    return values[0] if values else None


def _run(args: argparse.Namespace) -> None:
    _validate(argparse.ArgumentParser(prog="spinlabs run"), args)
    if args.query:
        _run_query(args)
        return
    run_full_pipeline(
        sim_folder=args.sim_folder[0] if args.sim_folder and len(args.sim_folder) == 1 else args.sim_folder,
        raw_zfs_file=args.raw_zfs_file[0] if args.raw_zfs_file and len(args.raw_zfs_file) == 1 else args.raw_zfs_file,
        raw_zfs_file_1d=args.raw_zfs_file_1d,
        raw_zfs_file_2d=args.raw_zfs_file_2d,
        coupling_file=args.coupling_file,
        phonon_file=args.phonon_file,
        two_phonon_file=args.two_phonon,
        calc_method=_METHOD_MAP[_infer_method(args)][0],
        zfs_folder=_METHOD_MAP[_infer_method(args)][1],
        order=parse_orders(args.order) if args.order else None,
        pert_scale=_first(args.pert_scale),
        defect=_first(args.defect),
        cell_size=_first(args.cell_size),
        t_start=args.t_start,
        t_end=args.t_end,
        t_step=args.t_step,
        temperatures=args.temperatures,
        init_state=args.init_state,
        output_root=args.output_root,
        run_name=args.run_name,
        save_plots=args.plot,
        show_plots=False,
        keep_inconsistent_pairs=args.keep_inconsistent_pairs,
        sigma=args.sigma,
    )


def _run_query(args: argparse.Namespace) -> None:
    """
    Resolve packaged data from the registry and run each dataset.

    Each configuration runs on its newest package, even when it was analyzed
    before: every run gets a new timestamped folder, and ``--latest`` picks
    the newest when plotting or tabulating.

    Every option accepted by ``run`` is forwarded to the pipeline exactly as in the
    non-query path, so ``--query`` never silently computes something other than what
    was asked for. ``--defect``/``--cell-size``/``--method``/``--order``/``--pert-scale``
    do double duty: they filter the registry *and* are passed through to the run.
    ``--phonon-file`` is optional here -- when omitted the pipeline resolves the phonon
    from the registry; when given it takes precedence, as it does for any other input mode.
    """
    from spinlabs.registry import find_packaged, unregistered_packages, unregistered_warning

    conn = ensure_db()
    warning = unregistered_warning(unregistered_packages(conn))
    if warning:
        LOGGER.warning(warning)
    # Multi-valued filters query every combination; each dataset is run with
    # the filter values that matched it.
    files: list[tuple[Path, str | None, int | None, float | None]] = []
    seen: set[Path] = set()
    for defect, cell_size, method, pert_scale in product(
        *[values or [None] for _, values in _filter_flags(args)]
    ):
        matched = find_packaged(
            conn,
            defect=defect,
            cell_size=cell_size,
            calc_method=normalize_calc_method(method),
            pert_scale=pert_scale,
            pert_order=parse_orders(args.order) if args.order else None,
            latest=True,
        )
        for path in matched:
            if path not in seen:
                seen.add(path)
                files.append((path, defect, cell_size, pert_scale))
    if not files:
        LOGGER.warning("No packaged data matching the query.")
        return
    # With multiple matched datasets, disambiguate the run-name so each
    # dataset lands in its own <run_name>_<stem> directory instead of
    # silently sharing/creating duplicate <run_name>_1, _2, ... folders.
    failed: list[tuple[Path, Exception]] = []
    for n, (path, defect, cell_size, pert_scale) in enumerate(files, start=1):
        # Shown without -v, so warnings and errors below can be traced to their dataset.
        print(f"[{n}/{len(files)}] {path}", flush=True)
        method = _infer_method_from_file(Path(path))
        if args.run_name and len(files) > 1:
            run_name = f"{args.run_name}_{Path(path).stem}"
        else:
            run_name = args.run_name
        try:
            _run_one_query_dataset(args, path, method, pert_scale, defect, cell_size, run_name)
        except Exception as error:  # one bad dataset must not stop the rest of the query
            LOGGER.error(f"Failed: {path}: {error}")
            failed.append((Path(path), error))
    print(f"Done: {len(files) - len(failed)} of {len(files)} dataset(s) analyzed.")
    if failed:
        raise SystemExit(
            f"{len(failed)} dataset(s) failed:\n" + "\n".join(f"  {path}: {error}" for path, error in failed)
        )


def _run_one_query_dataset(
    args: argparse.Namespace,
    path: Path,
    method: str,
    pert_scale: float | None,
    defect: str | None,
    cell_size: int | None,
    run_name: str | None,
) -> None:
    """Run the pipeline on one packaged dataset with the query's options."""
    run_full_pipeline(
        raw_zfs_file=str(path),
        phonon_file=args.phonon_file,
        two_phonon_file=args.two_phonon,
        calc_method=_METHOD_MAP[method][0],
        zfs_folder=_METHOD_MAP[method][1],
        order=parse_orders(args.order) if args.order else None,
        pert_scale=pert_scale,
        defect=defect,
        cell_size=cell_size,
        t_start=args.t_start,
        t_end=args.t_end,
        t_step=args.t_step,
        temperatures=args.temperatures,
        init_state=args.init_state,
        output_root=args.output_root if args.output_root else "runs",
        run_name=run_name,
        save_plots=args.plot,
        show_plots=False,
        keep_inconsistent_pairs=args.keep_inconsistent_pairs,
        sigma=args.sigma,
    )
