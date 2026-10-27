"""CLI subcommands for the plotting functions."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from spinlabs.cli import add_verbosity_args
from spinlabs.convergence import SNR_TARGET, TRUNC_TOL, plot_pert_convergence, read_conv_metadata
from spinlabs.plotter import plot_run_coupling, plot_run_rates, plot_t1
from spinlabs.registry import ensure_db, find_convergence, find_runs, parse_orders
from spinlabs.relaxation_dynamics import INIT_STATES, find_t1_files, normalize_init_state
from spinlabs.logging_utils import get_logger, section

LOGGER = get_logger(__name__)

_METHODS = {"all": "all_bands", "approx": "defect_band_approx"}  # --method values -> registry calc_method


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o", "--output", type=Path, help="Output file (or directory for run plots); defaults to figures/<kind>/"
    )
    parser.add_argument("--fmt", default="png", help="Image format, e.g. png or pdf")
    parser.add_argument("--dpi", type=int, default=300, help="Image resolution")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    add_verbosity_args(parser)


def build_plot_parser(subparsers: argparse._SubParsersAction) -> None:  # noqa: SLF001
    """Add the ``plot`` subcommands to a subparsers object."""
    coupling = subparsers.add_parser("coupling", help="Plot run coupling from a run directory")
    coupling.add_argument("run_dir", type=Path, nargs="?")
    _add_query_args(coupling)
    _add_output_args(coupling)
    coupling.set_defaults(func=_run_coupling)

    rates = subparsers.add_parser("rates", help="Plot transition rates from a run directory")
    rates.add_argument("run_dir", type=Path, nargs="?")
    _add_query_args(rates)
    _add_output_args(rates)
    rates.set_defaults(func=_run_rates)

    t1 = subparsers.add_parser("t1", help="Plot T1 curves from one or more npz files or run directories in one figure")
    t1.add_argument("inputs", nargs="*", type=Path)
    _add_query_args(t1)
    t1.add_argument(
        "--init-state",
        nargs="+",
        type=normalize_init_state,
        choices=INIT_STATES,
        metavar="STATE",
        help="Only plot these initial spin state(s): ms_1 ms_0 ms_-1, or 1 0 -1 (default: every state found)",
    )
    t1.add_argument(
        "--rate", action="store_true", help="Plot the relaxation rate 1/T1 (s^-1) instead of T1 (saved as t1_rate)"
    )
    t1.add_argument(
        "--log-temperature", action="store_true", help="Log temperature axis (power laws such as T^5 become lines)"
    )
    t1.add_argument("--plain-name", action="store_true", help="Do not append metadata to the output filename")
    _add_output_args(t1)
    t1.set_defaults(func=_run_t1)

    conv = subparsers.add_parser(
        "convergence", help="Plot ZFS principal components vs perturbation scale from a packed npz"
    )
    conv.add_argument("npz", type=Path, nargs="*", help="npz files from 'spinlabs package --conv'")
    _add_query_args(conv, runs=False)
    conv.add_argument("-m", "--mode", type=int, action="append", help="plot only this mode (repeatable; default: all)")
    conv.add_argument(
        "--quad", action="store_true", help="also fit y = a + b x + c x^2 and report the second-order derivative 2c"
    )
    conv.add_argument(
        "--deviation",
        action="store_true",
        help="plot D(Q) - D(Q=0) instead of raw D, so the response fills the y-range",
    )
    conv.add_argument(
        "--csv", type=Path, default=None, help="write slopes (and quad coefficients with --quad) to this csv"
    )
    conv.add_argument(
        "--drop-anchor",
        action="store_true",
        help="exclude the Q=0 anchor from fits and residuals (curves still extrapolate to Q=0)",
    )
    conv.add_argument(
        "--noise", type=float, help="noise sigma per tensor element in MHz (default: cubic-fit scatter of all modes)"
    )
    conv.add_argument("--snr-target", type=float, default=SNR_TARGET, help=f"required SNR (default {SNR_TARGET:g})")
    conv.add_argument(
        "--trunc-tol", type=float, default=TRUNC_TOL, help=f"allowed relative truncation error (default {TRUNC_TOL:g})"
    )
    _add_output_args(conv)
    conv.set_defaults(func=_run_convergence)


def _run_convergence(args: argparse.Namespace) -> None:
    if args.query:
        if args.npz:
            raise SystemExit("error: pass either npz paths or --query, not both")
        files = find_convergence(
            ensure_db(),
            defects=args.defect or [],
            cell_sizes=args.cell_size or [],
            calc_methods=[_METHODS[m] for m in args.method or []],
            latest=args.latest,
        )
        if not files:
            raise SystemExit("error: no convergence files in the registry match the query")
    else:
        if not args.npz:
            raise SystemExit("error: provide npz paths, or use --query to select registry files")
        files = args.npz
    if len(files) > 1 and (args.output or args.csv):
        raise SystemExit(f"error: -o/--csv name one file but {len(files)} convergence files were selected")
    for n, npz in enumerate(files, start=1):
        section(LOGGER, f"Plot convergence {n}/{len(files)}: {npz}", major=True)
        _plot_convergence(args, Path(npz))


def _plot_convergence(args: argparse.Namespace, npz: Path) -> None:
    meta = read_conv_metadata(npz)
    stem = "_".join(str(meta[k]) for k in ("defect", "cell") if meta.get(k) is not None)
    # method_key names the folder (test_all vs test_all_mass_weighted); method alone would let them overwrite
    method = meta.get("method_key") or meta.get("method")
    if method:
        stem = f"{stem}_{method}" if stem else str(method)
    if not stem:
        if not args.output:
            raise SystemExit(
                f"error: {npz} has no defect/cell metadata; "
                "repack it with 'spinlabs package --conv' or pass -o explicitly"
            )
        stem = npz.stem
    out = args.output or _default_output(args, "convergence", stem)
    plot_pert_convergence(
        npz,
        output_path=out,
        mode=args.mode,
        quad=args.quad,
        deviation=args.deviation,
        csv_path=args.csv,
        show=args.show,
        drop_anchor=args.drop_anchor,
        noise=args.noise,
        snr_target=args.snr_target,
        trunc_tol=args.trunc_tol,
    )


def _figures_dir(kind: str) -> Path:
    return Path("figures") / kind


def _default_output(args: argparse.Namespace, kind: str, stem: str) -> Path:
    """Default output file under figures/<kind>/ when -o is omitted."""
    p = _figures_dir(kind) / f"{stem}.{args.fmt}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _explicit_inputs(args: argparse.Namespace) -> list[Path]:
    """Explicit run paths given on the command line, if any."""
    if hasattr(args, "inputs"):
        return list(args.inputs)
    run_dir = getattr(args, "run_dir", None)
    return [run_dir] if run_dir else []


def _resolve_query_runs(args: argparse.Namespace) -> list[Path]:
    """Completed run dirs in the registry matching the --query filters."""
    if _explicit_inputs(args):
        raise SystemExit("error: pass either explicit run paths or --query, not both")
    conn = ensure_db()
    dirs = find_runs(conn, _query_filters(args), latest=args.latest, output_root=args.output_root)
    if not dirs:
        raise SystemExit("error: no completed runs in the registry match the query")
    return dirs


def _resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    """Run directories to plot: explicit paths, or registry matches via --query."""
    if args.query:
        return _resolve_query_runs(args)
    provided = _explicit_inputs(args)
    if not provided:
        raise SystemExit("error: provide a run directory, or use --query to select registry runs")
    return [p if p.is_dir() else p.parent for p in provided]


def _run_meta(run_dir: Path) -> tuple | None:
    """Registry (defect, cell, method, pert) of a run directory, or None if it is not registered."""
    conn = ensure_db()
    row = conn.execute(
        "SELECT defect, cell, method, pert FROM runs WHERE run_dir = ? ORDER BY id DESC LIMIT 1", (str(run_dir),)
    ).fetchone()
    if row is None:
        # Runs stored from another host are matched by directory name.
        rows = conn.execute("SELECT run_dir, defect, cell, method, pert FROM runs ORDER BY id DESC")
        row = next((r[1:] for r in rows if r[0] and Path(r[0]).name == run_dir.name), None)
    return row if row is not None and row[0] else None


def _run_label(run_dir: Path, meta: tuple | None) -> str:
    """Readable run configuration (e.g. ``NV 512, all bands, pert 0.1``), else the dir name."""
    if meta is None:
        return run_dir.name
    defect, cell, method, pert = meta
    parts = [f"{defect} {cell}" if cell is not None else defect]
    if method:
        parts.append(method.replace("_", " "))
    if pert is not None:
        parts.append(f"pert {pert:g}")
    return ", ".join(parts) + f" ({run_dir.name})"


def _run_figure_subdir(run_dir: Path, meta: tuple | None) -> Path:
    """Figure folder of a run, mirroring the run layout: ``<defect>_<cell>/pert_<x>/<method>/<run name>``.

    The run name (its timestamp) is the leaf. Unregistered runs keep their
    last four path parts when they sit in that layout, else just their name.
    """
    if meta is None:
        parts = run_dir.parts
        if len(parts) >= 4 and parts[-3].startswith("pert_"):
            return Path(*parts[-4:])
        return Path(run_dir.name)
    defect, cell, method, pert = meta
    head = f"{defect}_{cell}" if cell is not None else str(defect)
    pert_dir = f"pert_{pert:g}" if pert is not None else "pert_none"
    return Path(head) / pert_dir / (method or "unknown_method") / run_dir.name


def _combined_figure_subdir(run_dirs: list[Path]) -> Path:
    """Figure folder for one figure drawn from several runs, in the per-run layout.

    Levels every run shares are kept; differing values are joined with ``+``
    (e.g. ``NV_64+NV_512/pert_0.1/all_bands``). The leaf is the newest run's
    name (the last run, as queries list runs oldest first). Folders that are
    neither registered runs nor in the run layout (loose npz files) add no level.
    """
    subdirs = [
        sub.parts
        for d in dict.fromkeys(run_dirs)
        if len((sub := _run_figure_subdir(d, _run_meta(d))).parts) > 1
    ]
    if not subdirs:
        return Path()
    if len(subdirs) == 1:
        return Path(*subdirs[0])
    if len({len(parts) for parts in subdirs}) != 1:
        return Path(subdirs[-1][-1])  # mixed layouts: only the newest run's name
    levels = ["+".join(dict.fromkeys(level)) for level in zip(*(parts[:-1] for parts in subdirs))]
    return Path(*levels, subdirs[-1][-1])


def _plot_each_run(args: argparse.Namespace, kind: str, plot_run) -> None:
    """Plot every selected run in its own figure(s); with --show all windows open together."""
    shown = []
    for run_dir in _resolve_run_dirs(args):
        meta = _run_meta(run_dir)
        section(LOGGER, f"Plot {kind}: {_run_label(run_dir, meta)} ({run_dir})")
        out_dir = (args.output or _figures_dir(kind)) / _run_figure_subdir(run_dir, meta)
        out_dir.mkdir(parents=True, exist_ok=True)
        shown += plot_run(run_dir, out_dir, args.fmt, args.dpi, args.show, title=_run_label(run_dir, meta)) or []
    if shown:
        plt.show()


def _run_coupling(args: argparse.Namespace) -> None:
    _plot_each_run(args, "coupling", plot_run_coupling)


def _run_rates(args: argparse.Namespace) -> None:
    _plot_each_run(args, "rates", plot_run_rates)


def _add_query_args(parser: argparse.ArgumentParser, runs: bool = True) -> None:
    """Registry-query filters shared by all plot subcommands (same as 'run --query').

    ``runs=False`` (plot convergence) queries packaged convergence files, which have no
    order, perturbation scale or run directory: only --defect/--cell-size/--method/--latest.
    """
    parser.add_argument(
        "--query",
        action="store_true",
        help="Plot the completed runs in the registry matching the filters "
        "(--defect/--cell-size/--method/--order/--pert-scale). Same syntax as 'run --query'"
        if runs
        else "Plot the convergence files in the registry ('package --conv') matching --defect/--cell-size/--method",
    )
    parser.add_argument("--defect", nargs="+", help="Filter by defect name(s) (e.g. NV ClV)")
    parser.add_argument("--cell-size", nargs="+", type=int, help="Filter by supercell size(s) (e.g. 64 128)")
    parser.add_argument(
        "--method",
        nargs="+",
        choices=["all", "approx"],
        help="Filter by calculation method(s): 'all' for all_bands, 'approx' for defect_band_approx",
    )
    if not runs:
        parser.add_argument(
            "--latest",
            action="store_true",
            help="With --query, use only the newest file of each convergence folder (defect, cell size, folder)",
        )
        return
    parser.add_argument(
        "--order",
        nargs="+",
        type=int,
        choices=[1, 2, 12],
        help="Filter by perturbation order(s), e.g. --order 1 2 (12 is the legacy alias for 1 2)",
    )
    parser.add_argument(
        "--pert-scale", nargs="+", type=float, help="Filter by perturbation scale(s) (e.g. 0.1 0.025)"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="With --query, use only the most recent matching run of each configuration "
        "(defect, cell size, method, order, pert-scale)",
    )
    parser.add_argument(
        "-r", "--output-root", default="runs",
        help="Root folder holding run directories (default: runs); used to resolve runs queried from the registry",
    )


def _query_filters(args: argparse.Namespace) -> list[tuple[str, str]]:
    """--query flag values as (column, value) filters for registry.find_runs.

    Each value of a multi-valued flag becomes its own filter; find_runs ORs
    repeated fields, so ``--defect NV ClV`` matches runs of either defect.
    """
    out = []
    out += [("defect", d) for d in args.defect or []]
    out += [("cell", c) for c in args.cell_size or []]
    out += [("method", _METHODS[m]) for m in args.method or []]
    if args.order is not None:
        out.append(("order", parse_orders(args.order)))
    out += [("pert", p) for p in args.pert_scale or []]
    out += [("init_state", s) for s in getattr(args, "init_state", None) or []]
    return out


def _run_t1(args: argparse.Namespace) -> None:
    if args.query:
        dirs = _resolve_query_runs(args)
        missing = [d for d in dirs if not find_t1_files(d)]
        if missing:
            raise SystemExit(f"error: missing t1_relaxation*.npz in: {', '.join(str(m) for m in missing)}")
        inputs = [f for d in dirs for f in find_t1_files(d)]
    else:
        if not args.inputs:
            raise SystemExit("error: provide npz/run paths, or use --query to select registry runs")
        inputs = args.inputs
    if args.output:
        out = args.output
    else:
        run_dirs = dirs if args.query else [Path(p) if Path(p).is_dir() else Path(p).parent for p in inputs]
        name = "t1_rate" if args.rate else "t1"
        out = _figures_dir("t1") / _combined_figure_subdir(run_dirs) / f"{name}.{args.fmt}"
        out.parent.mkdir(parents=True, exist_ok=True)
    section(LOGGER, f"Plot {'1/T1' if args.rate else 'T1'}: {len(inputs)} file(s)")
    for path in inputs:
        LOGGER.debug(f"T1 data: {path}")
    plot_t1(
        inputs,
        output_path=out,
        show=args.show,
        plain_name=args.plain_name,
        init_states=args.init_state,
        rate=args.rate,
        log_temperature=args.log_temperature,
    )



