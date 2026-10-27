"""CLI subcommands for generating LaTeX tables from the registry."""

import argparse
from itertools import product
import sys
from pathlib import Path

from spinlabs.cli import add_verbosity_args
from spinlabs.logging_utils import get_logger
from spinlabs.registry import ensure_db, find_packaged, more_complete, parse_orders, stored_orders
from spinlabs.tables import (
    collect_d_rows,
    collect_t1_rows,
    format_t1_table,
    merge_rows,
    render_defect_table,
    run_packages,
)

LOGGER = get_logger(__name__)


def build_table_parser(subparsers: argparse._SubParsersAction) -> None:  # noqa: SLF001
    """Add the ``table`` subcommands to a subparsers object."""
    t1 = subparsers.add_parser(
        "t1",
        help="LaTeX table of the latest T1 per defect, method and pert-scale for each initial spin state",
    )
    _add_run_filter_args(t1)
    pert = t1.add_mutually_exclusive_group()
    pert.add_argument(
        "--pert-column", dest="show_pert", action="store_true", default=None,
        help="Always add a perturbation-scale column (default: only when the table spans several scales)",
    )
    pert.add_argument("--no-pert-column", dest="show_pert", action="store_false", help="Never add the perturbation-scale column")
    t1.add_argument("--label", default="tab:t1", help="LaTeX \\label (default: tab:t1)")
    _add_output_args(t1)
    t1.set_defaults(func=_run_t1)

    defects = subparsers.add_parser(
        "defects",
        help="LaTeX table of ground-state D and the latest T1 per initial state for each defect",
        description="Ground-state D (corrected/contaminated) per defect, with the latest "
        "T1 per initial spin state and method appended. Values of different supercells share a row "
        "and get a superscript letter explained in the caption.",
    )
    defects.add_argument(
        "inputs", nargs="*", type=Path,
        help="Defect folders (searched for packaged npz and ZFS_hyp/ZFS_occup OUTCARs) or packaged ZFS npz "
        "files to read D from (default: every packaged ZFS file in the registry)",
    )
    _add_run_filter_args(defects)
    defects.add_argument("--no-t1", action="store_true", help="Leave out the T1 columns (D only)")
    defects.add_argument("--label", default="tab:defects", help="LaTeX \\label (default: tab:defects)")
    _add_output_args(defects)
    defects.set_defaults(func=_run_defects)


def _add_run_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in K to read T1 at (default: 300)")
    parser.add_argument("--defect", nargs="+", help="Only these defect(s) (e.g. NV ClV)")
    parser.add_argument("--cell-size", nargs="+", type=int, help="Only these supercell size(s) (e.g. 64 512)")
    parser.add_argument(
        "--method",
        nargs="+",
        choices=["all", "approx"],
        help="Only 'all' (all_bands) and/or 'approx' (defect_band_approx) runs",
    )
    parser.add_argument(
        "--order", nargs="+", type=int, choices=[1, 2, 12], help="Only runs containing these perturbation orders"
    )
    parser.add_argument("--pert-scale", nargs="+", type=float, help="Only these perturbation scale(s) (e.g. 0.1 0.025)")


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-r", "--output-root", default="runs",
        help="Root folder holding run directories (default: runs); used to resolve runs stored from another host",
    )
    parser.add_argument("-o", "--output", type=Path, help="Write the table to this .tex file (default: print to stdout)")
    add_verbosity_args(parser)


_METHODS = {"all": "all_bands", "approx": "defect_band_approx"}


def _filters(args: argparse.Namespace) -> list[tuple[str, object]]:
    # One filter per value; find_runs ORs repeated fields.
    out: list[tuple[str, object]] = []
    out += [("defect", d) for d in args.defect or []]
    out += [("cell", c) for c in args.cell_size or []]
    out += [("method", _METHODS[m]) for m in args.method or []]
    if args.order is not None:
        out.append(("order", parse_orders(args.order)))
    out += [("pert", p) for p in args.pert_scale or []]
    return out


def _run_t1(args: argparse.Namespace) -> None:
    rows = collect_t1_rows(ensure_db(), args.temperature, _filters(args), output_root=args.output_root)
    if not rows:
        raise SystemExit("error: no completed runs with T1 data in the registry match the filters")
    _write(args, format_t1_table(rows, args.temperature, label=args.label, show_pert=args.show_pert))


def _run_defects(args: argparse.Namespace) -> None:
    for path in args.inputs:
        if not path.exists():
            raise SystemExit(f"error: input not found: {path}")
    conn = ensure_db()
    t1_rows = [] if args.no_t1 else collect_t1_rows(
        conn, args.temperature, _filters(args), output_root=args.output_root, by_pert=False
    )
    if args.inputs:
        sources = args.inputs
    else:
        # D from the package each chosen T1 run analysed, so D and T1 describe the same data
        # (an older package may hold an older ground state); configurations without a T1
        # run take the latest, most complete package.
        sources = run_packages(conn, t1_rows)
        for path in _latest_package_per_method(conn, args):
            if path not in sources:
                sources.append(path)
    d_rows = collect_d_rows(list(sources))
    if args.inputs:
        d_rows = [
            r for r in d_rows
            if (args.defect is None or r["defect"] in args.defect)
            and (args.cell_size is None or r["cell"] in {str(c) for c in args.cell_size})
        ]
    merged = merge_rows(d_rows, t1_rows)
    if not merged:
        raise SystemExit("error: no ZFS or T1 data found")
    _write(args, render_defect_table(merged, args.temperature, include_t1=not args.no_t1, label=args.label))


def _latest_package_per_method(conn, args: argparse.Namespace) -> list[Path]:
    """One package per defect/cell/method: the most orders, then the newest (registry.more_complete)."""
    best: dict[tuple, tuple[frozenset[int], Path, int]] = {}
    for defect, cell, method in product(args.defect or [None], args.cell_size or [None], args.method or [None]):
        for path in find_packaged(conn, defect=defect, cell_size=cell, calc_method=_METHODS.get(method), latest=True):
            row = conn.execute(
                "SELECT defect, cell_size, calc_method, pert_order, id FROM files WHERE path = ?", (str(path),)
            ).fetchone()
            if row is None:
                continue
            key, orders, rid = tuple(row[:3]), stored_orders(row[3]), row[4]
            previous = best.get(key)
            if previous is None or more_complete(orders, previous[0]) and (
                len(orders) > len(previous[0]) or rid > previous[2]
            ):
                best[key] = (orders, path, rid)
    return [path for _, path, _ in best.values()]


def _write(args: argparse.Namespace, tex: str) -> None:
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(tex)
        LOGGER.info(f"saved table: {args.output}")
    else:
        sys.stdout.write(tex)
