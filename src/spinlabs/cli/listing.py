"""``spinlabs list``: what the latest packaged data, runs and plots are, and when they were made.

One row per configuration (defect, cell, method, pert-scale), picked like every
``--latest`` query (most perturbation orders, then newest; ``registry.more_complete``).
Runs analysed from an older package than the newest one, and plots made from an older
run than the newest one, are marked so it is clear what needs rerunning.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from spinlabs.cli import add_verbosity_args
from spinlabs.registry import ensure_db, find_packaged, find_runs, stored_orders

FIGURE_SUFFIXES = (".png", ".pdf", ".svg", ".jpg", ".eps")


def build_list_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--package", action="store_true", help="Latest packaged data per configuration")
    parser.add_argument("--runs", action="store_true", help="Latest completed run per configuration")
    parser.add_argument("--plot", action="store_true", help="Plots made from the latest run of each configuration")
    parser.add_argument("--defect", nargs="+", help="Only these defect name(s) (e.g. NV ClV)")
    parser.add_argument("--cell-size", nargs="+", type=int, help="Only these supercell size(s)")
    parser.add_argument(
        "--figures-dir", type=Path, default=Path("figures"), help="Where the plot commands wrote (default: ./figures)"
    )
    add_verbosity_args(parser)
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    conn = ensure_db()
    show_all = not (args.package or args.runs or args.plot)
    sections = []
    if args.package or show_all:
        sections.append(package_table(conn, args.defect, args.cell_size))
    if args.runs or show_all:
        sections.append(runs_table(conn, args.defect, args.cell_size))
    if args.plot or show_all:
        sections.append(plots_table(conn, args.defect, args.cell_size, args.figures_dir))
    print("\n\n".join(sections))


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def local_time(stamp: str | None) -> str:
    """Registry UTC timestamp as local 'YYYY-MM-DD HH:MM'."""
    if not stamp:
        return "?"
    try:
        return datetime.fromisoformat(stamp).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return stamp


def packaged_time(path, registered: str | None) -> str:
    """When a package was written: the file's mtime (registering again rewrites registered_at)."""
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return local_time(registered)


def _config(defect, cell, method, pert) -> str:
    return f"{defect}_{cell}" if cell is not None else str(defect)


def _orders(raw) -> str:
    orders = sorted(stored_orders(raw))
    return "+".join(str(o) for o in orders) if orders else "?"


def _table(title: str, header: list[str], rows: list[list[str]], notes: list[str] = ()) -> str:
    if not rows:
        return f"{title}\n  (none)"
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    line = lambda cells: "  " + "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()  # noqa: E731
    out = [title, line(header), "  " + "  ".join("-" * w for w in widths)] + [line(r) for r in rows]
    return "\n".join(out + [f"  {n}" for n in notes])


def _sort_key(defect, cell, method, pert):
    return (str(defect), cell or 0, str(method), pert if pert is not None else -1.0)


def _wanted(defect, cell, defects, cells) -> bool:
    return (not defects or defect in defects) and (not cells or cell in cells)


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
def _latest_packages(conn) -> dict[tuple, tuple]:
    """(defect, cell, method, pert) -> (path, orders, registered_at, n_versions) of the latest package."""
    latest = {}
    for path in find_packaged(conn, latest=True):
        row = conn.execute(
            "SELECT defect, cell_size, calc_method, pert_scale, pert_order, registered_at FROM files "
            "WHERE path = ? AND kind = 'packaged_zfs'",
            (str(path),),
        ).fetchone()
        if row is None:
            continue
        defect, cell, method, pert, orders, registered = row
        n = conn.execute(
            "SELECT COUNT(*) FROM files WHERE kind = 'packaged_zfs' AND defect = ? AND cell_size = ? "
            "AND calc_method = ? AND pert_scale = ?",
            (defect, cell, method, pert),
        ).fetchone()[0]
        latest[(defect, cell, method, pert)] = (path, orders, registered, n)
    return latest


def package_table(conn, defects=None, cells=None) -> str:
    rows = []
    for key, (path, orders, registered, n) in sorted(_latest_packages(conn).items(), key=lambda kv: _sort_key(*kv[0])):
        defect, cell, method, pert = key
        if not _wanted(defect, cell, defects, cells):
            continue
        rows.append([
            _config(*key), str(method), f"{pert:g}", _orders(orders), packaged_time(path, registered),
            str(n - 1) if n > 1 else "", _short(path),
        ])
    return _table(
        "Packaged data (latest per configuration)",
        ["defect", "method", "pert", "orders", "packaged", "older", "file"],
        rows,
    )


def _latest_runs(conn) -> dict[tuple, tuple]:
    """(defect, cell, method, pert) -> (run_dir, orders, started_at, git_hash, zfs_files) of the latest run."""
    latest = {}
    for run_dir in find_runs(conn, latest=True):
        row = conn.execute(
            'SELECT defect, cell, method, pert, "order", started_at, git_hash, zfs_files FROM runs '
            "WHERE run_dir = ? AND status = 'done' ORDER BY id DESC LIMIT 1",
            (str(run_dir),),
        ).fetchone()
        if row is None or row[0] is None:
            continue
        defect, cell, method, pert, *rest = row
        latest[(defect, cell, method, pert)] = (run_dir, *rest)
    return latest


def runs_table(conn, defects=None, cells=None) -> str:
    packages = _latest_packages(conn)
    rows, stale = [], 0
    for key, (run_dir, orders, started, git_hash, zfs_files) in sorted(
        _latest_runs(conn).items(), key=lambda kv: _sort_key(*kv[0])
    ):
        defect, cell, method, pert = key
        if not _wanted(defect, cell, defects, cells):
            continue
        used = set(json.loads(zfs_files)) if zfs_files else set()
        newest = packages.get(key)
        if newest is None:
            note = ""
        elif str(Path(newest[0]).resolve()) in {str(Path(p).resolve()) for p in used}:
            note = "newest package"
        else:
            note = f"OLDER package (newest {packaged_time(newest[0], newest[2])})"
            stale += 1
        rows.append([
            _config(*key), str(method), f"{pert:g}" if pert is not None else "?", _orders(orders),
            local_time(started), git_hash or "?", note, Path(run_dir).name,
        ])
    notes = [f"{stale} run(s) analysed an older package: rerun with spinlabs run --query"] if stale else []
    return _table(
        "Runs (latest per configuration)",
        ["defect", "method", "pert", "orders", "started", "git", "data", "run"],
        rows,
        notes,
    )


def _figures_by_run(figures_dir: Path) -> dict[str, dict[str, tuple[int, float]]]:
    """Run folder name -> plot kind -> (number of figures, newest mtime), from the per-run figure layout."""
    found: dict[str, dict[str, list]] = {}
    if not figures_dir.is_dir():
        return {}
    for path in figures_dir.rglob("*"):
        if path.suffix.lower() not in FIGURE_SUFFIXES or not path.is_file():
            continue
        rel = path.relative_to(figures_dir)
        if len(rel.parts) < 3:
            continue  # loose figures (convergence, method comparison) belong to no run
        kind, run_name = rel.parts[0], rel.parts[-2]
        entry = found.setdefault(run_name, {}).setdefault(kind, [0, 0.0])
        entry[0] += 1
        entry[1] = max(entry[1], path.stat().st_mtime)
    return {run: {k: (n, t) for k, (n, t) in kinds.items()} for run, kinds in found.items()}


def plots_table(conn, defects=None, cells=None, figures_dir: Path = Path("figures")) -> str:
    figures = _figures_by_run(figures_dir)
    # every done run of a configuration, newest first, to tell which older run the plots belong to
    history: dict[tuple, list[str]] = {}
    for defect, cell, method, pert, run_dir in conn.execute(
        "SELECT defect, cell, method, pert, run_dir FROM runs WHERE status = 'done' ORDER BY started_at DESC, id DESC"
    ):
        if run_dir:
            history.setdefault((defect, cell, method, pert), []).append(Path(run_dir).name)
    rows, missing = [], 0
    for key, (run_dir, *_rest) in sorted(_latest_runs(conn).items(), key=lambda kv: _sort_key(*kv[0])):
        defect, cell, method, pert = key
        if not _wanted(defect, cell, defects, cells):
            continue
        name = Path(run_dir).name
        kinds = figures.get(name, {})
        if kinds:
            made = ", ".join(f"{k} ({n}, {datetime.fromtimestamp(t):%m-%d %H:%M})" for k, (n, t) in sorted(kinds.items()))
        else:
            missing += 1
            older = next((r for r in history.get(key, []) if r != name and r in figures), None)
            made = f"none; last plotted from older run {older}: {', '.join(sorted(figures[older]))}" if older else "none"
        rows.append([_config(*key), str(method), f"{pert:g}" if pert is not None else "?", name, made])
    notes = [f"{missing} latest run(s) without plots under {figures_dir}/"] if missing else []
    return _table(
        f"Plots of the latest runs (under {figures_dir}/; kind (figures, last written))",
        ["defect", "method", "pert", "run", "plots"],
        rows,
        notes,
    )


def _short(path) -> str:
    """Path relative to the working directory when it lies below it."""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)
