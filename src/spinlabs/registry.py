"""SQLite registry of files and analysis runs.

Lives at ~/.spinlabs/registry.db. Initialized lazily by ``spinlabs run`` /
``spinlabs package`` (and any other entry point that calls :func:`ensure_db`),
so there is nothing to set up by hand.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_env_dir = os.environ.get("SPINLABS_REGISTRY_DIR")
REGISTRY_DIR = Path(_env_dir) if _env_dir else Path.home() / ".spinlabs"
REGISTRY_PATH = REGISTRY_DIR / "registry.db"

def parse_orders(value: object) -> list[int]:
    """Perturbation orders named by ``value``, as a sorted list.

    Accepts a list of ints, a single int, or a string ('1', '1 2', '1,2', '1+2').
    The legacy scalar 12 means both orders. Anything other than 1, 2 or 12
    raises ValueError. This is the one place the 12 convention is expanded.
    """
    if isinstance(value, (list, tuple)):
        values = list(value)
    elif isinstance(value, int):
        values = [value]
    else:
        values = [v for v in str(value).replace(",", "+").replace(" ", "+").split("+") if v.strip()]
    out: set[int] = set()
    for v in values:
        n = int(v)
        if n == 12:
            out.update((1, 2))
        elif n in (1, 2):
            out.add(n)
        else:
            raise ValueError(f"unknown perturbation order {n}")
    return sorted(out)


def stored_orders(raw: object) -> frozenset[int]:
    """Orders of a stored ``pert_order``/``order`` value: a JSON list, or a legacy scalar (1, 2, 12, '1+2')."""
    if raw is None:
        return frozenset()
    try:
        items = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        items = raw
    try:
        return frozenset(parse_orders(items))
    except (TypeError, ValueError):
        return frozenset()


def more_complete(orders: frozenset[int], than: frozenset[int] | None) -> bool:
    """Whether data with ``orders`` should replace data with ``than`` when picking the latest.

    More perturbation orders is more complete data (a first-order-only package or run lacks the
    two-phonon channel), so it wins over newer data with fewer orders; between equally
    complete data the newer one wins, so callers go oldest first and replace on True.
    """
    return than is None or len(orders) >= len(than)


def _normalize_orders(value: object) -> str | None:
    """Normalize a perturbation order value to a JSON list string, e.g. '[1, 2]'.

    ``None`` and an empty selection are stored as NULL.
    """
    if value is None:
        return None
    orders = parse_orders(value)
    return json.dumps(orders) if orders else None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    kind TEXT NOT NULL,              -- raw_zfs | packaged_zfs | phonon | coupling | two_phonon | convergence
    defect TEXT,
    cell_size INTEGER,
    calc_method TEXT,
    pert_order TEXT,                 -- JSON list of orders, e.g. '[1, 2]' for combined data
    pert_scale REAL,
    sha256 TEXT,
    registered_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    run_dir TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',   -- running | done | failed
    zfs_files TEXT,                           -- JSON list of file paths
    phonon_file TEXT,
    method TEXT,
    init_state TEXT,                          -- JSON list of initial states, e.g. '["ms_1", "ms_0", "ms_-1"]'
    command TEXT,
    defect TEXT,
    cell INTEGER,
    "order" TEXT,                             -- JSON list of orders present in the data, e.g. '[1, 2]'
    pert REAL,
    temp_min REAL,
    temp_max REAL,
    n_temps INTEGER,
    npz_files TEXT,                           -- JSON list of npz names
    git_hash TEXT
);
CREATE INDEX IF NOT EXISTS idx_files_defect ON files(defect, cell_size, calc_method);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE TABLE IF NOT EXISTS run_logs (
    id INTEGER PRIMARY KEY,
    run_id INTEGER,                  -- runs.id once the run is registered; NULL before
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    logger TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_run_logs_run ON run_logs(run_id);
"""


def ensure_db() -> sqlite3.Connection:
    """Open (creating if needed) the registry database."""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(REGISTRY_PATH)
    conn.executescript(_SCHEMA)
    # Old rows: combined files had pert_order=NULL (before the 12 convention),
    # then the integer 12. Legacy integer 1/2 rows are left alone — they were
    # registered as single-order data and rewriting them to [1, 2] would be wrong.
    cur = conn.execute(
        "SELECT COUNT(*) FROM files WHERE kind = 'packaged_zfs' AND (pert_order IS NULL OR pert_order = '12')"
    )
    if cur.fetchone()[0]:
        conn.execute(
            "UPDATE files SET pert_order = '[1, 2]' WHERE kind = 'packaged_zfs' AND (pert_order IS NULL OR pert_order = '12')",
        )
        conn.commit()
    # Migration: registrys created before the runs table grew the analysis-
    # index columns (replacing runs_index.csv) need the columns added.
    existing = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    for column in (
        "defect", "cell", '"order"', "pert",
        "temp_min", "temp_max", "n_temps", "npz_files", "git_hash",
    ):
        if column.strip('"') not in existing:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {column}")
    # Migration: ``order`` used to be a scalar int (1, 2 or 12) and ``n_dims``
    # carried the actual data content as a JSON list. The two described the
    # same thing; they are consolidated into one list column. Old scalar
    # values become lists (12 -> '[1, 2]'), NULLs are backfilled from
    # ``n_dims``, and ``n_dims`` is dropped.
    if "n_dims" in existing:
        for run_id, order_val, n_dims_val in conn.execute(
            'SELECT id, "order", n_dims FROM runs'
        ):
            if isinstance(order_val, int):
                orders = [1, 2] if order_val == 12 else [order_val]
            elif isinstance(order_val, str):
                try:
                    parsed = json.loads(order_val)
                except (TypeError, json.JSONDecodeError):
                    parsed = None
                orders = parsed if isinstance(parsed, list) else None
            else:
                orders = None
            if orders is None and isinstance(n_dims_val, str):
                try:
                    parsed = json.loads(n_dims_val)
                except (TypeError, json.JSONDecodeError):
                    parsed = None
                orders = parsed if isinstance(parsed, list) else None
            if orders is not None:
                orders = sorted(set(int(o) for o in orders if o in (1, 2)))
                conn.execute(
                    'UPDATE runs SET "order" = ? WHERE id = ?',
                    (json.dumps(orders) if orders else None, run_id),
                )
        # DROP COLUMN needs SQLite >= 3.35 (2021-03); older libraries raise
        # OperationalError here.
        conn.execute('ALTER TABLE runs DROP COLUMN n_dims')
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def register_file(
    conn: sqlite3.Connection,
    path: str | Path,
    kind: str,
    *,
    defect: str | None = None,
    cell_size: int | None = None,
    calc_method: str | None = None,
    pert_order: list[int] | int | None = None,
    pert_scale: float | None = None,
) -> int:
    """Insert or update a file row (re-registering overwrites metadata and hash).

    ``pert_order`` is normalized to a JSON list (e.g. ``[1, 2]`` for combined data).
    """
    path = str(Path(path).resolve())
    digest = sha256_of(path) if Path(path).exists() else None
    orders = _normalize_orders(pert_order)
    conn.execute(
        """INSERT INTO files (path, kind, defect, cell_size, calc_method, pert_order, pert_scale, sha256, registered_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(path) DO UPDATE SET
             kind=excluded.kind, defect=excluded.defect, cell_size=excluded.cell_size,
             calc_method=excluded.calc_method, pert_order=excluded.pert_order,
             pert_scale=excluded.pert_scale, sha256=excluded.sha256,
             registered_at=excluded.registered_at""",
        (
            path,
            kind,
            defect,
            cell_size,
            calc_method,
            orders,
            pert_scale,
            digest,
            _now(),
        ),
    )
    conn.commit()
    cur = conn.execute("SELECT id FROM files WHERE path = ?", (path,))
    return int(cur.fetchone()[0])


def insert_run_log(
    conn: sqlite3.Connection,
    level: str,
    logger_name: str,
    message: str,
    run_id: int | None = None,
) -> None:
    """Append one log record to the ``run_logs`` table."""
    conn.execute(
        "INSERT INTO run_logs (run_id, ts, level, logger, message) VALUES (?, ?, ?, ?, ?)",
        (run_id, _now(), level, logger_name, message),
    )
    conn.commit()


def list_run_logs(conn: sqlite3.Connection, run_id: int | None = None) -> list[dict]:
    """Return log records from ``run_logs`` (optionally for one run), oldest first."""
    if run_id is None:
        cur = conn.execute(
            "SELECT id, run_id, ts, level, logger, message FROM run_logs ORDER BY id"
        )
    else:
        cur = conn.execute(
            "SELECT id, run_id, ts, level, logger, message FROM run_logs"
            " WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def register_run(
    conn: sqlite3.Connection,
    *,
    run_dir: str | Path | None = None,
    zfs_files: list[str | Path] | None = None,
    phonon_file: str | Path | None = None,
    method: str | None = None,
    init_state: str | Sequence[str] | None = None,
    command: str | None = None,
    defect: str | None = None,
    cell: int | None = None,
    order: list[int] | int | None = None,
    pert: float | None = None,
    temp_min: float | None = None,
    temp_max: float | None = None,
    n_temps: int | None = None,
    npz_files: Sequence[str | Path] = (),
    git_hash: str | None = None,
) -> int:
    """Start a run row; returns its id for :func:`finish_run`.

    The analysis metadata (defect, cell, order, pert, temperatures, outputs)
    replaces what used to be appended to ``runs_index.csv``; it makes the
    ``runs`` table the single source of truth for analysis runs.

    ``order`` is the perturbation orders actually present in the analyzed
    data, normalized to a JSON list (e.g. ``[1, 2]`` for combined data).
    ``init_state`` is the initial state(s) whose dynamics the run solved,
    stored as a JSON list (one run directory holds every state).
    """
    if isinstance(init_state, str):
        init_state = [init_state]
    conn.execute(
        """INSERT INTO runs (run_dir, started_at, status, zfs_files, phonon_file, method, init_state, command,
                            defect, cell, "order", pert, temp_min, temp_max, n_temps, npz_files, git_hash)
           VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(Path(run_dir).resolve()) if run_dir else None,
            _now(),
            json.dumps([str(Path(p).resolve()) for p in (zfs_files or [])]),
            str(Path(phonon_file).resolve()) if phonon_file else None,
            method,
            json.dumps(list(init_state)) if init_state else None,
            command,
            defect,
            cell,
            _normalize_orders(order),
            pert,
            temp_min,
            temp_max,
            n_temps,
            json.dumps([Path(p).name for p in npz_files]) if npz_files else None,
            git_hash,
        ),
    )
    conn.commit()
    return int(conn.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1").fetchone()[0])


def finish_run(conn: sqlite3.Connection, run_id: int, status: str) -> None:
    if status not in ("done", "failed"):
        raise ValueError(f"status must be 'done' or 'failed', got {status!r}")
    conn.execute(
        "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?", (status, _now(), run_id)
    )
    conn.commit()


def unregistered_packages(conn: sqlite3.Connection, data_roots: Iterable[str | Path] = ("data",)) -> list[Path]:
    """Packaged raw ZFS files on disk that the registry does not know.

    ``--query`` and ``--latest`` only see registered files, so a package copied in
    without ``spinlabs package --register-only`` is silently ignored. Looks in
    ``data_roots`` and in every folder that already holds registered packages
    (``<root>/<defect>_<cell>/pert_<x>/<method>/<date>/*raw_zfs_data*.npz``).
    """
    known = set()
    roots = {Path(r).resolve() for r in data_roots if Path(r).is_dir()}
    for (path,) in conn.execute("SELECT path FROM files WHERE kind = 'packaged_zfs'"):
        p = Path(path)
        known.add(str(p.resolve()) if p.exists() else str(p))
        if len(p.parents) > 4 and p.parents[4].is_dir():
            roots.add(p.parents[4].resolve())
    found = []
    for root in sorted(roots):
        for npz in sorted(root.glob("*/pert_*/*/*/*raw_zfs_data*.npz")):
            if str(npz.resolve()) not in known:
                found.append(npz)
    return found


def unregistered_warning(paths: list[Path]) -> str | None:
    """One message naming the unregistered packages and the command that registers them."""
    if not paths:
        return None
    defects = sorted({p.parents[3] for p in paths})
    shown = ", ".join(str(p) for p in paths[:3]) + (", ..." if len(paths) > 3 else "")
    commands = "; ".join(f"spinlabs package --register-only {d}" for d in defects)
    return (
        f"{len(paths)} packaged file(s) on disk are not in the registry, so --query and --latest "
        f"ignore them: {shown}. Register them with: {commands}"
    )


def find_packaged(
    conn: sqlite3.Connection,
    *,
    defect: str | None = None,
    cell_size: int | str | None = None,
    calc_method: str | None = None,
    pert_scale: float | None = None,
    pert_order: list[int] | int | None = None,
    latest: bool = False,
) -> list[Path]:
    """Packaged ZFS files matching the filters.

    With ``latest`` one package per configuration (defect, cell, method,
    pert-scale) is kept: the one with the most perturbation orders, and among
    those the newest (:func:`more_complete`), so repackaged data replaces older
    versions and a first-order-only package never hides one with second-order
    data. Newest is the latest first registration (highest id): re-registering
    a file keeps its id but rewrites ``registered_at``.
    """
    sql = (
        "SELECT path, pert_order, defect, cell_size, calc_method, pert_scale FROM files"
        " WHERE kind = 'packaged_zfs'"
    )
    params: list[object] = []
    if defect is not None:
        sql += " AND defect = ?"
        params.append(defect)
    if cell_size is not None:
        sql += " AND cell_size = ?"
        params.append(int(cell_size))
    if calc_method is not None:
        sql += " AND calc_method = ?"
        params.append(calc_method)
    if pert_scale is not None:
        sql += " AND pert_scale = ?"
        params.append(pert_scale)
    if pert_order is not None:
        wanted = set(parse_orders(pert_order))
    sql += " ORDER BY id"
    cur = conn.execute(sql, params)  # type: ignore[arg-type]
    results = []
    newest: dict[tuple, tuple[frozenset[int], Path]] = {}
    for p_, stored, *config in cur.fetchall():
        if pert_order is not None:
            # Exact set membership on the stored JSON list, not substring matching.
            try:
                have = set(json.loads(stored)) if stored else set()
            except (TypeError, json.JSONDecodeError):
                have = set()
            if not wanted.issubset(have):
                continue
        if not Path(p_).exists():
            continue
        if latest:
            # Rows are oldest first; a later package of the same configuration replaces it
            # unless it has fewer orders.
            orders = stored_orders(stored)
            key = tuple(config)
            if more_complete(orders, newest[key][0] if key in newest else None):
                newest.pop(key, None)
                newest[key] = (orders, Path(p_))
        else:
            results.append(Path(p_))
    return [path for _, path in newest.values()] if latest else results


def find_runs(
    conn: sqlite3.Connection,
    filters: Iterable[tuple[str, str]] = (),
    *,
    latest: bool = False,
    output_root: str | Path | None = None,
) -> list[Path]:
    """Run directories of completed runs matching ``filters``.

    Replaces ``runs_index.csv``: the ``runs`` table is the index. Filter
    fields are the ``runs`` table columns (``defect``, ``cell``, ``order``,
    ``pert``, ``temp_min``, ``temp_max``, ``n_temps``,
    ``npz_files``, ``method``, ``init_state``, ``git_hash``, ``run_dir``,
    plus ``timestamp`` as an alias for ``started_at``); unknown fields raise
    SystemExit so typos fail loudly. Values match like the old CSV index:
    exact and case-insensitive, with numeric fields compared numerically and
    list fields (``npz_files`` per name, ``init_state`` per state; legacy
    scalar ``init_state`` values count as one-element lists) matching any element. ``order`` is a
    JSON list of the orders present in the data (e.g. ``'[1, 2]'``); a filter
    matches when every requested order is present (``--order 1`` matches
    ``[1]`` and ``[1, 2]`` runs, ``--order 1 2`` matches only combined runs).

    With ``latest`` one run per configuration (defect, cell, method, pert) is
    returned: the one with the most perturbation orders, and among those the
    most recent (by ``started_at``; :func:`more_complete`), so a query spanning
    several configurations yields one run of each and a first-order-only run
    never hides one with second-order data. Directories stored from another host are resolved against
    ``output_root`` by name when the stored path does not exist.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    numeric = {"cell", "pert", "temp_min", "temp_max", "n_temps"}
    list_like = {"npz_files", "init_state"}
    # Repeating a field ORs its values: [("defect", "NV"), ("defect", "ClV")]
    # matches runs of either defect; distinct fields are ANDed.
    by_column: dict[str, list[object]] = {}
    for field, value in filters:
        column = "started_at" if field == "timestamp" else field
        if column not in columns | {"timestamp"}:
            raise SystemExit(
                f"error: unknown run filter '{field}'; valid fields: "
                + ", ".join(sorted(columns | {"timestamp"}))
            )
        by_column.setdefault(column, []).append(value)

    def clause(column: str, value: object) -> tuple[str, list[object]]:
        if column in list_like:
            # Matches when any stored element equals the value (case-insensitive).
            wanted = str(value).strip().lower()
            keep = []
            for (raw,) in conn.execute(
                f'SELECT DISTINCT "{column}" FROM runs WHERE status = \'done\''
            ):
                try:
                    items = json.loads(raw) if isinstance(raw, str) else raw
                except (TypeError, json.JSONDecodeError):
                    items = [raw]  # legacy scalar, e.g. init_state 'ms_0'
                if not isinstance(items, list):
                    items = [items]
                if wanted in [str(i).lower() for i in items]:
                    keep.append(raw)
            if not keep:
                return "0", []  # nothing matches this filter
            return f'"{column}" IN ({", ".join("?" * len(keep))})', keep
        if column == "order":
            # Subset semantics: every requested order must be present in the
            # run's data, mirroring find_packaged(pert_order=...).
            try:
                wanted_orders = parse_orders(value)
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"error: invalid --order value {value!r}: {exc}")
            if not wanted_orders:
                raise SystemExit("error: empty --order filter; give at least one of 1, 2")
            keep = []
            for (raw,) in conn.execute(
                f'SELECT DISTINCT "{column}" FROM runs WHERE status = \'done\''
            ):
                try:
                    have = set(json.loads(raw)) if isinstance(raw, str) else set()
                except (TypeError, json.JSONDecodeError):
                    # Legacy scalar convention: 1, 2, 12 or '1+2'.
                    text = str(raw).replace("+", " ")
                    have = {int(v) for v in text.split() if v.strip().isdigit() and int(v) in (1, 2)}
                if set(wanted_orders).issubset(have):
                    keep.append(raw)
            if not keep:
                return "0", []
            return f'"{column}" IN ({", ".join("?" * len(keep))})', keep
        if column in numeric:
            try:
                number = float(value)
                if number.is_integer() and column in ("cell", "n_temps"):
                    number = int(number)
                value = number
            except ValueError:
                pass
        return f'"{column}" = ?', [value]

    group_columns = [c for c in ("defect", "cell", "method", "pert") if c in columns]
    select = ", ".join(['"run_dir"', '"order"'] + [f'"{c}"' for c in group_columns])
    sql = f"SELECT {select} FROM runs WHERE status = 'done'"
    params: list[object] = []
    for column, values in by_column.items():
        parts = [clause(column, v) for v in values]
        sql += " AND (" + " OR ".join(c for c, _ in parts) + ")"
        params.extend(p for _, ps in parts for p in ps)
    sql += " ORDER BY started_at, id"
    rows = conn.execute(sql, params).fetchall()
    if latest:
        # Rows are oldest first, so the last row per configuration is the
        # newest (ties broken by the higher, later-registered id). The pop
        # moves a configuration to the end on each newer run, so the result
        # stays ordered by each configuration's newest run.
        newest: dict[tuple, tuple] = {}
        for row in rows:
            key, orders = row[2:], stored_orders(row[1])
            if more_complete(orders, stored_orders(newest[key][1]) if key in newest else None):
                newest.pop(key, None)
                newest[key] = row
        rows = list(newest.values())
    results: list[Path] = []
    root = Path(output_root).resolve() if output_root else None
    for run_dir, *_ in rows:
        if not run_dir:
            continue
        path = Path(run_dir)
        if not path.exists() and root is not None:
            candidate = root / path.name
            if candidate.exists():
                path = candidate
        results.append(path)
    return results


def find_convergence(
    conn: sqlite3.Connection,
    *,
    defects: Iterable[str] = (),
    cell_sizes: Iterable[int | str] = (),
    calc_methods: Iterable[str] = (),
    latest: bool = False,
) -> list[Path]:
    """Packaged convergence files ('package --conv') matching the filters, oldest first.

    Several values of one filter match any of them. With ``latest`` one file per
    convergence folder is kept, the one with the newest date in its name: files are named
    ``<defect>_<cell>_<date>_conv_<folder suffix>.npz``, so the folder is the
    configuration plus the suffix (two folders of one method, e.g. ``test_all`` and
    ``test_all_mass_weighted``, are kept apart).
    """
    sql = "SELECT path, defect, cell_size, calc_method FROM files WHERE kind = 'convergence'"
    params: list[object] = []
    for column, values in (("defect", list(defects)), ("cell_size", [int(c) for c in cell_sizes]),
                           ("calc_method", list(calc_methods))):
        if values:
            sql += f" AND {column} IN ({', '.join('?' * len(values))})"
            params.extend(values)
    rows = [(Path(p), config) for p, *config in conn.execute(sql + " ORDER BY id", params) if Path(p).exists()]
    if not latest:
        return [path for path, _ in rows]
    # Newest by the packaging date in the name; registration order says little, since
    # --register-only registers a whole data folder at once. Stable sort: same date -> id order.
    rows.sort(key=lambda row: re.findall(r"_(\d{8})_conv_", row[0].name)[:1])
    newest: dict[tuple, Path] = {}
    for path, config in rows:
        key = (*config, path.stem.partition("_conv_")[2])
        newest.pop(key, None)  # re-insert so the result stays in order of the newest files
        newest[key] = path
    return list(newest.values())


def find_phonon(
    conn: sqlite3.Connection, defect: str, cell_size: int | str | None
) -> list[Path]:
    """Phonon files registered for this defect/cell that still exist on disk."""
    cur = conn.execute(
        "SELECT path FROM files WHERE kind = 'phonon' AND defect = ? AND cell_size = ? ORDER BY id",
        (defect, int(cell_size) if cell_size is not None else None),
    )
    return [Path(p) for (p,) in cur.fetchall() if Path(p).exists()]
