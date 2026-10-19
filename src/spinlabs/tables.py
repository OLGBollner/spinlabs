"""LaTeX tables built from the analysis runs in the registry."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spinlabs.logging_utils import get_logger
from spinlabs.registry import find_runs
from spinlabs.relaxation_dynamics import INIT_STATES, find_t1_files

LOGGER = get_logger(__name__)

# Significant figures of every number in the tables.
SIG_FIGS = 3
_INIT_STATE_HEADERS = {"ms_1": "$m_s=+1$", "ms_0": "$m_s=0$", "ms_-1": "$m_s=-1$"}
_METHOD_SUFFIX = {"all_bands": "", "defect_band_approx": " (spin corrected)"}
# all_bands before the spin-corrected row of the same defect.
_METHOD_RANK = {"all_bands": 0, "defect_band_approx": 1}
_PERT_RE = re.compile(r"(?<![A-Za-z])pert_([0-9.]+)")


@dataclass
class T1Row:
    """Latest T1 (s) per initial state for one defect/cell/method/pert-scale."""

    defect: str
    cell: int | None
    method: str
    pert: float | None
    t1: dict[str, float] = field(default_factory=dict)
    run_dirs: dict[str, Path] = field(default_factory=dict)

    @property
    def label(self) -> str:
        head = f"{self.defect} {self.cell}" if self.cell is not None else self.defect
        return head + _METHOD_SUFFIX.get(self.method, f" ({self.method.replace('_', ' ')})")

    def sort_key(self) -> tuple:
        # Largest supercell first within a defect, as in the reference table.
        return (
            self.defect,
            -(self.cell or 0),
            self.pert if self.pert is not None else float("inf"),
            _METHOD_RANK.get(self.method, 99),
            self.method,
        )


def t1_at(npz_path: str | Path, temperature: float) -> float:
    """ODE-fit T1 (s) at ``temperature``, interpolated linearly in log T1.

    Returns NaN when the temperature lies outside the run's grid or there is
    no valid T1 to interpolate.
    """
    data = np.load(str(npz_path), allow_pickle=True)
    temps = np.asarray(data["temperatures"], dtype=float)
    t1 = np.asarray(data["t1_fit"], dtype=float)
    valid = np.isfinite(t1) & (t1 > 0)
    temps, t1 = temps[valid], t1[valid]
    if temps.size == 0 or not temps.min() <= temperature <= temps.max():
        return float("nan")
    order = np.argsort(temps)
    return float(np.exp(np.interp(temperature, temps[order], np.log(t1[order]))))


def _npz_str(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in data.files:
        return None
    value = str(data[key])
    return value or None


def _single_state(run_states: str | None, npz: Path) -> str | None:
    """The registry's initial state for a T1 file lacking one, if unambiguous.

    The registry stores a JSON list (legacy rows: one state as a plain string).
    A run that solved several states cannot tell which one the file holds.
    """
    if not run_states:
        return None
    try:
        states = json.loads(run_states)
    except json.JSONDecodeError:
        return run_states
    if isinstance(states, str):
        return states
    if len(states) == 1:
        return states[0]
    LOGGER.warning(f"{npz.name} has no init_state and its run solved {', '.join(states)}; cannot tell which")
    return None


def collect_t1_rows(
    conn: sqlite3.Connection,
    temperature: float = 300.0,
    filters: list[tuple[str, object]] | None = None,
    output_root: str | Path | None = None,
    by_pert: bool = True,
) -> list[T1Row]:
    """Latest completed run per defect, cell, method, pert-scale and initial state.

    With ``by_pert=False`` runs of all pert-scales share one row per
    defect/cell/method (the row's ``pert`` is then ``None``). A run with
    second-order data is preferred over a first-order-only one, then the
    newest wins; rows that had runs at several pert-scales or order sets to
    choose from are reported with the one used.

    Runs are those matching ``filters`` (same fields as :func:`find_runs`).
    Each run directory holds one T1 file per initial state it solved
    (:func:`find_t1_files`). Older registry rows lack defect/cell/pert; those
    fall back to the T1 file metadata and a ``pert_<scale>`` path component.
    Runs without a T1 file or metadata to place them are skipped with a warning.
    """
    rows: dict[tuple, T1Row] = {}
    chosen: dict[tuple, tuple] = {}  # (row key, state) -> (has 2nd order, pert, orders) of the run used
    seen: dict[tuple, set] = {}  # row key -> {(pert, orders)} of every candidate run
    # find_runs yields oldest first, so later runs overwrite earlier ones.
    run_dirs = find_runs(conn, filters or [], output_root=output_root)
    meta = {
        row[0]: row[1:]
        for row in conn.execute(
            'SELECT run_dir, defect, cell, method, pert, init_state, "order" FROM runs WHERE status = \'done\''
        )
    }
    by_name = {Path(k).name: v for k, v in meta.items() if k}
    for run_dir in run_dirs:
        t1_files = find_t1_files(run_dir)
        if not t1_files:
            LOGGER.warning(f"Skipping {run_dir}: no t1_relaxation*.npz")
            continue
        run_defect, run_cell, run_method, pert, run_states, run_orders = meta.get(str(run_dir)) or by_name.get(
            run_dir.name, (None,) * 6
        )
        if pert is None:
            match = _PERT_RE.search(str(run_dir))
            pert = float(match.group(1).rstrip(".")) if match else None
        run_pert = float(pert) if pert is not None else None
        row_pert = run_pert if by_pert else None
        for npz in t1_files:
            data = np.load(str(npz), allow_pickle=True)
            defect = run_defect or _npz_str(data, "defect")
            cell = run_cell if run_cell is not None else _npz_str(data, "cell_size")
            method = run_method or _npz_str(data, "calc_method")
            init_state = _npz_str(data, "init_state") or _single_state(run_states, npz)
            if not (defect and method and init_state):
                LOGGER.warning(f"Skipping {npz}: missing defect/method/initial state metadata")
                continue
            cell = int(cell) if cell is not None else None
            orders = _run_orders(run_orders, data)
            complete = 2 in orders
            key = (defect, cell, method, row_pert)
            seen.setdefault(key, set()).add((run_pert, orders))
            # A first-order-only run lacks the two-phonon channel, so it never
            # replaces a run with second-order data; otherwise the newer run wins.
            previous = chosen.get((key, init_state))
            if previous is not None and previous[0] and not complete:
                continue
            chosen[(key, init_state)] = (complete, run_pert, orders)
            row = rows.setdefault(key, T1Row(defect, cell, method, row_pert))
            row.t1[init_state] = t1_at(npz, temperature)
            row.run_dirs[init_state] = run_dir
    for key, configs in seen.items():
        if len(configs) > 1 and key in rows:
            used = {chosen[(key, state)][1:] for state in rows[key].t1}
            LOGGER.warning(
                f"{rows[key].label}: runs at {_fmt_configs(configs)}; using {_fmt_configs(used)}"
            )
    return sorted(rows.values(), key=T1Row.sort_key)


def _run_orders(registry_orders: str | None, data: np.lib.npyio.NpzFile) -> tuple[int, ...]:
    """Perturbation orders a run analyzed: the registry's JSON list, else the T1 file's ``order``."""
    try:
        orders = json.loads(registry_orders) if registry_orders else None
    except json.JSONDecodeError:
        orders = None
    if orders is None and "order" in data.files:
        orders = np.atleast_1d(data["order"]).tolist()
    return tuple(sorted(int(o) for o in orders or []))


def _fmt_configs(configs: set[tuple]) -> str:
    """'pert 0.025 orders [1], pert 0.1 orders [1, 2]' for a set of (pert, orders)."""
    ordered = sorted(configs, key=lambda c: (c[0] is None, c[0] or 0.0, c[1]))
    return ", ".join(f"pert {'?' if p is None else f'{p:g}'} orders {list(o)}" for p, o in ordered)


def _fmt_ms(seconds: float | None) -> str:
    """T1 in ms with :data:`SIG_FIGS` significant figures (e.g. 1350, 92.1, 7.53, 13.0)."""
    if seconds is None or not np.isfinite(seconds):
        return "---"
    return _fmt_sig(seconds * 1e3)


def _fmt_pert(pert: float | None) -> str:
    return "---" if pert is None else f"{pert:g}"


def format_t1_table(
    rows: list[T1Row],
    temperature: float = 300.0,
    label: str = "tab:t1",
    show_pert: bool | None = None,
) -> str:
    """LaTeX table of T1 (ms) per initial state, one row per defect/method/pert.

    A perturbation-scale column is added when the rows span more than one
    scale (``show_pert=None``), otherwise the scale goes in the caption.
    Requires ``\\usepackage{multirow}``.
    """
    perts = sorted({r.pert for r in rows}, key=lambda p: (p is None, p))
    if show_pert is None:
        show_pert = len(perts) > 1
    n_lead = 2 if show_pert else 1
    cols = "c|" + ("c|" if show_pert else "") + "ccc"
    lead = r"\multirow{2}{*}{Defect}" + (r" & \multirow{2}{*}{Pert.\ scale}" if show_pert else "")
    states = " & ".join(_INIT_STATE_HEADERS[s] for s in INIT_STATES)

    caption = f"Spin-lattice relaxation time $T_1$ at $T = {temperature:g}$~K for each initial spin state."
    if not show_pert and perts and perts[0] is not None:
        caption += f" Perturbation scale {_fmt_pert(perts[0])}."

    lines = [
        "% generated by spinlabs table t1",
        r"% requires \usepackage{multirow}",
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \begin{{tabular}}{{{cols}}}",
        rf"    {lead} & \multicolumn{{3}}{{c}}{{$T_1$ (ms)}} \\",
        rf"    {' & ' * (n_lead - 1)} & {states} \\",
        r"    \hline",
    ]
    for row in rows:
        cells = [row.label]
        if show_pert:
            cells.append(_fmt_pert(row.pert))
        cells.extend(_fmt_ms(row.t1.get(s)) for s in INIT_STATES)
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [
        r"  \end{tabular}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ── Defect table: ground-state D and T1 per defect ───────────────────────────

# calc_method -> table column for D and T1.
METHOD_COLUMN = {
    "all_bands": "contaminated",  # ZFS_hyp: all-electron bands, no charge correction applied
    "defect_band_approx": "corrected",  # ZFS_occup: defect-band approximation, charge-corrected
}
COLUMNS = ("corrected", "contaminated")

# Values from different sources (OUTCAR vs npz, 1d vs 2d npz) closer than this
# are the same ground state; anything further apart is reported.
CONFLICT_TOL_MHZ = 0.01

# Base crystal per defect family. NV sits in diamond; the rest of the
# defects studied here are point defects in the 4H polytype of SiC.
BASE_CRYSTAL = {"NV": "Diamond"}
DEFAULT_BASE_CRYSTAL = "4H-SiC"


def base_crystal_of(defect: str) -> str:
    return BASE_CRYSTAL.get(defect.upper(), DEFAULT_BASE_CRYSTAL)


def infer_cell(path: Path) -> str:
    """Pull the supercell size (e.g. '512' in NV_512_k) out of a path, or '' if absent."""
    for part in reversed(path.parts):  # innermost folder first, never the user's home path
        match = re.search(r"(?<![\dA-Za-z])(1?\d{2,3})(?![\dA-Za-z])", part)
        if match:
            return match.group(1)
    return ""


def infer_defect(folder: Path) -> str:
    """Defect name from a folder like NV_512, V_Si_128 or NV_512_kk (everything before the cell size)."""
    match = re.fullmatch(r"(.+?)_\d+(?:_.*)?", folder.name)
    return match.group(1) if match else folder.name


def _method_column(npz_path: Path, calc_method: str | None = None) -> str | None:
    """Table column from the stored calc_method, else from the filename."""
    from spinlabs.models import normalize_calc_method

    stored = normalize_calc_method(calc_method)
    if stored in METHOD_COLUMN:
        return METHOD_COLUMN[stored]
    for fragment, column in METHOD_COLUMN.items():
        if fragment in npz_path.name:
            return column
    return None


def collect_d_rows(paths: list[Path]) -> list[dict]:
    """One ground-state D (MHz) per data source.

    ``paths`` are packaged ZFS npz files and/or folders, which are searched
    for packaged npz files (``*_raw_zfs_data_*.npz``) and relaxed-structure
    ground states (``ZFS_hyp/OUTCAR``, ``ZFS_occup/OUTCAR``).
    """
    from spinlabs.models import RawZFSData

    rows: list[dict] = []
    for path in paths:
        if path.is_dir():
            rows += collect_outcar_rows(path)
        npz_files = sorted(path.rglob("*_raw_zfs_data_*.npz")) if path.is_dir() else [path]
        for npz_path in npz_files:
            try:
                data = RawZFSData.load(str(npz_path))
            except Exception as error:
                LOGGER.warning(
                    f"Cannot load {npz_path} ({error.__class__.__name__}, possibly a git-lfs pointer stub), skipping"
                )
                continue
            zfs = data.ground_state_zfs
            if zfs is None:
                LOGGER.warning(f"No ground-state ZFS in {npz_path}, skipping")
                continue
            column = _method_column(npz_path, data.calc_method)
            if column is None:
                LOGGER.warning(f"Unknown method in {npz_path.name}, skipping")
                continue
            rows.append({
                "defect": data.defect or infer_defect(npz_path.parent),
                "column": column,
                "cell": str(data.cell_size) if data.cell_size else (infer_cell(npz_path.parent) or infer_cell(path)),
                "d_mhz": zfs.to_unit("MHz").D,
                "source": npz_path,
            })
    return rows


def collect_outcar_rows(folder: Path) -> list[dict]:
    """Read ground-state ZFS straight from relaxed-structure folders.

    ``folder`` is either the defect folder itself (containing ZFS_hyp/ and
    ZFS_occup/) or a parent folder containing defect folders; ZFS_* dirs are
    found recursively. ZFS_hyp/OUTCAR is the contaminated (all-electron bands)
    ground state, ZFS_occup/OUTCAR the corrected one (defect-band approx),
    which gets the same spin-correction scaling as RawZFSData.load applies
    to packaged approx data.
    """
    from spinlabs.models import SPIN_CORRECTION_SCALE, apply_spin_correction
    from spinlabs.parsers import parse_outcar_zfs

    rows: list[dict] = []
    for zfs_dir in sorted(folder.rglob("ZFS_hyp")) + sorted(folder.rglob("ZFS_occup")):
        defect_root = zfs_dir.parent
        corrected = zfs_dir.name == "ZFS_occup"
        outcar = zfs_dir / "OUTCAR"
        if not outcar.is_file():
            continue
        try:
            zfs = parse_outcar_zfs(outcar)
        except Exception as error:
            LOGGER.warning(f"Cannot parse {outcar} ({error.__class__.__name__}), skipping")
            continue
        if zfs is None:
            LOGGER.warning(f"No ZFS block in {outcar}, skipping")
            continue
        if corrected:
            LOGGER.debug(f"Applied spin correction scaling: {SPIN_CORRECTION_SCALE}")
            zfs = apply_spin_correction(zfs)
        rows.append({
            "defect": infer_defect(defect_root),
            "column": "corrected" if corrected else "contaminated",
            "cell": infer_cell(defect_root),
            "d_mhz": zfs.to_unit("MHz").D,
            "source": outcar,
        })
    return rows


def _entry(merged: dict, defect: str, cell: str) -> dict:
    return merged.setdefault((defect, cell), {
        "defect": defect,
        "cell": cell,
        "d": {c: None for c in COLUMNS},
        "t1": {},
        "sources": {},
    })


def merge_rows(d_rows: list[dict], t1_rows: list[T1Row] = ()) -> list[dict]:
    """One entry per defect+cell holding D and T1 per column.

    D sources disagreeing by more than :data:`CONFLICT_TOL_MHZ` are reported
    and the first one kept (OUTCARs are collected before packaged npz files).
    ``entry["t1"]`` maps ``(column, init_state)`` to T1 in seconds.
    """
    merged: dict[tuple[str, str], dict] = {}
    for row in d_rows:
        entry = _entry(merged, row["defect"], row["cell"])
        column = row["column"]
        previous = entry["d"][column]
        if previous is None:
            entry["d"][column] = row["d_mhz"]
            entry["sources"][column] = row["source"]
        elif abs(previous - row["d_mhz"]) > CONFLICT_TOL_MHZ:
            name = " ".join(filter(None, (row["defect"], row["cell"])))
            LOGGER.warning(
                f"Conflicting {column} D for {name}: {previous:.3f} MHz ({entry['sources'][column]}) "
                f"vs {row['d_mhz']:.3f} MHz ({row['source']}); keeping the first"
            )
    for t1_row in t1_rows:
        column = METHOD_COLUMN.get(t1_row.method)
        if column is None:
            continue
        cell = str(t1_row.cell) if t1_row.cell is not None else ""
        entry = _entry(merged, t1_row.defect, cell)
        for state, value in t1_row.t1.items():
            entry["t1"][(column, state)] = value
    # Largest supercell first within a defect.
    return sorted(merged.values(), key=lambda e: (e["defect"], -int(e["cell"] or 0)))


def assign_letters(merged: list[dict]) -> dict[str, str]:
    """Map each cell size to one letter for the whole table, in order of first
    occurrence (rows already sorted by defect, then cell).

    Letters are shared across defects so the caption footnote is unambiguous.
    """
    cells: list[str] = []
    for row in merged:
        if row["cell"] and row["cell"] not in cells:
            cells.append(row["cell"])
    return {c: chr(ord("a") + i) for i, c in enumerate(cells)}


def tex_defect(defect: str) -> str:
    """LaTeX label: V_Si -> V$_\\mathrm{Si}$."""
    base, _, sub = defect.partition("_")
    sub = sub.replace("_", r"\_")
    return f"{base}$_\\mathrm{{{sub}}}$" if sub else base


def _fmt_sig(value: float, digits: int = SIG_FIGS) -> str:
    """``digits`` significant figures: 2946 -> 2950, 3.249 -> 3.25, 13 -> 13.0.

    Magnitudes of 10^4 and above are written as a power of ten
    (``$2.11\\times10^{6}$``) so trailing zeros never pose as digits.
    """
    if value == 0:
        return "0"
    value = float(f"{value:.{digits - 1}e}")  # round first so 99.96 -> 100 sets the magnitude
    magnitude = int(np.floor(np.log10(abs(value))))
    if magnitude >= 4:
        mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"
    if magnitude >= digits - 1:
        return f"{round(value, digits - 1 - magnitude):.0f}"
    return f"{value:#.{digits}g}"


def render_defect_table(
    merged: list[dict],
    temperature: float = 300.0,
    include_t1: bool = True,
    label: str = "tab:defects",
) -> str:
    """LaTeX table of D (MHz) and, optionally, T1 (ms) per initial state.

    One row per defect; the values of the different supercells
    are comma-separated with a superscript letter per cell, explained in the
    caption. Rows are grouped under base-crystal headers (4H-SiC, Diamond).
    Requires ``booktabs`` and ``siunitx``.
    """
    letters = assign_letters(merged)

    def joined(values: list[tuple[float | None, str]], scale: float = 1.0) -> str:
        parts = []
        for value, cell in values:
            if value is None or not np.isfinite(value):
                continue
            letter = letters.get(cell, "")
            parts.append(_fmt_sig(value * scale) + (rf"$^{{({letter})}}$" if letter else ""))
        return ", ".join(parts) or "---"

    grouped: dict[str, list[dict]] = {}
    for row in merged:
        grouped.setdefault(row["defect"], []).append(row)

    t1_keys = [(c, s) for c in COLUMNS for s in INIT_STATES] if include_t1 else []
    n_cols = 5 + len(t1_keys)
    mhz = r"(\si{\mega\hertz})"
    head = [r"Defect", "Configuration", r"$D_\text{experiment}$", r"$D_\text{corrected}$", r"$D_\text{contaminated}$"]
    sub = ["", "", mhz, mhz, mhz]
    rules = ""
    if include_t1:
        head += [rf"\multicolumn{{3}}{{c}}{{$T_{{1,\text{{{c}}}}}$ (ms)}}" for c in COLUMNS]
        sub += [_INIT_STATE_HEADERS[s] for _, s in t1_keys]
        rules = r"    \cmidrule(lr){6-8} \cmidrule(lr){9-11}" + "\n"

    lines = [
        "% generated by spinlabs table defects",
        r"% requires \usepackage{booktabs} and \usepackage{siunitx}",
        r"\begin{table*}[htbp]",
        r"  \centering",
        rf"  \begin{{tabular}}{{ll{'c' * (n_cols - 2)}}}",
        r"    \toprule",
        "    " + " & ".join(head) + r" \\",
        rules + "    " + " & ".join(sub) + r" \\",
        r"    \midrule",
    ]
    last_base = None
    for defect in sorted(grouped, key=lambda d: (base_crystal_of(d) == "Diamond", d)):
        base = base_crystal_of(defect)
        if base != last_base:
            if last_base is not None:
                lines.append(r"    \midrule")
            lines.append(rf"    \multicolumn{{{n_cols}}}{{c}}{{\textit{{{base}}}}} \\")
            last_base = base
        cells = grouped[defect]
        # Configuration and experimental D are filled in by hand.
        values = [tex_defect(defect), "---", "---"]
        values += [joined([(e["d"][c], e["cell"]) for e in cells]) for c in COLUMNS]
        values += [joined([(e["t1"].get(key), e["cell"]) for e in cells], scale=1e3) for key in t1_keys]
        lines.append("    " + " & ".join(values) + r" \\")

    footnote = "; ".join(f"({letter}) cell size {cell}" for cell, letter in letters.items())
    caption = "Zero-field splitting parameter $D$ (MHz)"
    if include_t1:
        caption += (
            " and longitudinal relaxation $T_1$ of the treated point defects. "
            f"The presented $T_1$ value is for a system at $T = {temperature:g}$~K"
        )
    caption += (
        ". Corrected: defect-band approximation (ZFS\\_occup); contaminated: all-electron bands (ZFS\\_hyp)."
    )
    if footnote:
        caption += f" The superscripts refer to different defect supercells: {footnote}."
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"
