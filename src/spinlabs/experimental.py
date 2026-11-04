"""Experimental data digitized from papers, one CSV per source (``experimental_data/<defect>/<Author>_<year>.csv``).

The CSV layout is WebPlotDigitizer's export of several datasets from one figure, under an optional
citation line:

    Jarmola et al. (2012)    citation, as the legend shows it (optional; one cell)
    K,Hz                     units: one x and one y unit for every series (or one pair per series)
    S8,,S3,,S2,              series names, each over its X and Y column
    X,Y,X,Y,X,Y              column roles
    59.97,0.106,6.556,...    data; shorter series leave their cells empty

Rows are in digitizing order, so each series is sorted by x when read. The source (the paper) is
the file name, e.g. ``Jarmola_2012``.

A series measures either the relaxation itself, T1 or 1/T1 of a polarized state (``kind`` "relaxation",
e.g. Jarmola et al. 2012), or a single-transition rate, named by its label: ``omega`` (or Ω) for
|0> <-> |+-1>, ``gamma`` (or γ) for |+1> <-> |-1> (e.g. Lin et al. 2021). relaxation_series turns a
source's Omega and gamma into T1 with the spinlabs rate-equation solver.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

# Scale factors to a base unit per dimension (K, s, Hz). Within a dimension a conversion rescales;
# between time and rate it inverts (a T1 in ms is a 1/T1 in Hz), so data reported either way compare.
UNITS: dict[str, tuple[str, float]] = {
    "K": ("temperature", 1.0),
    "s": ("time", 1.0),
    "ms": ("time", 1e-3),
    "us": ("time", 1e-6),
    "µs": ("time", 1e-6),
    "ns": ("time", 1e-9),
    "Hz": ("rate", 1.0),
    "1/s": ("rate", 1.0),
    "s^-1": ("rate", 1.0),
    "kHz": ("rate", 1e3),
    "MHz": ("rate", 1e6),
}


# Series labels that name a single-transition rate (case-insensitive) and the kind they get.
RATE_LABELS = {"omega": "omega", "ω": "omega", "gamma": "gamma", "γ": "gamma"}


def series_kind(label: str) -> str:
    """'omega' or 'gamma' for a single-transition rate, else 'relaxation' (T1 or 1/T1 of a polarized state)."""
    return RATE_LABELS.get(label.strip().lower(), "relaxation")


@dataclass(frozen=True)
class ExperimentalSeries:
    """One digitized dataset: ``y`` against ``x``, sorted by ``x``.

    ``kind`` is what y measures: "relaxation" (T1 or 1/T1), "omega" or "gamma" (see the module docstring).
    """

    label: str
    x: np.ndarray
    y: np.ndarray
    x_unit: str
    y_unit: str
    source: str
    kind: str = "relaxation"
    citation: str | None = None

    def __len__(self) -> int:
        return len(self.x)

    def to_units(self, x_unit: str | None = None, y_unit: str | None = None) -> ExperimentalSeries:
        """The same series in other units: rescaled within a dimension (Hz -> kHz, ms -> s), inverted between
        time and rate (ms -> Hz gives 1/T1 from T1). Re-sorted by x, which an inverted x would reverse."""
        x = self.x if x_unit is None else _convert(self.x, self.x_unit, x_unit)
        y = self.y if y_unit is None else _convert(self.y, self.y_unit, y_unit)
        order = np.argsort(x, kind="stable")
        return replace(self, x=x[order], y=y[order], x_unit=x_unit or self.x_unit, y_unit=y_unit or self.y_unit)


def _convert(values: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    for unit in (from_unit, to_unit):
        if unit not in UNITS:
            raise ValueError(f"unknown unit {unit!r}; known: {', '.join(UNITS)}")
    (dim_from, scale_from), (dim_to, scale_to) = UNITS[from_unit], UNITS[to_unit]
    base = np.asarray(values, dtype=float) * scale_from
    if dim_from == dim_to:
        return base / scale_to
    if {dim_from, dim_to} == {"time", "rate"}:
        with np.errstate(divide="ignore"):
            return 1.0 / base / scale_to  # 1/s = Hz: a time and its rate are reciprocal
    raise ValueError(f"cannot convert {from_unit} ({dim_from}) to {to_unit} ({dim_to})")


def read_experimental_csv(path: str | Path) -> list[ExperimentalSeries]:
    """The series of one digitized CSV (see the module docstring), in the file's column order."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [[cell.strip() for cell in row] for row in csv.reader(fh)]
    citation = None
    if len(rows) >= 4 and _is_roles_row(rows[3]) and not _is_roles_row(rows[2]):
        citation = ",".join(c for c in rows[0] if c) or None  # the optional citation line
        rows = rows[1:]
    if len(rows) < 3:
        raise ValueError(f"{path}: expected a units, a names and an X,Y header row before the data")
    units_row, names_row, roles_row, data = rows[0], rows[1], rows[2], rows[3:]

    roles = [r.upper() for r in roles_row]
    while roles and roles[-1] == "":
        roles.pop()
    if not roles or len(roles) % 2 or any(roles[k : k + 2] != ["X", "Y"] for k in range(0, len(roles), 2)):
        raise ValueError(f"{path}: third row must be X,Y pairs, got {roles_row}")
    n_series = len(roles) // 2

    units = [u for u in units_row if u]
    if len(units) == 2:
        units = units * n_series
    elif len(units) != 2 * n_series:
        raise ValueError(
            f"{path}: first row must hold one x and one y unit, or one pair per series ({n_series}), got {units_row}"
        )

    series = []
    for k in range(n_series):
        name = names_row[2 * k] if 2 * k < len(names_row) and names_row[2 * k] else f"series {k + 1}"
        points = []
        for line, row in enumerate(data, start=4):
            cells = row[2 * k : 2 * k + 2] + ["", ""]
            x_cell, y_cell = cells[0], cells[1]
            if not x_cell and not y_cell:
                continue
            if not x_cell or not y_cell:
                raise ValueError(f"{path}:{line}: series {name!r} has only one of X and Y")
            try:
                points.append((float(x_cell), float(y_cell)))
            except ValueError:
                raise ValueError(f"{path}:{line}: series {name!r}: not a number: {x_cell!r}, {y_cell!r}") from None
        if not points:
            raise ValueError(f"{path}: series {name!r} has no data")
        xy = np.array(sorted(points))
        series.append(
            ExperimentalSeries(
                label=name,
                x=xy[:, 0],
                y=xy[:, 1],
                x_unit=units[2 * k],
                y_unit=units[2 * k + 1],
                source=path.stem,
                kind=series_kind(name),
                citation=citation,
            )
        )
    return series


def _is_roles_row(row: list[str]) -> bool:
    """Whether a CSV row is the X,Y column-roles header."""
    roles = [r.upper() for r in row if r]
    return bool(roles) and roles == ["X", "Y"] * (len(roles) // 2) and len(roles) % 2 == 0


def _log_interp(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y at x_new, linear in log(y): the rates grow about exponentially with temperature."""
    return np.exp(np.interp(x_new, x, np.log(y)))


def t1_from_rates(omega: ExperimentalSeries, gamma: ExperimentalSeries, init_state: str = "ms_0") -> ExperimentalSeries:
    """T1 (s) of ``init_state`` from measured Omega and gamma, by the spinlabs rate-equation solver.

    Omega drives |0> <-> |+1> and |0> <-> |-1>, gamma |+1> <-> |-1>, each the same in both directions
    (detailed balance differs by exp(D/kT) - 1 ~ 5e-4 at 300 K). The two series are taken at the union
    of their temperatures where both were measured, each interpolated linearly in log(rate). T1 is the
    fitted decay of RelaxationDynamics, as for the computed rates (pipeline.solve_t1).
    """
    from spinlabs.relaxation_dynamics import INIT_POPULATIONS, RelaxationDynamics

    if init_state not in INIT_POPULATIONS:
        raise ValueError(f"init_state must be one of {', '.join(INIT_POPULATIONS)}, got {init_state!r}")
    om, ga = omega.to_units(x_unit="K", y_unit="Hz"), gamma.to_units(x_unit="K", y_unit="Hz")
    lo, hi = max(om.x[0], ga.x[0]), min(om.x[-1], ga.x[-1])
    temps = np.unique(np.concatenate([om.x, ga.x]))
    temps = temps[(temps >= lo) & (temps <= hi)]
    if len(temps) == 0:
        raise ValueError(
            f"{omega.source}: Omega ({om.x[0]:g}-{om.x[-1]:g} K) and gamma ({ga.x[0]:g}-{ga.x[-1]:g} K) "
            "share no temperature range"
        )
    w, g = _log_interp(temps, om.x, om.y), _log_interp(temps, ga.x, ga.y)
    rates = {"0_to_1": w, "0_to_-1": w, "1_to_0": w, "-1_to_0": w, "1_to_-1": g, "-1_to_1": g}
    buffer = io.BytesIO()
    np.savez(buffer, temperatures=temps, **rates)
    buffer.seek(0)
    t1 = np.asarray(RelaxationDynamics(INIT_POPULATIONS[init_state], buffer).compute_T1_range(), dtype=float)
    label = f"T1 {init_state} from {omega.label} and {gamma.label}"
    return ExperimentalSeries(label, temps, t1, "K", "s", omega.source, "relaxation", omega.citation)


def relaxation_series(series: list[ExperimentalSeries], init_state: str = "ms_0") -> list[ExperimentalSeries]:
    """The T1 / 1/T1 data of ``series``, plus T1 of ``init_state`` for every source that gives Omega and gamma.

    A source with only one of Omega and gamma contributes nothing from it. The relaxation data are
    returned as measured; which state they start from is the source's (Jarmola et al.: m_s = 0).
    """
    out = [s for s in series if s.kind == "relaxation"]
    by_source: dict[str, dict[str, ExperimentalSeries]] = {}
    for s in series:
        if s.kind != "relaxation":
            by_source.setdefault(s.source, {})[s.kind] = s
    for rates in by_source.values():
        if "omega" in rates and "gamma" in rates:
            out.append(t1_from_rates(rates["omega"], rates["gamma"], init_state))
    return out


def load_experimental_data(folder: str | Path = "experimental_data") -> dict[str, list[ExperimentalSeries]]:
    """Every ``*.csv`` in ``folder``, keyed by source (the file name without .csv), sorted by name."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"no experimental data folder: {folder}")
    return {p.stem: read_experimental_csv(p) for p in sorted(folder.glob("*.csv"))}
