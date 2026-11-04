"""Experimental data digitized from papers, one CSV per source (``experimental_data/<Author>_<year>.csv``).

The CSV layout is WebPlotDigitizer's export of several datasets from one figure:

    K,Hz                     units: one x and one y unit for every series (or one pair per series)
    S8,,S3,,S2,              series names, each over its X and Y column
    X,Y,X,Y,X,Y              column roles
    59.97,0.106,6.556,...    data; shorter series leave their cells empty

Rows are in digitizing order, so each series is sorted by x when read. The source (the paper) is
the file name, e.g. ``Jarmola_2012``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

# Scale factors to a base unit per dimension; conversions stay within one dimension.
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


@dataclass(frozen=True)
class ExperimentalSeries:
    """One digitized dataset: ``y`` against ``x``, sorted by ``x``."""

    label: str
    x: np.ndarray
    y: np.ndarray
    x_unit: str
    y_unit: str
    source: str

    def __len__(self) -> int:
        return len(self.x)

    def to_units(self, x_unit: str | None = None, y_unit: str | None = None) -> ExperimentalSeries:
        """The same series in other units of the same dimension (e.g. Hz -> kHz, ms -> s)."""
        x = self.x if x_unit is None else _convert(self.x, self.x_unit, x_unit)
        y = self.y if y_unit is None else _convert(self.y, self.y_unit, y_unit)
        return replace(self, x=x, y=y, x_unit=x_unit or self.x_unit, y_unit=y_unit or self.y_unit)


def _convert(values: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    for unit in (from_unit, to_unit):
        if unit not in UNITS:
            raise ValueError(f"unknown unit {unit!r}; known: {', '.join(UNITS)}")
    (dim_from, scale_from), (dim_to, scale_to) = UNITS[from_unit], UNITS[to_unit]
    if dim_from != dim_to:
        raise ValueError(f"cannot convert {from_unit} ({dim_from}) to {to_unit} ({dim_to})")
    return np.asarray(values, dtype=float) * (scale_from / scale_to)


def read_experimental_csv(path: str | Path) -> list[ExperimentalSeries]:
    """The series of one digitized CSV (see the module docstring), in the file's column order."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [[cell.strip() for cell in row] for row in csv.reader(fh)]
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
            )
        )
    return series


def load_experimental_data(folder: str | Path = "experimental_data") -> dict[str, list[ExperimentalSeries]]:
    """Every ``*.csv`` in ``folder``, keyed by source (the file name without .csv), sorted by name."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"no experimental data folder: {folder}")
    return {p.stem: read_experimental_csv(p) for p in sorted(folder.glob("*.csv"))}
