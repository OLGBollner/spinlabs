"""Headless analysis layer of the convergence GUI.

Everything here works without Tk: registry tabs and the tests use it directly.
The Tk application only turns these results into widgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spinlabs.convergence import SNR_TARGET, TRUNC_TOL, load_convergence
from spinlabs.convergence_channels import ChannelAnalysis, analyse_channels
from spinlabs.convergence_summary import COLORS, WEAK_FRACTION, analyse, mode_details
from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)

# One categorical slot per mode, extended beyond the CLI's six.  The first six
# match convergence_summary.COLORS, so GUI screenshots and CLI figures agree.
PALETTE = list(COLORS) + [
    "#7b5aa6", "#8c6d31", "#d65f5f", "#3a9db8", "#b07aa1", "#8c8c8c",
    "#59a14f", "#e15759", "#4e79a7", "#f28e2b", "#76b7b2", "#edc948",
    "#b6992d", "#9c755f",
]


def mode_color(index: int) -> str:
    """Colour of the index-th selected mode (cycles after the palette)."""
    return PALETTE[index % len(PALETTE)]


@dataclass(frozen=True)
class ModeEntry:
    mode: int
    symmetry: str | None = None
    frequency: float | None = None

    @property
    def label(self) -> str:
        parts = [str(self.mode)]
        if self.symmetry:
            parts.append(self.symmetry)
        if self.frequency is not None:
            parts.append(f"{self.frequency:.2f} meV")
        return "  ".join(parts)


@dataclass
class PackAvailability:
    has_q0: bool
    has_plus: bool
    has_minus: bool
    has_pair: bool
    perts: list[float]
    unfinished: list[float]
    modes: list[int]


@dataclass
class NormResult:
    rows: list
    sigma: float
    offset: float
    central: bool
    warnings: list[str] = field(default_factory=list)


class PackSession:
    """One convergence pack with cached analyses.

    A GUI tab holds one PackSession per column.  Modes and metadata come from
    the pack; phonon symmetry/frequency lookup uses the registry once per
    defect/cell and is shared through ``mode_cache``.
    """

    def __init__(self, path: Path, label: str, displacement: str, mode_cache: dict | None = None):
        self.path = Path(path)
        self.label = label
        self.displacement = displacement
        self.mode_cache = {} if mode_cache is None else mode_cache
        self._conv = None
        self._modes: list[ModeEntry] | None = None
        self._mode_warnings: list[str] = []
        self._norm_cache: dict[tuple, NormResult] = {}
        self._channel_cache: dict[tuple, ChannelAnalysis] = {}

    @property
    def conv(self):
        if self._conv is None:
            self._conv = load_convergence(self.path)
        return self._conv

    @property
    def metadata(self) -> dict:
        return dict(self.conv.metadata or {})

    def availability(self) -> PackAvailability:
        conv = self.conv
        perts = [float(q) for q in conv.perts]
        bad = np.isnan(conv.tensors).any(axis=(2, 3))  # (n_modes, n_perts)
        unfinished = sorted({float(conv.perts[j]) for j in np.flatnonzero(bad.any(axis=0))})
        has_plus = any(q > 0 for q in perts)
        has_minus = any(q < 0 for q in perts)
        has_pair = any(
            np.isclose(perts, 2 * q).any() for q in perts if not np.isclose(q, 0.0)
        )
        return PackAvailability(
            has_q0=conv.q0 is not None,
            has_plus=has_plus,
            has_minus=has_minus,
            has_pair=has_pair,
            perts=perts,
            unfinished=sorted(set(unfinished)),
            modes=[int(m) for m in conv.modes],
        )

    def modes(self) -> list[ModeEntry]:
        """All modes in the pack, with symmetry and frequency when registered."""
        if self._modes is not None:
            return self._modes
        availability = self.availability()
        info = mode_details(self.path, availability.modes, self.mode_cache, meta=self.metadata)
        if not info and availability.modes:
            self._mode_warnings.append(
                "no registered phonon data covers this pack; mode symmetry and frequency are unavailable"
            )
        self._modes = [
            ModeEntry(int(m), *(info.get(int(m), (None, None)))) for m in availability.modes
        ]
        return self._modes

    @property
    def mode_warnings(self) -> list[str]:
        if self._modes is None:
            self.modes()
        return list(self._mode_warnings)

    def norm(
        self,
        modes: set[int],
        *,
        noise: float | None = None,
        target: float = SNR_TARGET,
        tol: float = TRUNC_TOL,
        central: bool = False,
    ) -> NormResult:
        """The CLI summary analysis for one scheme (forward or central)."""
        key = (tuple(sorted(modes)), noise, target, tol, bool(central))
        if key in self._norm_cache:
            return self._norm_cache[key]
        if not modes:
            result = NormResult([], float("nan"), float("nan"), central, ["no modes selected"])
            self._norm_cache[key] = result
            return result
        warnings: list[str] = []
        availability = self.availability()
        if not availability.has_q0:
            warnings.append("no Q=0 anchor (no pert=0 run and no d0 key); differences cannot be measured")
            result = NormResult([], float("nan"), float("nan"), central, warnings)
            self._norm_cache[key] = result
            return result
        if availability.unfinished:
            warnings.append(
                "unfinished runs dropped: " + ", ".join(f"{q:g}" for q in availability.unfinished)
            )
        if central and not availability.has_minus:
            warnings.append("no -Q runs; central differences unavailable")
            result = NormResult([], float("nan"), float("nan"), True, warnings)
            self._norm_cache[key] = result
            return result
        try:
            rows, sigma, offset = analyse(
                self.path, set(modes), noise=noise, target=target, tol=tol, central=central
            )
        except ValueError as exc:
            warnings.append(str(exc))
            result = NormResult([], float("nan"), float("nan"), central, warnings)
            self._norm_cache[key] = result
            return result
        analyzed = {int(row[0]) for row in rows}
        missing = sorted(set(modes) - analyzed)
        not_in_pack = [m for m in missing if m not in set(availability.modes)]
        too_few = [m for m in missing if m in set(availability.modes)]
        if not_in_pack:
            warnings.append("modes not in this pack: " + ", ".join(str(m) for m in not_in_pack))
        if too_few:
            warnings.append(
                "modes without enough finished amplitudes for an SNR estimate (not plotted): "
                + ", ".join(str(m) for m in too_few)
            )
        result = NormResult(rows, float(sigma), float(offset), central, warnings)
        self._norm_cache[key] = result
        return result

    def channels(
        self,
        modes: set[int],
        *,
        schemes: tuple[str, ...] = ("forward", "central"),
        noise: float | None = None,
        target: float = SNR_TARGET,
        tol: float = TRUNC_TOL,
        weak_fraction: float = WEAK_FRACTION,
    ) -> ChannelAnalysis:
        """The V_{Delta n} channel analysis for this pack."""
        key = (tuple(sorted(modes)), tuple(schemes), noise, target, tol, weak_fraction)
        if key in self._channel_cache:
            return self._channel_cache[key]
        if not self.availability().has_q0:
            analysis = ChannelAnalysis(
                path=self.path, sigma=float("nan"), offset=float("nan"), modes=[], errors={}, common={},
                warnings=["no Q=0 anchor; coupling channels cannot be measured"],
            )
            self._channel_cache[key] = analysis
            return analysis
        info = {
            entry.mode: (entry.symmetry, entry.frequency)
            for entry in self.modes()
            if entry.mode in modes
        }
        try:
            analysis = analyse_channels(
                self.path,
                modes=set(modes),
                noise=noise,
                target=target,
                tol=tol,
                weak_fraction=weak_fraction,
                schemes=schemes,
                mode_info=info,
            )
        except ValueError as exc:
            analysis = ChannelAnalysis(
                path=self.path, sigma=float("nan"), offset=float("nan"), modes=[], errors={}, common={},
                warnings=[str(exc)],
            )
        self._channel_cache[key] = analysis
        return analysis

    def warnings(self) -> list[str]:
        """Pack-level warnings shown in the report pane."""
        out = list(self.mode_warnings)
        availability = self.availability()
        if not availability.has_q0:
            out.append("no Q=0 anchor; the Summary cannot measure finite differences")
        if availability.unfinished:
            out.append(
                "unfinished runs (NaN tensors): " + ", ".join(f"{q:g}" for q in availability.unfinished)
            )
        if not availability.has_minus:
            out.append("no -Q runs; Central differences are unavailable")
        if not availability.has_pair:
            out.append("no 2Q run; forward second-order differences are unavailable")
        return out
