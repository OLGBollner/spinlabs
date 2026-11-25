"""Production couplings, spectral functions, rates and T1 for the GUI's comparison tabs.

Every number comes from the production code in :mod:`spinlabs.pipeline`: ``coupling_manager`` builds the
couplings exactly as ``spinlabs run`` does (production options, the method read from the data),
``rate_series`` gives the rates and ``t1_details`` T1. The spectral functions are those the rates integrate:
the one-phonon density of |V|^2 and the two-phonon density of TransitionRate.pair_weights with the zero-gap
pairs left out. Nothing here imports Tk, so it runs headless (tests, and the GUI's worker process: the GUI
queues every computation on a separate process, ``worker_executor``, so the window never waits for numpy and
no Tk object is ever touched outside the Tk thread).

Building the couplings of a large cell takes on the order of a minute, so they are cached on disk under
``~/.spinlabs/gui_cache/``, keyed on the package's registered sha256 (registry.package_digest; a file
content hash for an unregistered package) and the coupling settings; the rates likewise, with sigma, the
direct model and the temperatures in the key.
"""

from __future__ import annotations

from concurrent.futures import BrokenExecutor, CancelledError, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import threading

import numpy as np

from spinlabs import pipeline, registry
from spinlabs.models import PhononSpectrum, SpinPhononCouplingData
from spinlabs.relaxation_dynamics import init_populations
from spinlabs.spin import direction_key, spin_projections
from spinlabs.transition_rate import TransitionRate
from spinlabs.utils import MathUtils

# Channel -> (first-order key, second-order key) of a coupling file; the names of convergence_channels
CHANNELS = ("d0", "d1", "d2")
CHANNEL_KEYS = {"d0": ("V_0_0", "V2_0_0"), "d1": ("V_0_pm", "V2_0_pm"), "d2": ("V_p_m", "V2_p_m")}
CHANNEL_LABELS = {"d0": r"$\Delta 0$", "d1": r"$\Delta 1$", "d2": r"$\Delta 2$"}
SPIN_FLIP = ("d1", "d2")  # the channels that drive transitions (d0 only modulates the levels)
PAIR_TREATMENTS = tuple(pipeline.PAIR_MODES)
DIRECT_MODELS = ("debye", "gaussian")
# Rate processes shown: production total (direct + two-phonon), and each process alone
PROCESSES = {
    "total": pipeline.PRODUCTION_PROCESSES,
    "direct": ("first_order",),
    "two-phonon": ("two_phonon",),
    "Raman (1st order)": ("second_order",),
}
# T1 initial states: the lowest |m_s| (optically pumped), m_s = +S, and m_s = -S (differs from +S in a B field;
# for S = 3/2 the T1 states are the +-m doublets, so +S and -S are the same doublet)
T1_STATES = ("lowest |m_s|", "highest |m_s|", "m_s = -S")
_DATE = re.compile(r"(?<!\d)(\d{8})(?!\d)")


CACHE_FORMAT = 2  # bump when the layout of the cached files changes
KEEP_RESULTS = 3  # results kept in memory per kind (and T1 state); older ones come back from the disk cache
_CODE_FINGERPRINT: str | None = None


def code_fingerprint() -> str:
    """A hash of the spinlabs code that computes couplings and rates (everything but the GUI, the CLI and the
    plotting), in the cache key: a change to that code recomputes instead of showing stale numbers."""
    global _CODE_FINGERPRINT
    if _CODE_FINGERPRINT is None:
        root = Path(pipeline.__file__).resolve().parent
        h = hashlib.sha1(f"format {CACHE_FORMAT}".encode())
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(root)
            if rel.parts[0] in ("gui", "cli") or rel.name.startswith("plot"):
                continue
            h.update(str(rel).encode())
            h.update(path.read_bytes())
        _CODE_FINGERPRINT = h.hexdigest()[:16]
    return _CODE_FINGERPRINT


def default_cache_dir() -> Path:
    return Path.home() / ".spinlabs" / "gui_cache"


# ---------------------------------------------------------------------------------------------------- packages


@dataclass(frozen=True)
class ProductionPackage:
    """One production dataset: a packaged ZFS file, or a full-decontamination folder of sublevel files."""

    path: Path
    defect: str
    cell: int | None = None
    method: str | None = None
    step: float | None = None
    date: str = "unknown"
    # Whether it holds second-order data (pairs, or central -Q runs): without it there is no two-phonon spectral
    # function, rate or T1. Older first-order-only packages (*_1d.npz) and empty ones are not complete.
    complete: bool = True
    # Registry id (a full-decontamination set: its newest file). Within a setup the highest id is the latest
    # data, as registry.find_packaged(latest=True) picks it.
    registered: int = 0

    @property
    def key(self) -> str:
        return str(self.path)

    @property
    def setup(self) -> tuple:
        """What the registry groups versions by: defect, cell, canonical method and step."""
        return (self.defect, self.cell or 0, str(registry._canonical(self.method) or ""),
                self.step if self.step is not None else float("nan"))

    @property
    def setup_key(self) -> str:
        defect, cell, method, step = self.setup
        return f"{defect}|{cell}|{method}|{step:g}"

    @property
    def setup_label(self) -> str:
        defect, cell, method, step = self.setup
        cell_text = f" ({cell})" if cell else ""
        step_text = f" {step:g}" if self.step is not None else ""
        return f"{defect}{cell_text} {method or '?'}{step_text}"

    @property
    def label(self) -> str:
        cell = f" ({self.cell})" if self.cell else ""
        step = f" {self.step:g}" if self.step is not None else ""
        return f"{self.defect}{cell} {self.method or '?'}{step}"

    @property
    def long_label(self) -> str:
        return f"{self.label} [{self.date}]"


def _date_of(path: Path) -> str:
    for part in reversed(Path(path).parts):
        match = _DATE.search(part)
        if match:
            return match.group(1)
    return "unknown"


def _has_second_order(pert_order) -> bool:
    """Whether a registry pert_order ('[1, 2]', '[1]', legacy 12 or None) includes the second order."""
    text = str(pert_order or "")
    return "2" in text  # '[1, 2]' and the legacy '12'; '[1]' is first order only


def production_packages(conn=None, *, complete_only: bool = True) -> list[ProductionPackage]:
    """Every registered production dataset that still exists: packaged ZFS files, and full-decontamination
    sets (their sublevel files grouped by folder), sorted by defect, cell, method and date.

    ``complete_only`` (default) leaves out the packages without second-order data (registry pert_order [1]:
    the older first-order-only *_1d.npz files, and empty ones): they have no two-phonon spectral function,
    rates or T1. Central-difference packages are complete (their second order comes from the -Q runs).
    """
    conn = conn or registry.ensure_db()
    rows = conn.execute(
        "SELECT id, kind, path, defect, cell_size, calc_method, pert_scale, pert_order FROM files "
        "WHERE kind IN ('packaged_zfs', 'packaged_zfs_sublevel')"
    ).fetchall()
    out: dict[str, ProductionPackage] = {}
    for row_id, kind, path, defect, cell, method, step, pert_order in rows:
        complete = _has_second_order(pert_order)
        path = Path(path)
        if kind == "packaged_zfs_sublevel":
            path = path.parent  # the set is the folder of its sublevel files
            method = method or "full_decont"
        if not path.exists():
            continue
        try:
            step = float(step) if step is not None else None
        except (TypeError, ValueError):
            step = None
        previous = out.get(str(path))
        out[str(path)] = ProductionPackage(
            path=path, defect=str(defect or "?"), cell=int(cell) if cell else None, method=method, step=step,
            date=_date_of(path), complete=complete and (previous is None or previous.complete),
            registered=max(int(row_id), previous.registered if previous else 0),
        )
    # Filtered after a decontamination set's sublevel files are merged: one first-order-only file makes the set
    # incomplete
    packages = [p for p in out.values() if p.complete or not complete_only]
    packages.sort(key=lambda p: -p.registered)  # newest first within each setup
    return sorted(packages, key=lambda p: (p.defect, p.cell or 0, p.setup[2], p.step if p.step is not None else -1))


def production_setups(packages: list[ProductionPackage]) -> dict[str, list[ProductionPackage]]:
    """The packages grouped by setup (defect, cell, method, step) as the registry groups them, each group
    newest first: its first package is the latest data for that setup (registry ``--latest``)."""
    groups: dict[str, list[ProductionPackage]] = {}
    for package in packages:
        groups.setdefault(package.setup_key, []).append(package)
    for versions in groups.values():
        versions.sort(key=lambda p: -p.registered)
    return groups


def version_labels(versions: list[ProductionPackage]) -> list[str]:
    """Dropdown labels of a setup's versions (newest first): the date, the file name when dates repeat."""
    dates = [p.date for p in versions]
    labels = []
    for index, package in enumerate(versions):
        label = package.date if dates.count(package.date) == 1 else f"{package.date} · {package.path.name}"
        labels.append(f"{label} (latest)" if index == 0 else label)
    return labels


def package_for_path(path: Path) -> ProductionPackage:
    """A package opened by hand: the registry's description if it is registered, else its own metadata.

    Raises ValueError for a package without second-order data (nothing to compare beyond the first order).
    """
    path = Path(path).expanduser().resolve()
    from spinlabs.models import RawZFSData

    first = sorted(path.glob("*.npz"))[0] if path.is_dir() else path
    raw = RawZFSData.load(first)
    if not (raw.second_order or raw.first_order_minus):
        what = "no ZFS data at all (an empty package)" if not raw.first_order else "first-order data only"
        raise ValueError(f"{path.name} has {what}: no second-order couplings, so no two-phonon spectral function, "
                         "rates or T1. Use the package of the same step with both orders (*_12d.npz or central).")
    for package in production_packages(complete_only=False):
        if package.path == path:
            return package
    method = "full_decont" if path.is_dir() else raw.calc_method
    return ProductionPackage(
        path=path, defect=str(raw.defect or path.stem), cell=int(raw.cell_size) if raw.cell_size else None,
        method=method, step=float(raw.pert_scale) if raw.pert_scale is not None else None, date=_date_of(path),
    )


def package_digest(path: Path) -> str:
    """Content key of a package: its registered sha256 (re-hashed when rewritten), else a hash of its bytes."""
    try:
        return registry.package_digest(registry.ensure_db(), path)
    except registry.UnregisteredPackageError:
        h = hashlib.sha256()
        files = sorted(Path(path).glob("*.npz")) if Path(path).is_dir() else [Path(path)]
        for f in files:
            h.update(f.name.encode())
            h.update(f.read_bytes())
        return h.hexdigest()


# ---------------------------------------------------------------------------------------------------- experiment

EXPERIMENT_DIRNAME = "experimental_data"  # <root>/experimental_data/<defect>/*.csv (spinlabs.experimental)
# Series a new comparison leaves unticked, as the paper figures do: Jarmola et al. (2012) S2 (plot_t1_vs_experiment
# EXPERIMENT_EXCLUDE; M2 "S2 present but not used")
EXPERIMENT_OFF_BY_DEFAULT = ("NV/Jarmola_2012/S2",)


@dataclass(frozen=True)
class ExperimentSeries:
    """One digitized experimental series of a defect, in K and Hz (rates) or s (T1)."""

    defect: str
    source: str  # the CSV's file name, e.g. Jarmola_2012
    label: str  # the series name in the CSV, e.g. S8, omega
    kind: str  # "relaxation" (T1 or 1/T1 of a polarized state), "omega" (0 <-> +-1) or "gamma" (+1 <-> -1)
    citation: str
    x: np.ndarray  # K
    y: np.ndarray  # Hz for omega/gamma; s (T1) for relaxation

    @property
    def key(self) -> str:
        return f"{self.defect}/{self.source}/{self.label}"

    @property
    def legend(self) -> str:
        name = {"omega": "Ω", "gamma": "γ"}.get(self.kind, self.label)
        return f"{self.citation} {name}".strip()


def experiment_folder(defect: str, near: Path | None = None) -> Path | None:
    """``experimental_data/<defect>`` in the working directory or a parent of ``near`` (the package: the data
    tree lives next to it in the repository)."""
    roots = [Path.cwd()] + (list(Path(near).resolve().parents) if near is not None else [])
    for root in roots:
        folder = root / EXPERIMENT_DIRNAME / defect
        if folder.is_dir():
            return folder
    return None


def experimental_series(defect: str, near: Path | None = None) -> list[ExperimentSeries]:
    """Every experimental series of ``defect`` (none without a folder): relaxation data as T1 (s), Omega and
    gamma as rates (Hz), temperatures in K, positive values only."""
    from spinlabs.experimental import load_experimental_data

    folder = experiment_folder(defect, near)
    if folder is None:
        return []
    out = []
    for source, series_list in load_experimental_data(folder).items():
        for s in series_list:
            s = s.to_units(x_unit="K", y_unit="s" if s.kind == "relaxation" else "Hz")
            keep = np.isfinite(s.y) & (s.y > 0)
            out.append(ExperimentSeries(defect, source, s.label, s.kind, s.citation or source.replace("_", " "),
                                        np.asarray(s.x[keep], dtype=float), np.asarray(s.y[keep], dtype=float)))
    return out


def experiment_rates(series: list[ExperimentSeries], channel: str) -> list[ExperimentSeries]:
    """The measured single-transition rates of a spin-flip channel: Omega for Delta1 (the m_s 1 -> 0 rate of the
    Rates view), gamma for Delta2 (1 -> -1). S = 1 defects only (that is what Omega and gamma name)."""
    kind = {"d1": "omega", "d2": "gamma"}.get(channel)
    return [s for s in series if s.kind == kind and len(s.x)]


def experiment_t1(series: list[ExperimentSeries], which: str) -> list[tuple[ExperimentSeries, np.ndarray, np.ndarray]]:
    """(series, temperatures, T1 in s) to draw with the computed T1 of initial state ``which`` (T1_STATES).

    The relaxation data are the T1 of the polarization the experiment prepares (optical pumping: the lowest
    |m_s|), so they are drawn for that state only. Omega and gamma of a source give T1 of any S = 1 state by the
    rate-equation solver (spinlabs.experimental.t1_from_rates), drawn as a series of the Omega entry.
    """
    from spinlabs.experimental import ExperimentalSeries as Raw, t1_from_rates

    out = []
    if which == T1_STATES[0]:
        out += [(s, s.x, s.y) for s in series if s.kind == "relaxation" and len(s.x)]
    by_source: dict[tuple, dict[str, ExperimentSeries]] = {}
    for s in series:
        if s.kind in ("omega", "gamma") and len(s.x):
            by_source.setdefault((s.defect, s.source), {})[s.kind] = s
    state = initial_state(1.0, which)
    for rates in by_source.values():
        if "omega" not in rates or "gamma" not in rates:
            continue
        omega, gamma = rates["omega"], rates["gamma"]
        try:
            t1 = t1_from_rates(Raw(omega.label, omega.x, omega.y, "K", "Hz", omega.source, "omega"),
                               Raw(gamma.label, gamma.x, gamma.y, "K", "Hz", gamma.source, "gamma"), state)
        except ValueError:  # Omega and gamma share no temperature range
            continue
        label = f"T1 {state} from Ω, γ"
        out.append((ExperimentSeries(omega.defect, omega.source, label, "relaxation", omega.citation, t1.x, t1.y),
                    t1.x, t1.y))
    return out


# ---------------------------------------------------------------------------------------------------- settings


@dataclass
class Settings:
    """What the couplings and rates of a comparison are computed with (production defaults)."""

    pairs: str = "sqrt"
    mirror_projection: bool = pipeline.PRODUCTION_COUPLING_OPTIONS["mirror_projection"]
    direct: str = pipeline.PRODUCTION_DIRECT
    sigma: float = pipeline.DEFAULT_SIGMA_MEV
    t_min: float = 5.0
    t_max: float = pipeline.PRODUCTION_T_END
    t_step: float = pipeline.PRODUCTION_T_STEP
    b_field_mt: float = 0.0  # axial magnetic field (mT): Zeeman-splits +m and -m, moves every gap

    def __post_init__(self):
        if self.pairs not in PAIR_TREATMENTS:
            raise ValueError(f"pair treatment must be one of {', '.join(PAIR_TREATMENTS)}, got {self.pairs!r}")
        if self.direct not in DIRECT_MODELS:
            raise ValueError(f"direct model must be one of {', '.join(DIRECT_MODELS)}, got {self.direct!r}")
        if not self.sigma > 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if not (0 < self.t_min < self.t_max and self.t_step > 0):
            raise ValueError("temperatures: need 0 < T min < T max and a positive step")
        if not np.isfinite(self.b_field_mt):
            raise ValueError(f"B must be a number of mT, got {self.b_field_mt}")

    @property
    def temperatures(self) -> np.ndarray:
        return np.arange(self.t_min, self.t_max + 0.5 * self.t_step, self.t_step)

    def coupling_key(self) -> tuple:
        return (self.pairs, bool(self.mirror_projection))

    def rate_key(self) -> tuple:
        key = (*self.coupling_key(), self.direct, float(self.sigma), float(self.t_min), float(self.t_max),
               float(self.t_step))
        return key + (float(self.b_field_mt),) if self.b_field_mt else key  # B = 0 keeps the earlier cache keys

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> Settings:
        known = {k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__}
        try:
            return cls(**known)
        except (TypeError, ValueError):
            return cls()


# ---------------------------------------------------------------------------------------------------- results


@dataclass
class Couplings:
    """Production couplings of one package, in MHz (energies in meV)."""

    frequencies: np.ndarray
    first: dict[str, np.ndarray]  # channel -> |V| per mode
    second: dict[str, np.ndarray]  # channel -> V2 (N x N), the fill included
    symmetries: list[str]
    deg_groups: list[list[int]]
    spin: float
    convention: str
    phonon_file: str | None
    coupling_path: Path  # the coupling file as the manager wrote it (J), for rate_series
    spectrum_path: Path  # its phonon spectrum (labels, groups), for the Debye direct rate


@dataclass
class Rates:
    temperatures: np.ndarray
    rates: dict[str, dict[str, np.ndarray]]  # process -> direction -> 1/s
    spin: float
    zfs_mev: float
    direct: str  # the direct model used (gaussian when Debye could not be fitted)
    warnings: list[str] = field(default_factory=list)
    b_field_mt: float = 0.0

    def total(self, processes) -> dict[str, np.ndarray]:
        directions = next(iter(self.rates.values())).keys()
        return {d: sum(np.asarray(self.rates[p][d], dtype=float) for p in processes) for d in directions}


def rate_direction(spin: float, channel: str, sign: int = 1) -> str:
    """The transition a channel's rate is shown for: from m = S down by 1 (d1) or 2 (d2), or with ``sign`` -1
    its mirror from m = -S up by 1 or 2.

    S = 1: 1 -> 0 (Omega) and 1 -> -1 (gamma), mirrors -1 -> 0 and -1 -> 1; S = 3/2: +3/2 -> +1/2 and
    +3/2 -> -1/2. Without a field the mirror rates are equal; at the temperatures shown each rate also equals
    its reverse (the levels are split by ~ D << kT).
    """
    top = max(spin_projections(spin))
    step = {"d1": 1, "d2": 2}[channel]
    if top - step < -top:
        raise ValueError(f"no {channel} transition for spin {spin:g}")
    return direction_key(sign * top, sign * (top - step))


def initial_state(spin: float, which: str) -> str:
    """The T1 initial state (relaxation_dynamics.init_populations names) of the lowest |m_s|, m_s = +S or
    m_s = -S: S = 1: ms_0, ms_1, ms_-1; S = 3/2: ms_pm1_2, ms_pm3_2, ms_pm3_2 (doublets). Lowest is the
    optically pumped state of NV- and V2."""
    states = list(init_populations(spin))
    if which == T1_STATES[0]:
        target = min(_abs_m(s) for s in states)
        return next(s for s in states if _abs_m(s) == target)
    target = max(_abs_m(s) for s in states)
    top = [s for s in states if _abs_m(s) == target]
    negative = [s for s in top if s.startswith("ms_-")]
    if which == T1_STATES[2] and negative:
        return negative[0]
    return next((s for s in top if not s.startswith("ms_-")), top[0])


def _abs_m(state: str) -> float:
    """|m| of a state name: ms_0, ms_1, ms_-1, ms_pm1_2, ms_pm3_2, ..."""
    text = state.removeprefix("ms_").removeprefix("pm")
    if "_" in text:
        num, den = text.split("_")
        return abs(float(num) / float(den))
    return abs(float(text))


# ---------------------------------------------------------------------------------------------------- session


class ProductionSession:
    """The couplings, spectral functions, rates and T1 of one package, computed on demand and cached.

    Each ``compute_*`` method runs the work and returns the result (call it from a worker thread); ``get``
    returns a finished result, or None after queueing the work on ``executor`` (the GUI polls). Results are
    kept in memory per settings key, and the couplings and rates on disk as well.
    """

    def __init__(self, package: ProductionPackage, *, cache_dir: Path | None = None, phonon_file: Path | None = None):
        self.package = package
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self.phonon_file = phonon_file
        self._results: dict[tuple, object] = {}
        self._errors: dict[tuple, str] = {}
        self._pending: set[tuple] = set()
        self._died: dict[tuple, int] = {}  # times the worker process died on a computation
        self._lock = threading.Lock()
        self._digest: str | None = None

    # -- the asynchronous front -------------------------------------------------------------------------

    def get(self, kind: str, settings: Settings, *args, executor=None):
        """A finished result of ``compute_<kind>``, or None while it is computed on ``executor``.

        Raises RuntimeError with the message of a computation that failed. Without an executor the work
        runs here and now.
        """
        key = (kind, self._settings_key(kind, settings), args)
        with self._lock:
            if key in self._results:
                self._results[key] = self._results.pop(key)  # most recently used last
                return self._results[key]
            if key in self._errors:
                raise RuntimeError(self._errors[key])
            if executor is not None and key in self._pending:
                return None
            if executor is not None:
                self._pending.add(key)
        if executor is None:
            return self._run(key, kind, settings, args, swallow=False)
        try:
            future = executor.submit(compute_remote, self.package, self.cache_dir, self.phonon_file, kind,
                                     settings, args)
        except Exception as exc:  # noqa: BLE001 -- a shut-down or broken pool
            with self._lock:
                self._pending.discard(key)
                self._errors[key] = f"{type(exc).__name__}: {exc}"
            return None
        future.add_done_callback(lambda f, key=key: self._finish(key, f))
        return None

    def _finish(self, key, future) -> None:
        """Keep a queued job's outcome (runs on the executor's thread, never touches Tk)."""
        try:
            status, value = future.result()
        except CancelledError:
            with self._lock:
                self._pending.discard(key)
            return
        except BrokenExecutor as exc:  # the worker process died (killed, out of memory): the Worker restarts it
            with self._lock:
                self._pending.discard(key)
                self._died[key] = self._died.get(key, 0) + 1
                if self._died[key] > 1:  # this job killed it twice: report instead of retrying forever
                    self._errors[key] = (f"the worker process died twice on this computation ({exc or 'killed'}); "
                                         "out of memory?")
            return  # else 'missing' again: the next draw queues it on the restarted worker
        except Exception as exc:  # noqa: BLE001
            status, value = "error", f"{type(exc).__name__}: {exc}"
        with self._lock:
            if status == "ok":
                self._store(key, value)
            else:
                self._errors[key] = value
            self._pending.discard(key)

    def _store(self, key, value) -> None:
        """Keep a result (call with the lock held): at most KEEP_RESULTS per kind and arguments, the least
        recently used dropped (couplings and rates come back from the disk cache)."""
        self._results.pop(key, None)
        self._results[key] = value
        same = [k for k in self._results if k[0] == key[0] and k[2] == key[2]]
        for old in same[:-KEEP_RESULTS]:
            del self._results[old]

    def status(self, kind: str, settings: Settings, *args) -> str:
        """'ready', 'failed', 'pending' or 'missing' (never queued); never starts any work."""
        key = (kind, self._settings_key(kind, settings), args)
        with self._lock:
            if key in self._results:
                return "ready"
            if key in self._errors:
                return "failed"
            return "pending" if key in self._pending else "missing"

    def _run(self, key, kind, settings, args, *, swallow: bool):
        """Compute and keep one result; a failure is recorded (shown by the tab) and raised unless ``swallow``
        (a queued job: nobody waits for it)."""
        try:
            result = getattr(self, f"compute_{kind}")(settings, *args)
        except Exception as exc:  # noqa: BLE001 -- reported in the tab, never fatal
            message = str(exc) if isinstance(exc, RuntimeError) else f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._errors[key] = message
                self._pending.discard(key)
            if swallow:
                return None
            raise RuntimeError(message) from exc
        with self._lock:
            self._store(key, result)
            self._pending.discard(key)
        return result

    def peek(self, kind: str, settings: Settings, *args):
        """A finished result, or None; never starts any work (for reports of what is already there)."""
        key = (kind, self._settings_key(kind, settings), args)
        with self._lock:
            return self._results.get(key)

    def busy(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def errors(self, settings: Settings | None = None) -> list[str]:
        """The errors of failed computations; with ``settings`` only those of these settings (an entry that
        failed under other settings is not failed now)."""
        with self._lock:
            items = list(self._errors.items())
        if settings is not None:
            items = [(k, v) for k, v in items if k[1] == self._settings_key(k[0], settings)]
        return list(dict.fromkeys(v for _k, v in items))

    @staticmethod
    def _settings_key(kind: str, settings: Settings) -> tuple:
        if kind == "couplings":
            return settings.coupling_key()
        if kind in ("spectral", "spectral_map"):
            return (*settings.coupling_key(), float(settings.sigma))
        return settings.rate_key()

    # -- couplings --------------------------------------------------------------------------------------

    def digest(self) -> str:
        if self._digest is None:
            self._digest = package_digest(self.package.path)
        return self._digest

    def _folder(self, kind: str, key: tuple) -> Path:
        """The cache folder of a result: the package's content, the settings, the phonon spectrum override and
        the code that computes it (code_fingerprint)."""
        phonon = str(Path(self.phonon_file).expanduser().resolve()) if self.phonon_file is not None else None
        text = json.dumps([self.digest(), kind, list(key), phonon, code_fingerprint()], default=str)
        return self.cache_dir / f"{kind}_{hashlib.sha1(text.encode()).hexdigest()[:16]}"

    def compute_couplings(self, settings: Settings) -> Couplings:
        folder = self._folder("couplings", settings.coupling_key())
        coupling_path, spectrum_path, meta_path = folder / "coupling.npz", folder / "spectrum.npz", folder / "meta.json"
        if not (coupling_path.is_file() and spectrum_path.is_file() and meta_path.is_file()):
            manager, phonon = pipeline.coupling_manager(
                raw_zfs_file=self.package.path, phonon_file=self.phonon_file, pairs=settings.pairs,
                mirror_projection=settings.mirror_projection,
            )
            folder.mkdir(parents=True, exist_ok=True)
            manager.process_both_orders(output_filename=str(folder / "coupling"))
            manager.spectrum.save(spectrum_path)
            raw = manager.raw_data
            meta = {
                "deg_groups": [list(map(int, g)) for g in (manager.spectrum.deg_groups or [])],
                "spin": float(raw.spin) if raw.spin is not None else 1.0,
                "convention": f"{raw.defect}_{raw.cell_size} step {raw.pert_scale:g}: {manager.displacement}",
                "phonon_file": str(phonon) if phonon is not None else None,
            }
            _write_atomic(meta_path, lambda tmp: tmp.write_text(json.dumps(meta)))
        meta = json.loads(meta_path.read_text())
        data = SpinPhononCouplingData.load(str(coupling_path)).to_unit("MHz").frequencies_to_unit("meV")
        symmetries = [str(s) for s in (PhononSpectrum.load(spectrum_path).symmetries or [])]
        return Couplings(
            frequencies=np.asarray(data.frequencies, dtype=float),
            first={c: np.abs(np.asarray(getattr(data, k1), dtype=float)) for c, (k1, _k2) in CHANNEL_KEYS.items()},
            second={c: np.asarray(getattr(data, k2), dtype=float) for c, (_k1, k2) in CHANNEL_KEYS.items()},
            symmetries=symmetries, deg_groups=meta["deg_groups"], spin=float(meta["spin"]),
            convention=meta["convention"], phonon_file=meta["phonon_file"], coupling_path=coupling_path,
            spectrum_path=spectrum_path,
        )

    # -- spectral functions -----------------------------------------------------------------------------

    def compute_spectral(self, settings: Settings) -> dict:
        """One-phonon J1(w) = sum_i |V_i|^2 g(w - w_i) per channel (MHz^2/meV) and the two-phonon density along
        its diagonal, J2(w, w), of the pair weights the two-phonon rate integrates (MHz^2/meV^2)."""
        c = self.get("couplings", settings)
        sigma = float(settings.sigma)
        out = {"one": {}, "two": {}}
        res1 = min(0.1, sigma / 10)
        for channel in CHANNELS:
            omega, density = MathUtils.smear_data(c.frequencies, c.first[channel] ** 2, res1, sigma)
            out["one"][channel] = (np.asarray(omega), np.asarray(density))
        maps = self.get("spectral_map", settings)
        for channel, (omega, _y, density) in maps.items():
            out["two"][channel] = (omega, np.diag(density))
        return out

    def compute_spectral_map(self, settings: Settings) -> dict:
        """J2(w_x, w_y) per channel: (w grid, w grid, density), zero-gap pairs left out as in the rates."""
        c = self.get("couplings", settings)
        sigma = float(settings.sigma)
        zero_gap = TransitionRate.symmetry_zero_gap_pairs(len(c.frequencies), c.deg_groups)
        res = min(0.5, sigma / 4)
        out = {}
        for channel in CHANNELS:
            weights = TransitionRate.pair_weights(np.where(zero_gap, 0.0, c.second[channel]))
            X, Y, J = MathUtils.get_2d_spectral_density(c.frequencies, weights, res, sigma)
            omega = X[0] if X.ndim == 2 else X
            out[channel] = (np.asarray(omega), np.asarray(Y[:, 0] if Y.ndim == 2 else Y), np.asarray(J))
        return out

    # -- rates and T1 -----------------------------------------------------------------------------------

    def compute_rates(self, settings: Settings) -> Rates:
        c = self.get("couplings", settings)
        path = self._folder("rates", settings.rate_key()) / "rates.npz"
        if not path.is_file():
            temps = settings.temperatures
            warnings: list[str] = []
            direct = settings.direct
            spectrum = PhononSpectrum.load(c.spectrum_path) if direct == "debye" else None
            b_field = float(settings.b_field_mt)
            try:
                series = pipeline.rate_series(
                    c.coupling_path, temps, sigma=settings.sigma, has_second_order=True,
                    degenerate_groups=c.deg_groups, direct=direct, spectrum=spectrum, b_field_mt=b_field,
                )
            except Exception as exc:  # noqa: BLE001 -- a spectrum the Debye fit cannot use
                if direct != "debye":
                    raise
                warnings.append(f"Debye direct rate not available ({exc}); Gaussian direct rate used")
                direct = "gaussian"
                series = pipeline.rate_series(
                    c.coupling_path, temps, sigma=settings.sigma, has_second_order=True,
                    degenerate_groups=c.deg_groups, direct=direct, b_field_mt=b_field,
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            arrays = {f"{p}:{d}": np.asarray(v, dtype=float) for p, by_d in series.rates.items() for d, v in by_d.items()}

            def write(tmp: Path) -> None:
                with open(tmp, "wb") as handle:
                    np.savez(handle, temperatures=series.temperatures, spin=series.spin, zfs_mev=series.zfs_mev,
                             direct=direct, warnings=np.array(json.dumps(warnings)), **arrays)

            _write_atomic(path, write)
        with np.load(path, allow_pickle=True) as d:
            rates: dict[str, dict[str, np.ndarray]] = {}
            for key in d.files:
                if ":" in key:
                    process, direction = key.split(":", 1)
                    rates.setdefault(process, {})[direction] = np.asarray(d[key], dtype=float)
            return Rates(
                temperatures=np.asarray(d["temperatures"], dtype=float), rates=rates, spin=float(d["spin"]),
                zfs_mev=float(d["zfs_mev"]), direct=str(d["direct"]), warnings=json.loads(str(d["warnings"])),
                b_field_mt=float(settings.b_field_mt),
            )

    def compute_t1(self, settings: Settings, which: str) -> tuple[np.ndarray, np.ndarray, str]:
        """(temperatures, T1 in s, initial state) from the production processes (direct + two-phonon)."""
        r = self.get("rates", settings)
        state = initial_state(r.spin, which)
        t1, _eig, _source = pipeline.t1_details(r.temperatures, r.total(pipeline.PRODUCTION_PROCESSES), state, r.spin)
        return r.temperatures, t1, state


def _write_atomic(path: Path, write) -> None:
    """Write a cache file through a temporary name, so a job killed mid-write (the GUI closed) leaves no
    truncated file behind."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------------- worker

_WORKER_SESSIONS: dict[tuple, ProductionSession] = {}
_WORKER_LOCK = threading.Lock()


def compute_remote(package: ProductionPackage, cache_dir: Path, phonon_file: Path | None, kind: str,
                   settings: Settings, args: tuple) -> tuple[str, object]:
    """One queued computation, run in the worker (a process, or a thread in tests): ("ok", result) or
    ("error", message). The worker keeps a session per package, so a spectral function or T1 reuses the
    couplings and rates it already holds."""
    key = (package.key, str(cache_dir), str(phonon_file))
    with _WORKER_LOCK:
        session = _WORKER_SESSIONS.get(key)
        if session is None:
            session = _WORKER_SESSIONS[key] = ProductionSession(package, cache_dir=cache_dir,
                                                                phonon_file=phonon_file)
    try:
        return "ok", session.get(kind, settings, *args)
    except RuntimeError as exc:
        return "error", str(exc)


def _init_worker() -> None:
    import matplotlib

    matplotlib.use("Agg")  # the worker never draws on screen


def _process_pool() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_init_worker)


class Worker:
    """The GUI's computation worker: one separate process (spawned, so it shares no Tk or thread state with
    the window), jobs in the order queued. If the process dies (killed, out of memory) the pool is broken; the
    next submit starts a fresh one, so one lost job does not fail every later computation."""

    def __init__(self, factory=_process_pool):
        self._factory = factory
        self._pool = None
        self._lock = threading.Lock()
        self.restarts = 0

    def submit(self, fn, *args, **kwargs):
        with self._lock:
            if self._pool is None:
                self._pool = self._factory()
            try:
                return self._pool.submit(fn, *args, **kwargs)
            except BrokenExecutor:
                _stop_pool(self._pool)
                self._pool = self._factory()
                self.restarts += 1
                return self._pool.submit(fn, *args, **kwargs)

    def stop(self) -> None:
        with self._lock:
            if self._pool is not None:
                _stop_pool(self._pool)
                self._pool = None


def worker_executor() -> Worker:
    return Worker()


def _stop_pool(pool) -> None:
    pool.shutdown(wait=False, cancel_futures=True)
    for process in list((getattr(pool, "_processes", None) or {}).values()):
        try:
            process.terminate()
        except Exception:  # noqa: BLE001 -- already gone
            pass


def stop_worker(executor) -> None:
    """Drop the queue and end the worker now, also mid-job (cache files are written atomically)."""
    if isinstance(executor, Worker):
        executor.stop()
    else:
        _stop_pool(executor)
