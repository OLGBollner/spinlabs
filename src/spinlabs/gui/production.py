"""Production couplings, spectral functions, rates and T1 for the GUI's comparison tabs.

Every number comes from the production code in :mod:`spinlabs.pipeline`: ``coupling_manager`` builds the
couplings exactly as ``spinlabs run`` does (production options, the method read from the data),
``rate_series`` gives the rates and ``t1_details`` T1. The spectral functions are those the rates integrate:
the one-phonon density of |V|^2 and the two-phonon density of TransitionRate.pair_weights with the zero-gap
pairs left out. Nothing here imports Tk, so it runs headless (tests, worker threads).

Building the couplings of a large cell takes on the order of a minute, so they are cached on disk under
``~/.spinlabs/gui_cache/``, keyed on the package's registered sha256 (registry.package_digest; a file
content hash for an unregistered package) and the coupling settings; the rates likewise, with sigma, the
direct model and the temperatures in the key.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
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
T1_STATES = ("lowest |m_s|", "highest |m_s|")
_DATE = re.compile(r"(?<!\d)(\d{8})(?!\d)")


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
        if complete_only and not complete:
            continue
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
    packages = sorted(out.values(), key=lambda p: -p.registered)  # newest first within each setup
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

    def __post_init__(self):
        if self.pairs not in PAIR_TREATMENTS:
            raise ValueError(f"pair treatment must be one of {', '.join(PAIR_TREATMENTS)}, got {self.pairs!r}")
        if self.direct not in DIRECT_MODELS:
            raise ValueError(f"direct model must be one of {', '.join(DIRECT_MODELS)}, got {self.direct!r}")
        if not self.sigma > 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if not (0 < self.t_min < self.t_max and self.t_step > 0):
            raise ValueError("temperatures: need 0 < T min < T max and a positive step")

    @property
    def temperatures(self) -> np.ndarray:
        return np.arange(self.t_min, self.t_max + 0.5 * self.t_step, self.t_step)

    def coupling_key(self) -> tuple:
        return (self.pairs, bool(self.mirror_projection))

    def rate_key(self) -> tuple:
        return (*self.coupling_key(), self.direct, float(self.sigma), float(self.t_min), float(self.t_max),
                float(self.t_step))

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

    def total(self, processes) -> dict[str, np.ndarray]:
        directions = next(iter(self.rates.values())).keys()
        return {d: sum(np.asarray(self.rates[p][d], dtype=float) for p in processes) for d in directions}


def rate_direction(spin: float, channel: str) -> str:
    """The transition a channel's rate is shown for: from m = S down by 1 (d1) or 2 (d2).

    S = 1: 1 -> 0 (Omega) and 1 -> -1 (gamma); S = 3/2: +3/2 -> +1/2 and +3/2 -> -1/2. At the temperatures
    shown each rate equals its reverse and its time-reversed partner (the levels are split by ~ D << kT).
    """
    top = max(spin_projections(spin))
    step = {"d1": 1, "d2": 2}[channel]
    if top - step < -top:
        raise ValueError(f"no {channel} transition for spin {spin:g}")
    return direction_key(top, top - step)


def initial_state(spin: float, which: str) -> str:
    """The T1 initial state of the lowest or highest |m_s| (relaxation_dynamics.init_populations names):
    S = 1: ms_0 or ms_1; S = 3/2: ms_pm1_2 or ms_pm3_2. Lowest is the optically pumped state of NV- and V2."""
    states = list(init_populations(spin))
    pick = min if which == T1_STATES[0] else max
    target = pick(_abs_m(s) for s in states)
    return next(s for s in states if _abs_m(s) == target)


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
                return self._results[key]
            if key in self._errors:
                raise RuntimeError(self._errors[key])
            if executor is not None and key in self._pending:
                return None
            if executor is not None:
                self._pending.add(key)
        if executor is None:
            return self._run(key, kind, settings, args, swallow=False)
        executor.submit(self._run, key, kind, settings, args, swallow=True)
        return None

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
            self._results[key] = result
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

    def errors(self) -> list[str]:
        with self._lock:
            return list(dict.fromkeys(self._errors.values()))

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
        text = json.dumps([self.digest(), kind, list(key)], default=str)
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
            meta_path.write_text(json.dumps(meta))
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
            try:
                series = pipeline.rate_series(
                    c.coupling_path, temps, sigma=settings.sigma, has_second_order=True,
                    degenerate_groups=c.deg_groups, direct=direct, spectrum=spectrum,
                )
            except Exception as exc:  # noqa: BLE001 -- a spectrum the Debye fit cannot use
                if direct != "debye":
                    raise
                warnings.append(f"Debye direct rate not available ({exc}); Gaussian direct rate used")
                direct = "gaussian"
                series = pipeline.rate_series(
                    c.coupling_path, temps, sigma=settings.sigma, has_second_order=True,
                    degenerate_groups=c.deg_groups, direct=direct,
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path, temperatures=series.temperatures, spin=series.spin, zfs_mev=series.zfs_mev, direct=direct,
                warnings=np.array(json.dumps(warnings)),
                **{f"{p}:{d}": np.asarray(v, dtype=float) for p, by_d in series.rates.items() for d, v in by_d.items()},
            )
        with np.load(path, allow_pickle=True) as d:
            rates: dict[str, dict[str, np.ndarray]] = {}
            for key in d.files:
                if ":" in key:
                    process, direction = key.split(":", 1)
                    rates.setdefault(process, {})[direction] = np.asarray(d[key], dtype=float)
            return Rates(
                temperatures=np.asarray(d["temperatures"], dtype=float), rates=rates, spin=float(d["spin"]),
                zfs_mev=float(d["zfs_mev"]), direct=str(d["direct"]), warnings=json.loads(str(d["warnings"])),
            )

    def compute_t1(self, settings: Settings, which: str) -> tuple[np.ndarray, np.ndarray, str]:
        """(temperatures, T1 in s, initial state) from the production processes (direct + two-phonon)."""
        r = self.get("rates", settings)
        state = initial_state(r.spin, which)
        t1, _eig, _source = pipeline.t1_details(r.temperatures, r.total(pipeline.PRODUCTION_PROCESSES), state, r.spin)
        return r.temperatures, t1, state
