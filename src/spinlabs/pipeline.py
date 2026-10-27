from __future__ import annotations

import json
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from spinlabs.logging_utils import (
    attach_db_logging,
    db_logging_attached,
    detach_db_logging,
    get_logger,
    section,
    set_db_run_id,
)

from spinlabs.constants import CONSTANTS
from spinlabs.models import PhononSpectrum, RawZFSData, SpinPhononCouplingData
from spinlabs.registry import ensure_db, find_phonon, finish_run, parse_orders, register_file, register_run
from spinlabs.parsers import (
    parse_phonon_data,
    parse_zfs_dataset_npz,
    parse_zfs_simulation_dataset,
    sim_folder_order,
)
from spinlabs.plotter import (
    plot_1d_spectral_functions,
    plot_t1,
    plot_transition_rates_stacked,
)
from spinlabs.relaxation_dynamics import INIT_POPULATIONS, INIT_STATES, RelaxationDynamics, t1_file_name
from spinlabs.transition_rate import TransitionRate
from spinlabs.zfs_manager import ZFSManager

LOGGER = get_logger(__name__)


def _pert_dir_name(pert_scale: float | None) -> str:
    """Folder name for a perturbation scale, e.g. 'pert_1.0' or 'pert_none'."""
    if pert_scale is None:
        return "pert_none"
    return f"pert_{pert_scale:g}"


def _git_hash() -> str | None:
    """Short git hash of the spinlabs repo, or None outside a repo."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def get_unique_run_dir(
    output_root: str | Path = "runs",
    defect: str = "defect",
    cell_size: str | int = "",
    calc_method: str = "all_bands",
    order: int = 1,
    pert_scale: float | None = None,
    run_name: str | None = None,
) -> Path:
    """
    Creates a unique run directory so prior results are never overwritten.

    Layout: <output_root>/<defect>_<cell>/pert_<scale>/<method>/<YYYYmmdd_HHMMSS>
    With --run-name, <output_root>/<run_name>[_N] is used instead.
    """
    base = Path(output_root)

    if run_name:
        base.mkdir(parents=True, exist_ok=True)
        candidate = base / run_name
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        idx = 1
        while (base / f"{run_name}_{idx}").exists():
            idx += 1
        final_dir = base / f"{run_name}_{idx}"
        final_dir.mkdir(parents=True, exist_ok=True)
        return final_dir

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    leaf = (
        base / f"{defect}_{cell_size}" / _pert_dir_name(pert_scale)
        / calc_method / now_str
    )
    if not leaf.exists():
        leaf.mkdir(parents=True, exist_ok=True)
        return leaf

    idx = 1
    while (leaf.parent / f"{now_str}_{idx}").exists():
        idx += 1
    final_dir = leaf.parent / f"{now_str}_{idx}"
    final_dir.mkdir(parents=True, exist_ok=True)
    return final_dir


def find_default_phonon_file(sim_folder: Path | None) -> Path | None:
    """Search conventional defect locations for phonopy.yaml or phonon_data.npz."""
    if sim_folder is None:
        return None
    candidates = [
        sim_folder / "phonopy.yaml",
        sim_folder / "phonon_data.npz",
        sim_folder.parent / "phonopy.yaml",
        sim_folder.parent / "phonon_data.npz",
        sim_folder.parent.parent / "data" / "phonon_data.npz",
        sim_folder.parent.parent / "phonon_data.npz",
        sim_folder.parent.parent.parent / "data" / "phonon_data.npz",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def find_default_phonon_file_for_raw(raw_zfs_file: str | Path | Sequence[str | Path]) -> Path | None:
    """Search next to the raw ZFS .npz file(s) for phonon_data.npz / phonopy.yaml.

    Looks from the file's folder up to the defect folder of the packaged layout
    ``data/<defect>/pert_<x>/<method>/<date>/<file>.npz``, nearest first.
    """
    if raw_zfs_file is None:
        return None
    if isinstance(raw_zfs_file, (list, tuple)):
        if not raw_zfs_file:
            return None
        anchor = Path(raw_zfs_file[0])
    else:
        anchor = Path(raw_zfs_file)
    anchor_dir = anchor.parent
    candidates = [
        anchor_dir / "phonon_data.npz",
        anchor_dir / "phonopy.yaml",
        anchor_dir.parent / "phonon_data.npz",
        anchor_dir.parent / "phonopy.yaml",
        anchor_dir.parent.parent / "phonon_data.npz",
        anchor_dir.parent.parent / "data" / "phonon_data.npz",
        anchor_dir.parent.parent.parent / "phonon_data.npz",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def raw_mode_count(raw: RawZFSData) -> int:
    """Number of modes a spectrum must have to hold every mode index of the raw data."""
    indices = list(raw.first_order) + [i for pair in raw.second_order for i in pair]
    return max(indices) + 1 if indices else 0


def _covers(spectrum: PhononSpectrum, n_needed: int) -> bool:
    """A spectrum fits the raw data when its full mode count reaches the highest raw mode index."""
    n_full = getattr(spectrum, "n_full", None) or spectrum.n_modes
    return n_full >= n_needed


def _check_requested_orders(user_orders: Sequence[int], *, has_1d: bool | None, has_2d: bool) -> None:
    """Raise when ``--order`` asks for an order the input data lacks.

    ``has_1d=None`` means first-order presence is unknown and is not checked.
    """
    for req, present, label in ((1, has_1d, "first-order (1d)"), (2, has_2d, "second-order (2d)")):
        if req in user_orders and present is False:
            raise ValueError(
                f"--order {req} requested, but the input dataset contains no "
                f"{label} tensors. Rerun without --order {req} or with input that "
                f"includes {label.split(' ')[0]} perturbation data."
            )


def load_raw_inputs(
    *,
    sim_folder: str | Path | Sequence[str | Path] | None = None,
    raw_zfs_file: str | Path | Sequence[str | Path] | None = None,
    raw_zfs_file_1d: str | Path | None = None,
    raw_zfs_file_2d: str | Path | None = None,
    phonon_file: str | Path | None = None,
    calc_method: str = "all_bands",
    zfs_folder: str | None = None,
    order: int | None = None,
    pert_scale: float | None = None,
    defect: str | None = None,
    cell_size: int | None = None,
) -> tuple[RawZFSData, PhononSpectrum | None, Path | None]:
    """Raw ZFS data and phonon spectrum for an analysis, loaded as ``spinlabs run`` does.

    Exactly one ZFS source is used: simulation folder(s), one raw .npz, or a
    1d + 2d pair. The phonon file is ``phonon_file``, else one found next to
    the inputs, else the registry's match for the defect and cell. Returns
    ``(raw_data, spectrum, phonon_path)``; the spectrum is None without phonons.
    """
    combine_orders = raw_zfs_file_1d is not None and raw_zfs_file_2d is not None
    spectrum: PhononSpectrum | None = None
    # Load phonons
    sim_p = sim_folder[0] if isinstance(sim_folder, (list, tuple)) else sim_folder
    ph_from_sim = find_default_phonon_file(Path(sim_p)) if sim_p else None
    ph_from_raw = find_default_phonon_file_for_raw(raw_zfs_file) if raw_zfs_file is not None else None
    ph_path = Path(phonon_file) if phonon_file else (ph_from_sim or ph_from_raw)
    if ph_path is not None and ph_path.exists():
        spectrum = parse_phonon_data(ph_path)
    elif phonon_file is not None:
        raise FileNotFoundError(f"Phonon file not found: {phonon_file}")
    elif ph_path is not None and not ph_path.exists():
        LOGGER.warning(f"phonon file {ph_path} not found; looking in the registry instead")

    # Load raw ZFS
    if sim_folder is not None:
        if isinstance(sim_folder, (list, tuple)) and len(sim_folder) > 1:
            datasets = []
            for sf in sim_folder:
                # Each folder's own name (first_order/..., second_order/...) gives its order;
                # `order` is the fallback for folders whose name does not say.
                ds = parse_zfs_simulation_dataset(
                    sim_folder=sf,
                    calc_method=calc_method,
                    zfs_folder=zfs_folder,
                    order=sim_folder_order(sf) or order,
                    pert_scale=pert_scale,
                    defect=defect,
                    cell_size=cell_size,
                )
                datasets.append(ds)

            raw_data = datasets[0]
            for other_ds in datasets[1:]:
                raw_data = raw_data.combine(other_ds)
        else:
            sf = sim_folder[0] if isinstance(sim_folder, (list, tuple)) else sim_folder
            raw_data = parse_zfs_simulation_dataset(
                sim_folder=sf,
                calc_method=calc_method,
                zfs_folder=zfs_folder,
                order=order,
                pert_scale=pert_scale,
                defect=defect,
                cell_size=cell_size,
            )
    elif combine_orders:
        # Combine 1d + 2d raw datasets into one RawZFSData with both
        # first_order and second_order perturbation sets.
        raw_data = RawZFSData.load([str(raw_zfs_file_1d), str(raw_zfs_file_2d)])
    elif raw_zfs_file is not None:
        raw_data = parse_zfs_dataset_npz(raw_zfs_file)
    else:
        raise ValueError("Must provide either sim_folder, raw_zfs_file, raw_zfs_file_1d+2d, or coupling_file.")

    defect = raw_data.defect or defect or "defect"
    cell_size = raw_data.cell_size or cell_size or 0
    n_needed = raw_mode_count(raw_data)

    # A phonon file found by searching (not passed explicitly) must hold the raw data's modes;
    # otherwise it belongs to another cell and the registry is asked instead.
    if spectrum is not None and phonon_file is None and not _covers(spectrum, n_needed):
        LOGGER.warning(
            f"phonon file {ph_path} has {spectrum.n_modes} modes but the raw data uses mode indices up to "
            f"{n_needed} (another cell?); looking in the registry instead"
        )
        spectrum, ph_path = None, None

    # Registry fallback: resolve phonon by defect/cell when no file found next to the inputs
    if spectrum is None:
        reg = ensure_db()
        reg_hits = find_phonon(reg, str(defect), cell_size)
        candidates = [(p, parse_phonon_data(p)) for p in reg_hits]
        fitting = [(p, s) for p, s in candidates if _covers(s, n_needed)]
        for p, s in candidates:
            if not _covers(s, n_needed):
                LOGGER.warning(
                    f"registry phonon file {p} for {defect}_{cell_size} has {s.n_modes} modes, "
                    f"fewer than the raw data's {n_needed}; skipped (wrong registry entry?)"
                )
        if fitting:
            ph_path, spectrum = fitting[0]
            if len(fitting) > 1:
                LOGGER.warning(
                    f"{len(fitting)} registry phonon files fit {defect}_{cell_size}; using {ph_path} "
                    "(pass --phonon-file to pick another)"
                )
        elif candidates:
            LOGGER.warning(f"no registry phonon file for {defect}_{cell_size} covers the raw data's modes")
    if spectrum is not None:
        LOGGER.info(f"phonon data: {ph_path}")
    else:
        LOGGER.warning(f"no phonon data for {defect}_{cell_size}; mode frequencies and symmetries are unknown")

    return raw_data, spectrum, ph_path


DEFAULT_SIGMA_MEV = 7.5  # Gaussian broadening of the spectral functions (meV)


def temperature_grid(
    t_start: float = 0.0, t_end: float = 300.0, t_step: float = 10.0, temperatures: Sequence[float] | None = None
) -> np.ndarray:
    """The run's temperature grid in K: ``temperatures`` if given, else t_start..t_end (finer below 1 K)."""
    if temperatures is not None:
        return np.asarray(temperatures, dtype=float)
    if t_start < 1.0:
        low_range = np.arange(t_start, 1.0, max(0.01, t_step / 100))
        high_range = np.arange(1.0 + t_step, t_end + t_step, t_step)
        return np.unique(np.concatenate([low_range, high_range]))
    return np.arange(t_start, t_end + t_step, t_step)


def compute_rates(
    coupling_path: str | Path,
    temps: Sequence[float],
    *,
    sigma: float = DEFAULT_SIGMA_MEV,
    two_phonon_file: str | Path | None = None,
    has_second_order: bool = False,
) -> tuple[np.ndarray, dict, dict]:
    """Transition rates of a coupling file at each temperature.

    ``sigma`` (meV) broadens each mode in the 1D and 2D spectral functions,
    standing in for the continuous phonon bands of a macroscopic crystal. The
    2D grid spacing follows it (at most sigma/4, 0.5 meV by default); the rates
    do not depend on the spacing beyond that resolution. Returns the
    temperatures with a result, the total rates per process and the
    directional rates.
    """
    two_ph_p = Path(two_phonon_file) if two_phonon_file else None
    calculator = TransitionRate(str(coupling_path), two_phonon_data_file=two_ph_p)
    two_phonon = two_ph_p is not None or has_second_order
    # For combined 1d+2d runs the coupling file itself carries the second-order
    # coefficients (V2_*); make sure the calculator sees them as the 2-phonon source.
    if two_ph_p is None and has_second_order:
        calculator.data_2ph = calculator.data

    omega, J_0_pm, J_p_m, J_0_0 = calculator.get_spectral_density(res=0.02, sigma=sigma)
    if two_phonon:
        omega_x, omega_y, J2_0_pm, J2_p_m, J2_0_0 = calculator.get_2d_spectral_density(
            res=min(0.5, sigma / 4), sigma=sigma
        )
    omega_zfs = calculator.data["zfs"] / CONSTANTS["meV2J"]

    rate_results = {k: {"0_1": [], "1_-1": []} for k in ("first_order", "second_order", "two_phonon")}
    directional_results = {k: [] for k in ("0_to_1", "0_to_-1", "1_to_0", "-1_to_0", "1_to_-1", "-1_to_1")}
    valid_temps = []
    for T in temps:
        calculator.compute_transition_rates(T, omega, J_0_pm, J_p_m, J_0_0, omega_zfs)
        if two_phonon:
            calculator.compute_two_phonon_rates(T, omega_x, omega_y, J2_0_pm, J2_p_m, J2_0_0, omega_zfs)
        total_rates = calculator.get_total_rates()
        dir_rates = calculator.get_directional_rates()
        if total_rates and dir_rates:
            for k, val in dir_rates.items():
                directional_results[k].append(val)
            for ord_k in rate_results:
                for trans_k in ("0_1", "1_-1"):
                    rate_results[ord_k][trans_k].append(total_rates[ord_k][trans_k])
            valid_temps.append(float(T))
    return np.asarray(valid_temps, dtype=float), rate_results, directional_results


def solve_t1(directional_rates_path: str | Path, init_state: str) -> tuple[np.ndarray, np.ndarray]:
    """ODE-fit and eigenvalue T1 (s) at each temperature of a directional-rates file."""
    dynamics = RelaxationDynamics(init_state=INIT_POPULATIONS[init_state], rates_data=str(directional_rates_path))
    t1_fit = np.asarray(dynamics.compute_T1_range(), dtype=float)
    t1_eig = np.asarray(
        [dynamics.get_T1_eigenval(temp_index=i)[0] for i in range(len(dynamics.temperatures))], dtype=float
    )
    return t1_fit, t1_eig


def run_full_pipeline(
    *,
    sim_folder: str | Path | Sequence[str | Path] | None = None,
    raw_zfs_file: str | Path | Sequence[str | Path] | None = None,
    raw_zfs_file_1d: str | Path | None = None,
    raw_zfs_file_2d: str | Path | None = None,
    coupling_file: str | Path | None = None,
    phonon_file: str | Path | None = None,
    two_phonon_file: str | Path | None = None,
    calc_method: str = "all_bands",
    zfs_folder: str | None = None,
    order: list[int] | int | None = None,
    pert_scale: float | None = None,
    defect: str | None = None,
    cell_size: int | None = None,
    t_start: float = 0.0,
    t_end: float = 300.0,
    t_step: float = 10.0,
    temperatures: Sequence[float] | None = None,
    init_state: str | Sequence[str] = INIT_STATES,
    output_root: str | Path = "runs",
    run_name: str | None = None,
    save_plots: bool = True,
    show_plots: bool = False,
    exclude_inconsistent_pairs: bool = False,
    degenerate_pairs_from_symmetry: bool = False,
    sigma: float = DEFAULT_SIGMA_MEV,
) -> dict[str, Any]:
    """
    Executes the entire end-to-end pipeline:
      1. Ingest raw simulation (OUTCARs or raw .npz)
      2. Calculate ZFS spin-phonon couplings and derivatives
      3. Compute 1-phonon, Raman and two-phonon transition rates vs temperature
      4. Compute population relaxation dynamics and T_1 times, once per
         initial state in ``init_state`` (one state or several; default all)
      5. Save all results bunched together in a dedicated unique run folder
      6. (Optional) Generate and save publication-ready figures
    Keyword-only wrapper: see :func:`_execute_pipeline` for the pipeline body.
    """
    owns_db_logging = not db_logging_attached()
    if owns_db_logging:
        # Library callers may have attached their own DB handler already; only
        # attach (and later detach) when none is present, so ownership is clear.
        attach_db_logging(ensure_db())
    try:
        return _execute_pipeline(
            sim_folder=sim_folder,
            raw_zfs_file=raw_zfs_file,
            raw_zfs_file_1d=raw_zfs_file_1d,
            raw_zfs_file_2d=raw_zfs_file_2d,
            coupling_file=coupling_file,
            phonon_file=phonon_file,
            two_phonon_file=two_phonon_file,
            calc_method=calc_method,
            zfs_folder=zfs_folder,
            order=order,
            pert_scale=pert_scale,
            defect=defect,
            cell_size=cell_size,
            t_start=t_start,
            t_end=t_end,
            t_step=t_step,
            temperatures=temperatures,
            init_state=init_state,
            output_root=output_root,
            run_name=run_name,
            save_plots=save_plots,
            show_plots=show_plots,
            exclude_inconsistent_pairs=exclude_inconsistent_pairs,
            degenerate_pairs_from_symmetry=degenerate_pairs_from_symmetry,
            sigma=sigma,
        )
    finally:
        if owns_db_logging:
            detach_db_logging()


def _execute_pipeline(
    *,
    sim_folder: str | Path | Sequence[str | Path] | None = None,
    raw_zfs_file: str | Path | Sequence[str | Path] | None = None,
    raw_zfs_file_1d: str | Path | None = None,
    raw_zfs_file_2d: str | Path | None = None,
    coupling_file: str | Path | None = None,
    phonon_file: str | Path | None = None,
    two_phonon_file: str | Path | None = None,
    calc_method: str = "all_bands",
    zfs_folder: str | None = None,
    order: int | None = None,
    pert_scale: float | None = None,
    defect: str | None = None,
    cell_size: int | None = None,
    t_start: float = 0.0,
    t_end: float = 300.0,
    t_step: float = 10.0,
    temperatures: Sequence[float] | None = None,
    init_state: str | Sequence[str] = INIT_STATES,
    output_root: str | Path = "runs",
    run_name: str | None = None,
    save_plots: bool = True,
    show_plots: bool = False,
    exclude_inconsistent_pairs: bool = False,
    degenerate_pairs_from_symmetry: bool = False,
    sigma: float = DEFAULT_SIGMA_MEV,
) -> dict[str, Any]:
    """Pipeline body of :func:`run_full_pipeline` (DB logging already owned)."""

    section(LOGGER, "Load input data")
    init_states = list(dict.fromkeys([init_state] if isinstance(init_state, str) else init_state))
    unknown = [s for s in init_states if s not in INIT_POPULATIONS]
    if unknown or not init_states:
        raise ValueError(f"init_state must be one or more of {', '.join(INIT_STATES)}, got {init_state!r}")
    # ── 1. Resolve order and method ──────────────────────────────────────────
    # ``--order`` is a filter: which orders to analyze / look for. It accepts
    # one or more integers (1, 2); the legacy scalar 12 means "both".
    user_orders: list[int] | None = None
    if order is not None:
        user_orders = parse_orders(order) or None
    is_multi_sim = isinstance(sim_folder, (list, tuple)) and len(sim_folder) > 1
    combine_orders = (raw_zfs_file_1d is not None and raw_zfs_file_2d is not None) or is_multi_sim
    if combine_orders:
        order = 2 if order is None else order  # combined run is handled below
    elif order is None:
        if sim_folder is not None:
            s = str(sim_folder).lower()
            if any(k in s for k in ("second_order", "2d", "2nd")):
                order = 2
            else:
                order = 1
        elif raw_zfs_file is not None and any(k in str(raw_zfs_file).lower() for k in ("second_order", "2d", "2nd")):
            order = 2
        else:
            order = 1

    if user_orders is not None and len(user_orders) == 1:
        # A single explicitly requested order pins the parsing/branch scalar.
        order = user_orders[0]

    if zfs_folder is None:
        zfs_folder = "ZFS_hyp" if calc_method == "all_bands" else "ZFS_occup"

    raw_data: RawZFSData | None = None
    spectrum: PhononSpectrum | None = None
    coupling_data: SpinPhononCouplingData | None = None

    # ── 2. Ingest Input Data ─────────────────────────────────────────────────
    ph_path = None
    _zfs_paths: list[Path] = []
    if coupling_file is not None:
        coupling_data = SpinPhononCouplingData.load(coupling_file)
        defect = coupling_data.defect or defect or "defect"
        cell_size = coupling_data.cell_size or cell_size or 0
        calc_method = coupling_data.calc_method or calc_method
        order = coupling_data.order or order
        dims = [1, 2] if coupling_data.has_second_order else [1]
        if user_orders is not None:
            _check_requested_orders(user_orders, has_1d=None, has_2d=coupling_data.has_second_order)
            analyzed = list(user_orders)
        else:
            analyzed = list(dims)
    else:
        raw_data, spectrum, ph_path = load_raw_inputs(
            sim_folder=sim_folder,
            raw_zfs_file=raw_zfs_file,
            raw_zfs_file_1d=raw_zfs_file_1d,
            raw_zfs_file_2d=raw_zfs_file_2d,
            phonon_file=phonon_file,
            calc_method=calc_method,
            zfs_folder=zfs_folder,
            order=order,
            pert_scale=pert_scale,
            defect=defect,
            cell_size=cell_size,
        )
        if combine_orders:
            order = 2  # both orders present; run combined analysis
        if not (raw_data.first_order or raw_data.second_order):
            raise ValueError(f"No perturbation tensors in the input ({raw_zfs_file or sim_folder}); nothing to analyze")
        defect = raw_data.defect or defect or "defect"
        cell_size = raw_data.cell_size or cell_size or 0

        # Resolve calc_method (avoid string "None")
        if calc_method in (None, "None", ""):
            if raw_data.calc_method not in (None, "None", ""):
                calc_method = raw_data.calc_method
            elif raw_zfs_file:
                for cm in ("all_bands", "defect_band_approx", "approx"):
                    if cm in str(raw_zfs_file):
                        calc_method = "all_bands" if cm == "all_bands" else "defect_band_approx"
                        break

        # Orders actually present in the input data; ``dims`` is the single
        # source of truth for what the data contains, ``order`` keeps the
        # legacy scalar for parsing and branch selection below.
        dims: list[int] = []
        if raw_data is not None:
            if len(raw_data.second_order) > 0:
                dims.append(2)
            if len(raw_data.first_order) > 0:
                dims.append(1)
        if user_orders is not None:
            has_1d = bool(raw_data is not None and len(raw_data.first_order) > 0)
            has_2d = (raw_data is not None and len(raw_data.second_order) > 0) or (
                coupling_data is not None and coupling_data.has_second_order
            )
            _check_requested_orders(user_orders, has_1d=has_1d if raw_data is not None else None, has_2d=has_2d)
            # ``analyzed`` is what this run will actually compute and register:
            # the user filter intersected with reality (validated above).
            analyzed = list(user_orders)
        elif combine_orders or (raw_data is not None and len(raw_data.second_order) > 0):
            order = 2
        elif raw_data is not None and raw_data.order is not None and str(raw_data.order) != "None":
            order = int(raw_data.order)
        elif raw_zfs_file is not None and any(k in str(raw_zfs_file).lower() for k in ("2d", "second_order", "2nd")):
            order = 2
        elif raw_data is not None and len(raw_data.first_order) > 0:
            order = 1
        # No explicit filter: analyze everything the data contains.
        if user_orders is None:
            analyzed = (
                list(dims)
                if dims
                else ([1, 2] if (coupling_data is not None and coupling_data.has_second_order) else [1])
            )

    # ── 2b. Registry: record input files so future lookups and queries work ──
    reg = ensure_db()
    if isinstance(raw_zfs_file, (list, tuple)):
        _zfs_paths = [Path(p) for p in raw_zfs_file]
    elif raw_zfs_file:
        _zfs_paths = [Path(raw_zfs_file)]
    _zfs_paths += [Path(p) for p in (raw_zfs_file_1d, raw_zfs_file_2d) if p]
    for p in _zfs_paths:
        # A packaged file registered by `spinlabs package` keeps its richer
        # metadata (pert_order/pert_scale/calc_method); don't overwrite it with
        # a bare 'raw_zfs' row just because the pipeline read the file.
        cur = reg.execute("SELECT kind FROM files WHERE path = ?", (str(Path(p).resolve()),))
        row = cur.fetchone()
        if row is not None and row[0] == "packaged_zfs":
            continue
        register_file(reg, p, "raw_zfs", defect=str(defect), cell_size=cell_size, calc_method=calc_method)
    if coupling_file is not None:
        register_file(
            reg, coupling_file, "coupling", defect=str(defect), cell_size=cell_size, calc_method=calc_method
        )
    if ph_path is not None and ph_path.exists():
        register_file(reg, ph_path, "phonon", defect=str(defect), cell_size=cell_size)

    # ── 3. Create Unique Run Folder ──────────────────────────────────────────
    pert_eff = (
        pert_scale
        if pert_scale is not None
        else (raw_data.pert_scale if raw_data else coupling_data.pert_scale)
    )
    run_dir = get_unique_run_dir(
        output_root=output_root,
        defect=str(defect),
        cell_size=cell_size,
        calc_method=calc_method,
        order=order,
        pert_scale=pert_eff,
        run_name=run_name,
    )
    name = f"{defect}_{cell_size} {calc_method}"  # names the data in every chapter below
    LOGGER.info(f"data: {name}, orders {sorted(analyzed)}")
    LOGGER.info(f"run folder: {run_dir}")

    # ── 4. Save Raw ZFS Data (if generated/loaded) ───────────────────────────
    raw_path = run_dir / "raw_zfs_data.npz"
    if raw_data is not None:
        raw_data.save(raw_path, spectrum=spectrum)
        LOGGER.info(f"saved raw ZFS data: {raw_path}")

    # ── 5. Calculate Spin-Phonon Couplings ───────────────────────────────────
    section(LOGGER, f"Spin-phonon coupling: {name}")
    coupling_path = run_dir / "spin_phonon_coupling.npz"
    inconsistent_pairs: list[tuple[int, int, float, float]] = []
    symmetry_derived_pairs: list[tuple[int, int]] = []
    if coupling_data is not None:
        coupling_data.save(coupling_path)
        LOGGER.info(f"saved coupling data (copied from the input): {coupling_path}")
    else:
        manager = ZFSManager(
            raw_data=raw_data,
            spectrum=spectrum,
            exclude_inconsistent_pairs=exclude_inconsistent_pairs,
            degenerate_pairs_from_symmetry=degenerate_pairs_from_symmetry,
        )
        if calc_method:
            manager.calc_method = calc_method
        out_base = str(run_dir / "spin_phonon_coupling")
        has_both = bool(manager.zfs_tensors_2d and manager.zfs_tensors)

        # ``analyzed`` drives which analyses run: the user filter (or, when no
        # filter was given, everything the data contains).
        if 1 in analyzed and 2 in analyzed and has_both:
            # Combined 1d + 2d analysis: compute both and save in one file.
            manager.process_both_orders(output_filename=out_base)
        elif 2 in analyzed and manager.zfs_tensors_2d:
            manager.process_second_order_perturbations(output_filename=out_base)
        elif 1 in analyzed:
            manager.process_first_order_perturbations(output_filename=out_base)
        elif manager.zfs_tensors_2d:
            manager.process_second_order_perturbations(output_filename=out_base)
        else:
            manager.process_first_order_perturbations(output_filename=out_base)
        coupling_data = SpinPhononCouplingData.load(coupling_path)
        inconsistent_pairs = manager.inconsistent_pairs
        symmetry_derived_pairs = manager.symmetry_derived_pairs

    # ── 6. Transition Rates Calculation ──────────────────────────────────────
    temps = temperature_grid(t_start, t_end, t_step, temperatures)
    section(LOGGER, f"Transition rates: {name}, {len(temps)} temperatures, sigma {sigma:g} meV")
    started = time.perf_counter()
    valid_temps_arr, rate_results, directional_results = compute_rates(
        coupling_path,
        temps,
        sigma=sigma,
        two_phonon_file=two_phonon_file,
        has_second_order=coupling_data is not None and coupling_data.has_second_order,
    )

    # Save rates
    rates_path = run_dir / "transition_rates.npz"
    np.savez(
        rates_path,
        defect=defect,
        cell_size=cell_size,
        calc_method=calc_method,
        pert_scale=raw_data.pert_scale if raw_data else coupling_data.pert_scale,
        temperatures=valid_temps_arr,
        **rate_results,
    )

    dir_rates_path = run_dir / "directional_rates.npz"
    np.savez(
        dir_rates_path,
        defect=defect,
        cell_size=cell_size,
        calc_method=calc_method,
        pert_scale=raw_data.pert_scale if raw_data else coupling_data.pert_scale,
        temperatures=valid_temps_arr,
        **directional_results,
    )
    LOGGER.info(f"saved transition rates: {rates_path} ({time.perf_counter() - started:.1f} s)")
    LOGGER.info(f"saved directional rates: {dir_rates_path}")

    # ── 7. Population Relaxation Dynamics & T1 Times ─────────────────────────
    # Only this step depends on the initial state, so every state shares the
    # run directory (and the couplings/rates above) and gets its own T1 file.
    section(LOGGER, f"T1 relaxation: {name}, initial state(s) {', '.join(init_states)}")
    t1_paths: dict[str, Path] = {}
    for state in init_states:
        started = time.perf_counter()
        t1_fit_arr, t1_eig_arr = solve_t1(dir_rates_path, state)
        rates_fit = np.where(np.isfinite(t1_fit_arr) & (t1_fit_arr > 0), 1.0 / t1_fit_arr, np.nan)
        rates_eig = np.where(np.isfinite(t1_eig_arr) & (t1_eig_arr > 0), 1.0 / t1_eig_arr, np.nan)

        t1_path = run_dir / t1_file_name(state)
        np.savez(
            t1_path,
            defect=defect,
            cell_size=cell_size,
            calc_method=calc_method,
            order=analyzed,
            init_state=state,
            pert_scale=pert_eff if pert_eff is not None else np.nan,
            sigma_mev=sigma,
            temperatures=valid_temps_arr,
            t1_fit=t1_fit_arr,
            t1_eigenval=t1_eig_arr,
            rates_fit=rates_fit,
            rates_eigenval=rates_eig,
        )
        t1_paths[state] = t1_path
        LOGGER.info(f"saved T1 ({state}): {t1_path} ({time.perf_counter() - started:.1f} s)")

    # ── 8. Summary and Metadata ──────────────────────────────────────────────
    run_info = {
        "defect": defect,
        "cell_size": cell_size,
        "calc_method": calc_method,
        "order": analyzed,
        "timestamp": datetime.now().isoformat(),
        "inputs": {
            "sim_folder": str(sim_folder) if sim_folder else None,
            "raw_zfs_file": str(raw_zfs_file) if raw_zfs_file else None,
            "coupling_file": str(coupling_file) if coupling_file else None,
            "phonon_file": str(phonon_file) if phonon_file else None,
            "two_phonon_file": str(two_phonon_file) if two_phonon_file else None,
        },
        "outputs": {
            "raw_zfs_data": str(raw_path) if raw_data else None,
            "spin_phonon_coupling": str(coupling_path),
            "transition_rates": str(rates_path),
            "directional_rates": str(dir_rates_path),
            "t1_relaxation": {state: str(p) for state, p in t1_paths.items()},
        },
        "temperatures": {
            "min": float(np.min(valid_temps_arr)),
            "max": float(np.max(valid_temps_arr)),
            "count": len(valid_temps_arr),
            "points": len(valid_temps_arr),
        },
        "init_state": init_states,
        "sigma_mev": sigma,
        # Second-order entries departing from the first-order expectation (run folders, 1-based).
        "inconsistent_second_order": {
            "excluded": bool(inconsistent_pairs) and exclude_inconsistent_pairs,
            "pairs": [
                {"pair": f"{i + 1}_{j + 1}", "alpha": round(a, 3), "remainder_mhz": round(r, 3)}
                for i, j, a, r in inconsistent_pairs
            ],
        },
        # Degenerate (Ex, Ey) pairs (1-based modes) whose couplings came from the Ex-Ex curvature.
        "degenerate_pairs_from_symmetry": {
            "enabled": degenerate_pairs_from_symmetry,
            "pairs": [f"{x + 1}_{y + 1}" for x, y in symmetry_derived_pairs],
        },
    }

    info_path = run_dir / "run_info.json"
    with open(info_path, "w") as f:
        json.dump(run_info, f, indent=2)

    # ── 8b. Registry ──
    # The runs table is the only index: one row per analysis run, carrying
    # the metadata the old runs_index.csv used to hold.
    reg = ensure_db()
    run_id = register_run(
        reg,
        run_dir=run_dir,
        zfs_files=[str(Path(p).resolve()) for p in _zfs_paths] if raw_data is not None else None,
        phonon_file=ph_path if ph_path is not None else None,
        method=calc_method,
        init_state=init_states,
        defect=str(defect),
        cell=cell_size,
        order=analyzed,
        pert=pert_eff,
        temp_min=float(np.min(valid_temps_arr)),
        temp_max=float(np.max(valid_temps_arr)),
        n_temps=len(valid_temps_arr),
        npz_files=[p.name for p in sorted(run_dir.glob("*.npz"))],
        git_hash=_git_hash(),
    )
    reg.commit()
    set_db_run_id(run_id)
    LOGGER.debug(f"registered run {run_id}: {run_dir}")

    # Mark the run failed on any unhandled error so queries don't skip
    # files that were never fully analyzed.
    try:
        # ── 9. Optional Plotting ─────────────────────────────────────────────────
        figures_saved = []
        if save_plots:
            fig_dir = run_dir / "figures"
            fig_dir.mkdir(exist_ok=True)
            section(LOGGER, f"Figures: {name}")

            # Coupling plots (1D and 2D diagonal)
            try:
                coupling_display = coupling_data.to_unit("MHz").frequencies_to_unit("meV")

                # 1D coupling plot
                fig_c1, _, _ = plot_1d_spectral_functions(
                    frequencies_mev=coupling_display.frequencies,
                    V_0_0=coupling_display.V_0_0,
                    V_p_m=coupling_display.V_p_m,
                    V_0_pm=coupling_display.V_0_pm,
                    order=1,
                )
                fc1_path = fig_dir / "coupling_spectral_1d.png"
                fig_c1.savefig(fc1_path, dpi=300)
                figures_saved.append(fc1_path)

                # Also save alias coupling_spectral.png for backward compatibility
                fc_path = fig_dir / "coupling_spectral.png"
                fig_c1.savefig(fc_path, dpi=300)
                figures_saved.append(fc_path)

                # 2D coupling plot (diagonal phonons i == j)
                if coupling_display.V2_0_0 is not None:
                    v2_00 = coupling_display.V2_0_0
                    v2_pm = coupling_display.V2_p_m
                    v2_0pm = coupling_display.V2_0_pm

                    v2_00_diag = np.diag(v2_00) if v2_00.ndim == 2 else v2_00
                    v2_pm_diag = (
                        np.diag(v2_pm)
                        if (v2_pm is not None and v2_pm.ndim == 2)
                        else (v2_pm if v2_pm is not None else np.zeros_like(v2_00_diag))
                    )
                    v2_0pm_diag = (
                        np.diag(v2_0pm)
                        if (v2_0pm is not None and v2_0pm.ndim == 2)
                        else (v2_0pm if v2_0pm is not None else np.zeros_like(v2_00_diag))
                    )

                    fig_c2, _, _ = plot_1d_spectral_functions(
                        frequencies_mev=coupling_display.frequencies,
                        V_0_0=v2_00_diag,
                        V_p_m=v2_pm_diag,
                        V_0_pm=v2_0pm_diag,
                        order=2,
                    )
                    fc2_path = fig_dir / "coupling_spectral_2d.png"
                    fig_c2.savefig(fc2_path, dpi=300)
                    figures_saved.append(fc2_path)
                for path in figures_saved:
                    LOGGER.info(f"saved figure: {path}")
            except Exception as e:
                LOGGER.warning(f"{name}: coupling plot failed: {e}")

            # Transition rate stacked plots
            try:
                fr_path = fig_dir / "transition_rates.png"
                plot_transition_rates_stacked(rates_path, output_path=fr_path)
                figures_saved.append(fig_dir / "transition_rates_0_1.png")
                figures_saved.append(fig_dir / "transition_rates_1_-1.png")
            except Exception as e:
                LOGGER.warning(f"{name}: transition rate plots failed: {e}")

            # T1 plot
            try:
                ft1_path = fig_dir / "t1_vs_temperature.png"
                plot_t1(list(t1_paths.values()), output_path=ft1_path, plain_name=True)
                figures_saved.append(ft1_path)
            except Exception as e:
                LOGGER.warning(f"{name}: T1 plot failed: {e}")

        LOGGER.info(f"run finished: {run_dir}")

        finish_run(reg, run_id, "done")
        reg.commit()

        return {
            "run_dir": run_dir,
            "run_info": run_info,
            "raw_zfs_path": raw_path if raw_data else None,
            "coupling_path": coupling_path,
            "rates_path": rates_path,
            "directional_rates_path": dir_rates_path,
            "t1_paths": t1_paths,
            "figures": figures_saved,
        }
    except Exception:
        finish_run(reg, run_id, 'failed')
        reg.commit()
        raise

