"""Shared computations of the paper figures: couplings, rates per process, T1 with fit details, JSON records."""

import hashlib
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.constants as Cn

import config as C

sys.path.insert(0, str(C.ROOT / "scripts"))  # coupling_manager and the spectral helpers live in scripts/
from plot_t1_comparison import coupling_manager  # noqa: E402

from spinlabs.constants import CONSTANTS  # noqa: E402
from spinlabs.models import SpinPhononCouplingData  # noqa: E402
from spinlabs.relaxation_dynamics import INIT_POPULATIONS, RelaxationDynamics  # noqa: E402
from spinlabs.transition_rate import TransitionRate  # noqa: E402

logging.disable(logging.WARNING)  # per-run pipeline warnings would bury the summaries

DIRECTIONS = ("0_to_1", "0_to_-1", "1_to_0", "-1_to_0", "1_to_-1", "-1_to_1")
ALL_PROCESSES = ("first_order", "second_order", "two_phonon")


def drop_a_x_e(manager):
    """Leave out the computed A x E pairs (as plot_spectral_comparison.drop_a_x_e)."""
    a, e = {"A1", "A2"}, {"Ex", "Ey"}
    manager.zfs_tensors_2d = {k: v for k, v in manager.zfs_tensors_2d.items()
                              if not (set(v["symmetry"]) & a and set(v["symmetry"]) & e)}
    return manager


def manager_for(system, scheme, pairs):
    """ZFSManager of a production package with a second-order pair treatment.

    pairs: "sqrt" (computed + fill for the rest: production), "computed", "diagonal", "diagonal-sqrt"
    (diagonal + fill everywhere off the diagonal), "no_axe" (computed without the A x E pairs).
    """
    mode = "computed" if pairs == "no_axe" else pairs
    manager, convention = coupling_manager(C.package(system, scheme), mode)
    if pairs == "no_axe":
        drop_a_x_e(manager)
    return manager, convention


def _key(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def couplings(system, scheme, pairs=C.PAIRS):
    """Coupling arrays in MHz (V_0_0, V_p_m, V_0_pm per mode; V2_* N x N) and the mode energies in meV."""
    cache = C.CACHE / f"couplings_{system}_{scheme}_{pairs}_{_key(C.package(system, scheme))}.npz"
    if not cache.exists():
        manager, convention = manager_for(system, scheme, pairs)
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / "coupling")
            manager.process_both_orders(output_filename=base)
            data = SpinPhononCouplingData.load(f"{base}.npz").to_unit("MHz").frequencies_to_unit("meV")
        arrays = {k: np.asarray(getattr(data, k), dtype=float)
                  for k in ("V_0_0", "V_p_m", "V_0_pm", "V2_0_0", "V2_p_m", "V2_0_pm")}
        C.CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(cache, frequencies=np.asarray(data.frequencies, dtype=float), convention=convention,
                 symmetries=np.asarray([str(s) for s in manager.spectrum.symmetries]), **arrays)
    d = np.load(cache, allow_pickle=True)
    return {k: d[k] for k in d.files}


def rates(system, scheme, pairs=C.PAIRS, nogap=C.NOGAP, sigma=C.SIGMA, direct=C.DIRECT, temps=C.TEMPERATURES):
    """Directional rates (1/s) per process at ``temps``: {"temperatures", "<process>:<direction>", "debye_C", ...}."""
    pkg = C.package(system, scheme)
    key = _key(str(pkg), pairs, nogap, sigma, direct, list(map(float, temps)))
    cache = C.CACHE / f"rates_{system}_{scheme}_{pairs}_{'nogap' if nogap else 'gap'}_s{sigma:g}_{direct}_{key}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}
    print(f"  computing rates: {system} {scheme} pairs={pairs} nogap={nogap} sigma={sigma:g} direct={direct}")
    manager, convention = manager_for(system, scheme, pairs)
    out = {"temperatures": np.asarray(temps, dtype=float), "convention": convention, "package": str(pkg)}
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "coupling")
        manager.process_both_orders(output_filename=base)
        calc = TransitionRate(f"{base}.npz")
        calc.data_2ph = calc.data  # the coupling file carries the second-order coefficients
        if direct == "debye":
            fit = calc.set_debye_from_spectrum(manager.spectrum)
            out["debye_C"] = fit.C
        else:
            calc.direct = direct
        omega, j0pm, jpm, j00 = calc.get_spectral_density(res=0.02, sigma=sigma)
        groups = (manager.spectrum.deg_groups or []) if nogap else None
        lines = calc.get_line_spectral_density(sigma, degenerate_groups=groups)
        zfs = calc.data["zfs"] / CONSTANTS["meV2J"]
        out["zfs_mev"] = float(zfs)
        series = {f"{p}:{d}": [] for p in ALL_PROCESSES for d in DIRECTIONS}
        for T in temps:
            calc.compute_transition_rates(T, omega, j0pm, jpm, j00, zfs)
            calc.compute_two_phonon_rates(T, lines)
            for p in ALL_PROCESSES:
                for d in DIRECTIONS:
                    series[f"{p}:{d}"].append(float(calc.transition_rate[p].get(d, 0.0)))
    out.update({k: np.asarray(v) for k, v in series.items()})
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def directional(rate_data, processes=C.PROCESSES):
    """Total directional rates over ``processes`` (production: direct + two-phonon)."""
    return {d: sum(np.asarray(rate_data[f"{p}:{d}"], dtype=float) for p in processes) for d in DIRECTIONS}


def t1_details(temps, directional_rates, state):
    """T1 (s) by the population fit, T1 by the eigenvalue, and the source per temperature
    (RelaxationDynamics.get_T1_fit_details: "fit", "eigenvalue" fallback, or "none" at 0 K)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "directional.npz"
        np.savez(path, temperatures=np.asarray(temps, dtype=float), **directional_rates)
        dyn = RelaxationDynamics(INIT_POPULATIONS[state], str(path))
        rows = [dyn.get_T1_fit_details(temp_index=i) for i in range(len(dyn.temperatures))]
    t1, t1_eig, source = zip(*rows)
    return np.asarray(t1, dtype=float), np.asarray(t1_eig, dtype=float), list(source)


def cambria_rates(temps, p=C.CAMBRIA):
    """Directional rates of the Cambria et al. (2023) model: Omega on 0 <-> +-1, gamma on +1 <-> -1."""
    kT = Cn.k / Cn.e * 1e3 * np.asarray(temps, dtype=float)  # meV

    def nn(delta):
        with np.errstate(over="ignore", divide="ignore"):
            n = np.where(kT > 0, 1.0 / np.expm1(delta / np.where(kT > 0, kT, 1.0)), 0.0)
        return n * (n + 1)

    omega = p["A1"] * nn(p["Delta1"]) + p["A2"] * nn(p["Delta2"]) + p["A3"]
    gamma = p["B1"] * nn(p["Delta1"]) + p["B2"] * nn(p["Delta2"]) + p["B3"]
    return {"0_to_1": omega, "0_to_-1": omega, "1_to_0": omega, "-1_to_0": omega,
            "1_to_-1": gamma, "-1_to_1": gamma}, omega, gamma


def at(temps, values, targets):
    """values at the grid temperatures ``targets`` (exact grid points, no interpolation)."""
    temps = np.asarray(temps, dtype=float)
    out = {}
    for T in targets:
        idx = np.flatnonzero(np.isclose(temps, T))
        if not len(idx):
            raise ValueError(f"{T} K is not on the temperature grid")
        out[f"{T:g}"] = float(np.asarray(values)[idx[0]])
    return out


class Record:
    """Settings, numbers and T1 fit diagnostics of one figure, written to figures/out/records/<fig>.json."""

    def __init__(self, fig_id, title, script, kind):
        self.data = {"id": fig_id, "title": title, "script": script, "kind": kind, "files": [],
                     "settings": {}, "numbers": {}, "fallbacks": [], "fit_vs_eigen": []}

    def t1_curve(self, curve_label, temps, t1, t1_eig, source):
        """Log fallback points and the largest |fit - eig| / eig of a curve where both exist."""
        temps = np.asarray(temps, dtype=float)
        for T, s in zip(temps, source):
            if s == "eigenvalue":
                self.data["fallbacks"].append({"curve": curve_label, "T_K": float(T)})
        both = np.array([s == "fit" for s in source]) & np.isfinite(t1_eig) & (t1_eig > 0)
        if both.any():
            rel = np.abs(t1[both] - t1_eig[both]) / t1_eig[both]
            k = int(np.argmax(rel))
            self.data["fit_vs_eigen"].append({"curve": curve_label, "max_rel_diff": float(rel[k]),
                                              "at_T_K": float(temps[both][k]), "n_points": int(both.sum())})

    def check_fallbacks(self):
        """Stop before plotting when an eigenvalue-fallback point lies above FALLBACK_LIMIT_K."""
        high = [f for f in self.data["fallbacks"] if f["T_K"] > C.FALLBACK_LIMIT_K]
        if high:
            for f in high:
                print(f"  FALLBACK above {C.FALLBACK_LIMIT_K:g} K: {f['curve']} at {f['T_K']:g} K")
            self.write()
            raise SystemExit(f"{len(high)} eigenvalue-fallback T1 points above {C.FALLBACK_LIMIT_K:g} K; not plotting")

    def write(self):
        C.RECORDS.mkdir(parents=True, exist_ok=True)
        path = C.RECORDS / f"{self.data['id']}.json"
        path.write_text(json.dumps(self.data, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        print("record:", path.relative_to(C.OUT))
