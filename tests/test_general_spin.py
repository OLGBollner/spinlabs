"""Rates and T1 for any spin S: matrix elements, selection rules, detailed balance, time reversal, and the spin-1
regression against the frozen spin-1 code (spinlabs.legacy_spin1)."""

from pathlib import Path

import numpy as np
import pytest
from scipy import constants as Cn

from spinlabs.constants import CONSTANTS
from spinlabs.debye import DebyeFit
from spinlabs.legacy_spin1 import RelaxationDynamicsSpin1, TransitionRateSpin1
from spinlabs.models import SpinPhononCouplingData, ZFSTensor
from spinlabs.relaxation_dynamics import INIT_POPULATIONS, RelaxationDynamics, init_populations, init_states
from spinlabs.spin import (
    direction_key,
    directions,
    element_dm1,
    element_dm2,
    level_energies,
    spin_factor,
    spin_matrices,
    spin_projections,
    transition_class,
    zfs_hamiltonian,
)
from spinlabs.transition_rate import TransitionRate

MEV = CONSTANTS["meV2J"]
KB_MEV = Cn.k / Cn.e * 1000
SPINS = (1.0, 1.5, 2.0, 2.5)


def _random_traceless(rng):
    d = rng.normal(size=(3, 3))
    d = (d + d.T) / 2
    return d - np.trace(d) / 3 * np.eye(3)


def _channel_amplitudes(d):
    """V_0pm and V_+- of d by the formulas of ZFSManager.calculate_first_order_derivatives (the spin-1 elements)."""
    v0pm = np.sqrt(d[0, 2] ** 2 + d[1, 2] ** 2) / np.sqrt(2)
    vpm = 0.5 * np.sqrt((d[0, 0] - d[1, 1]) ** 2 + 4.0 * d[0, 1] ** 2)
    return {1: v0pm, 2: vpm}


# ------------------------------------------------------------------ spin operators and matrix elements


@pytest.mark.parametrize("spin", SPINS)
def test_spin_matrices_are_angular_momentum(spin):
    sx, sy, sz = spin_matrices(spin)
    assert np.allclose(sx @ sy - sy @ sx, 1j * sz)
    assert np.allclose(sx @ sx + sy @ sy + sz @ sz, spin * (spin + 1) * np.eye(int(2 * spin + 1)))
    assert spin_projections(spin) == [spin - k for k in range(int(2 * spin + 1))]


@pytest.mark.parametrize("spin", (1.0, 1.5, 2.0))
@pytest.mark.parametrize("seed", range(5))
def test_analytic_elements_match_built_hamiltonian(spin, seed):
    """<m+1|S.d.S|m> = (1/2) p* (2m+1) sqrt(S(S+1) - m(m+1)), <m+2|S.d.S|m> = (1/4) w* sqrt(...), diagonal
    (3/2) d_zz (m^2 - S(S+1)/3), for random traceless symmetric d."""
    d = _random_traceless(np.random.default_rng(seed))
    h = zfs_hamiltonian(spin, d)
    ms = spin_projections(spin)
    idx = {m: k for k, m in enumerate(ms)}
    p = d[0, 2] + 1j * d[1, 2]
    w = (d[0, 0] - d[1, 1]) + 2j * d[0, 1]
    for m in ms:
        assert h[idx[m], idx[m]] == pytest.approx(1.5 * d[2, 2] * (m * m - spin * (spin + 1) / 3), abs=1e-12)
        if m + 1 in idx:
            assert h[idx[m + 1], idx[m]] == pytest.approx(0.5 * np.conj(p) * element_dm1(spin, m), abs=1e-12)
        if m + 2 in idx:
            assert h[idx[m + 2], idx[m]] == pytest.approx(0.25 * np.conj(w) * element_dm2(spin, m), abs=1e-12)
        for mp in ms:
            if abs(mp - m) > 2:
                assert abs(h[idx[mp], idx[m]]) < 1e-12


@pytest.mark.parametrize("spin", (1.0, 1.5, 2.0))
@pytest.mark.parametrize("seed", range(5))
def test_weight_is_channel_amplitude_squared_times_spin_factor(spin, seed):
    """|<m'|S.d.S|m>|^2 = V^2 spin_factor(S, m, m') with V the zfs_manager channel amplitude of d."""
    d = _random_traceless(np.random.default_rng(100 + seed))
    h = zfs_hamiltonian(spin, d)
    amplitudes = _channel_amplitudes(d)
    ms = spin_projections(spin)
    for i, m in enumerate(ms):
        for j, mp in enumerate(ms):
            dm = abs(m - mp)
            if dm in (1, 2):
                assert abs(h[j, i]) ** 2 == pytest.approx(amplitudes[dm] ** 2 * spin_factor(spin, m, mp), rel=1e-12)
            elif dm > 2:
                assert spin_factor(spin, m, mp) == 0.0


def test_spin_one_factors_are_exactly_one():
    for m in (1.0, 0.0, -1.0):
        for mp in (1.0, 0.0, -1.0):
            if m != mp:
                assert spin_factor(1.0, m, mp) == 1.0


def test_spin_three_halves_factors():
    assert spin_factor(1.5, 0.5, 1.5) == spin_factor(1.5, -1.5, -0.5) == 6.0  # dm = 1 between the doublets
    assert spin_factor(1.5, -0.5, 1.5) == spin_factor(1.5, -1.5, 0.5) == 3.0  # dm = 2
    assert transition_class(1.5, 0.5) == transition_class(-0.5, -1.5) == "0.5_1.5"
    assert transition_class(-0.5, 1.5) == transition_class(0.5, -1.5) == "0.5_-1.5"
    assert transition_class(0.0, -1.0) == "0_1" and transition_class(-1.0, 1.0) == "1_-1"


def test_get_V2ph_is_an_amplitude():
    """_get_V2ph: F_z V_00 on the diagonal (not F_z^2), sqrt(spin_factor) times V_0pm / V_+- off it."""
    v0pm, vpm, v00 = np.array([2.0, -3.0]), np.array([5.0, 7.0]), np.array([0.5, -1.5])

    def amp(spin, m1, m2):
        return TransitionRate(spin=spin)._get_V2ph(m1, m2, v0pm, vpm, v00)

    # S = 1: F_z = 1, -2, 1 and every spin factor is 1
    np.testing.assert_allclose(amp(1.0, 1.0, 1.0), v00)
    np.testing.assert_allclose(amp(1.0, 0.0, 0.0), -2.0 * v00)
    np.testing.assert_allclose(amp(1.0, 0.0, -1.0), v0pm)
    np.testing.assert_allclose(amp(1.0, 1.0, -1.0), vpm)
    # S = 3/2: F_z = 3, -3; spin factors 6 (dm = 1 between the doublets) and 3 (dm = 2); Kramers and dm = 3 zero
    np.testing.assert_allclose(amp(1.5, 1.5, 1.5), 3.0 * v00)
    np.testing.assert_allclose(amp(1.5, -0.5, -0.5), -3.0 * v00)
    np.testing.assert_allclose(amp(1.5, 0.5, 1.5), np.sqrt(6.0) * v0pm)
    np.testing.assert_allclose(amp(1.5, -0.5, 1.5), np.sqrt(3.0) * vpm)
    np.testing.assert_array_equal(amp(1.5, 0.5, -0.5), 0.0)
    np.testing.assert_array_equal(amp(1.5, 1.5, -1.5), 0.0)


# ------------------------------------------------------------------ selection rules


@pytest.mark.parametrize("spin", (1.5, 2.5))
@pytest.mark.parametrize("seed", range(5))
def test_kramers_and_dm3_weights_vanish(spin, seed):
    d = _random_traceless(np.random.default_rng(200 + seed))
    h = zfs_hamiltonian(spin, d)
    ms = spin_projections(spin)
    scale = np.abs(h).max()
    for i, m in enumerate(ms):
        for j, mp in enumerate(ms):
            if mp == -m or abs(m - mp) >= 3:
                assert abs(h[j, i]) < 1e-13 * scale
                assert spin_factor(spin, m, mp) == 0.0


# ------------------------------------------------------------------ rates on synthetic couplings

FREQS_MEV = np.array([12.0, 12.0, 30.0, 42.0, 55.0, 61.0])


def _coupling(zfs_mev, spin=None, seed=0):
    rng = np.random.default_rng(seed)
    n = len(FREQS_MEV)
    v2 = [rng.uniform(0.5, 1.5, (n, n)) for _ in range(3)]
    v2 = [(v + v.T) / 2 for v in v2]
    return SpinPhononCouplingData(
        order=2,
        defect="X",
        cell_size=64,
        pert_scale=0.1,
        calc_method="all_bands",
        frequencies=FREQS_MEV * MEV,
        frequency_unit="J",
        ground_state_zfs=ZFSTensor(matrix=np.diag([-1.0, -1.0, 2.0]) * zfs_mev / 3 * MEV, unit="J"),
        V_0_0=rng.uniform(0.5, 2.0, n) * MEV,
        V_p_m=rng.uniform(0.5, 2.0, n) * MEV,
        V_0_pm=rng.uniform(0.5, 2.0, n) * MEV,
        V2_0_0=v2[0] * MEV,
        V2_p_m=v2[1] * MEV,
        V2_0_pm=v2[2] * MEV,
        coupling_unit="J",
        symmetries=["Ex", "Ey", "A1", "A1", "Ex", "A2"],
        iprs=np.full(n, 0.2),
        spin=spin,
    )


def _fit(C):
    return DebyeFit(C=C, q=0.44, branches={}, acoustic_modes=np.array([]), weights=np.array([]))


def _rates(calc, T, sigma=3.0, groups=None):
    """Directional rates per process at T, Debye direct (C_0pm, C_+-) and two-phonon lines."""
    calc.debye_fit, calc.debye_fit_pm, calc.direct = _fit(2e-3), _fit(5e-3), "debye"
    omega, j0pm, jpm, j00 = calc.get_spectral_density(res=0.02, sigma=sigma)
    lines = calc.get_line_spectral_density(sigma, degenerate_groups=groups)
    calc.compute_transition_rates(T, omega, j0pm, jpm, j00, calc.data["zfs"] / MEV)
    calc.compute_two_phonon_rates(T, lines)
    return {p: dict(r) for p, r in calc.transition_rate.items()}


def test_spin_travels_with_the_coupling_file(tmp_path):
    path = tmp_path / "c.npz"
    _coupling(0.1, spin=1.5).save(path)
    calc = TransitionRate(str(path))
    assert calc.spin == 1.5 and len(calc.directions) == 12
    assert TransitionRate(str(path), spin=2.0).spin == 2.0  # explicit spin wins
    _coupling(0.1, spin=None).save(path)
    assert TransitionRate(str(path)).spin == 1.0  # no recorded spin: S = 1


@pytest.mark.parametrize("spin", (1.5, 2.5))
@pytest.mark.parametrize("T", (10.0, 300.0))
def test_kramers_and_dm3_rates_are_zero(spin, T):
    calc = TransitionRate(spin=spin)
    calc.load_data(_coupling(2.0))
    calc.spin = spin
    rates = _rates(calc, T)
    for m in spin_projections(spin):
        for mp in spin_projections(spin):
            if m != mp and (mp == -m or abs(m - mp) >= 3):
                key = direction_key(m, mp)
                assert rates["first_order"][key] == 0.0 and rates["two_phonon"][key] == 0.0
    assert rates["second_order"] == {}  # the incoherent first-order Raman is S = 1 only
    assert rates["two_phonon"][direction_key(0.5, 1.5)] > 0 and rates["two_phonon"][direction_key(-0.5, 1.5)] > 0


@pytest.mark.parametrize("spin", SPINS)
@pytest.mark.parametrize("T", (20.0, 150.0))
def test_detailed_balance(spin, T):
    """Gamma(m -> m') / Gamma(m' -> m) = exp(-(E_m' - E_m) / kT) for the direct and the two-phonon rates."""
    D = 2.0  # meV: gaps of 2-10 meV, so the Boltzmann factors are far from 1
    calc = TransitionRate()
    calc.load_data(_coupling(D))
    calc.spin = spin
    rates = _rates(calc, T)
    energies = level_energies(spin, D)
    for m in spin_projections(spin):
        for mp in spin_projections(spin):
            if m == mp or energies[m] == energies[mp] or spin_factor(spin, m, mp) == 0:
                continue
            boltzmann = np.exp(-(energies[mp] - energies[m]) / (KB_MEV * T))
            fwd, back = direction_key(m, mp), direction_key(mp, m)
            assert rates["first_order"][fwd] / rates["first_order"][back] == pytest.approx(boltzmann, rel=1e-12)
            assert rates["two_phonon"][fwd] / rates["two_phonon"][back] == pytest.approx(boltzmann, rel=2e-3)


@pytest.mark.parametrize("spin", SPINS)
def test_time_reversal(spin):
    calc = TransitionRate()
    calc.load_data(_coupling(0.7))
    calc.spin = spin
    rates = _rates(calc, 120.0, groups=[[0, 1]])
    for process in ("first_order", "two_phonon"):
        for m in spin_projections(spin):
            for mp in spin_projections(spin):
                if m != mp:
                    assert rates[process][direction_key(m, mp)] == pytest.approx(
                        rates[process][direction_key(-m, -mp)], rel=1e-13, abs=0
                    )


# ------------------------------------------------------------------ T1


def _rates_file(path, temps, rates, spin):
    np.savez(path, temperatures=np.asarray(temps, dtype=float), spin=spin,
             **{k: np.asarray(v, dtype=float) for k, v in rates.items()})
    return path


def test_three_halves_doublet_t1_is_one_over_two_a_plus_b(tmp_path):
    """Symmetric rates a (dm = 1 between the doublets) and b (dm = 2): P(+-1/2) - 1/2 ~ exp(-2 (a + b) t)."""
    a, b = 3.0, 1.0
    rates = {}
    for m in spin_projections(1.5):
        for mp in spin_projections(1.5):
            if m != mp:
                f = spin_factor(1.5, m, mp)
                rates[direction_key(m, mp)] = [a if f == 6.0 else b if f == 3.0 else 0.0] * 2
    path = _rates_file(tmp_path / "r.npz", [100.0, 200.0], rates, 1.5)
    for state in ("ms_pm1_2", "ms_pm3_2"):
        dyn = RelaxationDynamics(init_populations(1.5)[state], str(path))
        t1, _, source = dyn.get_T1_fit_details(temp_index=0)
        assert source == "fit" and t1 == pytest.approx(1 / (2 * (a + b)), rel=1e-6)
    assert init_states(1.5) == ("ms_pm1_2", "ms_pm3_2")
    assert init_populations(1.5)["ms_pm1_2"] == [0.0, 0.5, 0.5, 0.0]


# ------------------------------------------------------------------ spin-1 regression against the frozen code

REG_TEMPS = np.arange(1.0, 601.0, 1.0)
REG_REL = 1e-10


def _assert_close(new, old, what):
    new, old = np.asarray(new, dtype=float), np.asarray(old, dtype=float)
    scale = np.maximum(np.abs(old), 1e-300)
    worst = np.nanmax(np.where(np.isfinite(old) | np.isfinite(new), np.abs(new - old) / scale, 0.0))
    assert np.array_equal(np.isnan(new), np.isnan(old)), what
    assert worst <= REG_REL, f"{what}: max relative difference {worst:.3e}"


def _regression(make_new, make_old, temps, tmp_path, processes, prepare, sigma, groups):
    """Directional rates per process and T1 (ms_0, ms_1, ms_-1) of the general (S = 1) and the frozen code."""
    out = {}
    for name, make, dyn_cls in (("new", make_new, RelaxationDynamics), ("old", make_old, RelaxationDynamicsSpin1)):
        calc = make()
        prepare(calc)
        omega, j0pm, jpm, j00 = calc.get_spectral_density(res=0.02, sigma=sigma)
        lines = calc.get_line_spectral_density(sigma, degenerate_groups=groups)
        zfs = calc.data["zfs"] / MEV
        series = {p: {d: [] for d in directions(1.0)} for p in ("first_order", "second_order", "two_phonon")}
        totals = []
        for T in temps:
            calc.compute_transition_rates(T, omega, j0pm, jpm, j00, zfs)
            calc.compute_two_phonon_rates(T, lines)
            for p in series:
                for d in series[p]:
                    series[p][d].append(calc.transition_rate[p].get(d, 0.0))
            totals.append(calc.get_total_rates())
        directional = {d: np.sum([series[p][d] for p in processes], axis=0) for d in directions(1.0)}
        path = _rates_file(tmp_path / f"{name}.npz", temps, directional, 1.0)
        t1 = {}
        for state in INIT_POPULATIONS:
            dyn = dyn_cls(INIT_POPULATIONS[state], str(path))
            t1[state] = [dyn.get_T1_fit_details(temp_index=i)[:2] for i in range(len(temps))]
        out[name] = (series, totals, t1)
    (s_new, tot_new, t1_new), (s_old, tot_old, t1_old) = out["new"], out["old"]
    for p in s_old:
        for d in s_old[p]:
            _assert_close(s_new[p][d], s_old[p][d], f"{p} {d}")
    for k in ("0_1", "1_-1"):
        for p in s_old:
            _assert_close([t[p][k] for t in tot_new], [t[p][k] for t in tot_old], f"total {p} {k}")
    for state in t1_old:
        _assert_close(np.array(t1_new[state]), np.array(t1_old[state]), f"T1 {state}")


def test_spin_one_regression_synthetic(tmp_path):
    """General code at S = 1 = frozen spin-1 code on random couplings: every process (incl. the first-order
    Raman), Gaussian and Debye direct rates, with and without the zero-gap pairs."""
    coupling = _coupling(0.0119, seed=3)

    def make(cls):
        def build():
            calc = cls()
            calc.load_data(coupling)
            return calc

        return build

    def debye(calc):
        calc.debye_fit, calc.direct = _fit(2e-3), "debye"

    temps = np.arange(1.0, 601.0, 7.0)
    for prepare, groups in ((debye, [[0, 1]]), (lambda c: None, None)):
        _regression(make(TransitionRate), make(TransitionRateSpin1), temps, tmp_path,
                    ("first_order", "second_order", "two_phonon"), prepare, 3.5, groups)


# Production packages of scripts/paper/config.py (sigma 3.5 meV, Debye direct, zero-gap pairs excluded, computed
# pairs plus the geometric-mean fill, inconsistent second-order entries excluded, direct + two-phonon)
DATA = Path(__file__).resolve().parents[1] / "data"
PRODUCTION = {
    "NV_64 AO": "NV_64/pert_0.1/all_bands/20261029/NV_64_raw_zfs_data_all_bands_12d.npz",
    "NV_64 UO": "NV_64/pert_0.1/defect_band_approx/20261021/NV_64_raw_zfs_data_defect_band_approx_12d.npz",
    "DiV_128 AO": "DiV_128/pert_0.05/all_bands/20261105/DiV_128_raw_zfs_data_all_bands_12d.npz",
    "DiV_128 UO": "DiV_128/pert_0.05/defect_band_approx/20261105/DiV_128_raw_zfs_data_defect_band_approx_12d.npz",
}


@pytest.mark.parametrize("case", list(PRODUCTION))
def test_spin_one_regression_production(case, tmp_path):
    raw_file = DATA / PRODUCTION[case]
    if not raw_file.exists():
        pytest.skip(f"production package not available: {raw_file}")
    from spinlabs.pipeline import load_raw_inputs
    from spinlabs.zfs_manager import ZFSManager

    method = "defect_band_approx" if "defect_band_approx" in raw_file.name else "all_bands"
    raw, spectrum, _ = load_raw_inputs(raw_zfs_file=raw_file, calc_method=method, zfs_folder="ZFS_hyp", order=2)
    manager = ZFSManager(raw_data=raw, spectrum=spectrum, exclude_inconsistent_pairs=True, offdiag_from_diagonal=True)
    manager.calc_method = method
    base = str(tmp_path / "coupling")
    manager.process_both_orders(output_filename=base)

    def make(cls):
        def build():
            calc = cls(f"{base}.npz")
            calc.data_2ph = calc.data
            return calc

        return build

    def debye(calc):
        calc.set_debye_from_spectrum(manager.spectrum)

    assert make(TransitionRate)().spin == 1.0
    _regression(make(TransitionRate), make(TransitionRateSpin1), REG_TEMPS, tmp_path,
                ("first_order", "two_phonon"), debye, 3.5, manager.spectrum.deg_groups or [])
