"""Grid-based Fermi golden rule rates must not depend on the grid spacing."""

import numpy as np
import pytest
from scipy import constants as Cn

from spinlabs.constants import CONSTANTS
from spinlabs.models import SpinPhononCouplingData, ZFSTensor
from spinlabs.transition_rate import Phonons, TransitionRate

MEV = CONSTANTS["meV2J"]
FREQS_MEV = np.array([30.0, 30.0, 42.0, 55.0, 61.0])
SIGMA = 3.0
T = 300.0


def _calculator(zfs_mev=2.87e-3 * 4.1357):
    """Couplings with on-shell (degenerate) and off-shell pairs; zfs in meV."""
    rng = np.random.default_rng(0)
    n = len(FREQS_MEV)
    v2 = rng.uniform(0.5, 1.5, (n, n))
    v2 = (v2 + v2.T) / 2
    coupling = SpinPhononCouplingData(
        order=2,
        defect="NV",
        cell_size=64,
        pert_scale=0.1,
        calc_method="all_bands",
        frequencies=FREQS_MEV * MEV,
        frequency_unit="J",
        ground_state_zfs=ZFSTensor(matrix=np.diag([-1.0, -1.0, 2.0]) * zfs_mev / 3 * MEV, unit="J"),
        V_0_0=np.linspace(1.0, 2.0, n) * MEV,
        V_p_m=np.linspace(2.0, 1.0, n) * MEV,
        V_0_pm=np.linspace(1.5, 0.5, n) * MEV,
        V2_0_0=v2 * MEV,
        V2_p_m=2 * v2 * MEV,
        V2_0_pm=3 * v2 * MEV,
        coupling_unit="J",
        symmetries=["Ex", "Ey", "A1", "A1", "A1"],
        iprs=np.full(n, 0.2),
    )
    calc = TransitionRate()
    calc.load_data(coupling)
    return calc


def _gauss(x, width):
    return np.exp(-0.5 * (x / width) ** 2) / (np.sqrt(2 * np.pi) * width)


def _pair_sum(calc, key, omega_zfs):
    """Golden rule summed over the two-phonon final states; each mode smeared by SIGMA, so pairs by SIGMA*sqrt(2).

    V2 holds the 1/2 of the Taylor series, so a pair i != j enters H as 2 V2_ij q_i q_j:
    absorbing i and emitting j (one final state per ordered pair) has |M|^2 = 4 |V2_ij|^2 n_i (n_j + 1),
    absorbing both (one final state per unordered pair) 4 |V2_ij|^2 n_i n_j. Diagonal pairs keep the
    diagonal-only convention, 2 |V2_ii|^2 n (n + 1) (Cambria et al., Eq. S12).
    """
    v2 = (np.asarray(calc.data[key]) / MEV) ** 2
    off = 1.0 - np.eye(len(FREQS_MEV))
    n = Phonons.bose_einstein(FREQS_MEV, T)
    width = np.sqrt(2) * SIGMA
    w_a, w_e = FREQS_MEV[:, None], FREQS_MEV[None, :]
    raman = (2 + 2 * off) * v2 * n[:, None] * (n[None, :] + 1) * _gauss(w_a - w_e - omega_zfs, width)
    double_abs = (1 + off) * v2 * n[:, None] * n[None, :] * _gauss(w_a + w_e - omega_zfs, width)
    return 2 * np.pi / (Cn.hbar / MEV) * (raman.sum() + double_abs.sum())


def test_pair_weight_matches_exact_matrix_element():
    """|<+1, n_a - 1, n_e + 1| H2 |-1, n_a, n_e>|^2 for H2 = 1/2 sum_ij D_ij q_i q_j, q = b + b^dagger.

    compute_two_phonon_rates gives one ordered pair this final state twice (abs_em and em_abs),
    each with the pair weight, so the weight must be half of |M|^2 / (n_a (n_e + 1)).
    """
    nmax, n_a, n_e = 5, 2, 1
    b = np.diag(np.sqrt(np.arange(1, nmax)), 1)
    q = b + b.T
    s_plus = np.sqrt(2) * np.diag([1.0, 1.0], 1)
    spin = [(s_plus + s_plus.T) / 2, (s_plus - s_plus.T) / 2j, np.diag([1.0, 0.0, -1.0])]
    d = np.random.default_rng(1).normal(size=(3, 3))
    d = d + d.T  # mixed second derivative d2D/dq_a dq_e
    sds = sum(d[i, j] * spin[i] @ spin[j] for i in range(3) for j in range(3))
    h2 = np.kron(sds, np.kron(q, q))  # 1/2 (D_ae q_a q_e + D_ea q_e q_a)

    def ket(m, k1, k2):
        v = np.zeros(3 * nmax * nmax)
        v[(m * nmax + k1) * nmax + k2] = 1.0
        return v

    m2 = abs(ket(0, n_a - 1, n_e + 1) @ h2 @ ket(2, n_a, n_e)) ** 2
    v2 = 0.25 * np.hypot(d[0, 0] - d[1, 1], 2 * d[0, 1])  # V2_+- as zfs_manager extracts it
    weight = TransitionRate.pair_weights(np.array([[0.0, v2], [v2, 0.0]]))[0, 1]
    assert 2 * weight * n_a * (n_e + 1) == pytest.approx(m2, rel=1e-12)


def _two_phonon(calc, res):
    omega_zfs = calc.data["zfs"] / MEV
    ox, oy, j0pm, jpm, j00 = calc.get_2d_spectral_density(res=res, sigma=SIGMA)
    calc.compute_two_phonon_rates(T, ox, oy, j0pm, jpm, j00, omega_zfs)
    return dict(calc.transition_rate["two_phonon"])


def test_two_phonon_rate_is_independent_of_grid_spacing():
    calc = _calculator()
    coarse, fine = _two_phonon(calc, 0.5), _two_phonon(calc, 0.125)
    for key in ("0_to_1", "1_to_-1", "1_to_0"):
        assert fine[key] == pytest.approx(coarse[key], rel=0.02)


def test_two_phonon_rate_matches_golden_rule_pair_sum():
    calc = _calculator()
    rates = _two_phonon(calc, 0.25)
    omega_zfs = calc.data["zfs"] / MEV
    # 0 -> 1 (|dm| = 1) uses V2_0_pm, 1 -> -1 (|dm| = 2) uses V2_p_m.
    assert rates["0_to_1"] == pytest.approx(_pair_sum(calc, "V2_0_pm", omega_zfs), rel=0.03)
    assert rates["1_to_-1"] == pytest.approx(_pair_sum(calc, "V2_p_m", omega_zfs), rel=0.03)


def test_first_order_rate_is_independent_of_grid_spacing():
    # A splitting inside the phonon band, so the direct process is not negligible.
    calc = _calculator()
    rates = []
    for res in (0.04, 0.01):
        omega, j0pm, jpm, j00 = calc.get_spectral_density(res=res, sigma=SIGMA)
        calc.compute_transition_rates(T, omega, j0pm, jpm, j00, 42.0)
        rates.append(calc.transition_rate["first_order"]["1_to_0"])
    assert rates[0] > 0
    assert rates[1] == pytest.approx(rates[0], rel=0.01)


def test_direct_rate_is_golden_rule_at_the_splitting():
    calc = _calculator()
    omega_zfs = calc.data["zfs"] / MEV
    omega, j0pm, jpm, j00 = calc.get_spectral_density(res=0.01, sigma=SIGMA)
    calc.compute_transition_rates(T, omega, j0pm, jpm, j00, omega_zfs)
    first = calc.transition_rate["first_order"]

    j_split = np.sum((calc.data["V_0_pm"] / MEV) ** 2 * _gauss(FREQS_MEV - omega_zfs, SIGMA))
    n_split = 1 / np.expm1(omega_zfs / (Cn.k / Cn.e * 1000 * T))
    prefactor = 2 * np.pi / (Cn.hbar / MEV)
    assert first["0_to_1"] == pytest.approx(prefactor * j_split * n_split, rel=1e-3)
    assert first["1_to_0"] == pytest.approx(prefactor * j_split * (n_split + 1), rel=1e-3)
    assert first["0_to_-1"] == first["0_to_1"] and first["-1_to_0"] == first["1_to_0"]
    # detailed balance between the levels
    assert first["1_to_0"] / first["0_to_1"] == pytest.approx(np.exp(omega_zfs / (Cn.k / Cn.e * 1000 * T)))
    # +1 and -1 are degenerate at zero field
    assert first["1_to_-1"] == 0.0 and first["-1_to_1"] == 0.0


def test_direct_rate_with_negative_d_swaps_absorption_and_emission():
    calc = _calculator()
    omega, j0pm, jpm, j00 = calc.get_spectral_density(res=0.04, sigma=SIGMA)
    calc.compute_transition_rates(T, omega, j0pm, jpm, j00, 42.0)
    up_d = dict(calc.transition_rate["first_order"])
    calc.compute_transition_rates(T, omega, j0pm, jpm, j00, -42.0)
    down_d = calc.transition_rate["first_order"]
    assert down_d["0_to_1"] == up_d["1_to_0"] and down_d["1_to_0"] == up_d["0_to_1"]
    assert down_d["1_to_0"] < down_d["0_to_1"]


def test_direct_rate_vanishes_at_zero_temperature_only_for_absorption():
    calc = _calculator()
    omega, j0pm, jpm, j00 = calc.get_spectral_density(res=0.04, sigma=SIGMA)
    calc.compute_transition_rates(0.0, omega, j0pm, jpm, j00, 42.0)
    first = calc.transition_rate["first_order"]
    assert first["0_to_1"] == 0.0 and first["1_to_0"] > 0.0
