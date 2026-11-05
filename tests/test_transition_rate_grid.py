"""Grid-based Fermi golden rule rates must not depend on the grid spacing."""

import numpy as np
import pytest
from scipy import constants as Cn

from spinlabs.constants import CONSTANTS
from spinlabs.models import SpinPhononCouplingData, ZFSTensor
from spinlabs.transition_rate import TransitionRate

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


def _line_integral(calc, key, gap, temp=T, keep=None):
    """Raman rate straight from the definition, without binning.

    V2 holds the 1/2 of the Taylor series, so a pair i != j enters H as 2 V2_ij q_i q_j and
    absorbing i while emitting j has |M|^2 = 4 |V2_ij|^2 n_i (n_j + 1); diagonal pairs keep the
    diagonal-only convention, 2 |V2_ii|^2 n (n + 1) (Cambria et al., Eq. S12). Each mode is smeared
    by SIGMA, and energy conservation puts the emitted phonon at w - gap:
    Gamma = 2 pi / hbar * integral dw sum_ij |M_ij|^2 G(w - w_i) G(w - gap - w_j) n(w) (n(w - gap) + 1).
    """
    v2 = (np.asarray(calc.data[key]) / MEV) ** 2
    m2 = (2 + 2 * (1.0 - np.eye(len(FREQS_MEV)))) * v2
    if keep is not None:
        m2 = m2 * keep
    w = np.arange(0.005, 120.0, 0.01)
    g_abs = _gauss(w[:, None] - FREQS_MEV[None, :], SIGMA)
    g_em = _gauss(w[:, None] - gap - FREQS_MEV[None, :], SIGMA)
    j2 = np.einsum("ki,ij,kj->k", g_abs, m2, g_em)
    n_abs, n_em = TransitionRate._bose_grid(w, temp), TransitionRate._bose_grid(w - gap, temp)
    ok = w - gap > 0
    return 2 * np.pi / (Cn.hbar / MEV) * np.sum((j2 * n_abs * (n_em + 1))[ok]) * 0.01


def test_pair_weight_matches_exact_matrix_element():
    """|<+1, n_a - 1, n_e + 1| H2 |-1, n_a, n_e>|^2 for H2 = 1/2 sum_ij D_ij q_i q_j, q = b + b^dagger.

    compute_two_phonon_rates gives a pair its weight times 4 pi / hbar (two Raman orderings),
    so the weight must be half of |M|^2 / (n_a (n_e + 1)).
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


def _two_phonon(calc, res=None, temp=T):
    calc.compute_two_phonon_rates(temp, calc.get_line_spectral_density(SIGMA, res=res))
    return dict(calc.transition_rate["two_phonon"])


def test_two_phonon_rate_is_independent_of_grid_spacing():
    calc = _calculator()
    coarse, fine = _two_phonon(calc, 0.3), _two_phonon(calc, 0.02)
    for key in ("0_to_1", "1_to_-1", "1_to_0"):
        assert fine[key] == pytest.approx(coarse[key], rel=2e-3)


@pytest.mark.parametrize("temp", [30.0, T])
def test_two_phonon_rate_matches_the_line_integral(temp):
    calc = _calculator()
    rates = _two_phonon(calc, temp=temp)
    D = calc.data["zfs"] / MEV
    # 0 -> 1 (|dm| = 1, up by D) uses V2_0_pm, 1 -> 0 goes down by D, 1 -> -1 (|dm| = 2, no gap) V2_p_m.
    assert rates["0_to_1"] == pytest.approx(_line_integral(calc, "V2_0_pm", D, temp), rel=2e-3)
    assert rates["1_to_0"] == pytest.approx(_line_integral(calc, "V2_0_pm", -D, temp), rel=2e-3)
    assert rates["1_to_-1"] == pytest.approx(_line_integral(calc, "V2_p_m", 0.0, temp), rel=2e-3)
    assert rates["-1_to_1"] == rates["1_to_-1"]


def test_two_phonon_rates_obey_detailed_balance():
    """A splitting of 5 meV, so up and down differ visibly: down / up = exp(D / kT)."""
    calc = _calculator(zfs_mev=5.0)
    rates = _two_phonon(calc)
    kT = Cn.k / Cn.e * 1000 * T
    assert rates["1_to_0"] / rates["0_to_1"] == pytest.approx(np.exp(5.0 / kT), rel=2e-3)


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


def test_zero_gap_pairs_are_left_out():
    """FREQS_MEV holds a degenerate doublet (30, 30) as group [0, 1]: it and every (i, i) drop out."""
    calc = _calculator()
    calc.compute_two_phonon_rates(T, calc.get_line_spectral_density(SIGMA, degenerate_groups=[[0, 1]]))
    rates = dict(calc.transition_rate["two_phonon"])
    D = calc.data["zfs"] / MEV
    keep = ~np.eye(len(FREQS_MEV), dtype=bool)
    keep[0, 1] = keep[1, 0] = False
    assert rates["0_to_1"] == pytest.approx(_line_integral(calc, "V2_0_pm", D, keep=keep), rel=2e-3)
    assert rates["1_to_-1"] == pytest.approx(_line_integral(calc, "V2_p_m", 0.0, keep=keep), rel=2e-3)
    full = _two_phonon(calc)
    assert rates["0_to_1"] < full["0_to_1"]


def test_zero_gap_exclusion_follows_symmetry_not_frequency():
    """Modes 0 and 1 share a frequency, but outside a degenerate group they are an accidental pair and stay;
    only the diagonal goes."""
    calc = _calculator()
    calc.compute_two_phonon_rates(T, calc.get_line_spectral_density(SIGMA, degenerate_groups=[]))
    rates = dict(calc.transition_rate["two_phonon"])
    keep = ~np.eye(len(FREQS_MEV), dtype=bool)
    assert rates["1_to_-1"] == pytest.approx(_line_integral(calc, "V2_p_m", 0.0, keep=keep), rel=2e-3)
    pairs = calc.symmetry_zero_gap_pairs(len(FREQS_MEV), [[0, 1]])
    assert pairs[0, 1] and pairs[1, 0] and pairs[2, 2] and not pairs[0, 2]
    with pytest.raises(ValueError, match="outside"):
        calc.symmetry_zero_gap_pairs(3, [[2, 3]])
