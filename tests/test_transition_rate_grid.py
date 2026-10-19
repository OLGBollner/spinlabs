"""Grid-based Fermi golden rule rates must not depend on the grid spacing."""

import numpy as np
import pytest
from scipy import constants as Cn

from spinlabs.constants import CONSTANTS
from spinlabs.models import SpinPhononCouplingData, ZFSTensor
from spinlabs.transition_rate import PHONON_PROCESSES, Phonons, TransitionRate

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


def _pair_sum(calc, key, omega_zfs):
    """Golden rule summed over mode pairs; each mode smeared by SIGMA, so pairs by SIGMA*sqrt(2)."""
    v = np.asarray(calc.data[key]) / MEV
    n = Phonons.bose_einstein(FREQS_MEV, T)
    width = np.sqrt(2) * SIGMA
    total = 0.0
    for proc, (s, sp) in PHONON_PROCESSES.items():
        bose = Phonons._bose_factors_2ph(n[None, :], n[:, None])[proc]
        mismatch = s * FREQS_MEV[None, :] + sp * FREQS_MEV[:, None] - omega_zfs
        total += np.sum(v**2 * bose * np.exp(-0.5 * (mismatch / width) ** 2) / (np.sqrt(2 * np.pi) * width))
    return 2 * np.pi / (Cn.hbar / MEV) * total


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
