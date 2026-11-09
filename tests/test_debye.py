"""Debye direct-rate estimate from folded acoustic modes (spinlabs.debye, TransitionRate direct='debye')."""

import numpy as np
import pytest
from scipy import constants as Cn

from spinlabs.constants import CONSTANTS
from spinlabs.debye import DebyeFit, debye_coefficient, first_shell
from spinlabs.models import PhononSpectrum, SpinPhononCouplingData, ZFSTensor
from spinlabs.transition_rate import TransitionRate

A = 2.0  # Angstrom, simple-cubic spacing
N = 4  # atoms per side: an 8 Angstrom cubic supercell
E_LONG, E_TRANS, A_COUPLING = 40.0, 30.0, 2e-4  # meV; |V|^2 = A_COUPLING * E for acoustic modes (meV)


def _crystal():
    """Simple-cubic supercell with its 18 first-shell acoustic plane waves as modes, plus 6 random modes."""
    grid = np.array([(i, j, k) for i in range(N) for j in range(N) for k in range(N)], dtype=float)
    r = grid * A
    lattice = np.eye(3) * N * A
    modes, freqs, kinds = [], [], []
    for axis in range(3):
        q = np.zeros(3)
        q[axis] = 2 * np.pi / (N * A)
        for pol in np.eye(3):
            longitudinal = pol[axis] == 1
            for trig in (np.cos, np.sin):
                u = trig(r @ q)[:, None] * pol[None, :]
                modes.append(u / np.linalg.norm(u))
                freqs.append(E_LONG if longitudinal else E_TRANS)
                kinds.append("acoustic")
    acoustic = np.array([m.ravel() for m in modes])
    rng = np.random.default_rng(0)
    for _ in range(6):  # optical-like modes orthogonal to the acoustic ones
        v = rng.normal(size=acoustic.shape[1])
        v -= acoustic.T @ (acoustic @ v)
        modes.append((v / np.linalg.norm(v)).reshape(-1, 3))
        freqs.append(rng.uniform(60, 120))
        kinds.append("optical")
    n = len(freqs)
    spectrum = PhononSpectrum(
        frequencies_mev=np.array(freqs),
        eigenvectors=np.array(modes),
        atom_frac_coords=grid / N,
        atom_symbols=["C"] * len(r),
        atomic_masses=np.full(len(r), 12.011),
        lattice=lattice,
        symmetries=["A1"] * n,
        iprs=np.full(n, 0.1),
    )
    v = np.where(np.array(kinds) == "acoustic", np.sqrt(A_COUPLING * np.array(freqs)), 5.0)
    return spectrum, np.array(freqs), v, np.array(kinds)


def test_first_shell_is_the_shortest_reciprocal_vectors():
    qs = first_shell(np.eye(3) * 8.0)
    assert len(qs) == 3 and np.allclose(np.linalg.norm(qs, axis=1), 2 * np.pi / 8)
    qs = first_shell(np.diag([8.0, 8.0, 12.0]))  # the long axis has the shortest q
    assert len(qs) == 1 and np.allclose(np.abs(qs[0]), [0, 0, 2 * np.pi / 12])


def test_debye_fit_finds_the_acoustic_modes_and_the_analytic_coefficient():
    spectrum, freqs, v, kinds = _crystal()
    fit = debye_coefficient(spectrum, freqs, v)
    assert set(fit.acoustic_modes) == set(np.flatnonzero(kinds == "acoustic"))
    assert np.all(fit.weights[kinds == "acoustic"] > 0.99) and np.all(fit.weights[kinds == "optical"] < 0.5)
    assert len(fit.branches["longitudinal"]["modes"]) == 6 and len(fit.branches["transverse"]["modes"]) == 12
    q = 2 * np.pi / (N * A)
    volume = (N * A) ** 3
    expected = sum(n_s * volume * A_COUPLING / (2 * np.pi**2 * (E / q) ** 3) for n_s, E in ((1, E_LONG), (2, E_TRANS)))
    assert fit.q == pytest.approx(q)
    assert fit.C == pytest.approx(expected, rel=1e-10)


def test_no_acoustic_mode_is_an_error():
    spectrum, freqs, v, kinds = _crystal()
    with pytest.raises(ValueError, match="no mode"):
        debye_coefficient(spectrum, freqs, v, min_weight=1.01)


def _calculator():
    mev = CONSTANTS["meV2J"]
    coupling = SpinPhononCouplingData(
        order=1,
        defect="NV",
        cell_size=64,
        pert_scale=0.1,
        calc_method="all_bands",
        frequencies=np.array([30.0, 40.0]) * mev,
        frequency_unit="J",
        ground_state_zfs=ZFSTensor(matrix=np.diag([-1.0, -1.0, 2.0]) * 0.0119 / 3 * mev, unit="J"),
        V_0_0=np.zeros(2),
        V_p_m=np.zeros(2),
        V_0_pm=np.array([1.0, 1.0]) * mev,
        coupling_unit="J",
        symmetries=["Ex", "Ey"],
        iprs=np.full(2, 0.2),
    )
    calc = TransitionRate()
    calc.load_data(coupling)
    return calc


@pytest.mark.parametrize("T", [0.0, 10.0, 300.0])
def test_debye_direct_rate_is_golden_rule_with_c_d_cubed(T):
    calc = _calculator()
    calc.debye_fit = DebyeFit(C=2e-15, q=0.44, branches={}, acoustic_modes=np.array([]), weights=np.array([]))
    calc.direct = "debye"
    D = calc.data["zfs"] / CONSTANTS["meV2J"]
    omega, j0pm, jpm, j00 = calc.get_spectral_density(res=0.05, sigma=3.0)
    calc.compute_transition_rates(T, omega, j0pm, jpm, j00, D)
    first = calc.transition_rate["first_order"]
    n = 0.0 if T == 0 else 1 / np.expm1(D / (Cn.k / Cn.e * 1000 * T))
    rate = 2 * np.pi / (Cn.hbar / CONSTANTS["meV2J"]) * 2e-15 * D**3
    assert first["1_to_0"] == pytest.approx(rate * (n + 1), rel=1e-9)
    assert first["0_to_1"] == pytest.approx(rate * n, rel=1e-9, abs=0)
    assert first["1_to_-1"] == 0.0


def test_direct_none_and_bad_settings():
    calc = _calculator()
    omega, j0pm, jpm, j00 = calc.get_spectral_density(res=0.05, sigma=3.0)
    D = calc.data["zfs"] / CONSTANTS["meV2J"]
    calc.direct = "none"
    calc.compute_transition_rates(300.0, omega, j0pm, jpm, j00, D)
    assert calc.transition_rate["first_order"]["1_to_0"] == 0.0
    calc.direct = "debye"
    with pytest.raises(ValueError, match="set_debye_from_spectrum"):
        calc.compute_transition_rates(300.0, omega, j0pm, jpm, j00, D)
    calc.direct = "tails"
    with pytest.raises(ValueError, match="direct must be one of"):
        calc.compute_transition_rates(300.0, omega, j0pm, jpm, j00, D)


def test_set_debye_from_spectrum_uses_the_couplings_of_the_data():
    spectrum, freqs, v, kinds = _crystal()
    mev = CONSTANTS["meV2J"]
    n = len(freqs)
    coupling = SpinPhononCouplingData(
        order=1,
        defect="NV",
        cell_size=64,
        pert_scale=0.1,
        calc_method="all_bands",
        frequencies=freqs * mev,
        frequency_unit="J",
        ground_state_zfs=ZFSTensor(matrix=np.diag([-1.0, -1.0, 2.0]) * 0.0119 / 3 * mev, unit="J"),
        V_0_0=np.zeros(n),
        V_p_m=np.zeros(n),
        V_0_pm=v * mev,
        coupling_unit="J",
        symmetries=["A1"] * n,
        iprs=np.full(n, 0.2),
    )
    calc = TransitionRate()
    calc.load_data(coupling)
    fit = calc.set_debye_from_spectrum(spectrum)
    assert calc.direct == "debye" and calc.debye_fit is fit
    assert fit.C == pytest.approx(debye_coefficient(spectrum, freqs, v).C, rel=1e-9)


def test_axial_field_gives_a_c_pm_double_quantum_direct_rate():
    """In a field B, +1 and -1 split by 2 g mu_B B: the +1 <-> -1 rate uses C_+- at that splitting, and the
    0 <-> +-1 rates their own splittings D +- g mu_B B."""
    from spinlabs.transition_rate import ZEEMAN_MEV_PER_MT

    calc = _calculator()
    fit = lambda C: DebyeFit(C=C, q=0.44, branches={}, acoustic_modes=np.array([]), weights=np.array([]))  # noqa: E731
    calc.debye_fit, calc.debye_fit_pm, calc.direct = fit(2e-15), fit(5e-15), "debye"
    calc.b_field_mt = 100.0
    T = 300.0
    D = calc.data["zfs"] / CONSTANTS["meV2J"]
    z = ZEEMAN_MEV_PER_MT * 100.0
    assert z == pytest.approx(2.0028 * 5.7883818060e-2 * 0.1, rel=1e-8)  # g mu_B B in meV
    # E_m = D (m^2 - 2/3) + g mu_B B m: zero at the zero-field centroid
    assert calc.level_energies(D) == pytest.approx({1: D / 3 + z, 0: -2 * D / 3, -1: D / 3 - z})
    omega, j0pm, jpm, j00 = calc.get_spectral_density(res=0.05, sigma=3.0)
    calc.compute_transition_rates(T, omega, j0pm, jpm, j00, D)
    first = calc.transition_rate["first_order"]
    f1 = 2 * np.pi / (Cn.hbar / CONSTANTS["meV2J"])
    n = lambda E: 1 / np.expm1(E / (Cn.k / Cn.e * 1000 * T))  # noqa: E731
    assert first["-1_to_1"] == pytest.approx(f1 * 5e-15 * (2 * z) ** 3 * n(2 * z), rel=1e-9)
    assert first["1_to_-1"] == pytest.approx(f1 * 5e-15 * (2 * z) ** 3 * (n(2 * z) + 1), rel=1e-9)
    # with D > g mu_B B, 0 is lowest: both 0 -> +-1 absorb, at D + z and D - z
    assert first["0_to_1"] == pytest.approx(f1 * 2e-15 * (D + z) ** 3 * n(D + z), rel=1e-9)
    assert first["0_to_-1"] == pytest.approx(f1 * 2e-15 * (D - z) ** 3 * n(D - z), rel=1e-9)
    # the two-phonon lines follow the same gaps
    calc.data_2ph = {"V2_0_pm": np.eye(2) * CONSTANTS["meV2J"], "V2_p_m": np.eye(2) * CONSTANTS["meV2J"]}
    lines = calc.get_line_spectral_density(3.0)
    assert lines["1_to_-1"][2] == pytest.approx(-2 * z) and lines["0_to_-1"][2] == pytest.approx(D - z)


def test_zero_field_keeps_plus_minus_one_degenerate():
    calc = _calculator()
    calc.debye_fit = DebyeFit(C=2e-15, q=0.44, branches={}, acoustic_modes=np.array([]), weights=np.array([]))
    calc.direct = "debye"  # no C_+- fit needed: the +1 <-> -1 splitting is zero
    D = calc.data["zfs"] / CONSTANTS["meV2J"]
    omega, j0pm, jpm, j00 = calc.get_spectral_density(res=0.05, sigma=3.0)
    calc.compute_transition_rates(300.0, omega, j0pm, jpm, j00, D)
    first = calc.transition_rate["first_order"]
    assert first["1_to_-1"] == first["-1_to_1"] == 0.0
    assert first["0_to_1"] == first["0_to_-1"] > 0.0
