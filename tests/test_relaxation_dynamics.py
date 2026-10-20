"""T1 and population dynamics of the three-level rate equations.

The steady-state eigenvalue used to be told apart from the decays by |lambda| > 1e-9:
at GHz rates its round-off passed that test and became "T1" (1e8 s, or negative).
Decays slower than 1e-9 s^-1 (T1 > 1e9 s) still count as no relaxation, T1 = inf.
The populations came from odeint, which ran out of steps on these stiff systems.
"""

import warnings

import numpy as np
import pytest
from scipy.linalg import expm

from spinlabs.relaxation_dynamics import RelaxationDynamics

KEYS = ("1_to_0", "1_to_-1", "0_to_1", "0_to_-1", "-1_to_1", "-1_to_0")


def _rates_file(tmp_path, rates: dict[str, float]):
    path = tmp_path / "directional_rates.npz"
    np.savez(path, temperatures=np.array([300.0]), calc_method="all_bands",
             **{k: np.array([rates[k]]) for k in KEYS})
    return path


def _rates(scale: float) -> dict[str, float]:
    """Fast +-1 <-> 0 (scale) and slow +1 <-> -1 (1e-4 x scale), a stiff system."""
    return {"1_to_0": scale, "0_to_1": scale, "-1_to_0": scale, "0_to_-1": scale,
            "1_to_-1": 1e-4 * scale, "-1_to_1": 1e-4 * scale}


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e9])
def test_t1_is_the_slowest_decay_at_any_rate_scale(tmp_path, scale):
    rd = RelaxationDynamics([0.0, 1.0, 0.0], _rates_file(tmp_path, _rates(scale)))
    t1, _, _ = rd.get_T1_eigenval(temp_index=0)
    decays = np.sort(np.linalg.eigvals(rd.rate_matrices[0]).real)[:2]  # the two non-zero eigenvalues
    assert t1 == pytest.approx(-1.0 / decays.max(), rel=1e-6)
    assert 0 < t1 < np.inf


@pytest.mark.parametrize("scale", [1.0, 1e9])
def test_populations_are_exact_without_solver_warnings(tmp_path, scale):
    rd = RelaxationDynamics([0.0, 1.0, 0.0], _rates_file(tmp_path, _rates(scale)))
    t = rd.get_time_estimate(temp_index=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = rd.simulate_relaxation(t, temp_index=0)
    A = rd.rate_matrices[0]
    for k in (0, 10, 500, len(t) - 1):
        np.testing.assert_allclose(p[k], expm(A * t[k]) @ rd.population, atol=1e-9)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(p[-1], rd.get_steady_state(temp_index=0), atol=1e-6)


@pytest.mark.parametrize("scale, finite", [(1e-12, False), (5e-10, False), (2e-9, True)])
def test_decays_slower_than_the_cutoff_give_infinite_t1(tmp_path, scale, finite):
    """The slowest decay of _rates is about 1.0 x scale; below 1e-9 s^-1 T1 is inf (not drawn)."""
    rd = RelaxationDynamics([0.0, 1.0, 0.0], _rates_file(tmp_path, _rates(scale)))
    t1 = rd.get_T1_eigenval(temp_index=0)[0]
    assert bool(np.isfinite(t1)) == finite
    if not finite:
        assert rd.get_T1_fit(temp_index=0) == np.inf


def test_fitted_t1_matches_the_eigenvalue_at_ghz_rates(tmp_path):
    rd = RelaxationDynamics([0.0, 1.0, 0.0], _rates_file(tmp_path, _rates(1e9)))
    t1_eig, _, _ = rd.get_T1_eigenval(temp_index=0)
    assert rd.get_T1_fit(temp_index=0) == pytest.approx(t1_eig, rel=1e-3)
