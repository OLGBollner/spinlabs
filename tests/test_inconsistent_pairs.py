"""Second-order entries far off the response expected from the first-order runs are reported, and kept by default.

The expectation for a pair displaced by Q_i + Q_j is the sum of the single-mode changes. An A2 mode
cannot change D linearly (C3v), so its single-mode change is pure curvature and its diagonal pair,
displaced by 2Q, changes D by 4x as much, not 2x.
"""

import logging
import warnings

import numpy as np

from spinlabs.constants import CONSTANTS
from spinlabs.models import PhononSpectrum, RawZFSData, ZFSTensor
from spinlabs.zfs_manager import ZFSManager

GS = np.diag([-1000.0, -1000.0, 2000.0])
STEP = np.diag([-1.0, -1.0, 2.0])
CURVE = np.diag([-0.01, -0.01, 0.02])  # small genuine curvature
A2 = 4  # the A2 mode's index


def _manager(exclude=False, cancel_33=False, a2_pair_factor=4.0):
    spectrum = PhononSpectrum(
        frequencies_mev=np.array([10.0, 30.0, 30.0, 40.0, 50.0]),
        symmetries=["A1", "Ex", "Ey", "A1", "A2"],
        iprs=np.full(5, 0.5),
        eigenvectors=np.ones((5, 1, 3)) / np.sqrt(3),
        atom_frac_coords=np.zeros((1, 3)),
        atom_symbols=["C"],
        atomic_masses=np.array([12.011]),
        lattice=np.eye(3) * 5.0,
    )
    # first-order change of the two A1 modes (linear) and of the A2 mode (pure curvature, c Q^2)
    change = {0: 1.0, 3: 2.0, A2: 1.5}
    first = {i: {"tensor": GS + change.get(i, 0.0) * STEP, "unit": "MHz"} for i in range(5)}
    second = {
        (0, 0): {"tensor": GS + 2 * change[0] * STEP + CURVE, "unit": "MHz"},
        # cancel_33: the 2Q run shows no ZFS change at all, cancelling the linear response (alpha = -1).
        (3, 3): {"tensor": GS + (0.0 if cancel_33 else 2 * change[3]) * STEP + CURVE, "unit": "MHz"},
        # Displaced twice as far: twice the linear response, no real curvature.
        (0, 3): {"tensor": GS + 2 * (change[0] + change[3]) * STEP, "unit": "MHz"},
        # 2Q along the A2 mode: c (2Q)^2 = 4x its change at Q; 16x if displaced twice as far.
        (A2, A2): {"tensor": GS + a2_pair_factor * change[A2] * STEP, "unit": "MHz"},
    }
    raw = RawZFSData(defect="NV", cell_size=64, pert_scale=0.1, calc_method="all_bands",
                     ground_state_zfs=ZFSTensor(matrix=GS, unit="MHz"), first_order=first, second_order=second)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manager = ZFSManager(raw_data=raw.to_unit("J"), spectrum=spectrum, exclude_inconsistent_pairs=exclude)
    d1 = manager.calculate_first_order_derivatives()[0]
    return manager, manager.calculate_second_order_derivatives(d1)


def test_over_displaced_pair_is_listed_and_kept_by_default(caplog):
    with caplog.at_level(logging.WARNING):
        manager, (_, v00, _, _) = _manager()

    assert [(i, j) for i, j, *_ in manager.inconsistent_pairs] == [(0, 3)]
    assert abs(manager.inconsistent_pairs[0][2] - 1.0) < 1e-6  # alpha of a doubled run
    assert v00[0, 3] > 0
    assert "Kept 1 of 4 second-order entries" in caplog.text and "1_4 (alpha=1.00)" in caplog.text
    assert "--exclude-inconsistent-pairs drops them" in caplog.text


def test_exclude_inconsistent_pairs_drops_them(caplog):
    with caplog.at_level(logging.WARNING):
        manager, (d2, v00, _, _) = _manager(exclude=True)

    assert v00[0, 3] == 0 and np.all(d2[0, 3] == 0)
    assert v00[0, 0] > 0 and v00[3, 3] > 0  # healthy diagonal entries kept
    assert "Excluded (--exclude-inconsistent-pairs) 1 of 4" in caplog.text


def test_a2_diagonal_pair_with_pure_curvature_is_consistent():
    manager, (d2, v00, _, _) = _manager(exclude=True)

    assert (A2, A2) not in {(i, j) for i, j, *_ in manager.inconsistent_pairs}
    assert v00[A2, A2] > 0
    # D = D0 + c Q^2 with c Q^2 = 1.5 STEP at the run's Q: the second derivative is 2c
    q = manager.zfs_tensors_2d[(A2, A2)]["pert"][0]
    assert np.allclose(d2[A2, A2], 2 * 1.5 * STEP * CONSTANTS["MHz2J"] / q**2)


def test_a2_diagonal_pair_displaced_twice_as_far_is_flagged():
    manager, _ = _manager(a2_pair_factor=16.0)

    flagged = {(i, j): a for i, j, a, _ in manager.inconsistent_pairs}
    assert abs(flagged[(A2, A2)] - 3.0) < 1e-6  # 16x where 4x is expected


def test_entry_cancelling_the_linear_response_is_flagged(caplog):
    with caplog.at_level(logging.WARNING):
        manager, (_, v00, _, _) = _manager(exclude=True, cancel_33=True)

    flagged = {(i, j): a for i, j, a, _ in manager.inconsistent_pairs}
    assert set(flagged) == {(0, 3), (3, 3)}
    assert abs(flagged[(3, 3)] + 1.0) < 0.01
    assert v00[3, 3] == 0
    assert "4_4 (alpha=-1.00)" in caplog.text
