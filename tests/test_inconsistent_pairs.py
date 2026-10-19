"""Second-order entries that are a larger copy of the first-order response are excluded and reported."""

import logging
import warnings

import numpy as np

from spinlabs.models import PhononSpectrum, RawZFSData, ZFSTensor
from spinlabs.zfs_manager import ZFSManager

GS = np.diag([-1000.0, -1000.0, 2000.0])
STEP = np.diag([-1.0, -1.0, 2.0])
CURVE = np.diag([-0.01, -0.01, 0.02])  # small genuine curvature


def _manager(keep=False):
    spectrum = PhononSpectrum(
        frequencies_mev=np.array([10.0, 30.0, 30.0, 40.0]),
        symmetries=["A1", "Ex", "Ey", "A1"],
        iprs=np.full(4, 0.5),
        eigenvectors=np.ones((4, 1, 3)) / np.sqrt(3),
        atom_frac_coords=np.zeros((1, 3)),
        atom_symbols=["C"],
        atomic_masses=np.array([12.011]),
        lattice=np.eye(3) * 5.0,
    )
    lin = {0: 1.0, 3: 2.0}  # first-order response of the two A1 modes
    first = {i: {"tensor": GS + lin.get(i, 0.0) * STEP, "unit": "MHz"} for i in range(4)}
    second = {
        (0, 0): {"tensor": GS + 2 * lin[0] * STEP + CURVE, "unit": "MHz"},
        (3, 3): {"tensor": GS + 2 * lin[3] * STEP + CURVE, "unit": "MHz"},
        # Displaced twice as far: twice the linear response, no real curvature.
        (0, 3): {"tensor": GS + 2 * (lin[0] + lin[3]) * STEP, "unit": "MHz"},
    }
    raw = RawZFSData(defect="NV", cell_size=64, pert_scale=0.1, calc_method="all_bands",
                     ground_state_zfs=ZFSTensor(matrix=GS, unit="MHz"), first_order=first, second_order=second)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manager = ZFSManager(raw_data=raw.to_unit("J"), spectrum=spectrum, exclude_inconsistent_pairs=not keep)
    d1 = manager.calculate_first_order_derivatives()[0]
    return manager, manager.calculate_second_order_derivatives(d1)


def test_over_displaced_pair_is_excluded_and_listed(caplog):
    with caplog.at_level(logging.WARNING):
        manager, (d2, v00, _, _) = _manager()

    assert [(i, j) for i, j, *_ in manager.inconsistent_pairs] == [(0, 3)]
    assert manager.inconsistent_pairs[0][2] == np.float64(manager.inconsistent_pairs[0][2])
    assert abs(manager.inconsistent_pairs[0][2] - 1.0) < 1e-6  # alpha of a doubled run
    assert v00[0, 3] == 0 and np.all(d2[0, 3] == 0)
    assert v00[0, 0] > 0 and v00[3, 3] > 0  # healthy diagonal entries kept
    assert "Excluded 1 of 3 second-order entries" in caplog.text
    assert "1_4 (alpha=1.00)" in caplog.text


def test_keep_inconsistent_pairs_keeps_them_but_still_reports(caplog):
    with caplog.at_level(logging.WARNING):
        manager, (_, v00, _, _) = _manager(keep=True)

    assert len(manager.inconsistent_pairs) == 1
    assert v00[0, 3] > 0
    assert "Kept (--keep-inconsistent-pairs) 1 of 3" in caplog.text
