"""The full spin decontamination (Biktagirov) on tensors and on RawZFSData sublevel sets."""

import numpy as np
import pytest

from spinlabs.models import (
    PerturbationEntry,
    RawZFSData,
    ZFSTensor,
    decontaminate_tensors,
    vasp_dmatrix_scale,
)


def axial(dzz):
    return np.diag([-dzz / 2, -dzz / 2, dzz])


def test_vasp_scale_is_s_2s_minus_1_from_s_one_up():
    assert [vasp_dmatrix_scale(s) for s in (0, 0.5, 1, 1.5, 2, 2.5)] == [1.0, 1.0, 1.0, 3.0, 6.0, 10.0]


def test_spin_one_and_three_halves_follow_the_paper_with_vasp_output():
    """S = 1: (D_1/2 - mean D_0/2); S = 3/2: (3/2 D - mean D_1/2 / 2) / 2 (the flips come out unscaled)."""
    assert np.allclose(decontaminate_tensors(axial(2104.5), [axial(-1126.7)] * 2, 1), axial((2104.5 + 1126.7) / 2))
    lows = [axial(2149.5), axial(-1097.3), axial(-1096.5)]
    expected = (1.5 * axial(-12.94) - np.mean(lows, axis=0) / 2) / 2
    assert np.allclose(decontaminate_tensors(axial(-12.94), lows, 1.5), expected)


def test_flips_with_spin_one_or_more_are_unscaled_first():
    """S = 5/2: the m_s = 3/2 runs carry VASP's 1/(S'(2S'-1)) = 1/3, undone before they enter."""
    high, low = axial(10.0), axial(4.0)
    expected = (vasp_dmatrix_scale(2.5) * high / 2 - 3 * low / 2) / 4
    assert np.allclose(decontaminate_tensors(high, [low] * 5, 2.5), expected)
    with pytest.raises(ValueError, match="S >= 1"):
        decontaminate_tensors(high, [low], 0.5)


def test_tensor_classmethod_converts_the_flips_to_the_high_spin_unit():
    high = ZFSTensor(axial(2104.5), unit="MHz")
    flips = [ZFSTensor(axial(-1126.7), unit="MHz").to_unit("J")] * 2
    out = ZFSTensor.decontaminated(high, flips, 1)
    assert out.unit == "MHz" and np.allclose(out.matrix, axial(1615.6))


def _set(dzz_gs, dzz_modes, pert=0.025):
    entries = {k: PerturbationEntry(order=1, mode_indices=(k,), amplitude=pert, zfs_tensor=ZFSTensor(axial(v)))
               for k, v in dzz_modes.items()}
    return RawZFSData(defect="NV", cell_size=64, pert_scale=pert, calc_method="full_decont", spin=1.0,
                      displacement="normalized", ground_state_zfs=ZFSTensor(axial(dzz_gs)), first_order=entries)


def test_a_sublevel_set_becomes_one_dataset_over_the_common_entries():
    sets = {"ms_1": _set(2104.5, {0: 2200.0, 1: 2300.0}),
            "ms_0_127": _set(-1126.7, {0: -1100.0, 1: -1000.0}),
            "ms_0_128": _set(-1126.7, {0: -1100.0})}  # mode 1 missing in one flip
    for name, raw in sets.items():
        raw.spin_sublevel = name
    out = RawZFSData.decontaminated(sets)
    assert out.calc_method == "full_decont" and out.spin_sublevel is None and out.spin == 1.0
    assert np.allclose(out.ground_state_zfs.to_unit("MHz").matrix, axial((2104.5 + 1126.7) / 2))
    assert list(out.first_order) == [0]
    assert np.allclose(out.first_order[0].zfs_tensor.to_unit("MHz").matrix, axial((2200.0 + 1100.0) / 2))
    assert out.metadata["decontaminated_from"] == ["ms_0_127", "ms_0_128", "ms_1"]


def test_an_incomplete_or_inconsistent_set_is_refused():
    with pytest.raises(ValueError, match="expected 2"):
        RawZFSData.decontaminated({"ms_1": _set(1.0, {}), "ms_0_127": _set(1.0, {})})
    with pytest.raises(ValueError, match="pert_scale"):
        RawZFSData.decontaminated({"ms_1": _set(1.0, {}), "ms_0_127": _set(1.0, {}), "ms_0_128": _set(1.0, {}, pert=0.05)})
