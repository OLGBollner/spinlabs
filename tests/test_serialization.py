"""Dataclass .npz files are written compressed and read back unchanged."""

import numpy as np

from spinlabs.models import SpinPhononCouplingData, ZFSTensor


def _coupling(n=400):
    """Second-order arrays with only the diagonal filled, like a diagonal-pairs dataset."""
    rng = np.random.default_rng(1)
    v2 = np.diag(rng.uniform(1, 2, n)) * 1e-25
    d2 = np.zeros((n, n, 3, 3))
    d2[np.arange(n), np.arange(n)] = rng.normal(size=(n, 3, 3)) * 1e-25
    return SpinPhononCouplingData(
        order=2,
        defect="NV",
        cell_size=512,
        pert_scale=0.025,
        calc_method="all_bands",
        frequencies=np.linspace(1, 160, n) * 1.602e-22,
        frequency_unit="J",
        ground_state_zfs=ZFSTensor(matrix=np.diag([-1.0, -1.0, 2.0]) * 1e-25, unit="J"),
        V_0_0=rng.uniform(size=n) * 1e-25,
        V_p_m=rng.uniform(size=n) * 1e-25,
        V_0_pm=rng.uniform(size=n) * 1e-25,
        coupling_unit="J",
        V2_0_0=v2,
        V2_p_m=2 * v2,
        V2_0_pm=3 * v2,
        zfs_2nd_derivs=d2,
    )


def test_sparse_second_order_coupling_is_stored_compactly_and_round_trips(tmp_path):
    coupling = _coupling()
    path = coupling.save(tmp_path / "spin_phonon_coupling.npz")

    dense_bytes = sum(a.nbytes for a in (coupling.V2_0_0, coupling.V2_p_m, coupling.V2_0_pm, coupling.zfs_2nd_derivs))
    assert (tmp_path / "spin_phonon_coupling.npz").stat().st_size < dense_bytes / 20

    loaded = SpinPhononCouplingData.load(path)
    for name in ("V_0_0", "V2_0_0", "V2_p_m", "V2_0_pm", "zfs_2nd_derivs"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(coupling, name))


def test_uncompressed_files_still_load(tmp_path):
    path = _coupling(n=20).save(tmp_path / "compressed.npz")
    with np.load(path, allow_pickle=True) as data:
        np.savez(tmp_path / "plain.npz", **{k: data[k] for k in data.files})

    loaded = SpinPhononCouplingData.load(tmp_path / "plain.npz")
    np.testing.assert_array_equal(loaded.V2_p_m, _coupling(n=20).V2_p_m)
