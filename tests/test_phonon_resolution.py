"""Which phonon file an analysis of packaged data picks.

A ClV_576 run was analysed with the ClV_128 phonon file: the search next to the
dataset stopped two folders short of data/<defect>/phonon_data.npz, and the registry
had the 128-atom file registered for cell 576 too, with the first match used blindly.
"""

import numpy as np

from spinlabs.models import PhononSpectrum, RawZFSData, ZFSTensor
from spinlabs.pipeline import find_default_phonon_file_for_raw, load_raw_inputs, raw_mode_count
from spinlabs.registry import ensure_db, register_file


def _spectrum(n_atoms: int) -> PhononSpectrum:
    n = 3 * n_atoms
    rng = np.random.default_rng(n_atoms)
    eig = rng.normal(size=(n, n_atoms, 3))
    eig /= np.linalg.norm(eig.reshape(n, -1), axis=1)[:, None, None]
    return PhononSpectrum(
        frequencies_mev=np.linspace(1.0, 100.0, n),
        symmetries=["A1"] * n,
        eigenvectors=eig,
        atom_frac_coords=rng.random((n_atoms, 3)),
        atom_symbols=["C"] * n_atoms,
        atomic_masses=np.full(n_atoms, 12.011),
        lattice=np.eye(3) * 3.57 * n_atoms ** (1 / 3),
    )


def _packaged(root, defect: str, n_modes: int):
    """data/<defect>/pert_0.025/all_bands/<date>/<file>.npz with first-order entries up to n_modes."""
    folder = root / "data" / defect / "pert_0.025" / "all_bands" / "20261016"
    folder.mkdir(parents=True)
    d0 = np.diag([-1.0, -1.0, 2.0]) * 500
    raw = RawZFSData(
        defect=defect.split("_")[0], cell_size=int(defect.split("_")[1]), pert_scale=0.025, calc_method="all_bands",
        ground_state_zfs=ZFSTensor(matrix=d0, unit="MHz"),
        first_order={i: {"tensor": d0 + 0.1, "unit": "MHz"} for i in range(0, n_modes, max(1, n_modes // 10))}
        | {n_modes - 1: {"tensor": d0 + 0.1, "unit": "MHz"}},
    )
    return raw.save(folder / f"{defect}_raw_zfs_data_all_bands_1d.npz"), raw


def test_phonon_file_in_the_defect_folder_is_found(tmp_path):
    path, _ = _packaged(tmp_path, "ClV_8", 24)
    phonons = tmp_path / "data" / "ClV_8" / "phonon_data.npz"
    _spectrum(8).save(phonons)
    assert find_default_phonon_file_for_raw(path) == phonons


def test_registry_entry_of_another_cell_is_skipped(tmp_path):
    path, raw = _packaged(tmp_path, "ClV_8", 24)
    assert raw_mode_count(raw) == 24
    small, right = tmp_path / "small" / "phonon_data.npz", tmp_path / "right" / "phonon_data.npz"
    small.parent.mkdir()
    right.parent.mkdir()
    _spectrum(2).save(small)  # 6 modes: another cell's file, registered first for this cell
    _spectrum(8).save(right)
    conn = ensure_db()
    register_file(conn, small, "phonon", defect="ClV", cell_size=8)
    register_file(conn, right, "phonon", defect="ClV", cell_size=8)
    _, spectrum, ph_path = load_raw_inputs(raw_zfs_file=path)
    assert ph_path == right and spectrum.n_modes == 24


def test_nearby_phonon_file_of_another_cell_falls_back_to_the_registry(tmp_path):
    path, _ = _packaged(tmp_path, "ClV_8", 24)
    _spectrum(2).save(tmp_path / "data" / "ClV_8" / "phonon_data.npz")  # wrong file in the defect folder
    right = tmp_path / "right" / "phonon_data.npz"
    right.parent.mkdir()
    _spectrum(8).save(right)
    register_file(ensure_db(), right, "phonon", defect="ClV", cell_size=8)
    _, spectrum, ph_path = load_raw_inputs(raw_zfs_file=path)
    assert ph_path == right and spectrum.n_modes == 24
