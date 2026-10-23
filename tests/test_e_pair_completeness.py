import numpy as np
import pytest
from spinlabs.models import PhononSpectrum
from spinlabs.zfs_manager import ZFSManager


def test_e_pair_completeness_flagging():
    # 4 modes: A1 (10 meV), Ex (20.0 meV), Ey (20.02 meV), Ex (30.0 meV - lonely)
    freqs = np.array([10.0, 20.0, 20.02, 30.0])
    syms = ["A1", "Ex", "Ey", "Ex"]
    eigs = np.zeros((4, 2, 3))
    atoms = np.zeros((2, 3))
    masses = np.array([12.0, 12.0])
    lat = np.eye(3)

    spec = PhononSpectrum(
        frequencies_mev=freqs,
        eigenvectors=eigs,
        atom_frac_coords=atoms,
        atom_symbols=["C", "C"],
        atomic_masses=masses,
        lattice=lat,
        symmetries=syms,
    )

    # __post_init__ should have computed e_pair_complete
    assert spec.e_pair_complete is not None
    assert spec.e_pair_complete == [False, True, True, False]


def test_a_paired_mode_does_not_pair_again_with_a_nearby_lone_mode():
    # Ex(0) and Ey(1) are a pair; a lone Ex(2) 0.01 meV away must stay incomplete.
    spec = PhononSpectrum(
        frequencies_mev=np.array([20.0, 20.0, 20.01]),
        eigenvectors=np.zeros((3, 2, 3)),
        atom_frac_coords=np.zeros((2, 3)),
        atom_symbols=["C", "C"],
        atomic_masses=np.array([12.0, 12.0]),
        lattice=np.eye(3),
        symmetries=["Ex", "Ey", "Ex"],
    )
    assert spec.e_pair_complete == [True, True, False]


def test_zfs_manager_rejects_incomplete_spectrum():
    # The same 4-mode spectrum must be refused by ZFSManager now that the
    # silent expansion path is gone.
    freqs = np.array([10.0, 20.0, 20.02, 30.0])
    syms = ["A1", "Ex", "Ey", "Ex"]
    eigs = np.zeros((4, 2, 3))
    atoms = np.zeros((2, 3))
    masses = np.array([12.0, 12.0])
    lat = np.eye(3)

    spec = PhononSpectrum(
        frequencies_mev=freqs,
        eigenvectors=eigs,
        atom_frac_coords=atoms,
        atom_symbols=["C", "C"],
        atomic_masses=masses,
        lattice=lat,
        symmetries=syms,
    )
    with pytest.raises(ValueError, match="missing the degenerate partners"):
        ZFSManager(spectrum=spec)


def test_clv_128_lonely_e_modes_expansion():
    # Synthetic spectrum mimicking the lonely Ex mode case
    # (e.g. 2 atoms -> 6 total modes, 4 present: 2 A1 + 2 lonely Ex)
    n_atoms = 3
    # 3 atoms -> 3 * 3 = 9 modes total.
    # Suppose we have 2 A1 modes and 2 Ex modes without Ey partners (total 4 modes present)
    freqs = np.array([10.0, 15.0, 25.0, 35.0])
    syms = ["A1", "A1", "Ex", "Ex"]
    eigs = np.zeros((4, n_atoms, 3))
    atoms = np.zeros((n_atoms, 3))
    masses = np.ones(n_atoms) * 12.0
    lat = np.eye(3)

    spec = PhononSpectrum(
        frequencies_mev=freqs,
        eigenvectors=eigs,
        atom_frac_coords=atoms,
        atom_symbols=["C"] * n_atoms,
        atomic_masses=masses,
        lattice=lat,
        symmetries=syms,
    )

    assert spec.n_modes == 4
    assert spec.n_atoms == 3
    # All modes are either A1 (False) or lonely Ex (False)
    assert len(spec.e_pair_complete) == 4
    assert all(c is False for c in spec.e_pair_complete)

    # ZFSManager must refuse this incomplete spectrum (4 of 9 modes present)
    with pytest.raises(ValueError, match="missing the degenerate partners"):
        ZFSManager(spectrum=spec)


def test_save_and_load_preserves_e_pair_complete(tmp_path):
    out_file = tmp_path / "phonon_test.npz"
    freqs = np.array([15.0, 25.0])
    spec = PhononSpectrum(
        frequencies_mev=freqs,
        eigenvectors=np.zeros((2, 1, 3)),
        atom_frac_coords=np.zeros((1, 3)),
        atom_symbols=["C"],
        atomic_masses=np.array([12.0]),
        lattice=np.eye(3),
        symmetries=["A1", "Ex"],
    )
    assert spec.e_pair_complete == [False, False]

    spec.save(out_file)
    loaded = PhononSpectrum.load(out_file)
    assert loaded.n_modes == 2
    assert loaded.e_pair_complete == [False, False]


def test_filter_reduced_spectrum_keeps_original_indices_valid():
    """Regression: filtering an already-reduced spectrum (e.g. loading
    phonon_data_sym.npz and filtering again) must validate its carried
    original_indices against the full pre-reduction size, not the reduced
    size. classify_and_pair finds no degeneracies on this stub, so nothing
    is dropped -- but the index check would previously raise."""
    from spinlabs.models import PhononSpectrum
    from spinlabs.symmetry import filter_degenerate_partners

    reduced = PhononSpectrum(
        frequencies_mev=np.array([10.0, 20.0]),
        eigenvectors=np.zeros((2, 1, 3)),
        atom_frac_coords=np.zeros((1, 3)),
        atom_symbols=np.array(["N"]),
        atomic_masses=np.array([14.0]),
        lattice=np.eye(3),
        original_indices=np.array([0, 2]),
        n_full=4,  # as written by a first filter pass over the full spectrum
    )
    out = filter_degenerate_partners(reduced, tol_mev=0.01)
    assert out.n_modes == 2
    assert out.original_indices.tolist() == [0, 2]
