"""Non-mass-weighted perturbation data: stored flag and corrected perturbation amplitudes."""

import warnings

import numpy as np
import pytest

from spinlabs.models import PhononSpectrum, RawZFSData, ZFSTensor
from spinlabs.zfs_manager import ZFSManager

MASSES = np.array([12.011, 14.007])
SYMS = ["A1", "Ex", "Ey", "A1"]


def _spectrum():
    """Two atoms; mode weights on (C, N): all C, half/half, half/half, all N."""
    eigs = np.zeros((4, 2, 3))
    eigs[0, 0, 2] = 1.0
    eigs[1, :, 0] = np.sqrt(0.5)
    eigs[2, :, 1] = np.sqrt(0.5)
    eigs[3, 1, 2] = 1.0
    return PhononSpectrum(
        frequencies_mev=np.array([10.0, 30.0, 30.0, 40.0]),
        symmetries=SYMS,
        iprs=np.full(4, 0.5),
        eigenvectors=eigs,
        atom_frac_coords=np.zeros((2, 3)),
        atom_symbols=["C", "N"],
        atomic_masses=MASSES,
        lattice=np.eye(3) * 5.0,
    )


def _raw(mass_weighted=True):
    gs = np.diag([-1000.0, -1000.0, 2000.0])
    step = np.diag([-1.0, -1.0, 2.0])
    first = {i: {"tensor": gs + (i + 1) * step, "unit": "MHz"} for i in range(4)}
    second = {(i, i): {"tensor": gs + 2.1 * (i + 1) * step, "unit": "MHz"} for i in range(4)}
    return RawZFSData(
        defect="NV", cell_size=64, pert_scale=0.1, calc_method="all_bands", mass_weighted=mass_weighted,
        ground_state_zfs=ZFSTensor(matrix=gs, unit="MHz"), first_order=first, second_order=second,
    )


def test_effective_sqrt_mass_weights_atoms_by_mode_amplitude():
    expected = [np.sqrt(12.011), *[(np.sqrt(12.011) + np.sqrt(14.007)) / 2] * 2, np.sqrt(14.007)]
    np.testing.assert_allclose(_spectrum().effective_sqrt_mass(), expected)


def test_unweighted_amplitudes_are_scaled_by_effective_sqrt_mass():
    spectrum = _spectrum()
    weighted = spectrum.get_phonon_pert(1e-25)["disp"]
    unweighted = spectrum.get_phonon_pert(1e-25, mass_weighted=False)["disp"]
    np.testing.assert_allclose(np.array(unweighted) / np.array(weighted), spectrum.effective_sqrt_mass())


def test_flag_round_trips_and_legacy_files_read_as_mass_weighted(tmp_path):
    path = _raw(mass_weighted=False).save(tmp_path / "unweighted.npz")
    assert RawZFSData.load(path).mass_weighted is False

    with np.load(path, allow_pickle=True) as data:
        np.savez(tmp_path / "legacy.npz", **{k: data[k] for k in data.files if k != "mass_weighted"})
    assert RawZFSData.load(tmp_path / "legacy.npz").mass_weighted is True


def test_mixed_weighting_cannot_be_combined():
    with pytest.raises(ValueError, match="mass_weighted and unweighted"):
        _raw(True).combine(_raw(False))


def test_unweighted_couplings_are_divided_by_effective_sqrt_mass():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weighted = ZFSManager(raw_data=_raw(True).to_unit("J"), spectrum=_spectrum())
        unweighted = ZFSManager(raw_data=_raw(False).to_unit("J"), spectrum=_spectrum())
    f = _spectrum().effective_sqrt_mass()

    d1_w, *v1_w = weighted.calculate_first_order_derivatives()
    d1_u, *v1_u = unweighted.calculate_first_order_derivatives()
    np.testing.assert_allclose(d1_u, d1_w / f[:, None, None])
    for a, b in zip(v1_u, v1_w):
        np.testing.assert_allclose(a, b / f)

    d2_w = weighted.calculate_second_order_derivatives(d1_w)[0]
    d2_u = unweighted.calculate_second_order_derivatives(d1_u)[0]
    idx = np.arange(4)
    np.testing.assert_allclose(d2_u[idx, idx], d2_w[idx, idx] / (f**2)[:, None, None])


def test_unweighted_data_needs_a_spectrum():
    with pytest.raises(ValueError, match="phonon spectrum"):
        ZFSManager(raw_data=_raw(False).to_unit("J"))


def test_rigid_translations_get_no_displacement_whatever_their_frequency():
    """A Gamma translation with a small positive numerical frequency must not turn run noise into a coupling."""
    masses = np.array([12.011, 14.007])
    sqrt_m = np.sqrt(masses) / np.linalg.norm(np.sqrt(masses))
    eigs = np.zeros((6, 2, 3))
    for a in range(3):  # translations along x, y, z: sqrt(m_j), normalized
        eigs[a, :, a] = sqrt_m
    for a in range(3):  # the optical partners: C and N in antiphase, orthogonal to the translations
        eigs[3 + a, :, a] = [sqrt_m[1], -sqrt_m[0]]
    spectrum = PhononSpectrum(
        frequencies_mev=np.array([0.004, 0.006, -0.1, 50.0, 50.0, 60.0]),
        symmetries=["A1"] * 6, iprs=np.full(6, 0.5), eigenvectors=eigs,
        atom_frac_coords=np.zeros((2, 3)), atom_symbols=["C", "N"], atomic_masses=masses, lattice=np.eye(3) * 5.0,
    )
    np.testing.assert_array_equal(spectrum.translation_modes(), [True, True, True, False, False, False])
    disp = spectrum.get_phonon_pert(1e-25)["disp"]
    assert disp[:3] == [None, None, None]
    assert all(d is not None and d > 0 for d in disp[3:])
    assert not _spectrum().translation_modes().any()  # modes at 10-40 meV are not translations
