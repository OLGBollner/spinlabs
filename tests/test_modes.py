"""spinlabs.vasp.modes: the modes a tree runs, numbered like its runs/<i> folders."""

import numpy as np
import pytest

from spinlabs.models import PhononSpectrum, min_image_distances
from spinlabs.parsers import save_phonon_npz
from spinlabs.vasp.modes import main, mode_numbers, mode_table
from spinlabs.vasp.trees import n_modes_from_phonon


def _spectrum(kept, syms):
    """A spectrum filtered to the 0-based full-spectrum positions ``kept``."""
    n = len(kept)
    return PhononSpectrum(
        frequencies_mev=np.linspace(10.0, 30.0, n), eigenvectors=np.zeros((n, 1, 3)),
        atom_frac_coords=np.zeros((1, 3)), atom_symbols=["C"], atomic_masses=np.array([12.0]),
        lattice=np.eye(3), symmetries=list(syms), original_indices=np.asarray(kept),
    )


def test_filtered_spectrum_numbers_modes_by_their_full_spectrum_position(tmp_path):
    path = tmp_path / "phonon_sym.npz"
    save_phonon_npz(_spectrum([0, 3, 7], ["A1", "Ex", "A2"]), path)
    assert mode_numbers(path) == [1, 4, 8]
    assert n_modes_from_phonon(path) == 3
    table = mode_table(path)
    assert [(n, s) for n, _, s, *_ in table] == [(1, "A1"), (4, "Ex"), (8, "A2")]


def test_unfiltered_npz_runs_every_mode(tmp_path):
    path = tmp_path / "raw.npz"
    np.savez(path, freqs=np.array([100.0, 200.0, 300.0, 400.0]))
    assert mode_numbers(path) == [1, 2, 3, 4]


def test_cli_prints_the_mode_numbers_and_the_symmetry_table(tmp_path, capsys):
    path = tmp_path / "phonon_sym.npz"
    save_phonon_npz(_spectrum([1, 2], ["Ex", "Ey"]), path)
    assert main([str(path)]) == 0
    assert capsys.readouterr().out == "2 3\n"
    assert main([str(path), "--sym"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "mode freq[cm-1] sym r_mean[A] P_2bond" and lines[1].startswith("2 ")
    assert lines[2].split()[2] == "Ey" and len(lines[2].split()) == 5
    with pytest.raises(SystemExit):
        main([str(tmp_path / "phonons.yaml")])


A_DIAMOND = 3.567  # Angstrom
BOND = A_DIAMOND * np.sqrt(3) / 4


def _diamond(vacancy: bool):
    """2x2x2 conventional diamond cell (64 sites), with the site at the origin left empty if ``vacancy``.

    Mode 0 moves the four neighbours of the origin, mode 1 the atom farthest from it, mode 2 both equally.
    """
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    basis = np.vstack([basis, basis + 0.25])
    cells = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)])
    frac = ((basis[None, :, :] + cells[:, None, :]) / 2).reshape(-1, 3)
    if vacancy:
        frac = frac[1:]  # the site at the origin
    lattice = np.eye(3) * 2 * A_DIAMOND
    r = np.linalg.norm((frac - np.round(frac)) @ lattice, axis=1)
    neighbours, far = np.flatnonzero(np.isclose(r, BOND)), int(np.argmax(r))
    eigs = np.zeros((3, len(frac), 3))
    eigs[0, neighbours, 0] = 1 / np.sqrt(len(neighbours))
    eigs[1, far, 0] = 1.0
    eigs[2, neighbours, 1] = 1 / np.sqrt(2 * len(neighbours))
    eigs[2, far, 1] = 1 / np.sqrt(2)
    spec = PhononSpectrum(
        frequencies_mev=np.array([10.0, 20.0, 30.0]), eigenvectors=eigs, atom_frac_coords=frac,
        atom_symbols=["C"] * len(frac), atomic_masses=np.full(len(frac), 12.011), lattice=lattice,
        symmetries=["A1", "A1", "A1"],
    )
    return spec, r[far]


def test_distances_are_measured_from_the_vacancy_at_the_nearest_image():
    spec, _ = _diamond(vacancy=True)
    np.testing.assert_allclose(min_image_distances(spec.defect_centre(), np.eye(3)), 0.0, atol=1e-9)
    r = np.sort(spec.defect_distances())
    np.testing.assert_allclose(r[:4], BOND)  # the four atoms around the vacancy
    np.testing.assert_allclose(r[4:16], A_DIAMOND / np.sqrt(2))
    assert spec.defect_distances() is spec.defect_distances()  # the defect search runs once
    assert np.isclose(spec.bond_length(), BOND)


def test_mean_distance_and_near_weight_say_how_near_the_defect_a_mode_lives():
    spec, r_far = _diamond(vacancy=True)
    np.testing.assert_allclose(spec.mean_defect_distance, [BOND, r_far, (BOND + r_far) / 2])
    # within two bond lengths: the 4 + 12 + 12 nearest atoms
    assert np.sum(spec.defect_distances() <= 2 * spec.bond_length()) == 28
    np.testing.assert_allclose(spec.defect_participation, [1.0, 0.0, 0.5])
    # the IPR cannot tell the near mode 0 from the far mode 1 apart in the way that matters here
    assert spec.calc_ipr()[1] > spec.calc_ipr()[0]


def test_without_a_defect_the_localisation_is_nan(tmp_path):
    spec, _ = _diamond(vacancy=False)
    with pytest.raises(ValueError, match="no single defect centre"):
        spec.defect_centre()
    path = tmp_path / "perfect.npz"
    save_phonon_npz(spec, path)
    assert all(np.isnan(r) and np.isnan(p) for *_, r, p in mode_table(path))


def test_nearest_image_in_a_hexagonal_cell():
    """Rounding alone takes the offset (0.4, -0.4, 0) as is; in a 120 degree cell (-0.6, -0.4, 0) is nearer."""
    a = 12.37
    hexagonal = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 10.0]])
    offset = np.array([[0.4, -0.4, 0.0]])
    assert np.isclose(np.linalg.norm(offset @ hexagonal), 8.57, atol=0.01)
    np.testing.assert_allclose(min_image_distances(offset, hexagonal), 6.55, atol=0.01)
