"""spinlabs.vasp.modes: the modes a tree runs, numbered like its runs/<i> folders."""

import numpy as np
import pytest

from spinlabs.models import PhononSpectrum
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
    assert lines[0] == "mode freq[cm-1] sym r_mean[A] P_2shell" and lines[1].startswith("2 ")
    assert lines[2].split()[2] == "Ey" and len(lines[2].split()) == 5
    with pytest.raises(SystemExit):
        main([str(tmp_path / "phonons.yaml")])


def _defect_spectrum():
    """An N centre in a 10 A cube with C shells at 1.5 A (3 atoms) and 2.5 A (3 atoms) and one C at 4 A.

    Mode 0 moves the first shell only, mode 1 the far atom only, mode 2 the N and the far atom equally.
    """
    a = 10.0
    cart = [[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5], [-2.5, 0, 0], [0, -2.5, 0], [0, 0, -2.5], [4, 0, 0]]
    frac = (np.array(cart) / a + 0.5) % 1.0
    eigs = np.zeros((3, 8, 3))
    eigs[0, 1:4, 0] = 1 / np.sqrt(3)
    eigs[1, 7, 0] = 1.0
    eigs[2, 0, 1] = eigs[2, 7, 1] = 1 / np.sqrt(2)
    return PhononSpectrum(
        frequencies_mev=np.array([10.0, 20.0, 30.0]), eigenvectors=eigs, atom_frac_coords=frac,
        atom_symbols=["N"] + ["C"] * 7, atomic_masses=np.array([14.007] + [12.011] * 7),
        lattice=np.eye(3) * a, symmetries=["A1", "A1", "A1"],
    )


def test_defect_shells_and_distances_are_measured_from_the_defect_centre():
    spec = _defect_spectrum()
    np.testing.assert_allclose(np.sort(spec.defect_distances()), [0, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 4])
    np.testing.assert_allclose(spec.defect_shell_radii(), [1.5, 2.5, 4.0])  # the N on the centre is no shell


def test_mean_distance_and_two_shell_participation_say_how_near_the_defect_a_mode_lives():
    spec = _defect_spectrum()
    np.testing.assert_allclose(spec.mean_defect_distance, [1.5, 4.0, 2.0])
    np.testing.assert_allclose(spec.defect_participation, [1.0, 0.0, 0.5])  # the centre atom counts as near
    np.testing.assert_allclose(spec.defect_participation_shells(3), [1.0, 1.0, 1.0])
    # the IPR cannot tell the near mode 0 from the far mode 1 apart in the way that matters here
    assert spec.calc_ipr()[1] > spec.calc_ipr()[0]
