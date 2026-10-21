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
    assert [(n, s) for n, _, s in table] == [(1, "A1"), (4, "Ex"), (8, "A2")]


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
    assert lines[0] == "mode freq[cm-1] sym" and lines[1].startswith("2 ") and lines[2].endswith(" Ey")
    with pytest.raises(SystemExit):
        main([str(tmp_path / "phonons.yaml")])
