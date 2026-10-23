import numpy as np
import pytest

from spinlabs.models import PhononSpectrum
from spinlabs.parsers import (
    parse_phonon_data,
    parse_phonon_npz,
    parse_phonopy_yaml,
    save_phonon_npz,
)


def _spectrum() -> PhononSpectrum:
    return PhononSpectrum(
        frequencies_mev=np.array([1.0, 2.0, 3.0]),
        eigenvectors=np.ones((3, 2, 3)),
        atom_frac_coords=np.zeros((2, 3)),
        atom_symbols=["C", "C"],
        atomic_masses=np.array([12.0, 12.0]),
        lattice=np.eye(3),
    )


def test_dispatch_npz(tmp_path):
    path = tmp_path / "phonon_data.npz"
    save_phonon_npz(_spectrum(), path)
    result = parse_phonon_data(path)
    expected = parse_phonon_npz(path)
    np.testing.assert_allclose(result.frequencies_mev, expected.frequencies_mev)
    assert result.atom_symbols == expected.atom_symbols


def test_dispatch_yaml(tmp_path):
    path = tmp_path / "phonopy.yaml"
    path.write_text(
        "points:\n"
        "  - symbol: C\n"
        "    coordinates: [0.0, 0.0, 0.0]\n"
        "    mass: 12.0\n"
        "  - symbol: N\n"
        "    coordinates: [0.5, 0.5, 0.5]\n"
        "    mass: 14.0\n"
        "lattice: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]\n"
        "phonon:\n"
        "  - q-position: [0.0, 0.0, 0.0]\n"
        "    band:\n"
        + "".join(
            f"      - frequency: {f}\n"
            "        eigenvector:\n" + "".join(f"        - [[{a}, 0.0], [0.0, {a}], [0.0, 0.0]]\n" for a in (1.0, 2.0))
            for f in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
    )
    result = parse_phonon_data(path)
    expected = parse_phonopy_yaml(path)
    np.testing.assert_allclose(result.frequencies_mev, expected.frequencies_mev)
    assert result.atom_symbols == ["C", "N"]


def test_yaml_forward_poscar(tmp_path, monkeypatch):
    called = {}

    def fake_yaml(path, poscar_path=None):
        called["poscar"] = poscar_path
        return _spectrum()

    monkeypatch.setattr("spinlabs.parsers.parse_phonopy_yaml", fake_yaml)
    y = tmp_path / "phonopy.yaml"
    y.write_text("phonon: []")
    parse_phonon_data(y, poscar_path=tmp_path / "POSCAR")
    assert called["poscar"] == tmp_path / "POSCAR"


def test_dispatch_case_insensitive_suffix(tmp_path):
    with pytest.raises(ValueError, match="Unsupported phonon file format"):
        parse_phonon_data(tmp_path / "data.txt")


def test_missing_file_npz(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_phonon_data(tmp_path / "missing.npz")
