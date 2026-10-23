import numpy as np
import pytest
import yaml
from pymatgen.core import Structure

from spinlabs.parsers import parse_phonopy_yaml, prepare_phonopy_yaml


def _write_yaml(path, lattice):
    n = 8
    bands = []
    for i in range(3 * n):
        bands.append(
            {
                "frequency": 1.0 + i * 0.1,
                "eigenvector": [[[1.0, 0.0, 0.0]] for _ in range(n)],
            }
        )
    path.write_text(yaml.safe_dump({"phonon": [{"q-position": [0, 0, 0], "band": bands}]}))


def test_falls_back_to_sibling_poscar(tmp_path):
    struct = Structure(
        lattice=np.eye(3) * 5.0,
        species=["Si"] * 8,
        coords=np.random.RandomState(0).random((8, 3)),
        coords_are_cartesian=False,
    )
    struct.to(fmt="poscar", filename=str(tmp_path / "POSCAR"))
    yaml_path = tmp_path / "phonons.yaml"
    _write_yaml(yaml_path, np.eye(3))
    spectrum = parse_phonopy_yaml(yaml_path)
    assert spectrum.atom_symbols == ["Si"] * 8
    assert not np.allclose(spectrum.lattice, np.eye(3))


def test_no_poscar_still_degrades_to_defaults(tmp_path, capsys):
    yaml_path = tmp_path / "phonons.yaml"
    _write_yaml(yaml_path, np.eye(3))
    spectrum = parse_phonopy_yaml(yaml_path)
    assert spectrum.atom_symbols == ["X"] * 8


def test_mismatched_poscar_raises(tmp_path):
    struct = Structure(
        lattice=np.eye(3) * 5.0,
        species=["Si"] * 4,
        coords=np.random.RandomState(1).random((4, 3)),
        coords_are_cartesian=False,
    )
    struct.to(fmt="poscar", filename=str(tmp_path / "POSCAR"))
    yaml_path = tmp_path / "phonons.yaml"
    _write_yaml(yaml_path, np.eye(3))
    with pytest.raises(ValueError, match="Geometry mismatch"):
        parse_phonopy_yaml(yaml_path)


def test_prepare_phonopy_yaml(tmp_path):
    struct = Structure(
        lattice=np.eye(3) * 5.0,
        species=["Si"] * 8,
        coords=np.random.RandomState(2).random((8, 3)),
        coords_are_cartesian=False,
    )
    struct.to(fmt="poscar", filename=str(tmp_path / "POSCAR"))
    yaml_path = tmp_path / "phonons.yaml"
    _write_yaml(yaml_path, np.eye(3))
    spectrum, out = prepare_phonopy_yaml(yaml_path)
    assert out == tmp_path / "phonons_full.yaml"
    assert spectrum.atom_symbols == ["Si"] * 8
    # yaml already has structure -> no file written
    assert prepare_phonopy_yaml(out)[1] is None
    # no POSCAR next to file -> nothing written
    assert prepare_phonopy_yaml(yaml_path, poscar_path=tmp_path / "missing")[1] is None


def test_missing_structure_warns(tmp_path):
    yaml_path = tmp_path / "phonons.yaml"
    _write_yaml(yaml_path, np.eye(3))
    with pytest.warns(UserWarning, match="No structure data"):
        parse_phonopy_yaml(yaml_path)


def test_duplicate_qpoint_deduplicated(tmp_path, recwarn):
    """phonons.yaml with the same Gamma block twice: keep only the first."""
    n = 2
    bands = [{"frequency": 1.0 + i * 0.1, "eigenvector": [[[1.0, 0.0, 0.0]] for _ in range(n)]} for i in range(3 * n)]
    doc = {
        "phonon": [
            {"q-position": [0.0, 0.0, 0.0], "band": bands},
            {"q-position": [0.0, 0.0, 0.0], "band": bands},
        ]
    }
    yaml_path = tmp_path / "phonons.yaml"
    yaml_path.write_text(yaml.safe_dump(doc))

    spectrum = parse_phonopy_yaml(yaml_path)
    assert spectrum.frequencies_mev.shape == (6,)
    dup = [w for w in recwarn if "Duplicate q-point" in str(w.message)]
    assert len(dup) == 1


def test_write_full_removes_duplicate_qpoints(tmp_path):
    struct = Structure(
        lattice=np.eye(3) * 5.0,
        species=["Si"] * 8,
        coords=np.random.RandomState(2).random((8, 3)),
        coords_are_cartesian=False,
    )
    struct.to(fmt="poscar", filename=str(tmp_path / "POSCAR"))
    yaml_path = tmp_path / "phonons.yaml"
    _write_yaml(yaml_path, np.eye(3))
    # duplicate the q-position block
    text = yaml_path.read_text()
    yaml_path.write_text(text + text)

    _, out = prepare_phonopy_yaml(yaml_path)
    assert out is not None
    full = out.read_text()
    assert full.count("q-position") == 1
    assert "points:" in full
