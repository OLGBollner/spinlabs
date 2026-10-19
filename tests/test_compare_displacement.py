"""Tests for spinlabs.vasp.structures.compare_displacement."""

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Poscar

from spinlabs.vasp import compare_displacement, compare_displacement_cli


@pytest.fixture
def ref_poscar(tmp_path):
    lattice = Lattice.cubic(3.57)
    species = ["C"] * 4
    coords = [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0], [0.75, 0.75, 0.75]]
    struct = Structure(lattice, species, coords)
    path = tmp_path / "POSCAR_ref"
    Poscar(struct).write_file(path)
    return path, struct


def _write(path, struct):
    Poscar(struct).write_file(path)
    return path


def test_identical_structures(ref_poscar, tmp_path):
    ref_path, ref = ref_poscar
    pert = _write(tmp_path / "POSCAR_same", ref.copy())
    stats = compare_displacement(ref_path, pert)
    assert stats["max"] == pytest.approx(0.0, abs=1e-12)
    assert stats["rms"] == pytest.approx(0.0, abs=1e-12)
    assert stats["n_atoms"] == 4


def test_max_and_rms(ref_poscar, tmp_path):
    ref_path, ref = ref_poscar
    pert = ref.copy()
    pert.sites[0].coords = ref.sites[0].coords + [0.3, 0, 0]
    pert.sites[1].coords = ref.sites[1].coords + [0, 0, 0.1]
    pert_path = _write(tmp_path / "POSCAR_pert", pert)
    stats = compare_displacement(ref_path, pert_path)
    assert stats["max"] == pytest.approx(0.3)
    assert stats["max_index"] == 0
    assert stats["rms"] == pytest.approx(np.sqrt((0.3**2 + 0.1**2) / 4))
    assert stats["disp"].shape == (4, 3)


def test_atom_count_mismatch(ref_poscar, tmp_path):
    ref_path, ref = ref_poscar
    pert = Structure(ref.lattice, ["C"] * 3, [[0, 0, 0]] * 3)
    pert_path = _write(tmp_path / "POSCAR_bad", pert)
    with pytest.raises(ValueError, match="Atom count"):
        compare_displacement(ref_path, pert_path)


def test_composition_mismatch(ref_poscar, tmp_path):
    ref_path, ref = ref_poscar
    pert = Structure(ref.lattice, ["Si"] * 4, ref.frac_coords)
    pert_path = _write(tmp_path / "POSCAR_si", pert)
    with pytest.raises(ValueError, match="Composition"):
        compare_displacement(ref_path, pert_path)


def test_cli_table(ref_poscar, tmp_path, capsys):
    ref_path, ref = ref_poscar
    small = ref.copy()
    small.sites[0].coords = ref.sites[0].coords + [0.1, 0, 0]
    big = ref.copy()
    big.sites[0].coords = ref.sites[0].coords + [0.2, 0, 0]
    p1 = _write(tmp_path / "POSCAR_s", small)
    p2 = _write(tmp_path / "POSCAR_b", big)
    compare_displacement_cli(ref_path, [p1, p2])
    out = capsys.readouterr().out
    assert "max |disp|" in out
    assert "Relative sizes" in out
    assert "2.000" in out  # second entry is 2x the first


def test_cli_reports_missing_gracefully(ref_poscar, tmp_path, capsys):
    ref_path, _ = ref_poscar
    compare_displacement_cli(ref_path, [tmp_path / "nope"])
    out = capsys.readouterr().out
    assert "not found" in out or "No such file" in out


def test_wrapped_structure_matches(ref_poscar, tmp_path):
    """A structure shifted by a full lattice vector must compare as
    identical (nearest-periodic-image wrapping)."""
    ref_path, ref = ref_poscar
    pert = ref.copy()
    pert.sites[0].coords = ref.sites[0].coords + [3.57, 3.57, 0]
    pert_path = _write(tmp_path / "POSCAR_wrapped", pert)
    stats = compare_displacement(ref_path, pert_path)
    assert stats["max"] == pytest.approx(0.0, abs=1e-10)
    assert stats["rms"] == pytest.approx(0.0, abs=1e-10)


def test_outcar_and_run_directory_use_the_runs_poscar(ref_poscar, tmp_path):
    """Pointing at a run's OUTCAR or directory compares the POSCAR it started from."""
    ref_path, ref = ref_poscar
    for name in ("ref_run", "pert_run"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "OUTCAR").write_text(" executed on Dardel date 2026.02.19  14:42:53\n")
    _write(tmp_path / "ref_run" / "POSCAR", ref.copy())
    pert = ref.copy()
    pert.sites[2].coords = ref.sites[2].coords + [0, 0.2, 0]
    _write(tmp_path / "pert_run" / "POSCAR", pert)

    from_outcars = compare_displacement(tmp_path / "ref_run" / "OUTCAR", tmp_path / "pert_run" / "OUTCAR")
    from_dirs = compare_displacement(tmp_path / "ref_run", tmp_path / "pert_run")

    assert from_outcars["max"] == pytest.approx(0.2)
    assert from_dirs["max"] == pytest.approx(0.2)


def test_outcar_without_poscar_is_reported(ref_poscar, tmp_path, capsys):
    ref_path, _ = ref_poscar
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "OUTCAR").write_text("")

    compare_displacement_cli(ref_path, [tmp_path / "run" / "OUTCAR"])

    assert "No POSCAR for" in capsys.readouterr().out


def _pair_run(tmp_path, ref, *, mass_weighted):
    """A pert_0.1/runs/1_2 folder made by the generator, plus the spectrum it used."""
    from spinlabs.models import PhononSpectrum
    from spinlabs.vasp.structures import apply_combined_perturbation

    rng = np.random.default_rng(3)
    eigs = rng.normal(size=(3, len(ref), 3))
    eigs /= np.linalg.norm(eigs.reshape(3, -1), axis=1)[:, None, None]
    masses = np.array([12.011, 12.011, 14.007, 12.011])
    spectrum = PhononSpectrum(
        frequencies_mev=np.array([20.0, 40.0, 60.0]),
        eigenvectors=eigs,
        atom_frac_coords=np.array([s.frac_coords for s in ref]),
        atom_symbols=["C", "C", "N", "C"],
        atomic_masses=masses,
        lattice=ref.lattice.matrix,
    )
    used = masses if mass_weighted else np.ones(len(ref))
    perturbed = apply_combined_perturbation(ref, eigs, 1, 2, used, 0.1)
    run = tmp_path / ("w" if mass_weighted else "u") / "pert_0.1" / "runs" / "1_2"
    run.mkdir(parents=True)
    _write(run / "POSCAR", perturbed)
    return run, spectrum


@pytest.mark.parametrize("generated_weighted", [True, False])
def test_expected_displacement_scale_with_and_without_mass_weighting(ref_poscar, tmp_path, capsys, generated_weighted):
    ref_path, ref = ref_poscar
    run, spectrum = _pair_run(tmp_path, ref, mass_weighted=generated_weighted)

    compare_displacement_cli(ref_path, [run], spectrum=spectrum, mass_weighted=generated_weighted)
    matching = capsys.readouterr().out
    compare_displacement_cli(ref_path, [run], spectrum=spectrum, mass_weighted=not generated_weighted)
    other = capsys.readouterr().out

    assert "modes 1+2, Q=0.1" in matching
    assert "scale      = 1.000 x expected (residual 0.0%)" in matching
    # Assuming the other weighting gives a scale away from 1 (about sqrt(m) or 1/sqrt(m)).
    scale = float(other.split("scale      = ")[1].split(" x")[0])
    assert abs(scale - 1) > 0.5


def test_expected_needs_run_folder_and_amplitude(ref_poscar, tmp_path, capsys):
    ref_path, ref = ref_poscar
    run, spectrum = _pair_run(tmp_path, ref, mass_weighted=True)
    loose = _write(tmp_path / "POSCAR_loose", ref.copy())

    compare_displacement_cli(ref_path, [loose], spectrum=spectrum)

    assert "expected   = ? (modes need a runs/<i> or runs/<i>_<j> folder" in capsys.readouterr().out
