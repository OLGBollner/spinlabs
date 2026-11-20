"""Phonon files and the data displaced along them: aligned tree setup, PHONON_PATH, and packaging pairs."""

import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from spinlabs.models import PhononSpectrum
from spinlabs.parsers import parse_phonon_npz
from spinlabs.vasp.trees import full_spectrum_of, resolve_sym_phonon, tree_phonon

REPO_ROOT = Path(__file__).resolve().parents[1]
NV_64_PHONONS = REPO_ROOT / "NV_64" / "phonon_data.npz"  # plain git blob


def _spectrum(vectors, original=None, n_full=None):
    n = len(vectors)
    return PhononSpectrum(
        frequencies_mev=np.linspace(10.0, 30.0, n),
        eigenvectors=np.asarray(vectors, dtype=float).reshape(n, 1, 3),
        atom_frac_coords=np.zeros((1, 3)),
        atom_symbols=["C"],
        atomic_masses=np.array([12.0]),
        lattice=np.eye(3),
        symmetries=["A1"] * n,
        original_indices=None if original is None else np.asarray(original),
        n_full=n_full,
    )


FULL = np.eye(3)  # three modes along x, y, z


def _pair(folder, stem="phonon_data", stub="", full=FULL):
    """A full spectrum <stem><stub>.npz and its reduction <stem>_sym2<stub>.npz (modes 0 and 2)."""
    folder.mkdir(parents=True, exist_ok=True)
    full_path, reduced_path = folder / f"{stem}{stub}.npz", folder / f"{stem}_sym2{stub}.npz"
    _spectrum(full).save(full_path)
    _spectrum(FULL[[0, 2]], original=[0, 2], n_full=3).save(reduced_path)
    return full_path, reduced_path


# -- tree setup: aligned files by default -------------------------------------------------------------------


def test_tree_setup_writes_the_aligned_pair(tmp_path):
    out = tmp_path / "NV_64"
    (out / "data").mkdir(parents=True)
    shutil.copy(NV_64_PHONONS, out / "data" / "phonon_data.npz")
    target = resolve_sym_phonon(out, out / "data" / "phonon_data.npz")
    assert target == out / "data" / "phonon_data_sym126_aligned.npz"
    full = out / "data" / "phonon_data_aligned.npz"
    assert parse_phonon_npz(target).aligned is True and parse_phonon_npz(full).aligned is True
    assert parse_phonon_npz(full).n_modes == 189
    assert full_spectrum_of(target) == full
    # A second call reuses the files.
    stamp = target.stat().st_mtime_ns
    assert resolve_sym_phonon(out, out / "data" / "phonon_data.npz") == target
    assert target.stat().st_mtime_ns == stamp


def test_tree_setup_without_alignment_keeps_the_classified_file(tmp_path):
    out = tmp_path / "NV_64"
    (out / "data").mkdir(parents=True)
    shutil.copy(NV_64_PHONONS, out / "data" / "phonon_data.npz")
    target = resolve_sym_phonon(out, out / "data" / "phonon_data.npz", align=False)
    assert target.name == "phonon_data_sym126.npz"
    assert parse_phonon_npz(target).aligned is False
    assert full_spectrum_of(target) == out / "data" / "phonon_data.npz"


# -- PHONON_PATH from a tree's job script ----------------------------------------------------------------


@pytest.mark.parametrize(
    "line, expected",
    [
        ("PHONON_PATH=/x/data/phonon_data_sym126_aligned.npz", "/x/data/phonon_data_sym126_aligned.npz"),
        ('PHONON_PATH="/x/p q.npz"', "/x/p"),  # first token only, as the shell script reads it unquoted
        ("PHONON_PATH=${PHONON_PATH:-/x/data/phonon_data_sym126.npz}", "/x/data/phonon_data_sym126.npz"),
        ("  PHONON_PATH='/x/a.npz'  # comment", "/x/a.npz"),
    ],
)
def test_tree_phonon_reads_the_job_script(tmp_path, line, expected):
    (tmp_path / "run_first_order.sh").write_text(f"#!/bin/bash\n{line}\n")
    assert str(tree_phonon(tmp_path)).strip('"') == expected.strip('"')


def test_tree_phonon_is_none_without_it(tmp_path):
    assert tree_phonon(tmp_path) is None
    (tmp_path / "run_first_order.sh").write_text("#!/bin/bash\nBINARY=/x/vasp\n")
    assert tree_phonon(tmp_path) is None


# -- the full spectrum of the file a tree displaced along ---------------------------------------------------


def test_full_spectrum_of_a_full_and_a_reduced_file(tmp_path):
    full, reduced = _pair(tmp_path, stub="_aligned")
    assert full_spectrum_of(full) == full
    assert full_spectrum_of(reduced) == full


def test_full_spectrum_of_rejects_a_missing_or_mismatched_spectrum(tmp_path):
    _, reduced = _pair(tmp_path, stub="_aligned")
    (tmp_path / "phonon_data_aligned.npz").unlink()
    with pytest.raises(ValueError, match="no full spectrum"):
        full_spectrum_of(reduced)
    # The original (unrotated) spectrum renamed as the aligned one: same size, other vectors.
    rotated = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    _spectrum(rotated).save(tmp_path / "phonon_data_aligned.npz")
    with pytest.raises(ValueError, match="not those of"):
        full_spectrum_of(reduced)
    _spectrum(np.eye(3)[:2]).save(tmp_path / "phonon_data_aligned.npz")
    with pytest.raises(ValueError, match="has 2 modes"):
        full_spectrum_of(reduced)
    with pytest.raises(ValueError, match="not found"):
        full_spectrum_of(tmp_path / "missing.npz")


# -- spinlabs package pairs each dataset with its spectrum ----------------------------------------------------


def _package(tmp_path, monkeypatch, phonon_line):
    """Package the two-order tree of test_cli_package whose job scripts carry ``phonon_line``; return the dataset
    folder (a path that does not exist when nothing was saved)."""
    from test_cli_package import _make_args, _patch, _raw, _registry, _tree

    from spinlabs.cli.package import _run

    _tree(tmp_path)
    _registry(monkeypatch, tmp_path)
    from spinlabs.methods import basis_dir

    for order in ("first", "second"):
        # where the scan reads the tree (the fixture's data is packaged as AO)
        basis = basis_dir(tmp_path / "NV_512" / f"{order}_order" / "pert_0.025", "AO")
        basis.mkdir(parents=True, exist_ok=True)
        (basis / f"run_{order}_order.sh").write_text(f"#!/bin/bash\n{phonon_line}\n")
    with _patch(lambda **kw: _raw(order=kw["order"])):
        _run(_make_args(tmp_path))
    datasets = sorted((tmp_path / "data" / "NV_512" / "pert_0.025").glob(f"*/{date.today():%Y%m%d}"))
    assert len(datasets) <= 1
    return datasets[0] if datasets else tmp_path / "data" / "NV_512" / "pert_0.025" / "none"


def test_package_saves_the_job_scripts_spectrum_with_the_dataset(tmp_path, monkeypatch):
    full, reduced = _pair(tmp_path / "NV_512" / "data", stub="_aligned")
    dataset = _package(tmp_path, monkeypatch, f"PHONON_PATH={reduced}")
    saved = dataset / "phonon_data.npz"
    assert saved.is_file()
    assert saved.read_bytes() == full.read_bytes()  # the aligned full spectrum, not the original


def test_package_skips_a_dataset_it_cannot_pair(tmp_path, monkeypatch, caplog):
    _, reduced = _pair(tmp_path / "NV_512" / "data", stub="_aligned")
    (tmp_path / "NV_512" / "data" / "phonon_data_aligned.npz").unlink()
    dataset = _package(tmp_path, monkeypatch, f"PHONON_PATH={reduced}")
    assert not dataset.exists()
    assert "cannot pair the data with its phonon spectrum" in caplog.text


def test_package_without_phonon_path_keeps_the_old_behaviour(tmp_path, monkeypatch):
    dataset = _package(tmp_path, monkeypatch, "BINARY=/x/vasp")
    assert dataset.is_dir() and not (dataset / "phonon_data.npz").exists()
