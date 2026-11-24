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


def _spectrum(vectors, original=None, n_full=None, aligned=None):
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
        aligned=aligned,
    )


FULL = np.eye(3)  # three modes along x, y, z


def _pair(folder, stem="phonon_data", stub="", full=FULL):
    """A full spectrum <stem><stub>.npz and its reduction <stem>_sym2<stub>.npz (modes 0 and 2)."""
    folder.mkdir(parents=True, exist_ok=True)
    full_path, reduced_path = folder / f"{stem}{stub}.npz", folder / f"{stem}_sym2{stub}.npz"
    aligned = stub == "_aligned"
    _spectrum(full, aligned=aligned).save(full_path)
    _spectrum(FULL[[0, 2]], original=[0, 2], n_full=3, aligned=aligned).save(reduced_path)
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


def _reduced_nv64(folder):
    """NV_64's spectrum reduced to one mode per degenerate set (126 of 189), as spinlabs symmetrize writes it."""
    from spinlabs.parsers import save_phonon_npz
    from spinlabs.symmetry import classify_and_pair, filter_degenerate_partners, symmetrize_degenerate_groups

    spectrum = parse_phonon_npz(NV_64_PHONONS)
    classify_and_pair(spectrum)
    path = folder / "phonon_data.npz"
    save_phonon_npz(filter_degenerate_partners(symmetrize_degenerate_groups(spectrum)), path)
    return path


def test_tree_setup_refuses_a_reduced_spectrum(tmp_path):
    # NV_216: a reduced file where the full spectrum belongs passed through filter and alignment
    # unchanged and was written as its own "full" phonon_data_aligned.npz.
    out = tmp_path / "NV_64"
    (out / "data").mkdir(parents=True)
    reduced = _reduced_nv64(out / "data")
    with pytest.raises(ValueError, match=r"symmetry-reduced phonon spectrum \(126 of 189 modes\)"):
        resolve_sym_phonon(out, reduced)
    assert sorted(p.name for p in (out / "data").iterdir()) == ["phonon_data.npz"]  # nothing written


def test_reduced_spectra_are_told_from_full_ones(tmp_path):
    from spinlabs.models import reduced_spectrum_reason

    assert reduced_spectrum_reason(parse_phonon_npz(NV_64_PHONONS)) is None
    assert reduced_spectrum_reason(parse_phonon_npz(_reduced_nv64(tmp_path))) == "126 of 189 modes"
    # an older reduced file: the indices into the full spectrum, no n_full
    assert "indices into a larger spectrum" in reduced_spectrum_reason(_spectrum(FULL[[0, 2]], original=[0, 2]))
    # nothing but the mode count: 2 modes of a 1-atom cell
    assert "a full spectrum has 3" in reduced_spectrum_reason(_spectrum(FULL[:2]))


def test_symmetrize_refuses_a_reduced_spectrum(tmp_path):
    from spinlabs import cli

    reduced = _reduced_nv64(tmp_path)
    with pytest.raises(SystemExit, match="already symmetry-reduced"):
        cli.main(["symmetrize", str(reduced), "--align"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["phonon_data.npz"]


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


# -- spinlabs package tags each dataset; the analysis picks the defect's spectrum by the tag --------------------


def _package(tmp_path, monkeypatch, phonon_line):
    """Package the two-order tree of test_cli_package whose job scripts carry ``phonon_line``.

    Returns the packaged raw file, or None when nothing was saved.
    """
    from test_cli_package import _make_args, _patch, _raw, _registry, _tree

    from spinlabs.cli.package import _run
    from spinlabs.methods import basis_dir

    _tree(tmp_path)
    _registry(monkeypatch, tmp_path)
    for order in ("first", "second"):
        # where the scan reads the tree (the fixture's data is packaged as AO)
        basis = basis_dir(tmp_path / "NV_512" / f"{order}_order" / "pert_0.025", "AO")
        basis.mkdir(parents=True, exist_ok=True)
        (basis / f"run_{order}_order.sh").write_text(f"#!/bin/bash\n{phonon_line}\n")
    with _patch(lambda **kw: _raw(order=kw["order"])):
        _run(_make_args(tmp_path))
    files = sorted((tmp_path / "data" / "NV_512" / "pert_0.025").glob(f"*/{date.today():%Y%m%d}/*.npz"))
    assert len(files) <= 1
    return files[0] if files else None


def _tag(path):
    with np.load(path, allow_pickle=True) as data:
        return bool(data["aligned"]) if "aligned" in data.files else None


def test_aligned_data_is_tagged_and_its_spectrum_packaged_once(tmp_path, monkeypatch):
    from spinlabs.pipeline import load_raw_inputs

    full, reduced = _pair(tmp_path / "NV_512" / "data", stub="_aligned")
    raw_file = _package(tmp_path, monkeypatch, f"PHONON_PATH={reduced}")
    assert _tag(raw_file) is True
    assert [p.name for p in raw_file.parent.iterdir()] == [raw_file.name]  # no phonon copy with the data
    packaged = tmp_path / "data" / "NV_512" / "phonon_data_aligned.npz"
    assert packaged.read_bytes() == full.read_bytes()
    # The analysis takes the aligned spectrum by the tag, not the defect's phonon_data.npz.
    _spectrum(np.eye(3), aligned=None).save(tmp_path / "data" / "NV_512" / "phonon_data.npz")
    raw, spectrum, path = load_raw_inputs(raw_zfs_file=raw_file)
    assert raw.aligned is True and path == packaged and spectrum.aligned is True


def test_unaligned_data_is_tagged_and_uses_the_job_scripts_spectrum(tmp_path, monkeypatch):
    from spinlabs.pipeline import load_raw_inputs

    full, reduced = _pair(tmp_path / "NV_512" / "data")
    raw_file = _package(tmp_path, monkeypatch, f"PHONON_PATH={reduced}")
    assert _tag(raw_file) is False
    packaged = tmp_path / "data" / "NV_512" / "phonon_data.npz"
    assert packaged.read_bytes() == full.read_bytes()
    assert not (tmp_path / "data" / "NV_512" / "phonon_data_aligned.npz").exists()
    _, spectrum, path = load_raw_inputs(raw_zfs_file=raw_file)
    assert path == packaged and not spectrum.aligned


def test_analysis_refuses_a_spectrum_of_the_other_kind(tmp_path, monkeypatch):
    from spinlabs.pipeline import load_raw_inputs

    full, reduced = _pair(tmp_path / "NV_512" / "data", stub="_aligned")
    raw_file = _package(tmp_path, monkeypatch, f"PHONON_PATH={reduced}")
    plain = tmp_path / "plain.npz"
    _spectrum(np.eye(3)).save(plain)
    with pytest.raises(ValueError, match="pass the matching spectrum"):
        load_raw_inputs(raw_zfs_file=raw_file, phonon_file=plain)
    (tmp_path / "data" / "NV_512" / "phonon_data_aligned.npz").unlink()
    _spectrum(np.eye(3)).save(tmp_path / "data" / "NV_512" / "phonon_data.npz")  # only the unaligned one
    with pytest.raises(FileNotFoundError, match="mirror-aligned"):
        load_raw_inputs(raw_zfs_file=raw_file)


def test_package_refuses_to_replace_a_packaged_aligned_spectrum(tmp_path, monkeypatch):
    _, reduced = _pair(tmp_path / "NV_512" / "data", stub="_aligned")
    earlier = tmp_path / "data" / "NV_512" / "phonon_data_aligned.npz"
    earlier.parent.mkdir(parents=True)
    _spectrum(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), aligned=True).save(earlier)
    with pytest.raises(SystemExit, match="holds other aligned modes"):
        _package(tmp_path, monkeypatch, f"PHONON_PATH={reduced}")
    assert not list((tmp_path / "data" / "NV_512").glob("pert_*"))  # nothing saved


def test_package_skips_a_dataset_it_cannot_pair(tmp_path, monkeypatch, caplog):
    _, reduced = _pair(tmp_path / "NV_512" / "data", stub="_aligned")
    (tmp_path / "NV_512" / "data" / "phonon_data_aligned.npz").unlink()
    assert _package(tmp_path, monkeypatch, f"PHONON_PATH={reduced}") is None
    assert "cannot pair the data with its phonon spectrum" in caplog.text


def test_package_without_phonon_path_keeps_the_old_behaviour(tmp_path, monkeypatch):
    raw_file = _package(tmp_path, monkeypatch, "BINARY=/x/vasp")
    assert _tag(raw_file) is None


# -- a symmetry-reduced file is never packaged as the defect's full spectrum --------------------------------------


def _reduced_defect_spectrum(folder, name="phonon_data.npz"):
    """A reduced spectrum (modes 0 and 2 of 3) under the name of the defect's full one."""
    folder.mkdir(parents=True, exist_ok=True)
    _spectrum(FULL[[0, 2]], original=[0, 2], n_full=3).save(folder / name)
    return folder / name


def test_package_passes_over_a_reduced_phonon_data_npz(tmp_path, monkeypatch, caplog):
    _reduced_defect_spectrum(tmp_path / "NV_512" / "data")
    _package(tmp_path, monkeypatch, "BINARY=/x/vasp")  # no PHONON_PATH: the defect's file is found by name
    assert not (tmp_path / "data" / "NV_512" / "phonon_data.npz").exists()
    assert "no unaligned full phonon spectrum" in caplog.text
    assert "phonon_data.npz (2 of 3 modes): symmetry-reduced" in caplog.text


def test_aligned_data_needs_no_unaligned_spectrum_and_says_nothing(tmp_path, monkeypatch, caplog):
    # V_Si_576: data/ holds only the aligned pair; the job scripts name it, so nothing is missing
    full, reduced = _pair(tmp_path / "NV_512" / "data", stub="_aligned")
    _package(tmp_path, monkeypatch, f"PHONON_PATH={reduced}")
    assert (tmp_path / "data" / "NV_512" / "phonon_data_aligned.npz").read_bytes() == full.read_bytes()
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert not any("unaligned full phonon spectrum" in w for w in warnings)  # only a debug message


def test_missing_unaligned_spectrum_names_the_files_passed_over(tmp_path, monkeypatch, caplog):
    _pair(tmp_path / "NV_512" / "data", stub="_aligned")
    _package(tmp_path, monkeypatch, "BINARY=/x/vasp")  # no PHONON_PATH: the unaligned spectrum is needed
    assert "no unaligned full phonon spectrum" in caplog.text
    assert "phonon_data_aligned.npz (mirror-aligned" in caplog.text
    assert "phonon_data_sym2_aligned.npz (symmetry-reduced)" in caplog.text


def test_package_takes_the_full_spectrum_next_to_a_reduced_one(tmp_path, monkeypatch):
    _reduced_defect_spectrum(tmp_path / "NV_512" / "data")
    full = tmp_path / "NV_512" / "data" / "phonons.npz"
    _spectrum(FULL).save(full)
    _package(tmp_path, monkeypatch, "BINARY=/x/vasp")
    assert (tmp_path / "data" / "NV_512" / "phonon_data.npz").read_bytes() == full.read_bytes()


def test_conv_passes_over_a_reduced_phonon_data_npz(tmp_path, monkeypatch, caplog):
    _reduced_defect_spectrum(tmp_path / "NV_512" / "data")
    packs, registered = _conv_package(tmp_path, monkeypatch, {"convergence_test_all": "BINARY=/x/vasp"})
    assert len(packs) == 1  # the convergence data itself is packaged
    assert not (tmp_path / "data" / "NV_512" / "phonon_data.npz").exists()
    assert not any(kind == "phonon" for kind, _ in registered)
    assert "no unaligned full phonon spectrum" in caplog.text and "(2 of 3 modes): symmetry-reduced" in caplog.text


def test_older_reduced_files_are_recognised_from_their_keys(tmp_path):
    from spinlabs.cli.package import _npz_reduced_reason

    legacy = tmp_path / "phonon_data.npz"  # the legacy key names: freqs, idx, no n_full
    np.savez(legacy, freqs=np.array([10.0, 30.0]), eigs=np.zeros((2, 1, 3)), atoms=np.zeros((1, 3)),
             atom_symbols=np.array(["C"]), lattice=np.eye(3), idx=np.array([0, 2]))
    assert "indices into a larger spectrum" in _npz_reduced_reason(legacy)
    no_indices = tmp_path / "phonons.npz"
    np.savez(no_indices, freqs=np.array([10.0, 30.0]), atom_symbols=np.array(["C"]))
    assert "a full spectrum has 3" in _npz_reduced_reason(no_indices)
    _spectrum(FULL).save(tmp_path / "full.npz")
    assert _npz_reduced_reason(tmp_path / "full.npz") is None


def test_full_spectrum_of_refuses_a_reduced_file_without_indices(tmp_path):
    path = tmp_path / "phonon_data_sym2.npz"
    _spectrum(FULL[:2]).save(path)  # 2 modes of a 1-atom cell, indices 0, 1: looks like its own full spectrum
    with pytest.raises(ValueError, match="symmetry-reduced"):
        full_spectrum_of(path)


# -- spinlabs package --conv: the same pairing for the convergence runs ------------------------------------------


def _conv_package(tmp_path, monkeypatch, scripts):
    """Package NV_512's convergence folders, {folder name: job-script line}.

    Returns {pack name: metadata} of the saved packs and the registered (kind, file name) pairs.
    """
    import json
    import sqlite3

    from test_cli_package import _conv_args, _outcar, _registry

    from spinlabs.cli.package import _run_conv

    db = _registry(monkeypatch, tmp_path)
    for name, line in scripts.items():
        root = tmp_path / "NV_512" / name
        for p in (0.025, 0.05):
            _outcar(root / "runs" / f"1_pert_{p:g}" / "OUTCAR", (1.0, 1.0, 2.0))
        (root / "run_convergence.sh").write_text(f"#!/bin/bash\n{line}\n")
    _run_conv(args=_conv_args(tmp_path, None), defect_folder=tmp_path / "NV_512")
    packs = {}
    for path in sorted((tmp_path / "data").rglob("*_conv_*.npz")):
        with np.load(path, allow_pickle=True) as data:
            packs[path.name] = json.loads(str(data["metadata"]))
    rows = sqlite3.connect(db).execute("SELECT kind, path FROM files").fetchall() if db.exists() else []
    return packs, {(kind, Path(path).name) for kind, path in rows}


def test_conv_packages_the_aligned_spectrum_its_runs_used(tmp_path, monkeypatch):
    full, reduced = _pair(tmp_path / "NV_512" / "data", stub="_aligned")
    packs, registered = _conv_package(tmp_path, monkeypatch, {"convergence_test_all": f"PHONON_PATH={reduced}"})
    packaged = tmp_path / "data" / "NV_512" / "phonon_data_aligned.npz"
    assert packaged.read_bytes() == full.read_bytes()
    assert ("phonon", "phonon_data_aligned.npz") in registered
    # the defect has only aligned files: none of them stands in for its phonon code spectrum
    assert not (tmp_path / "data" / "NV_512" / "phonon_data.npz").exists()
    (meta,) = packs.values()
    assert meta["phonon_file"] == str(full) and meta["aligned"] is True


def test_conv_folder_it_cannot_pair_is_not_packaged(tmp_path, monkeypatch, caplog):
    _, reduced = _pair(tmp_path / "NV_512" / "data", stub="_aligned")
    (tmp_path / "NV_512" / "data" / "phonon_data_aligned.npz").unlink()  # the reduced file has no full spectrum
    packs, _ = _conv_package(tmp_path, monkeypatch, {"convergence_test_all": f"PHONON_PATH={reduced}"})
    assert packs == {}
    assert "cannot pair the data with its phonon spectrum" in caplog.text
    assert not (tmp_path / "data" / "NV_512" / "phonon_data_aligned.npz").exists()  # never the reduced file


def test_conv_folders_along_different_spectra_stop(tmp_path, monkeypatch):
    _, reduced = _pair(tmp_path / "NV_512" / "data", stub="_aligned")
    swapped = np.eye(3)[[1, 0, 2]]  # another aligned spectrum: modes 0 and 1 exchanged
    other_dir = tmp_path / "NV_512" / "other"
    other_dir.mkdir(parents=True)
    _spectrum(swapped, aligned=True).save(other_dir / "phonon_data_aligned.npz")
    other = other_dir / "phonon_data_sym2_aligned.npz"
    _spectrum(swapped[[0, 2]], original=[0, 2], n_full=3, aligned=True).save(other)
    with pytest.raises(SystemExit, match="different aligned spectra"):
        _conv_package(tmp_path, monkeypatch, {"convergence_test_all": f"PHONON_PATH={reduced}",
                                              "convergence_test_occup": f"PHONON_PATH={other}"})
    assert not (tmp_path / "data").exists()


def test_conv_refuses_to_replace_a_packaged_aligned_spectrum(tmp_path, monkeypatch):
    _, reduced = _pair(tmp_path / "NV_512" / "data", stub="_aligned")
    earlier = tmp_path / "data" / "NV_512" / "phonon_data_aligned.npz"
    earlier.parent.mkdir(parents=True)
    _spectrum(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), aligned=True).save(earlier)
    with pytest.raises(SystemExit, match="holds other aligned modes"):
        _conv_package(tmp_path, monkeypatch, {"convergence_test_all": f"PHONON_PATH={reduced}"})
    assert not list((tmp_path / "data").rglob("*_conv_*.npz"))  # nothing saved


def test_conv_without_phonon_path_packages_the_defects_spectrum(tmp_path, monkeypatch):
    full, _ = _pair(tmp_path / "NV_512" / "data")
    packs, registered = _conv_package(tmp_path, monkeypatch, {"convergence_test_all": "BINARY=/x/vasp"})
    assert (tmp_path / "data" / "NV_512" / "phonon_data.npz").read_bytes() == full.read_bytes()
    assert ("phonon", "phonon_data.npz") in registered
    (meta,) = packs.values()
    assert meta["phonon_file"] is None and meta["aligned"] is None
