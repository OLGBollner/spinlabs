"""Aligning degenerate E sets to a mirror plane (symmetry.align_degenerate_pairs, spinlabs symmetrize --align)."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from spinlabs import cli
from spinlabs.parsers import parse_phonon_npz
from spinlabs.symmetry import (
    ALIGN_MIRROR,
    _group_operator_apply,
    _op_mappings,
    align_degenerate_pairs,
    classify_and_pair,
    defect_frame_operations,
    filter_degenerate_partners,
    mirror_parities,
    symmetrize_degenerate_groups,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NV_64_PHONONS = REPO_ROOT / "NV_64" / "phonon_data.npz"  # plain git blob


def _chi(spec, mirror_index):
    """<e|M e> / <e|e> of every mode for one mirror of defect_frame_operations."""
    M = [defect_frame_operations(spec)[mirror_index]]
    mapping = _op_mappings(spec, M)
    vecs = np.asarray(spec.eigenvectors, dtype=float)
    out = []
    for v in vecs:
        mv = _group_operator_apply(mapping, M, v[None])[0]
        out.append(float(np.sum(v * mv) / np.sum(v * v)))
    return np.array(out)


@pytest.fixture(scope="module")
def nv64():
    spec = parse_phonon_npz(NV_64_PHONONS)
    classify_and_pair(spec)
    sym = symmetrize_degenerate_groups(spec)
    return sym, align_degenerate_pairs(sym)


def test_every_e_set_is_aligned_to_one_mirror(nv64):
    sym, aligned = nv64
    labels = np.array(aligned.symmetries)
    e = np.isin(labels, ["Ex", "Ey"])
    before, after = _chi(sym, ALIGN_MIRROR), _chi(aligned, ALIGN_MIRROR)
    assert np.min(np.abs(before[e])) < 0.9  # the phonon code's pairs are not aligned
    assert np.allclose(np.abs(after[e]), 1.0, atol=1e-6)
    # The kept partner (Ex: positive character on sigma_v1) is the sigma_v2-odd one.
    assert np.allclose(after[labels == "Ex"], -1.0, atol=1e-6)
    assert np.allclose(after[labels == "Ey"], 1.0, atol=1e-6)
    assert len(aligned.alignment_report["aligned"]) == 63 and aligned.alignment_report["skipped"] == []
    assert aligned.aligned is True


def test_alignment_keeps_frequencies_labels_and_orthonormality(nv64):
    sym, aligned = nv64
    assert np.array_equal(aligned.frequencies_mev, sym.frequencies_mev)
    v = aligned.eigenvectors.reshape(aligned.n_modes, -1)
    assert np.allclose(v @ v.T, np.eye(aligned.n_modes), atol=1e-10)
    for label in ("A1", "A2", "Ex", "Ey"):
        assert list(aligned.symmetries).count(label) == list(sym.symmetries).count(label)
    assert aligned.sym_check is None or aligned.sym_check.get("failures") in (None, [])
    # Non-E modes are untouched.
    a = np.isin(np.array(aligned.symmetries), ["A1", "A2"])
    assert np.array_equal(aligned.eigenvectors[a], sym.eigenvectors[a])


def test_filtered_aligned_spectrum_keeps_only_mirror_odd_e_modes(nv64):
    _, aligned = nv64
    kept = filter_degenerate_partners(aligned)
    assert kept.n_modes == 126 and kept.aligned is True
    par = mirror_parities(kept)
    chi = np.array([par[k][1] for k, s in enumerate(kept.symmetries) if s in ("Ex", "Ey")])
    assert len(chi) == 63 and np.allclose(chi, -1.0, atol=1e-6)


@pytest.mark.parametrize("flag", [True, False, None])
def test_aligned_key_round_trips(nv64, tmp_path, flag):
    sym, _ = nv64
    sym.aligned = flag
    path = tmp_path / "p.npz"
    sym.save(path)
    assert ("aligned" in np.load(path, allow_pickle=True).files) is (flag is not None)
    assert parse_phonon_npz(path).aligned is flag


def test_cli_writes_plain_and_aligned_files(tmp_path, capsys):
    src = tmp_path / "phonon_data.npz"
    shutil.copy(NV_64_PHONONS, src)

    cli.main(["symmetrize", str(src)])
    plain = parse_phonon_npz(tmp_path / "phonon_data_sym126.npz")
    assert plain.aligned is False and plain.n_modes == 126

    cli.main(["symmetrize", str(src), "--align"])
    out = capsys.readouterr().out
    reduced = parse_phonon_npz(tmp_path / "phonon_data_sym126_aligned.npz")
    full = parse_phonon_npz(tmp_path / "phonon_data_aligned.npz")
    assert reduced.aligned is True and reduced.n_modes == 126
    assert full.aligned is True and full.n_modes == 189
    assert "63 degenerate E set(s) aligned" in out and "aligned full spectrum" in out
    # The two aligned files share mode identity: each kept mode is the same vector in the full one.
    for k, original in enumerate(reduced.original_indices):
        assert np.allclose(reduced.eigenvectors[k], full.eigenvectors[original])
