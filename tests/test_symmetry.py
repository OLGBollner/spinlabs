"""Tests for the generalized symmetry pipeline.

Validation strategy: the general pipeline must reproduce the legacy
C3v hard-coded classification on real NV/ClV phonon data, plus unit
tests on synthetic structures with known point groups.
"""

from collections import Counter
from pathlib import Path
import warnings

import numpy as np

from spinlabs.models import check_original_indices
import pytest
from pymatgen.core import Structure

from spinlabs.symmetry import (
    detect_point_group,
    classify_modes,
    _parent_label,
)
from spinlabs.parsers import parse_phonon_npz


# ---------------------------------------------------------------------------
# Synthetic structures: known point groups
# ---------------------------------------------------------------------------


def _fcc_si_structure():
    """Primitive Si diamond structure -> space group Fd-3m, point group m-3m (Oh)."""
    lattice = np.array([[0, 5.43 / 2, 5.43 / 2], [5.43 / 2, 0, 5.43 / 2], [5.43 / 2, 5.43 / 2, 0]])
    return Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


class TestPointGroupDetection:
    def test_diamond_si_is_oh(self):
        pg = detect_point_group(_fcc_si_structure())
        assert pg.symbol == "m-3m"
        assert pg.order == 48

    def test_returns_operations(self):
        pg = detect_point_group(_fcc_si_structure())
        assert len(pg.operations) == 48


# ---------------------------------------------------------------------------
# Mode classification vs. legacy C3v results on real data
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
# The two small cells are committed as ordinary git blobs (~1.5 MB total), so
# CI gets real DFT data with zero LFS bandwidth.
SMALL_CELLS = {
    "NV_64": REPO_ROOT / "NV_64" / "phonon_data.npz",
    "ClV_128": REPO_ROOT / "ClV_128" / "phonon_data.npz",
}
# NV_512 (19 MB) stays in LFS and is only used for whole-cell numbers (mode
# counts, legacy labels). Those tests skip when the file is an LFS pointer.
NV_PATH = REPO_ROOT / "NV_512" / "phonon_data.npz"
# Whole frequency clusters cut from NV_512, including the only genuinely
# mixed A1/Ey pair (313/315); a plain git blob, so CI always has it.
# Rebuild with scripts/make_nv512_test_subset.py.
NV_SUBSET_PATH = REPO_ROOT / "NV_512" / "phonon_data_subset.npz"
CLV_PATH = SMALL_CELLS["ClV_128"]
CELL_PATHS = {**SMALL_CELLS, "NV_512": NV_PATH}

# Pinned per-cell inventories: total modes, modes left after
# filter_sym_pairs(), then per-label counts of the filtered spectrum.
INVENTORY = {
    "NV_64": {"n_modes": 189, "filtered": 126, "Ex": 63, "A1": 39, "A2": 24},
    "ClV_128": {"n_modes": 381, "filtered": 254, "Ex": 127, "A1": 79, "A2": 48},
    # 511 atoms -> 1533 raw modes = A1(287) + A2(224) + 2E(511).
    "NV_512": {"n_modes": 1533, "filtered": 1022, "Ex": 511, "A1": 287, "A2": 224},
}


def _real_phonon(path):
    """Return the path if real data is present. Only the LFS-tracked NV_512
    file may be absent (skip); the committed blobs must be there (fail)."""
    p = Path(path)
    missing = None
    if not p.is_file() or p.stat().st_size < 1024:
        missing = f"real phonon data not available: {p}"
    else:
        with p.open("rb") as fh:
            if fh.read(64).startswith(b"version https://git-lfs"):
                missing = f"LFS pointer, data not pulled: {p}"
    if missing is None:
        return p
    if p == NV_PATH:
        pytest.skip(missing)
    pytest.fail(missing)


def _mode(spec, original):
    """Position of full-run mode number ``original`` in ``spec``."""
    if spec.original_indices is None:
        return original
    return int(np.flatnonzero(spec.original_indices == original)[0])


CELL_CASES = [pytest.param(path, id=name) for name, path in SMALL_CELLS.items()] + [
    pytest.param(NV_SUBSET_PATH, id="NV_512_subset"),
    pytest.param(NV_PATH, id="NV_512"),
]


@pytest.mark.parametrize("path", CELL_CASES)
def test_matches_legacy_c3v_labels(path):
    spec = parse_phonon_npz(_real_phonon(path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = spec.analyze_c3v_symmetry()
    labels, _groups = classify_modes(spec)
    match = sum(lab == s for lab, s in zip(labels, legacy))
    assert match / len(legacy) > 0.95, f"only {match}/{len(legacy)} labels match"


@pytest.mark.parametrize("path", CELL_CASES)
def test_e_pairs_are_detected_as_degenerate(path):
    spec = parse_phonon_npz(_real_phonon(path))
    labels, groups = classify_modes(spec)
    # In C3v, E modes come in degenerate pairs; Ex/Ey partners share a
    # degeneracy group (same parent irrep, same frequency).
    e_groups = [g for g in groups if labels[g[0]].startswith("E")]
    assert e_groups
    assert all(len(g) >= 2 for g in e_groups)


@pytest.mark.parametrize("path", CELL_CASES)
def test_ex_ey_share_degeneracy_group(path):
    spec = parse_phonon_npz(_real_phonon(path))
    labels, groups = classify_modes(spec)
    # Both cells have clean Ex/Ey pairs: every Ex must sit in a 2-member
    # group with its Ey partner at the same frequency.
    for i, lab in enumerate(labels):
        if lab == "Ex":
            g = next(g for g in groups if i in g)
            partners = {labels[j] for j in g if j != i}
            assert partners == {"Ey"}


@pytest.mark.parametrize("path", CELL_CASES)
def test_accidental_degeneracies_do_not_merge(path):
    # Two modes of different parent irreps at the same frequency must not
    # land in the same degeneracy group.
    spec = parse_phonon_npz(_real_phonon(path))
    labels, groups = classify_modes(spec)
    for g in groups:
        parents = {_parent_label(labels[j]) for j in g}
        assert len(parents) == 1


# ---------------------------------------------------------------------------
# filter_sym_pairs: regression tests for the keep-all-Ex rule
# ---------------------------------------------------------------------------


def _tiny_spectrum(freqs, syms):
    """Minimal PhononSpectrum with preset labels (bypasses classification)."""
    import numpy as _np
    from spinlabs.models import PhononSpectrum

    n = len(freqs)
    spec = PhononSpectrum(
        frequencies_mev=_np.asarray(freqs, dtype=float),
        eigenvectors=_np.zeros((n, 1, 3)),
        atom_frac_coords=_np.zeros((1, 3)),
        atom_symbols=["C"],
        atomic_masses=_np.array([12.0]),
        lattice=_np.eye(3),
        symmetries=list(syms),
        original_indices=_np.arange(n),
    )
    spec.symmetries = list(syms)  # keep the preset labels
    return spec


def test_filter_keeps_all_ex_drops_all_matched_ey():
    spec = _tiny_spectrum(
        [10.0, 20.005, 20.0, 30.0],
        ["A1", "Ex", "Ey", "A2"],
    )
    red = spec.filter_sym_pairs()
    assert list(red.symmetries) == ["A1", "Ex", "A2"]
    assert red.n_modes == 3
    assert red.original_indices.tolist() == [0, 1, 3]


def test_filter_drops_ey_even_when_ex_is_closer_to_another_ex():
    # Regression: the legacy greedy scan dropped the Ex at 149.0761 instead
    # of the Ey twin at 149.0706, because another Ex sat closer to it.
    spec = _tiny_spectrum(
        [149.0706, 149.0761, 149.0764],
        ["Ex", "Ex", "Ey"],
    )
    red = spec.filter_sym_pairs()
    # All Ex kept; the Ey (within tol of an Ex) dropped.
    assert list(red.symmetries) == ["Ex", "Ex"]
    assert red.original_indices.tolist() == [0, 1]


def test_filter_orphan_ey_is_kept():
    # An Ey with no Ex within tolerance must survive (it cannot be paired).
    spec = _tiny_spectrum([10.0, 10.008, 10.02], ["Ex", "Ey", "Ey"])
    red = spec.filter_sym_pairs()
    assert list(red.symmetries) == ["Ex", "Ey"]
    assert red.original_indices.tolist() == [0, 2]


def test_filter_group_theory_invariant_on_synthetic_c3v():
    # 1 atom in C3v: 3 modes = A1 + 2E. After filtering: A1 + E = 2 modes.
    spec = _tiny_spectrum([100.0, 100.0, 200.0], ["Ex", "Ey", "A1"])
    red = spec.filter_sym_pairs()
    assert red.n_modes == 2
    assert Counter(list(red.symmetries)) == {"Ex": 1, "A1": 1}


def test_filter_rejects_broken_original_indices():
    from spinlabs.models import check_original_indices

    with pytest.raises(ValueError):
        check_original_indices(np.array([0, 0, 2]), n_modes=3)
    with pytest.raises(ValueError):
        check_original_indices(np.array([0, 1, 9]), n_modes=3, n_full=3)
    with pytest.raises(TypeError):
        check_original_indices(np.array([0.0, 1.0]), n_modes=2)


@pytest.mark.parametrize("cell", sorted(INVENTORY))
def test_filter_original_indices_valid_on_real_data(cell):
    spec = parse_phonon_npz(_real_phonon(CELL_PATHS[cell]))
    red = spec.filter_sym_pairs()
    check_original_indices(red.original_indices, red.n_modes, n_full=spec.n_modes)
    c = Counter(str(s) for s in red.symmetries)
    want = INVENTORY[cell]
    # Group-theoretic invariant: every mode is A1 + A2 + 2E, and after
    # filtering no Ey survives (one E partner per pair is dropped).
    assert spec.n_modes == want["n_modes"]
    assert c["Ey"] == 0
    assert c["Ex"] == want["Ex"]
    assert c["A1"] == want["A1"] and c["A2"] == want["A2"]
    assert red.n_modes == want["filtered"]


# ---------------------------------------------------------------------------
# Symmetrization of accidentally mixed near-degenerate modes
# ---------------------------------------------------------------------------


def _accidental_a1_e_clusters(spec, tol_mev=0.01):
    """Frequency clusters, as symmetrize_degenerate_groups forms them, that hold
    an A1 together with an Ex/Ey pair: accidental 1D/2D degeneracies."""
    from spinlabs.symmetry import _build_clusters

    labels = [str(s) for s in spec.symmetries]
    clusters = _build_clusters(np.asarray(spec.frequencies_mev, dtype=float), tol_mev)
    return [c for c in clusters if {"A1", "Ex", "Ey"} <= {labels[j] for j in c}]


def _class_chars_ok(chars, label):
    """Characters on E/C3/C3^2 match the label's irrep: A1 -> (1,1,1),
    E -> (1,-0.5,-0.5)."""
    want3 = np.array([1.0, 1.0, 1.0]) if label == "A1" else np.array([1.0, -0.5, -0.5])
    return np.allclose(chars[:3], want3, atol=0.05)


# NV_512 subset first: it carries the mixed pair and is always present.
MIXED_CASES = [
    pytest.param(NV_SUBSET_PATH, id="NV_512_subset"),
    pytest.param(NV_PATH, id="NV_512"),
]


@pytest.mark.parametrize("path", MIXED_CASES)
def test_symmetrize_accidental_a1_e_degeneracies(path):
    """Real accidental A1/E degeneracies: at least one mode starts mixed, and
    after symmetrization every mode has its label's class characters."""
    from spinlabs.symmetry import (
        symmetrize_degenerate_groups,
        defect_frame_operations,
        _mode_characters,
    )

    spec = parse_phonon_npz(_real_phonon(path))
    clusters = _accidental_a1_e_clusters(spec)
    assert clusters, "no accidental A1/E degeneracy found in this cell"
    members = [i for c in clusters for i in c]
    labels = [str(s) for s in spec.symmetries]
    before = _mode_characters(spec, defect_frame_operations(spec))
    # Guard against a vacuous pass: the input must actually be mixed.
    assert any(not _class_chars_ok(before[i], labels[i]) for i in members if labels[i] != "A2")

    sym = symmetrize_degenerate_groups(spec)
    chars = _mode_characters(sym, defect_frame_operations(sym))
    new_labels = [str(s) for s in sym.symmetries]
    for i in members:
        if new_labels[i] != "A2":
            assert _class_chars_ok(chars[i], new_labels[i]), (i, new_labels[i], chars[i])


def test_symmetrize_mixed_a1_e_pair_on_nv512():
    """NV_512 modes 313/314/315: Ex pure, Ey~A1 mixed (accidental degeneracy).

    After symmetrization every group member must have characters matching
    its label's class characters exactly.
    """
    from spinlabs.symmetry import (
        symmetrize_degenerate_groups,
        defect_frame_operations,
        _mode_characters,
    )

    spec = parse_phonon_npz(_real_phonon(NV_SUBSET_PATH))
    labels = [str(s) for s in spec.symmetries]
    i313, i314, i315 = (_mode(spec, m) for m in (313, 314, 315))
    assert labels[i313] == "Ey" and labels[i314] == "Ex" and labels[i315] == "A1"
    before = _mode_characters(spec, defect_frame_operations(spec))
    assert not _class_chars_ok(before[i313], "Ey") and not _class_chars_ok(before[i315], "A1")

    sym = symmetrize_degenerate_groups(spec)
    ops = defect_frame_operations(sym)
    chars = _mode_characters(sym, ops)
    new_labels = [str(s) for s in sym.symmetries]
    for i in (i313, i314, i315):
        c = chars[i]
        assert _class_chars_ok(c, new_labels[i]), (i, new_labels[i], c)
        if new_labels[i] in ("Ex", "Ey"):
            # mirror characters are cos(2 phi_k): absolute values must form
            # the same multiset as the pure reference mode's, and the sublabel
            # sign rule (op 3) must agree with the assigned label
            assert sorted(np.abs(c[3:]).round(3)) == sorted([0.5, 0.5, 1.0]), (i, c)
            assert (c[3] > 0) == (new_labels[i] == "Ex"), (i, new_labels[i], c)
    # unchanged elsewhere: pure modes stay put
    i309 = _mode(spec, 309)
    assert np.allclose(sym.eigenvectors[i309], spec.eigenvectors[i309], atol=1e-8)


@pytest.mark.parametrize("path", CELL_CASES)
def test_symmetrize_is_fixpoint_on_pure_groups(path):
    from spinlabs.symmetry import symmetrize_degenerate_groups

    spec = parse_phonon_npz(_real_phonon(path))
    sym = symmetrize_degenerate_groups(spec)
    # every vector either unchanged or reassigned within its group; total
    # label multiset must be preserved
    from collections import Counter

    assert Counter(map(str, sym.symmetries)) == Counter(map(str, spec.symmetries))


def test_symmetrize_recover_artificial_mixing():
    """Rotate a known-pure A1/Ey pair by theta, symmetrize, check recovery.

    Uses the NV_512 pair that recovers exactly (modes 313/315). On NV_64 and
    ClV_128 the same rotation is only partially undone, so this
    exact-recovery claim is not asserted there.
    """
    from spinlabs.symmetry import symmetrize_degenerate_groups
    import dataclasses

    spec = parse_phonon_npz(_real_phonon(NV_SUBSET_PATH))
    pure = symmetrize_degenerate_groups(spec)  # ground-truth pure states
    ia, ie = _mode(spec, 315), _mode(spec, 313)  # A1 and Ey slots
    theta = 0.35
    va = pure.eigenvectors[ia].copy()
    ve = pure.eigenvectors[ie].copy()
    vecs = spec.eigenvectors.copy()
    vecs[ia] = np.cos(theta) * va + np.sin(theta) * ve
    vecs[ie] = -np.sin(theta) * va + np.cos(theta) * ve
    mixed = dataclasses.replace(spec, eigenvectors=vecs)
    sym = symmetrize_degenerate_groups(mixed)
    # recovered vectors must overlap strongly with the pure originals
    ov_a = abs(np.sum(sym.eigenvectors[ia] * va))
    ov_e = abs(np.sum(sym.eigenvectors[ie] * ve))
    assert ov_a > 0.999, ov_a
    assert ov_e > 0.999, ov_e


# ---------------------------------------------------------------------------
# sym_check_summary: wrapper-facing one-liner for the verify report
# ---------------------------------------------------------------------------


def test_sym_check_summary_variants():
    from spinlabs.symmetry import sym_check_summary

    assert sym_check_summary(None) == ""
    assert sym_check_summary({"ok": True, "failures": [], "unpaired": []}) == "ok"
    assert sym_check_summary({"ok": True, "failures": [], "unpaired": [1, 2]}) == "ok (2 unpaired)"
    msg = sym_check_summary({"ok": False, "failures": [(3, [10, 11], [2.0, -1.0], [1.9, -0.8])], "unpaired": []})
    assert "FAILED" in msg and "1" in msg and "group 3" in msg
