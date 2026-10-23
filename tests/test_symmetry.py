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
NV_PATH = REPO_ROOT / "NV_512" / "phonon_data.npz"
CLV_PATH = REPO_ROOT / "ClV_128" / "phonon_data.npz"


@pytest.mark.parametrize("path", [NV_PATH, CLV_PATH])
def test_matches_legacy_c3v_labels(path):
    spec = parse_phonon_npz(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = spec.analyze_c3v_symmetry()
    labels, _groups = classify_modes(spec)
    match = sum(lab == s for lab, s in zip(labels, legacy))
    assert match / len(legacy) > 0.95, f"only {match}/{len(legacy)} labels match"


def test_e_pairs_are_detected_as_degenerate():
    spec = parse_phonon_npz(NV_PATH)
    labels, groups = classify_modes(spec)
    # In C3v, E modes come in degenerate pairs; Ex/Ey partners share a
    # degeneracy group (same parent irrep, same frequency).
    e_groups = [g for g in groups if labels[g[0]].startswith("E")]
    assert e_groups
    assert all(len(g) >= 2 for g in e_groups)


def test_ex_ey_share_degeneracy_group():
    spec = parse_phonon_npz(CLV_PATH)
    labels, groups = classify_modes(spec)
    # ClV-128 has clean Ex/Ey pairs: every Ex must sit in a 2-member group
    # with its Ey partner at the same frequency.
    for i, lab in enumerate(labels):
        if lab == "Ex":
            g = next(g for g in groups if i in g)
            partners = {labels[j] for j in g if j != i}
            assert partners == {"Ey"}


def test_accidental_degeneracies_do_not_merge():
    # Two modes of different parent irreps at the same frequency must not
    # land in the same degeneracy group.
    spec = parse_phonon_npz(NV_PATH)
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


def test_filter_original_indices_valid_on_real_data():
    spec = parse_phonon_npz(NV_PATH)
    red = spec.filter_sym_pairs()
    check_original_indices(red.original_indices, red.n_modes, n_full=spec.n_modes)
    labels = [str(s) for s in red.symmetries]
    c = Counter(labels)
    # Group-theoretic invariant for NV-512: 511 atoms -> 1533 raw modes
    # = A1(287) + A2(224) + 2E(511); after filtering, no Ey survives.
    assert c["Ey"] == 0
    assert c["Ex"] == 511
    assert c["A1"] == 287 and c["A2"] == 224
    assert red.n_modes == 1022


# ---------------------------------------------------------------------------
# Symmetrization of accidentally mixed near-degenerate modes
# ---------------------------------------------------------------------------


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

    spec = parse_phonon_npz(NV_PATH)
    labels = [str(s) for s in spec.symmetries]
    assert labels[313] == "Ey" and labels[314] == "Ex" and labels[315] == "A1"

    sym = symmetrize_degenerate_groups(spec)
    ops = defect_frame_operations(sym)
    chars = _mode_characters(sym, ops)
    new_labels = [str(s) for s in sym.symmetries]
    for i in (313, 314, 315):
        c = chars[i]
        # class characters: A1 -> (1,1,1); E -> (1, -0.5, -0.5) for E/C3/C3^2
        want3 = np.array([1.0, 1.0, 1.0]) if new_labels[i] == "A1" else np.array([1.0, -0.5, -0.5])
        assert np.allclose(c[:3], want3, atol=0.05), (i, new_labels[i], c)
        if new_labels[i] in ("Ex", "Ey"):
            # mirror characters are cos(2 phi_k): absolute values must form
            # the same multiset as the pure reference mode's, and the sublabel
            # sign rule (op 3) must agree with the assigned label
            assert sorted(np.abs(c[3:]).round(3)) == sorted([0.5, 0.5, 1.0]), (i, c)
            assert (c[3] > 0) == (new_labels[i] == "Ex"), (i, new_labels[i], c)
    # unchanged elsewhere: pure modes stay put
    assert np.allclose(sym.eigenvectors[309], spec.eigenvectors[309], atol=1e-8)


def test_symmetrize_is_fixpoint_on_pure_groups():
    from spinlabs.symmetry import symmetrize_degenerate_groups

    spec = parse_phonon_npz(CLV_PATH)
    sym = symmetrize_degenerate_groups(spec)
    # every vector either unchanged or reassigned within its group; total
    # label multiset must be preserved
    from collections import Counter

    assert Counter(map(str, sym.symmetries)) == Counter(map(str, spec.symmetries))


def test_symmetrize_recover_artificial_mixing():
    """Rotate a known-pure A1/Ey pair by theta, symmetrize, check recovery."""
    from spinlabs.symmetry import symmetrize_degenerate_groups
    import dataclasses

    spec = parse_phonon_npz(NV_PATH)
    pure = symmetrize_degenerate_groups(spec)  # ground-truth pure states
    ia, ie = 315, 313  # A1 and Ey slots
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
