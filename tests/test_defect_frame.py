"""The defect frame: stored in the phonon file, else derived from the structure (symmetry.defect_frame).

The C3 axis and the sigma_v1 normal fix the symmetry operations, the Ex/Ey labels and the alignment
mirror. A stored frame is used as it is (checked against the structure); a file without one gets
frame_from_structure: the C3 axis, and the mirror normal with the largest component along a1.
"""

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from spinlabs.models import PhononSpectrum
from spinlabs.parsers import parse_phonon_npz
from spinlabs.symmetry import (
    ALIGN_MIRROR,
    _mode_characters,
    align_degenerate_pairs,
    classify_and_pair,
    classify_modes,
    defect_frame,
    defect_frame_operations,
    filter_degenerate_partners,
    frame_from_structure,
    frame_matrix,
    symmetrize_degenerate_groups,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NV_64_PHONONS = REPO_ROOT / "NV_64" / "phonon_data.npz"  # plain git blob, frame stamped
NV_LEGACY_FRAME = frame_matrix([1.0, 1.0, 1.0], [1.0, -1.0, 0.0])


def _ov_like_spectrum() -> PhononSpectrum:
    """O on C next to a Si vacancy along c in 2H-SiC (3x3x2), two rigid-shift modes, no stored frame."""
    u = 0.375
    prim = Structure(
        Lattice.hexagonal(3.08, 5.04),
        ["Si", "Si", "C", "C"],
        [[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0.5], [1 / 3, 2 / 3, u], [2 / 3, 1 / 3, 0.5 + u]],
    )
    cell = prim.copy().make_supercell([3, 3, 2])
    c_site = next(i for i, s in enumerate(cell) if s.specie.symbol == "C")
    si_site = next(
        i
        for i, s in enumerate(cell)
        if s.specie.symbol == "Si"
        and np.allclose(s.frac_coords[:2], cell[c_site].frac_coords[:2])
        and cell.get_distance(c_site, i) < 2.0
    )
    cell.replace(c_site, "O")
    cell.remove_sites([si_site])
    n = len(cell)
    modes = np.zeros((2, n, 3))
    modes[0, :, 2] = 1.0
    modes[1, :, 0] = 1.0
    return PhononSpectrum(
        frequencies_mev=np.array([10.0, 20.0]),
        eigenvectors=modes / np.sqrt(n),
        atom_frac_coords=np.array([s.frac_coords for s in cell]),
        atom_symbols=[s.specie.symbol for s in cell],
        atomic_masses=np.ones(n),
        lattice=cell.lattice.matrix,
    )


@pytest.fixture(scope="module")
def nv64():
    return parse_phonon_npz(NV_64_PHONONS)


def test_hexagonal_frame_from_structure():
    frame = frame_from_structure(_ov_like_spectrum())
    np.testing.assert_allclose(frame[:, 2], [0, 0, 1], atol=1e-6)  # C3 axis along c
    np.testing.assert_allclose(frame[:, 0], [1, 0, 0], atol=1e-6)  # the normal along a1


def test_classification_stores_the_frame():
    spectrum = _ov_like_spectrum()  # classified on construction
    assert spectrum.symmetries == ["A1", spectrum.symmetries[1]] and spectrum.symmetries[1] in ("Ex", "Ey")
    np.testing.assert_allclose(spectrum.frame, frame_from_structure(spectrum))


def test_cubic_nv_frame_breaks_the_a1_tie_by_a2(nv64):
    # [1,0,-1] and [1,-1,0] have the same component along a1 = x; a2 = y breaks the tie, so a file
    # without a stored frame gets another mirror than the legacy [1,-1,0] (hence the stamped files).
    frame = frame_from_structure(nv64)
    np.testing.assert_allclose(frame[:, 2], np.ones(3) / np.sqrt(3), atol=1e-6)
    np.testing.assert_allclose(frame[:, 0], np.array([1, 0, -1]) / np.sqrt(2), atol=1e-6)


def test_stamped_file_keeps_its_legacy_frame(nv64, tmp_path):
    np.testing.assert_array_equal(nv64.frame, NV_LEGACY_FRAME)
    np.testing.assert_array_equal(defect_frame(nv64), NV_LEGACY_FRAME)
    out = tmp_path / "phonon.npz"
    nv64.save(out)
    np.testing.assert_array_equal(PhononSpectrum.load(out).frame, NV_LEGACY_FRAME)
    np.testing.assert_array_equal(filter_degenerate_partners(nv64).frame, NV_LEGACY_FRAME)


def test_stored_frame_that_is_no_symmetry_raises(nv64):
    wrong = replace(nv64, symmetries=None, frame=frame_matrix([0, 0, 1], [1, 0, 0]))
    with pytest.raises(ValueError, match="not a symmetry of the structure"):
        classify_modes(wrong)


def test_labels_without_a_frame_warn(nv64, caplog):
    unstamped = replace(nv64, frame=None)
    with caplog.at_level(logging.WARNING, logger="spinlabs.symmetry"):
        defect_frame(unstamped)
    assert "no stored defect frame" in caplog.text


def test_aligned_filter_keeps_the_mirror_odd_partner(nv64):
    spec = replace(nv64)
    classify_and_pair(spec)
    aligned = align_degenerate_pairs(symmetrize_degenerate_groups(spec))
    assert aligned.aligned is True
    kept = filter_degenerate_partners(aligned)
    labels = np.array(kept.symmetries)
    e = labels == "Ex"
    assert e.sum() == 63 and not np.isin(labels, ["Ey"]).any()
    eigs = np.asarray(kept.eigenvectors, dtype=float)
    chi = _mode_characters(kept, [defect_frame_operations(kept)[ALIGN_MIRROR]])[:, 0]
    chi = chi / np.sum(eigs.reshape(len(eigs), -1) ** 2, axis=1)
    np.testing.assert_allclose(chi[e], -1.0, atol=1e-6)


def test_aligned_flag_on_unaligned_pairs_raises(nv64):
    spec = replace(nv64)
    classify_and_pair(spec)
    unaligned = symmetrize_degenerate_groups(spec)
    unaligned.aligned = True  # claims alignment its pairs do not have
    with pytest.raises(ValueError, match="odd under the alignment mirror"):
        filter_degenerate_partners(unaligned)
