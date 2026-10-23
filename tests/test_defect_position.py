"""Tests for spinlabs.vasp.structures.find_defect.

Synthetic diamond-Si structures with known defects: a vacancy is
detected as the centroid of its under-coordinated neighbours, an
interstitial as a low-coordination atom. No pristine reference is
used, matching the self-contained heuristic.
"""

import numpy as np
import pytest
from pymatgen.core import Structure

from spinlabs.vasp.structures import detect_defect_position, find_defect


def _diamond_si_supercell(n: int = 3) -> Structure:
    """n x n x n supercell of diamond Si (2 atoms per cell, a = 5.43 A)."""
    prim = Structure(
        [[0, 2.715, 2.715], [2.715, 0, 2.715], [2.715, 2.715, 0]],
        ["Si", "Si"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )
    return prim.copy().make_supercell([n, n, n])


class TestFindDefect:
    def test_no_defect_in_pristine(self):
        struct = _diamond_si_supercell()
        assert find_defect(struct) == []

    def test_vacancy_position(self):
        struct = _diamond_si_supercell()
        removed = struct[5]
        defect_struct = Structure(
            struct.lattice,
            [s.specie for s in struct if s != removed],
            [s.frac_coords for s in struct if s != removed],
        )
        defects = find_defect(defect_struct)
        assert len(defects) == 1
        loc = defects[0]
        assert loc.defect_class == "vacancy"
        # Cartesian centroid of the under-coordinated neighbours should
        # coincide with the removed site (up to a periodic image).
        disp = (loc.frac_coords - removed.frac_coords + 0.5) % 1.0 - 0.5
        assert np.allclose(disp, 0.0, atol=1e-6)

    def test_interstitial_position(self):
        struct = _diamond_si_supercell(n=4)
        # Off-lattice-site position; a tetrahedral-site interstitial is
        # geometrically indistinguishable from a substitution and is
        # deliberately out of scope for this heuristic.
        inter_frac = np.array([0.6, 0.6, 0.6])
        struct.append("Si", inter_frac)
        defects = find_defect(struct)
        assert len(defects) == 1
        loc = defects[0]
        assert loc.defect_class == "interstitial"
        assert loc.site_index is None
        # The close pair: the added atom and the squeezed lattice atom.
        assert len(loc.neighbor_indices) == 2
        assert len(struct) - 1 in loc.neighbor_indices
        cart = np.mean([struct[i].coords for i in loc.neighbor_indices], axis=0)
        expected = struct.lattice.get_fractional_coords(cart) % 1.0
        assert np.allclose(loc.frac_coords, expected, atol=1e-6)


class TestFindDefectExtended:
    def test_divacancy_position(self):
        struct = _diamond_si_supercell()
        # Two nearest-neighbour Si sites (indices 5 and its NN).
        removed = [
            struct[5],
            min((s for i, s in enumerate(struct) if i != 5), key=lambda s: struct.get_distance(5, struct.index(s))),
        ]
        keep = [s for s in struct if all(not s.is_periodic_image(r) for r in removed)]
        defect_struct = Structure(struct.lattice, [s.specie for s in keep], [s.frac_coords for s in keep])
        defects = find_defect(defect_struct)
        assert len(defects) == 1
        loc = defects[0]
        assert loc.defect_class == "divacancy"
        assert loc.site_index is None
        # Centroid should sit on the bond centre between the two sites.
        midpoint = np.mean([r.frac_coords for r in removed], axis=0)
        disp = (loc.frac_coords - midpoint + 0.5) % 1.0 - 0.5
        assert np.allclose(disp, 0.0, atol=1e-6)

    def test_substitutional_identification(self):
        struct = _diamond_si_supercell()
        struct[5] = "Ge"
        defects = find_defect(struct)
        assert len(defects) == 1
        loc = defects[0]
        assert loc.defect_class == "substitutional"
        assert loc.site_index == 5
        assert loc.species == "Ge"


class TestDefectPositionIntegration:
    """find_defect wired into translate_defect_to_origin / _defect_center."""

    def _spectrum(self, structure: Structure):
        from spinlabs.models import PhononSpectrum

        n_atoms = len(structure)
        n_modes = 1
        return PhononSpectrum(
            frequencies_mev=np.array([10.0]),
            eigenvectors=np.zeros((n_modes, n_atoms, 3)),
            atom_frac_coords=np.array([s.frac_coords for s in structure]),
            atom_symbols=[s.specie.symbol for s in structure],
            atomic_masses=np.full(n_atoms, 28.0855),
            lattice=structure.lattice.matrix,
            symmetries=["A1"],
        )

    def test_vacancy_translation(self):
        from spinlabs.models import PhononSpectrum  # noqa: F401
        from spinlabs.vasp.structures import spectrum_structure
        from spinlabs.symmetry import _defect_center

        pristine = _diamond_si_supercell()
        removed = pristine[5]
        defect_struct = Structure(
            pristine.lattice,
            [s.specie for s in pristine if s != removed],
            [s.frac_coords for s in pristine if s != removed],
        )
        spectrum = self._spectrum(defect_struct)
        frac, defect_frac = spectrum.translate_defect_to_origin()
        np.testing.assert_allclose(defect_frac, np.mod(removed.frac_coords, 1.0), atol=1e-6)
        # The vacancy site is now at the origin: no atom should sit near it.
        d = (frac + 0.5) % 1.0 - 0.5
        assert np.all(np.linalg.norm(d @ spectrum.lattice, axis=1) > 1e-6)
        np.testing.assert_allclose(_defect_center(spectrum), np.mod(removed.frac_coords, 1.0), atol=1e-6)
        # Round trip: rebuilding a structure from the shifted coords puts
        # the vacancy at the origin.
        assert spectrum_structure(spectrum).num_sites == len(pristine) - 1

    def test_divacancy_translation(self):
        from spinlabs.symmetry import _defect_center

        pristine = _diamond_si_supercell(n=3)
        remove = [pristine[15], pristine[44]]
        defect_struct = Structure(
            pristine.lattice,
            [s.specie for s in pristine if s not in remove],
            [s.frac_coords for s in pristine if s not in remove],
        )
        spectrum = self._spectrum(defect_struct)
        frac, defect_frac = spectrum.translate_defect_to_origin()
        # Periodic minimum-image midpoint: the naive mean can cross the
        # cell boundary for divacancy partners in different images.
        a, b = remove
        diff = (b.frac_coords - a.frac_coords + 0.5) % 1.0 - 0.5
        expected = np.mod(a.frac_coords + diff / 2.0, 1.0)
        disp = (defect_frac - expected + 0.5) % 1.0 - 0.5
        assert np.linalg.norm(disp @ spectrum.lattice) < 0.25
        np.testing.assert_allclose(_defect_center(spectrum), defect_frac, atol=1e-6)


class TestDetectDefectPosition:
    """detect_defect_position: success, None, wrap, and multi-defect error."""

    def _spectrum(self, structure: Structure):
        from spinlabs.models import PhononSpectrum

        return PhononSpectrum(
            frequencies_mev=np.array([10.0]),
            eigenvectors=np.zeros((1, len(structure), 3)),
            atom_frac_coords=np.array([s.frac_coords for s in structure]),
            atom_symbols=[s.specie.symbol for s in structure],
            atomic_masses=np.full(len(structure), 28.0855),
            lattice=structure.lattice.matrix,
            symmetries=["A1"],
        )

    def _remove(self, pristine: Structure, indices):
        return Structure(
            pristine.lattice,
            [s.specie for s in pristine if s not in indices],
            [s.frac_coords for s in pristine if s not in indices],
        )

    def test_vacancy_position(self):
        pristine = _diamond_si_supercell()
        spectrum = self._spectrum(self._remove(pristine, [pristine[5]]))
        pos = detect_defect_position(spectrum)
        assert pos is not None
        np.testing.assert_allclose(pos, pristine[5].frac_coords, atol=1e-6)

    def test_divacancy_position_is_periodic_centroid(self):
        pristine = _diamond_si_supercell()
        remove = [pristine[15], pristine[44]]  # nearest neighbours
        spectrum = self._spectrum(self._remove(pristine, remove))
        pos = detect_defect_position(spectrum)
        assert pos is not None
        # Expected midpoint under the minimum-image convention: the two
        # sites can straddle the cell boundary, so the naive mean is not
        # the periodic midpoint.
        ref = remove[0].frac_coords
        offs = (remove[1].frac_coords - ref + 0.5) % 1.0 - 0.5
        expected = (ref + offs / 2) % 1.0
        np.testing.assert_allclose(pos, expected, atol=1e-6)

    def test_pristine_returns_none(self):
        assert detect_defect_position(self._spectrum(_diamond_si_supercell())) is None

    def test_coords_wrap_into_unit_cell(self):
        pristine = _diamond_si_supercell()
        # Remove a corner atom; its site lies at the boundary but the
        # centroid may need % 1.0 to land in [0, 1).
        pos = detect_defect_position(self._spectrum(self._remove(pristine, [pristine[0]])))
        assert pos is not None
        assert np.all(pos >= 0.0) and np.all(pos < 1.0)

    def test_multiple_defects_raise(self):
        # In a 3x3x3 cell any two vacancies have under-coordinated shells
        # close enough through the periodic boundary to merge into one
        # cluster, so a larger cell is needed for two separate defects.
        pristine = _diamond_si_supercell(4)
        # A pair whose centres are > 3 bond lengths apart, with shells
        # that stay further than the clustering cutoff (2.1 * bond).
        spectrum = self._spectrum(self._remove(pristine, [pristine[2], pristine[40]]))
        with pytest.raises(ValueError, match="found 2"):
            detect_defect_position(spectrum)
