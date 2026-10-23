import pytest
import numpy as np
from spinlabs.models import PhononSpectrum
from spinlabs.utils import MathUtils


class TestPhononSpectrumProperties:
    @pytest.fixture
    def sample_spectrum(self):
        n_modes = 6
        n_atoms = 4
        # Create normalized eigenvectors: sum_i (e_i)^2 == 1 for each mode
        raw_eigs = np.random.randn(n_modes, n_atoms, 3)
        norms = np.linalg.norm(raw_eigs.reshape(n_modes, -1), axis=1)[:, None, None]
        normalized_eigs = raw_eigs / norms

        return PhononSpectrum(
            frequencies_mev=np.array([10.0, 20.0, 20.0, 40.0, 50.0, 50.0]),
            eigenvectors=normalized_eigs,
            atom_frac_coords=np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]]),
            atom_symbols=["N", "C", "C", "C"],
            atomic_masses=np.array([14.007, 12.011, 12.011, 12.011]),
            lattice=np.eye(3) * 3.56,
            symmetries=["A1", "Ex", "Ey", "A1", "Ex", "Ey"],
            iprs=np.full(n_modes, 0.5),
        )

    def test_spectrum_properties(self, sample_spectrum):
        assert sample_spectrum.n_modes == 6
        assert sample_spectrum.n_atoms == 4
        assert len(sample_spectrum.frequencies_mev) == 6

    def test_translate_defect_to_origin(self, sample_spectrum):
        shifted_frac, defect_pos = sample_spectrum.translate_defect_to_origin()
        # N is at index 0, so shifted position of atom 0 should be at origin (0, 0, 0)
        assert np.allclose(shifted_frac[0], [0.0, 0.0, 0.0])

    def test_get_phonon_pert(self, sample_spectrum):
        pert = sample_spectrum.get_phonon_pert(perturbation_scale=0.01)
        assert "sym" in pert
        assert "eigs" in pert
        assert "freqs" in pert
        assert len(pert["freqs"]) == 6


class TestIPRCalculation:
    """Dedicated tests for Inverse Participation Ratio (IPR) and locality calculations."""

    def test_single_atom_localized_mode_ipr_is_one(self):
        """A mode vibrating exclusively on 1 atom should have IPR = 1.0."""
        n_atoms = 10
        # Mode localized strictly on atom index 0
        eigs = np.zeros((1, n_atoms, 3))
        eigs[0, 0, 2] = 1.0  # z-vibration on atom 0

        ipr = MathUtils.calc_ipr(eigs)
        assert len(ipr) == 1
        assert np.isclose(ipr[0], 1.0)

    def test_uniformly_delocalized_mode_ipr_is_one_over_n(self):
        """A mode uniformly spread over all N atoms should have IPR = 1 / N."""
        n_atoms = 64
        # Equal vibration amplitude on all atoms
        eigs = np.zeros((1, n_atoms, 3))
        eigs[0, :, 0] = 1.0 / np.sqrt(n_atoms)

        ipr = MathUtils.calc_ipr(eigs)
        assert len(ipr) == 1
        assert np.isclose(ipr[0], 1.0 / n_atoms)

    def test_two_atom_localized_mode_ipr(self):
        """A mode equally shared between 2 atoms out of N should have IPR = 1/2 = 0.5."""
        n_atoms = 32
        eigs = np.zeros((1, n_atoms, 3))
        eigs[0, 0, 0] = 1.0 / np.sqrt(2)
        eigs[0, 1, 0] = 1.0 / np.sqrt(2)

        ipr = MathUtils.calc_ipr(eigs)
        assert np.isclose(ipr[0], 0.5)

    def test_ipr_via_spectrum(self):
        """Verifies that PhononSpectrum.calc_ipr() computes and stores IPR directly on PhononSpectrum."""
        n_modes = 4
        n_atoms = 8
        eigs = np.zeros((n_modes, n_atoms, 3))

        # Mode 0: localized on 1 atom -> IPR = 1.0
        eigs[0, 0, 0] = 1.0
        # Mode 1: localized on 2 atoms -> IPR = 0.5
        eigs[1, 0, 0] = 1.0 / np.sqrt(2)
        eigs[1, 1, 0] = 1.0 / np.sqrt(2)
        # Mode 2: localized on 4 atoms -> IPR = 0.25
        eigs[2, :4, 0] = 1.0 / np.sqrt(4)
        # Mode 3: uniformly delocalized over 8 atoms -> IPR = 1/8 = 0.125
        eigs[3, :, 0] = 1.0 / np.sqrt(8)

        spectrum = PhononSpectrum(
            frequencies_mev=np.array([15.0, 25.0, 35.0, 45.0]),
            eigenvectors=eigs,
            atom_frac_coords=np.zeros((n_atoms, 3)),
            atom_symbols=["C"] * n_atoms,
            atomic_masses=np.full(n_atoms, 12.011),
            lattice=np.eye(3) * 5.0,
            iprs=None,  # not computed yet
        )

        assert spectrum.iprs is None

        computed_iprs = spectrum.calc_ipr()
        assert np.allclose(computed_iprs, [1.0, 0.5, 0.25, 0.125])
        assert spectrum.iprs is not None
        assert np.allclose(spectrum.get_ipr(), [1.0, 0.5, 0.25, 0.125])


class TestC3vSymmetryClassification:
    """Verifies that PhononSpectrum automatically computes C3v symmetry representations."""

    def test_symmetry_computed_automatically_if_none(self):
        n_modes = 4
        n_atoms = 4
        spectrum = PhononSpectrum(
            frequencies_mev=np.array([10.0, 20.0, 30.0, 40.0]),
            eigenvectors=np.ones((n_modes, n_atoms, 3)),
            atom_frac_coords=np.zeros((n_atoms, 3)),
            atom_symbols=["C"] * n_atoms,
            atomic_masses=np.full(n_atoms, 12.011),
            lattice=np.eye(3) * 5.0,
            symmetries=None,
        )
        assert spectrum.symmetries is not None
        assert len(spectrum.symmetries) == n_modes
        assert all(isinstance(s, str) for s in spectrum.symmetries)

    def test_c3v_symmetry_classification_a1_and_e_modes(self):
        # 4-atom cluster with C3v: N at origin, 3 C atoms at 120 deg
        axis = np.array([1.0, 1.0, 1.0])
        normal = np.array([1.0, -1.0, 0.0])
        R_C3 = MathUtils.rotation_around_symmetry_axis(axis, order=3)

        v1 = np.array([1.0, 1.0, -2.0])
        v1 = v1 / np.linalg.norm(v1) * 1.54
        pos = np.zeros((4, 3))
        pos[0] = [0.0, 0.0, 0.0]  # N
        pos[1] = v1  # C1
        pos[2] = R_C3 @ v1  # C2
        pos[3] = R_C3 @ pos[2]  # C3

        lattice = np.eye(3) * 10.0
        frac_coords = pos / 10.0
        symbols = ["N", "C", "C", "C"]
        masses = np.array([14.0, 12.0, 12.0, 12.0])

        # Mode 0: A1 mode along C3 axis
        eig_A1 = np.zeros((4, 3))
        eig_A1[0] = axis / np.linalg.norm(axis)
        eig_A1 /= np.linalg.norm(eig_A1)

        # Pure Ex (in mirror plane) and Ey (normal to mirror plane)
        eig_Ex_pure = np.zeros((4, 3))
        eig_Ex_pure[0] = v1 / np.linalg.norm(v1)
        eig_Ex_pure /= np.linalg.norm(eig_Ex_pure)

        eig_Ey_pure = np.zeros((4, 3))
        eig_Ey_pure[0] = normal / np.linalg.norm(normal)
        eig_Ey_pure /= np.linalg.norm(eig_Ey_pure)

        # Degenerate pair (arbitrary rotation in E subspace, e.g. theta = 0.55 rad)
        theta = 0.55
        e1 = np.cos(theta) * eig_Ex_pure + np.sin(theta) * eig_Ey_pure
        e2 = -np.sin(theta) * eig_Ex_pure + np.cos(theta) * eig_Ey_pure

        eigs = np.array([eig_A1, e1, e2])
        freqs = np.array([15.0, 30.0, 30.0])

        spectrum = PhononSpectrum(
            frequencies_mev=freqs,
            eigenvectors=eigs,
            atom_frac_coords=frac_coords,
            atom_symbols=symbols,
            atomic_masses=masses,
            lattice=lattice,
            symmetries=None,  # triggers automatic C3v classification
        )

        # Ex/Ey ordering within the degenerate pair is convention-dependent
        # (depends on which sigma_v plane the alignment picks); both members
        # must be present, A1 first.
        assert spectrum.symmetries[0] == "A1"
        assert set(spectrum.symmetries[1:]) == {"Ex", "Ey"}

    def test_locality_weight(self):
        """Verifies spatial defect-neighbourhood locality calculation."""
        n_atoms = 4
        lattice = np.eye(3) * 10.0  # 10x10x10 A supercell
        # Atom 0 at origin (defect), Atom 1 at 1.0 A, Atom 2 at 1.5 A, Atom 3 at 8.0 A
        frac_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],  # 1.0 A
                [0.15, 0.0, 0.0],  # 1.5 A
                [0.8, 0.0, 0.0],  # 8.0 A (or 2.0 A with PBC)
            ]
        )

        # Mode with vibration on atom 0 and atom 1 (both within radius 1.2 A)
        eigs = np.zeros((1, n_atoms, 3))
        eigs[0, 0, 0] = 1.0 / np.sqrt(2)
        eigs[0, 1, 0] = 1.0 / np.sqrt(2)

        lw = MathUtils.calc_locality_weight(eigs, frac_coords, lattice, radius=1.2)
        assert np.isclose(lw[0], 1.0)  # 100% inside radius

        # Mode with vibration on atom 3 (outside radius 1.2 A)
        eigs_outside = np.zeros((1, n_atoms, 3))
        eigs_outside[0, 3, 0] = 1.0

        lw_outside = MathUtils.calc_locality_weight(eigs_outside, frac_coords, lattice, radius=1.2)
        assert np.isclose(lw_outside[0], 0.0)
