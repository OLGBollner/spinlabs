import numpy as np
import pytest

from spinlabs.constants import CONSTANTS
from spinlabs.models import PhononSpectrum, RawZFSData, ZFSTensor
from spinlabs.zfs_manager import ZFSManager


class TestZFSManager:
    @pytest.fixture
    def setup_system(self):
        n_modes = 4
        n_atoms = 2
        spectrum = PhononSpectrum(
            frequencies_mev=np.array([10.0, 20.0, 20.0, 30.0]),
            eigenvectors=np.random.randn(n_modes, n_atoms, 3),
            atom_frac_coords=np.zeros((n_atoms, 3)),
            atom_symbols=["C", "C"],
            atomic_masses=np.array([12.011, 12.011]),
            lattice=np.eye(3) * 3.56,
            symmetries=["A1", "Ex", "Ey", "A1"],
            iprs=np.ones(n_modes) * 0.5,
        )

        relaxed_mat = np.diag([-2870.0 / 3, -2870.0 / 3, 2 * 2870.0 / 3])
        ground_state_zfs = ZFSTensor(matrix=relaxed_mat, unit="MHz")

        raw_data = RawZFSData(
            defect="NV",
            cell_size=64,
            pert_scale=0.01,
            ground_state_zfs=ground_state_zfs,
        )

        manager = ZFSManager(spectrum=spectrum, raw_data=raw_data)
        return manager, spectrum, ground_state_zfs

    def test_first_order_derivative_calculation(self, setup_system):
        manager, spectrum, gs_zfs = setup_system
        pert_SI = 0.01 * CONSTANTS["ang_amu2SI"]

        # Synthesize perturbation for A1 mode (idx 0) and Ex mode (idx 1)
        # A1: modulates D_zz
        # Ex: modulates D_xx - D_yy and D_xz
        d_a1 = gs_zfs.matrix.copy()
        d_a1[2, 2] += 50.0  # +50 MHz shift in zz

        d_ex = gs_zfs.matrix.copy()
        d_ex[0, 2] += 20.0
        d_ex[2, 0] += 20.0

        manager.zfs_tensors = {
            0: {"tensor": d_a1 * CONSTANTS["MHz2J"], "pert": pert_SI, "symmetry": "A1", "ipr": 0.5},
            1: {"tensor": d_ex * CONSTANTS["MHz2J"], "pert": pert_SI, "symmetry": "Ex", "ipr": 0.5},
        }
        manager.treated_modes = {0, 1}

        derivs, V_00, V_pm, V_0pm = manager.calculate_first_order_derivatives()

        # Check that V_00 is non-zero for A1 and V_0pm is non-zero for Ex
        assert V_00[0] > 0
        assert V_0pm[1] > 0
        assert np.isclose(V_pm[0], 0.0)

    def test_second_order_derivative_calculation(self, setup_system):
        manager, spectrum, gs_zfs = setup_system
        pert_SI = 0.01 * CONSTANTS["ang_amu2SI"]

        # Setup 1D derivatives first
        zfs_1d_derivs = np.zeros((4, 3, 3))
        zfs_1d_derivs[0, 2, 2] = 50.0 * CONSTANTS["MHz2J"] / pert_SI

        # Synthesize 2D diagonal perturbation (0, 0)
        d_2d = gs_zfs.matrix.copy()
        d_2d[2, 2] += 120.0

        manager.zfs_tensors_2d = {
            (0, 0): {
                "tensor": d_2d * CONSTANTS["MHz2J"],
                "pert": (pert_SI, pert_SI),
                "symmetry": ("A1", "A1"),
                "ipr": (0.5, 0.5),
            }
        }
        manager.treated_modes = {0}
        # The synthetic tensors are not a consistent first/second-order pair (the
        # linear-response check would exclude them); this test checks the formula.
        manager.exclude_inconsistent_pairs = False

        derivs_2d, V_00_2nd, V_pm_2nd, V_0pm_2nd = manager.calculate_second_order_derivatives(zfs_1d_derivs)

        assert V_00_2nd[0, 0] > 0
        assert np.isclose(V_00_2nd[0, 0], V_00_2nd[0, 0])

    PERT_SI = 0.01 * CONSTANTS["ang_amu2SI"]
    # Curvature unit: a ZFS change of 1 MHz at the displacement (q, q).
    D2_UNIT = CONSTANTS["MHz2J"] / PERT_SI**2

    @classmethod
    def _second_order_entry(cls, manager, dD_mhz, sym):
        # With zero first-order derivatives the entry's curvature is (D - D0) / q^2.
        manager.zfs_relaxed = np.diag([-2870.0 / 3, -2870.0 / 3, 2 * 2870.0 / 3]) * CONSTANTS["MHz2J"]
        tensor = manager.zfs_relaxed + dD_mhz * CONSTANTS["MHz2J"]
        return {"tensor": tensor, "pert": (cls.PERT_SI, cls.PERT_SI), "symmetry": sym, "ipr": (0.5, 0.5)}

    def test_degenerate_pairs_from_symmetry_replaces_ex_ey_coupling(self, setup_system):
        manager, spectrum, gs_zfs = setup_system
        # Ex-Ex change (MHz): E parts |w| = sqrt(6^2 + 4*4^2) = 10, |p| = sqrt(3^2 + 4^2) = 5.
        dD_xx = np.array([[2.0, 4.0, 3.0], [4.0, -4.0, 4.0], [3.0, 4.0, 2.0]])
        # A computed Ex-Ey entry that disagrees with symmetry (noisy run).
        dD_xy = np.array([[1.0, 0.5, 0.2], [0.5, -1.0, 0.1], [0.2, 0.1, 7.0]])
        manager.zfs_tensors_2d = {
            (1, 1): self._second_order_entry(manager, dD_xx, ("Ex", "Ex")),
            (1, 2): self._second_order_entry(manager, dD_xy, ("Ex", "Ey")),
        }
        manager.treated_modes = {1, 2}
        zfs_1d_derivs = np.zeros((4, 3, 3))

        _, V_00_comp, V_pm_comp, V_0pm_comp = manager.calculate_second_order_derivatives(zfs_1d_derivs)
        assert manager.symmetry_derived_pairs == []
        assert not np.isclose(V_pm_comp[1, 2] / self.D2_UNIT, 5.0)

        manager.degenerate_pairs_from_symmetry = True
        _, V_00, V_pm, V_0pm = manager.calculate_second_order_derivatives(zfs_1d_derivs)

        assert manager.symmetry_derived_pairs == [(1, 2)]
        for V in (V_pm, V_0pm, V_00):
            assert V[1, 2] == V[2, 1]
        # V_+- = 0.5 |w_xx| (twice the Ex-Ex value), V_0+- = |p_xx| / (2 sqrt 2), V_00 = 0.
        assert np.isclose(V_pm[1, 2] / self.D2_UNIT, 5.0)
        assert np.isclose(V_pm[1, 2], 2.0 * V_pm[1, 1])
        assert np.isclose(V_0pm[1, 2] / self.D2_UNIT, 5.0 / (2.0 * np.sqrt(2)))
        assert np.isclose(V_0pm[1, 2], V_0pm[1, 1])
        assert V_00[1, 2] == 0.0
        # Everything but the Ex-Ey pair is unchanged.
        mask = np.ones((4, 4), dtype=bool)
        mask[1, 2] = mask[2, 1] = False
        for V, V_comp in ((V_00, V_00_comp), (V_pm, V_pm_comp), (V_0pm, V_0pm_comp)):
            assert np.array_equal(V[mask], V_comp[mask])

    def test_degenerate_pairs_from_symmetry_keeps_pairs_without_ex_ex(self, setup_system):
        manager, spectrum, gs_zfs = setup_system
        dD_xy = np.array([[1.0, 0.5, 0.2], [0.5, -1.0, 0.1], [0.2, 0.1, 7.0]])
        manager.zfs_tensors_2d = {(1, 2): self._second_order_entry(manager, dD_xy, ("Ex", "Ey"))}
        manager.treated_modes = {1, 2}
        manager.degenerate_pairs_from_symmetry = True

        _, _, V_pm, V_0pm = manager.calculate_second_order_derivatives(np.zeros((4, 3, 3)))

        assert manager.symmetry_derived_pairs == []
        diff, off = dD_xy[0, 0] - dD_xy[1, 1], dD_xy[0, 1]
        assert np.isclose(V_pm[1, 2] / self.D2_UNIT, 0.5 * np.sqrt(diff**2 + 4 * off**2))

    def test_init_from_raw_data_with_legacy_scalar_pert_updates_to_si_displacements(self, setup_system):
        manager_base, spectrum, gs_zfs = setup_system

        # Raw data containing a static scalar pert (e.g. 0.025) like saved from legacy code
        raw_data = RawZFSData(
            defect="NV",
            cell_size=64,
            pert_scale=0.025,
            calc_method="all_bands",
            ground_state_zfs=gs_zfs,
            first_order={
                0: {"tensor": np.eye(3) * 1e-24, "unit": "J", "pert": 0.025},
                3: {"tensor": np.eye(3) * 1e-24, "unit": "J", "pert": 0.025},
            },
            second_order={
                (0, 3): {"tensor": np.eye(3) * 1e-24, "unit": "J", "pert": (0.025, 0.025)},
            },
        )

        manager = ZFSManager(spectrum=spectrum, raw_data=raw_data)

        # Displacements must be updated from spectrum, not remaining 0.025
        pert_0 = manager.first_order[0]["pert"]
        pert_3 = manager.first_order[3]["pert"]
        assert pert_0 != 0.025
        assert pert_3 != 0.025

        # Displacements must vary with frequency (mode 0 is 10 meV, mode 3 is 30 meV)
        assert pert_0 != pert_3
        pert_SI = 0.025 * CONSTANTS["ang_amu2SI"]
        expected_pert = spectrum.get_phonon_pert(pert_SI)
        assert np.isclose(pert_0, expected_pert["disp"][0])
        assert np.isclose(pert_3, expected_pert["disp"][3])

        # Second order perturbations must also be updated
        pert_2d = manager.second_order[(0, 3)]["pert"]
        assert pert_2d != (0.025, 0.025)
        assert np.isclose(pert_2d[0], expected_pert["disp"][0])
        assert np.isclose(pert_2d[1], expected_pert["disp"][3])

    def test_save_raw_zfs_data_with_spectrum_enriches_si_perturbations(self, setup_system, tmp_path):
        _, spectrum, gs_zfs = setup_system

        raw_data = RawZFSData(
            defect="NV",
            cell_size=64,
            pert_scale=0.025,
            calc_method="all_bands",
            ground_state_zfs=gs_zfs,
            first_order={
                0: {"tensor": np.eye(3) * 1e-24, "unit": "J", "pert": 0.025},
                3: {"tensor": np.eye(3) * 1e-24, "unit": "J", "pert": 0.025},
            },
            second_order={
                (0, 3): {"tensor": np.eye(3) * 1e-24, "unit": "J", "pert": (0.025, 0.025)},
            },
        )

        out_path = tmp_path / "raw_enriched.npz"
        raw_data.save(out_path, spectrum=spectrum)

        reloaded = RawZFSData.load(out_path)
        pert_SI = 0.025 * CONSTANTS["ang_amu2SI"]
        expected_pert = spectrum.get_phonon_pert(pert_SI)

        # 1D displacements are SI and mode-dependent
        assert np.isclose(reloaded.first_order[0]["pert"], expected_pert["disp"][0])
        assert np.isclose(reloaded.first_order[3]["pert"], expected_pert["disp"][3])
        assert reloaded.first_order[0]["pert"] != reloaded.first_order[3]["pert"]

        # 2D displacements are SI and mode-dependent tuple
        assert np.isclose(reloaded.second_order[(0, 3)]["pert"][0], expected_pert["disp"][0])
        assert np.isclose(reloaded.second_order[(0, 3)]["pert"][1], expected_pert["disp"][3])

    def test_ingest_rotates_tensors_into_principal_eigenframe(self, setup_system):
        """Regression test: loaded .npz (dict) tensors must be rotated into the
        ground-state principal eigenframe with R.T @ tensor @ R so that A1,A1
        modes produce a diagonal d2D. Using R @ tensor @ R.T (the wrong
        orientation) leaves off-diagonal terms at the same scale as the
        diagonal, which is what Oskar observed for mode (187,187)."""
        _, spectrum, gs_zfs = setup_system

        # Ground state with a non-trivial eigenframe: NV axis along [1,1,1]
        # gives an off-diagonal crystal-frame matrix (like the real NV_64 data).
        R = np.array(
            [
                [0.74867142, 0.32581862, -0.57735027],
                [-0.0921685, -0.81127778, -0.57735027],
                [-0.65650291, 0.48545915, -0.57735027],
            ]
        )
        D_gs_diag = np.diag([-2870.0 / 3, -2870.0 / 3, 2 * 2870.0 / 3])  # MHz
        gs_crystal = R @ D_gs_diag @ R.T  # MHz
        ground_state_zfs = ZFSTensor(matrix=gs_crystal, unit="MHz")

        # Perturbed tensor is diagonal *in the eigenframe* (A1,A1 mode should be
        # diagonal in dD). In the crystal frame it is R @ D_pert @ R.T.
        D_pert_diag = np.diag([-2870.0 / 3 + 35.0, -2870.0 / 3 + 35.0, 2 * 2870.0 / 3 - 70.0])
        T_crystal = R @ D_pert_diag @ R.T  # MHz

        raw_data = RawZFSData(
            defect="NV",
            cell_size=64,
            pert_scale=0.025,
            calc_method="all_bands",
            ground_state_zfs=ground_state_zfs,
            first_order={
                0: {"tensor": T_crystal * CONSTANTS["MHz2J"], "unit": "J", "pert": 0.025},
            },
            second_order={
                (0, 0): {"tensor": T_crystal * CONSTANTS["MHz2J"], "unit": "J", "pert": (0.025, 0.025)},
            },
        )

        manager = ZFSManager(spectrum=spectrum, raw_data=raw_data)

        # The ingested tensor must equal the eigenframe-diagonal tensor:
        # R.T @ T_crystal @ R == D_pert_diag.
        ingested = manager.second_order[(0, 0)]["tensor"]
        expected = D_pert_diag * CONSTANTS["MHz2J"]
        assert np.allclose(ingested, expected, atol=1e-24), (
            f"Tensor not rotated into eigenframe; off-diagonals remain.\ningested: {ingested}\nexpected: {expected}"
        )

        # Sanity: the wrong orientation R @ T @ R.T would NOT match.
        T_j = T_crystal * CONSTANTS["MHz2J"]
        wrong = R @ T_j @ R.T
        assert not np.allclose(wrong, expected, atol=1e-24)
