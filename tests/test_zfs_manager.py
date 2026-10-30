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
        assert not np.isclose(V_pm_comp[1, 2] / self.D2_UNIT, 2.5)

        manager.degenerate_pairs_from_symmetry = True
        _, V_00, V_pm, V_0pm = manager.calculate_second_order_derivatives(zfs_1d_derivs)

        assert manager.symmetry_derived_pairs == [(1, 2)]
        for V in (V_pm, V_0pm, V_00):
            assert V[1, 2] == V[2, 1]
        # V_+- = |w_xx| / 4 and V_0+- = |p_xx| / (2 sqrt 2), the Ex-Ex values; V_00 = 0.
        assert np.isclose(V_pm[1, 2] / self.D2_UNIT, 2.5)
        assert np.isclose(V_pm[1, 2], V_pm[1, 1])
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
        assert np.isclose(V_pm[1, 2] / self.D2_UNIT, 0.25 * np.sqrt(diff**2 + 4 * off**2))

    def test_symmetry_check_uses_the_shared_response_check(self, setup_system, caplog):
        manager, _, _ = setup_system
        mhz = CONSTANTS["MHz2J"]  # changes in J are ~1e-26: the check must not depend on the unit
        manager.zfs_relaxed = np.diag([-1000.0, -1000.0, 2000.0]) * mhz
        assert manager.point_group == "3m"  # synthetic structure: implied by the C3v labels
        a1_change = np.diag([-10.0, -10.0, 20.0]) * mhz
        e_change = np.array([[8.0, 0.0, 6.0], [0.0, -8.0, 0.0], [6.0, 0.0, 0.0]]) * mhz
        with caplog.at_level("WARNING", logger="spinlabs"):
            manager._check_symmetry(a1_change, ["A1"], 5, kind="displacement")
            manager._check_symmetry(e_change, ["A1", "Ex"], (5, 6), kind="derivative")
            manager._check_symmetry(a1_change, ["A1", "Ex"], (5, 6), kind="derivative")  # O(q D''') terms
            assert "symmetry mismatch" not in caplog.text
            manager._check_symmetry(a1_change + e_change, ["A1"], 7, kind="displacement")
            manager._check_symmetry(e_change, ["A1", "A1"], (8, 9), kind="derivative")
        assert "symmetry mismatch in run folder 8" in caplog.text
        assert "symmetry mismatch in run folder 9_10: modes A1/A1 allow only A1 in this derivative" in caplog.text

    def test_second_order_content_outside_the_exact_product_is_summarised(self, setup_system, caplog):
        manager, _, _ = setup_system
        # An A1 x Ex curvature that is pure A1: outside A1 x E = E, but within the O(q D''') terms.
        a1_only = np.diag([-1.0, -1.0, 2.0])
        manager.zfs_tensors_2d = {(0, 1): self._second_order_entry(manager, a1_only, ("A1", "Ex"))}
        manager.treated_modes = {0, 1}
        with caplog.at_level("DEBUG", logger="spinlabs"):
            manager.calculate_second_order_derivatives(np.zeros((4, 3, 3)))
        assert "symmetry mismatch" not in caplog.text
        assert "second-order symmetry: 1 of 1 entries hold content outside Gi x Gj" in caplog.text

    def test_ex_ey_pair_and_its_ex_ex_partner_give_the_same_v_pm(self, setup_system):
        """The D_xy of Ex x Ey partners the D_xx - D_yy of Ex x Ex: equal size, equal V_pm."""
        manager, _, _ = setup_system
        manager.zfs_tensors_2d = {
            (1, 1): self._second_order_entry(manager, np.diag([3.0, -3.0, 0.0]), ("Ex", "Ex")),
            (1, 2): self._second_order_entry(manager, np.array([[0.0, 3.0, 0.0], [3.0, 0.0, 0.0], [0.0] * 3]),
                                             ("Ex", "Ey")),
        }
        manager.treated_modes = {1, 2}
        _, _, Vpm, _ = manager.calculate_second_order_derivatives(np.zeros((4, 3, 3)))
        assert np.isclose(Vpm[1, 2], Vpm[1, 1]) and np.isclose(Vpm[1, 1] / self.D2_UNIT, 1.5)

    def test_a_x_e_pair_gives_the_e_couplings_of_its_curvature(self, setup_system):
        """A1 x Ex and A1 x Ey are pure E: V_pm and V_0pm as for Ex x Ex, and no V_00."""
        manager, _, _ = setup_system
        # E parts |w| = sqrt(6^2 + 4*4^2) = 10, |p| = sqrt(3^2 + 4^2) = 5, plus an A1 part that must not count.
        e_x = np.array([[3.0, 0.0, 3.0], [0.0, -3.0, 0.0], [3.0, 0.0, 0.0]])
        e_y = np.array([[0.0, 4.0, 0.0], [4.0, 0.0, 4.0], [0.0, 4.0, 0.0]])
        curvature = e_x + e_y + 0.2 * np.diag([-1.0, -1.0, 2.0])
        manager.zfs_tensors_2d = {
            (0, 1): self._second_order_entry(manager, curvature, ("A1", "Ex")),
            (2, 3): self._second_order_entry(manager, curvature, ("Ey", "A1")),
            (1, 1): self._second_order_entry(manager, curvature, ("Ex", "Ex")),
        }
        manager.treated_modes = {0, 1, 2, 3}

        _, V00, Vpm, V0pm = manager.calculate_second_order_derivatives(np.zeros((4, 3, 3)))

        for i, j in ((0, 1), (2, 3)):
            assert np.isclose(Vpm[i, j] / self.D2_UNIT, 10.0 / 4) and Vpm[j, i] == Vpm[i, j]
            assert np.isclose(V0pm[i, j] / self.D2_UNIT, 5.0 / (2.0 * np.sqrt(2))) and V0pm[j, i] == V0pm[i, j]
            assert V00[i, j] == 0.0
            # the same E couplings an Ex x Ex pair with this curvature gets
            assert np.isclose(Vpm[i, j], Vpm[1, 1]) and np.isclose(V0pm[i, j], V0pm[1, 1])

    def test_twin_inherits_a_diagonal_coupling_only_when_it_has_none(self, setup_system):
        manager, spectrum, gs_zfs = setup_system
        manager.treated_modes = {1, 2}  # fewer than the 4 modes: inheritance is on
        dD_x = np.array([[2.0, 0.0, 1.0], [0.0, -2.0, 0.0], [1.0, 0.0, 0.0]])
        dD_y = np.array([[6.0, 0.0, 3.0], [0.0, -6.0, 0.0], [3.0, 0.0, 0.0]])

        # Both twins ran their own diagonal: each keeps its own coupling.
        manager.zfs_tensors_2d = {
            (1, 1): self._second_order_entry(manager, dD_x, ("Ex", "Ex")),
            (2, 2): self._second_order_entry(manager, dD_y, ("Ey", "Ey")),
        }
        _, _, V_pm, V_0pm = manager.calculate_second_order_derivatives(np.zeros((4, 3, 3)))
        assert np.isclose(V_pm[2, 2], 3 * V_pm[1, 1]) and np.isclose(V_0pm[2, 2], 3 * V_0pm[1, 1])

        # Only Ex ran: Ey inherits it.
        manager.zfs_tensors_2d = {(1, 1): self._second_order_entry(manager, dD_x, ("Ex", "Ex"))}
        _, _, V_pm, V_0pm = manager.calculate_second_order_derivatives(np.zeros((4, 3, 3)))
        assert V_pm[1, 1] > 0 and V_pm[2, 2] == V_pm[1, 1] and V_0pm[2, 2] == V_0pm[1, 1]

    def _offdiag_couplings(self, manager, with_cross: bool, fill: bool):
        """Second-order couplings (V_00, V_pm, V_0pm) of modes A1, Ex, Ey, A1, with or without the fill."""
        a1 = np.diag([-1.0, -1.0, 2.0])
        dD_x = np.array([[2.0, 4.0, 3.0], [4.0, -4.0, 4.0], [3.0, 4.0, 2.0]])
        entries = {
            (0, 0): self._second_order_entry(manager, 2.0 * a1, ("A1", "A1")),
            (1, 1): self._second_order_entry(manager, dD_x, ("Ex", "Ex")),
            (3, 3): self._second_order_entry(manager, 5.0 * a1, ("A1", "A1")),
        }
        if with_cross:  # a run off-diagonal pair
            entries[(0, 3)] = self._second_order_entry(manager, 0.3 * a1, ("A1", "A1"))
        manager.treated_modes = {0, 1, 3}  # mode 2 (Ey) inherits mode 1's diagonal
        manager.zfs_tensors_2d = entries
        manager.offdiag_from_diagonal = fill
        return manager.calculate_second_order_derivatives(np.zeros((4, 3, 3)))[1:]

    def test_offdiag_from_diagonal_fills_symmetry_allowed_uncomputed_pairs(self, setup_system):
        """Uncomputed pairs get sqrt(V_ii V_jj) for the types their symmetry allows; a run pair keeps its value."""
        manager, _, _ = setup_system
        plain = self._offdiag_couplings(manager, with_cross=True, fill=False)
        assert manager.n_offdiag_filled == 0 and plain[1][1, 2] == 0  # no fill without the flag
        V00, Vpm, V0pm = self._offdiag_couplings(manager, with_cross=True, fill=True)
        for V_plain, V in zip(plain, (V00, Vpm, V0pm)):
            np.testing.assert_allclose(np.diag(V), np.diag(V_plain))  # the diagonal is untouched
            assert V[0, 3] == V_plain[0, 3]  # the computed A1 x A1 pair keeps its value
        # Ex x Ey (modes 1, 2): V_pm and V_0pm filled, no V_00
        for V in (Vpm, V0pm):
            assert V[1, 2] == pytest.approx(np.sqrt(V[1, 1] * V[2, 2])) and V[2, 1] == V[1, 2] and V[1, 2] > 0
        assert V00[1, 2] == 0
        # A1 x E pairs (an A1 mode has no diagonal V_pm or V_0pm): the A1 mode's V_00 with the E mode's
        # spin-flip coupling, and no V_00
        for a, e in ((0, 1), (0, 2), (3, 1), (3, 2)):
            for V in (Vpm, V0pm):
                assert V[a, e] == pytest.approx(np.sqrt(V00[a, a] * V[e, e])) and V[e, a] == V[a, e] > 0
            assert V00[a, e] == 0
        assert manager.n_offdiag_filled == 5 and manager.n_offdiag_filled_a_x_e == 4

        # without the run A1 x A1 pair, it is filled with V_00 only
        V00, Vpm, V0pm = self._offdiag_couplings(manager, with_cross=False, fill=True)
        assert V00[0, 3] == pytest.approx(np.sqrt(V00[0, 0] * V00[3, 3])) and V00[0, 3] > 0
        assert Vpm[0, 3] == V0pm[0, 3] == 0
        assert manager.n_offdiag_filled == 6 and manager.n_offdiag_filled_a_x_e == 4

    def test_offdiag_from_diagonal_keeps_a_computed_a_x_e_pair(self, setup_system):
        manager, _, _ = setup_system
        e_curvature = np.array([[1.0, 0.0, 2.0], [0.0, -1.0, 0.0], [2.0, 0.0, 0.0]])
        a1 = np.diag([-1.0, -1.0, 2.0])
        manager.zfs_tensors_2d = {
            (0, 0): self._second_order_entry(manager, 2.0 * a1, ("A1", "A1")),
            (1, 1): self._second_order_entry(manager, 3.0 * e_curvature, ("Ex", "Ex")),
            (0, 1): self._second_order_entry(manager, e_curvature, ("A1", "Ex")),
        }
        manager.treated_modes = {0, 1}
        computed = manager.calculate_second_order_derivatives(np.zeros((4, 3, 3)))[1:]
        manager.offdiag_from_diagonal = True
        filled = manager.calculate_second_order_derivatives(np.zeros((4, 3, 3)))[1:]
        for V_comp, V in zip(computed, filled):
            assert V[0, 1] == V_comp[0, 1]  # the run A1 x Ex pair
        assert filled[1][0, 2] > 0  # the uncomputed A1 x Ey pair (Ey inherits Ex's diagonal) is filled
        # the uncomputed A x E pairs of modes A1, Ex, Ey, A1: (0, 2), (3, 1), (3, 2); mode 3 has no
        # diagonal run, so its two roots are zero, but the fill covers them
        assert manager.n_offdiag_filled_a_x_e == 3

    def test_pair_with_a_mode_without_first_order_data_is_skipped(self, setup_system, caplog):
        manager, spectrum, gs_zfs = setup_system
        dD = np.array([[3.0, 0.0, 1.0], [0.0, -3.0, 0.0], [1.0, 0.0, 0.0]])
        manager.zfs_tensors_2d = {(1, 3): self._second_order_entry(manager, dD, ("Ex", "A1"))}
        manager.treated_modes = {1}  # mode 3 has no first-order run
        with caplog.at_level("WARNING", logger="spinlabs"):
            d2D, V_00, V_pm, V_0pm = manager.calculate_second_order_derivatives(np.zeros((4, 3, 3)))
        assert not np.any(d2D[1, 3]) and V_pm[1, 3] == 0 and V_0pm[1, 3] == 0 and V_00[1, 3] == 0
        assert "1 second-order entries skipped: a mode has no first-order derivative" in caplog.text
        assert "2_4" in caplog.text

    def test_raw_data_in_mhz_is_ingested_in_joules(self, setup_system):
        """Simulation folders are parsed in MHz; the manager must work in J like for loaded npz files."""
        from spinlabs.models import PerturbationEntry

        _, spectrum, _ = setup_system
        gs_mhz = np.diag([-1000.0, -1000.0, 2000.0])
        d_mhz = gs_mhz + np.array([[1.0, 0.0, 2.0], [0.0, -1.0, 0.0], [2.0, 0.0, 0.0]])

        def raw(unit):
            scale = CONSTANTS["MHz2J"] if unit == "J" else 1.0
            entry = PerturbationEntry(order=1, mode_indices=(1,), amplitude=0.01,
                                      zfs_tensor=ZFSTensor(matrix=d_mhz * scale, unit=unit))
            return RawZFSData(defect="NV", cell_size=64, pert_scale=0.01,
                              ground_state_zfs=ZFSTensor(matrix=gs_mhz * scale, unit=unit), first_order={1: entry})

        from_mhz = ZFSManager(spectrum=spectrum, raw_data=raw("MHz"))
        from_j = ZFSManager(spectrum=spectrum, raw_data=raw("J"))
        assert np.allclose(from_mhz.zfs_relaxed, gs_mhz * CONSTANTS["MHz2J"])
        assert np.allclose(from_mhz.zfs_relaxed, from_j.zfs_relaxed)
        assert np.allclose(from_mhz.first_order[1]["tensor"], from_j.first_order[1]["tensor"])

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
