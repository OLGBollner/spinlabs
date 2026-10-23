import numpy as np
import tempfile
from pathlib import Path

from spinlabs.parsers import save_phonon_npz, parse_phonon_npz
from spinlabs.models import ZFSTensor, PhononSpectrum, RawZFSData
from spinlabs.zfs_manager import ZFSManager
from spinlabs.constants import CONSTANTS


class TestFullIntegration:
    def test_full_pipeline_workflow(self):
        # 1. Create and serialize phonon spectrum
        n_modes = 9
        n_atoms = 4
        orig_spectrum = PhononSpectrum(
            # Degenerate Ex/Ey pairs (0.01 meV apart) so e-pair completeness passes
            frequencies_mev=np.array([10.0, 20.0, 20.01, 30.0, 40.0, 40.01, 50.0, 60.0, 60.01]),
            eigenvectors=np.random.randn(n_modes, n_atoms, 3),
            atom_frac_coords=np.zeros((n_atoms, 3)),
            atom_symbols=["C"] * n_atoms,
            atomic_masses=np.full(n_atoms, 12.011),
            lattice=np.eye(3) * 3.56,
            symmetries=["A1", "Ex", "Ey", "A1", "Ex", "Ey", "A1", "Ex", "Ey"],
            iprs=np.ones(n_modes) * 0.4,
        )

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp_ph:
            tmp_ph_path = Path(tmp_ph.name)

        try:
            save_phonon_npz(orig_spectrum, tmp_ph_path)
            spectrum = parse_phonon_npz(tmp_ph_path)
            assert spectrum.n_modes == n_modes

            # 2. Setup Ground State ZFSTensor (axial NV-like)
            gs_mat = np.diag([-2870.0 / 3, -2870.0 / 3, 2 * 2870.0 / 3])
            gs_zfs = ZFSTensor(matrix=gs_mat, unit="MHz")

            # 3. Create RawZFSData container
            raw_data = RawZFSData(
                defect="NV",
                cell_size=64,
                pert_scale=0.01,
                ground_state_zfs=gs_zfs,
            )

            # 4. Instantiate ZFSManager
            manager = ZFSManager(spectrum=spectrum, raw_data=raw_data)
            pert_SI = 0.01 * CONSTANTS["ang_amu2SI"]

            # 5. Populate perturbations
            manager.zfs_tensors = {
                0: {
                    "tensor": (gs_mat + np.diag([0, 0, 40])) * CONSTANTS["MHz2J"],
                    "pert": pert_SI,
                    "symmetry": "A1",
                    "ipr": 0.4,
                },
                1: {
                    "tensor": (gs_mat + np.array([[0, 0, 15], [0, 0, 0], [15, 0, 0]])) * CONSTANTS["MHz2J"],
                    "pert": pert_SI,
                    "symmetry": "Ex",
                    "ipr": 0.4,
                },
                2: {
                    "tensor": (gs_mat + np.array([[0, 15, 0], [15, 0, 0], [0, 0, 0]])) * CONSTANTS["MHz2J"],
                    "pert": pert_SI,
                    "symmetry": "Ey",
                    "ipr": 0.4,
                },
            }
            manager.treated_modes = {0, 1, 2}

            # 6. Calculate 1st order derivatives & coupling coefficients
            derivs_1d, V_00, V_pm, V_0pm = manager.calculate_first_order_derivatives()

            assert V_00[0] > 0
            assert V_0pm[1] > 0
            assert V_pm[2] > 0

            # 7. Save and verify derivatives archive
            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp_out:
                tmp_out_path = Path(tmp_out.name)

            try:
                manager.save_data(
                    str(tmp_out_path),
                    zfs_derivs=derivs_1d,
                    V_0_0=V_00,
                    V_p_m=V_pm,
                    V_0_pm=V_0pm,
                )
                loaded_out = np.load(tmp_out_path)
                assert "V_0_0" in loaded_out
                assert np.allclose(loaded_out["V_0_0"], V_00)
            finally:
                if tmp_out_path.exists():
                    tmp_out_path.unlink()

        finally:
            if tmp_ph_path.exists():
                tmp_ph_path.unlink()
