import tempfile
from pathlib import Path

import numpy as np
import copy

import pytest

from spinlabs.models import PhononSpectrum, RawZFSData, ZFSTensor
from spinlabs.parsers import (
    parse_outcar_energy,
    parse_outcar_zfs,
    parse_phonon_npz,
    parse_zfs_dataset_npz,
    parse_zfs_simulation_dataset,
    save_phonon_npz,
    split_defect_folder_name,
)


class TestParsers:
    def test_parse_real_outcar_if_exists(self):
        outcar_path = Path("NV_64/OUTCAR")
        if outcar_path.is_file():
            zfs = parse_outcar_zfs(outcar_path)
            assert isinstance(zfs, ZFSTensor)
            assert zfs.matrix.shape == (3, 3)
            assert np.isclose(zfs.D, 3143.464, atol=1.0)

            energy = parse_outcar_energy(outcar_path)
            assert energy is not None
            assert isinstance(energy, float)

    def test_parse_outcar_nonexistent_returns_none(self):
        assert parse_outcar_zfs("nonexistent_path/OUTCAR") is None
        assert parse_outcar_energy("nonexistent_path/OUTCAR") is None

    def test_phonon_npz_roundtrip(self):
        n_modes = 4
        n_atoms = 2
        spectrum = PhononSpectrum(
            frequencies_mev=np.array([15.0, 25.0, 35.0, 45.0]),
            eigenvectors=np.random.randn(n_modes, n_atoms, 3),
            atom_frac_coords=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            atom_symbols=["Si", "C"],
            atomic_masses=np.array([28.085, 12.011]),
            lattice=np.eye(3) * 4.36,
            symmetries=["A1", "Ex", "Ey", "A1"],
            iprs=np.array([0.5, 0.5, 0.5, 0.8]),
        )

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_phonon_npz(spectrum, tmp_path)
            loaded = parse_phonon_npz(tmp_path)

            assert loaded.n_modes == n_modes
            assert loaded.n_atoms == n_atoms
            assert np.allclose(loaded.frequencies_mev, spectrum.frequencies_mev)
            assert np.allclose(loaded.eigenvectors, spectrum.eigenvectors)
            assert loaded.atom_symbols == spectrum.atom_symbols
            assert loaded.symmetries == spectrum.symmetries
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_raw_zfs_npz_roundtrip_units_in_joule(self):
        """Verifies that saved npz tensors are in Joules and correctly converted."""
        from spinlabs.constants import CONSTANTS

        # Create dummy 1D and 2D npz archives matching ZFSManager.save_data
        gs_mhz = np.diag([-1000.0, -1000.0, 2000.0])
        gs_joule = gs_mhz * CONSTANTS["MHz2J"]
        eigen_rot = np.eye(3)

        tensors_1d = {
            0: {
                "tensor": gs_joule + np.diag([0, 0, 10.0]) * CONSTANTS["MHz2J"],
                "pert": 1e-12,
                "symmetry": "A1",
                "ipr": 0.5,
            }
        }
        tensors_2d = {
            (0, 0): {
                "tensor": gs_joule + np.diag([0, 0, 20.0]) * CONSTANTS["MHz2J"],
                "pert": (1e-12, 1e-12),
                "symmetry": ("A1", "A1"),
                "ipr": (0.5, 0.5),
            }
        }

        with (
            tempfile.NamedTemporaryFile(suffix="_1d.npz", delete=False) as f1,
            tempfile.NamedTemporaryFile(suffix="_2d.npz", delete=False) as f2,
        ):
            path_1d = Path(f1.name)
            path_2d = Path(f2.name)

        try:
            np.savez(
                str(path_1d),
                order=1,
                defect="NV",
                cell_size=64,
                pert_scale=0.025,
                calc_method="all_bands",
                zfs_relaxed=gs_joule,
                eigen_rotation=eigen_rot,
                zfs_tensors=tensors_1d,
            )
            np.savez(
                str(path_2d),
                order=2,
                defect="NV",
                cell_size=64,
                pert_scale=0.025,
                calc_method="all_bands",
                zfs_relaxed=gs_joule,
                eigen_rotation=eigen_rot,
                zfs_tensors_2d=tensors_2d,
            )

            raw = parse_zfs_dataset_npz([path_1d, path_2d])
            assert raw.defect == "NV"
            # As loaded from legacy file, stored unit is Joules
            assert raw.ground_state_zfs.unit == "J"
            assert np.allclose(raw.ground_state_zfs.matrix, gs_joule)

            # Converting via dataclass internal converter
            raw_mhz = raw.to_unit("MHz")
            assert raw_mhz.ground_state_zfs.unit == "MHz"
            assert np.allclose(raw_mhz.ground_state_zfs.matrix, gs_mhz)

            # The tensor in first_order dict is preserved and converted
            assert np.allclose(raw.first_order[0]["tensor"], tensors_1d[0]["tensor"])
            assert np.allclose(raw_mhz.first_order[0]["tensor"], tensors_1d[0]["tensor"] / CONSTANTS["MHz2J"])
        finally:
            if path_1d.exists():
                path_1d.unlink()
            if path_2d.exists():
                path_2d.unlink()

    def test_raw_zfs_saves_with_latest_naming_convention(self):
        """Verifies that saving RawZFSData uses latest keys (first_order/second_order), not legacy keys."""
        from spinlabs.models import RawZFSData, ZFSTensor

        gs = ZFSTensor(matrix=np.diag([-1000.0, -1000.0, 2000.0]), unit="MHz")
        first = {0: {"tensor": np.eye(3), "pert": 0.01}}
        second = {(0, 0): {"tensor": np.eye(3), "pert": (0.01, 0.01)}}

        raw = RawZFSData(
            defect="NV",
            cell_size=64,
            pert_scale=0.025,
            calc_method="all_bands",
            order=2,
            ground_state_zfs=gs,
            first_order=first,
            second_order=second,
        )

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            save_path = Path(f.name)

        try:
            raw.save(save_path)

            # Inspect raw keys in the npz file
            with np.load(str(save_path), allow_pickle=True) as npz:
                keys = set(npz.files)
                # Must use latest naming conventions
                assert "first_order" in keys
                assert "second_order" in keys
                assert "ground_state_zfs_matrix" in keys
                assert "ground_state_zfs_unit" in keys

                # Must NOT contain duplicate legacy keys when saving new files
                assert "zfs_tensors" not in keys
                assert "zfs_tensors_2d" not in keys
                assert "zfs_relaxed" not in keys

            # Re-load to verify full roundtrip (load() converts to Joules)
            reloaded = RawZFSData.load(save_path)
            assert reloaded.defect == "NV"
            assert reloaded.ground_state_zfs.unit in ("J", "MHz")
            assert 0 in reloaded.first_order
            assert (0, 0) in reloaded.second_order
        finally:
            if save_path.exists():
                save_path.unlink()

    @staticmethod
    def _create_mock_outcar(path: Path, d_diag=(-950.0, -950.0, 1900.0), energy=-500.12345):
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"""
 free  energy   TOTEN  =       {energy:.8f} eV

 Spin-spin contribution to zero-field splitting tensor (MHz)
 -------------------------------------------------------------
      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz
 -------------------------------------------------------------
   {d_diag[0]:.2f}   {d_diag[1]:.2f}   {d_diag[2]:.2f}      0.00      0.00      0.00
 -------------------------------------------------------------
"""
        path.write_text(content, encoding="utf-8")

    def test_parse_zfs_simulation_dataset_1d(self, tmp_path):
        """Tests parsing a 1D simulation directory structure (e.g. NV_512/first_order/pert_0.1)."""
        root = tmp_path / "NV_512"

        # Ground-state unperturbed OUTCAR in ZFS_occup
        relaxed_outcar = root / "ZFS_occup" / "OUTCAR"
        self._create_mock_outcar(relaxed_outcar, d_diag=(-1000.0, -1000.0, 2000.0))

        # 1D perturbation runs
        sim_dir = root / "first_order" / "pert_0.1"
        run_1060 = sim_dir / "defect_band_approx" / "runs" / "1060" / "OUTCAR"
        run_1307 = sim_dir / "defect_band_approx" / "runs" / "1307" / "OUTCAR"
        self._create_mock_outcar(run_1060, d_diag=(-990.0, -990.0, 1980.0), energy=-510.5)
        self._create_mock_outcar(run_1307, d_diag=(-980.0, -980.0, 1960.0), energy=-511.2)

        raw_data = parse_zfs_simulation_dataset(sim_dir, calc_method="approx")

        assert raw_data.defect == "NV"
        assert raw_data.cell_size == 512
        assert raw_data.pert_scale == pytest.approx(0.1)
        assert raw_data.calc_method == "defect_band_approx"
        assert raw_data.order == 1
        assert raw_data.ground_state_zfs is not None
        assert raw_data.spin_correction_applied is True
        # parser applies the 3/2 spin correction for defect_band_approx at intake
        assert np.isclose(raw_data.ground_state_zfs.matrix[2, 2], 3000.0)

        assert 1059 in raw_data.first_order
        assert 1306 in raw_data.first_order
        entry_1060 = raw_data.first_order[1059]
        assert entry_1060.order == 1
        assert entry_1060.mode_indices == (1059,)
        assert entry_1060.amplitude == pytest.approx(0.1)
        assert np.isclose(entry_1060.zfs_tensor.matrix[2, 2], 2970.0)
        assert entry_1060.energy == pytest.approx(-510.5, abs=1e-3)

    def test_parse_zfs_simulation_dataset_2d(self, tmp_path):
        """Tests parsing a 2D simulation directory structure (e.g. NV_512/second_order/pert_0.1)."""
        root = tmp_path / "NV_512"

        # Ground-state unperturbed OUTCAR in ZFS_hyp for all_bands
        relaxed_outcar = root / "ZFS_hyp" / "OUTCAR"
        self._create_mock_outcar(relaxed_outcar, d_diag=(-1000.0, -1000.0, 2000.0))

        # 2D perturbation runs
        sim_dir = root / "second_order" / "pert_0.1"
        run_412_412 = sim_dir / "all_bands" / "runs" / "412_412" / "OUTCAR"
        run_412_413 = sim_dir / "all_bands" / "runs" / "412_413" / "OUTCAR"
        self._create_mock_outcar(run_412_412, d_diag=(-995.0, -995.0, 1990.0), energy=-520.1)
        self._create_mock_outcar(run_412_413, d_diag=(-992.0, -992.0, 1984.0), energy=-520.3)

        raw_data = parse_zfs_simulation_dataset(sim_dir, calc_method="all_bands")

        assert raw_data.defect == "NV"
        assert raw_data.cell_size == 512
        assert raw_data.pert_scale == pytest.approx(0.1)
        assert raw_data.calc_method == "all_bands"
        assert raw_data.order == 2
        assert (411, 411) in raw_data.second_order
        assert (411, 412) in raw_data.second_order

        entry_2d = raw_data.second_order[(411, 411)]
        assert entry_2d.order == 2
        assert entry_2d.mode_indices == (411, 411)
        assert entry_2d.amplitude == (0.1, 0.1)
        assert np.isclose(entry_2d.zfs_tensor.matrix[2, 2], 1990.0)

    def test_parse_zfs_simulation_dataset_overrides_and_errors(self, tmp_path):
        """Tests parameter overrides and error handling in parse_zfs_simulation_dataset."""
        root = tmp_path / "Custom_256"
        relaxed = root / "ZFS_occup" / "OUTCAR"
        self._create_mock_outcar(relaxed)

        sim_dir = root / "first_order" / "pert_0.02"
        run_1 = sim_dir / "defect_band_approx" / "runs" / "1" / "OUTCAR"
        self._create_mock_outcar(run_1)

        # Test overrides
        raw = parse_zfs_simulation_dataset(
            sim_dir,
            calc_method="approx",
            defect="Divacancy",
            cell_size=128,
            pert_scale=0.05,
            order=1,
        )
        assert raw.defect == "Divacancy"
        assert raw.cell_size == 128
        assert raw.pert_scale == pytest.approx(0.05)
        assert raw.order == 1

        # Test non-existent directory
        with pytest.raises(FileNotFoundError):
            parse_zfs_simulation_dataset(tmp_path / "nonexistent", calc_method="approx")

        # Test folder name without "pert"
        bad_dir = root / "first_order" / "invalid_name"
        bad_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError, match="Perturbation scale not found"):
            parse_zfs_simulation_dataset(bad_dir, calc_method="approx")

        # Test missing relaxed OUTCAR
        sim_dir_no_relaxed = tmp_path / "Missing_64" / "first_order" / "pert_0.1"
        (sim_dir_no_relaxed / "defect_band_approx" / "runs" / "1").mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError, match="Relaxed ZFS tensor not found"):
            parse_zfs_simulation_dataset(sim_dir_no_relaxed, calc_method="approx")


class TestSpinCorrectionOnLoad:
    """The 3/2 spin-correction scaling lives in RawZFSData.load, not ZFSManager."""

    @staticmethod
    def _make_raw(calc_method: str) -> RawZFSData:
        from spinlabs.models import PerturbationEntry

        unit = "MHz"
        gs = ZFSTensor(matrix=np.diag([-1000.0, -1000.0, 2000.0]).astype(float), unit=unit)
        first = {
            1060: PerturbationEntry(
                order=1,
                mode_indices=(1060,),
                amplitude=0.1,
                zfs_tensor=ZFSTensor(matrix=np.diag([-990.0, -990.0, 1980.0]).astype(float), unit=unit),
            )
        }
        return RawZFSData(
            defect="NV",
            cell_size=512,
            pert_scale=0.025,
            calc_method=calc_method,
            order=1,
            ground_state_zfs=gs,
            first_order=first,
        )

    def _save(self, raw: RawZFSData, tmp_path: Path) -> Path:
        out = tmp_path / f"test_{raw.calc_method}.npz"
        raw.save(out)
        return out

    def test_approx_tensors_scaled_by_three_halves_once(self, tmp_path, caplog):
        raw = self._make_raw("defect_band_approx")
        out = self._save(raw, tmp_path)

        # Stored file is unscaled
        with np.load(str(out), allow_pickle=True) as npz:
            assert npz["ground_state_zfs_matrix"][2, 2] == pytest.approx(2000.0)
            assert not npz.get("spin_correction_applied", False)

        with caplog.at_level("INFO", logger="spinlabs"):
            loaded = RawZFSData.load(out)
        h = 6.62607015e-34
        assert loaded.spin_correction_applied is True
        assert loaded.ground_state_zfs.unit == "J"
        assert loaded.ground_state_zfs.matrix[2, 2] == pytest.approx(3000.0e6 * h)
        assert loaded.ground_state_zfs.matrix[0, 0] == pytest.approx(-1500.0e6 * h)
        entry = loaded.first_order[1060]
        assert entry["tensor"][2, 2] == pytest.approx(2970.0e6 * h)
        assert "applied spin correction (defect_band_approx, x1.5)" in caplog.text

        # Saving the corrected object persists the flag; re-loading must not rescale
        out2 = tmp_path / "resaved.npz"
        loaded.save(out2)
        reloaded = RawZFSData.load(out2)
        assert reloaded.ground_state_zfs.matrix[2, 2] == pytest.approx(3000.0e6 * h)

    def test_load_with_mixed_flags_scales_only_uncorrected_files(self, tmp_path):
        legacy = self._save(self._make_raw("defect_band_approx"), tmp_path)
        # Build an already-corrected companion file with a DIFFERENT mode key,
        # so both files' tensors are visible in the merge and can be checked.
        corrected = RawZFSData.load(legacy)
        corrected.first_order = {2061: dict(corrected.first_order[1060])}
        corrected.first_order[2061]["mode_indices"] = (2061,)
        out2 = tmp_path / "already_corrected.npz"
        corrected.save(out2)

        h = 6.62607015e-34
        merged = RawZFSData.load([legacy, out2])
        assert merged.spin_correction_applied is True
        # Legacy file scaled exactly once; corrected file untouched (no double-scale)
        assert merged.ground_state_zfs.matrix[2, 2] == pytest.approx(3000.0e6 * h)
        assert merged.first_order[1060]["tensor"][2, 2] == pytest.approx(2970.0e6 * h)
        assert merged.first_order[2061]["tensor"][2, 2] == pytest.approx(2970.0e6 * h)
        # Reversed order: same result
        merged_rev = RawZFSData.load([out2, legacy])
        assert merged_rev.first_order[1060]["tensor"][2, 2] == pytest.approx(2970.0e6 * h)
        assert merged_rev.first_order[2061]["tensor"][2, 2] == pytest.approx(2970.0e6 * h)

    def test_apply_corrections_false_leaves_raw_tensors(self, tmp_path):
        # also: merged flag is all(), not any()
        legacy = self._save(self._make_raw("defect_band_approx"), tmp_path)
        corrected = RawZFSData.load(legacy)
        out2 = tmp_path / "already_corrected.npz"
        corrected.save(out2)
        merged = RawZFSData.load([legacy, out2], apply_corrections=False)
        assert merged.spin_correction_applied is False
        out = self._save(self._make_raw("defect_band_approx"), tmp_path)
        raw = RawZFSData.load(out, apply_corrections=False)
        h = 6.62607015e-34
        assert raw.spin_correction_applied is False
        assert raw.ground_state_zfs.matrix[2, 2] == pytest.approx(2000.0e6 * h)
        assert raw.first_order[1060]["tensor"][2, 2] == pytest.approx(1980.0e6 * h)

    def test_all_bands_is_never_scaled(self, tmp_path, caplog):
        out = self._save(self._make_raw("all_bands"), tmp_path)
        with caplog.at_level("INFO", logger="spinlabs"):
            loaded = RawZFSData.load(out)
        assert loaded.spin_correction_applied is False
        assert loaded.ground_state_zfs.matrix[2, 2] == pytest.approx(2000.0e6 * 6.62607015e-34)
        assert "spin correction" not in caplog.text


def test_combine_flag_mismatch_raises():
    raw = TestSpinCorrectionOnLoad._make_raw("defect_band_approx")
    legacy = copy.deepcopy(raw)
    corrected = copy.deepcopy(raw)
    corrected.spin_correction_applied = True
    with pytest.raises(ValueError, match="spin-correction"):
        legacy + corrected


def test_combine_all_corrected_flag_is_true(tmp_path):
    raw = TestSpinCorrectionOnLoad._make_raw("defect_band_approx")
    out = tmp_path / "a.npz"
    raw.save(out)
    loaded = RawZFSData.load(out)
    out2 = tmp_path / "b.npz"
    loaded.save(out2)
    merged = RawZFSData.load([out, out2])
    assert merged.spin_correction_applied is True


class TestSplitDefectFolderName:
    def test_simple(self):
        assert split_defect_folder_name("NV_512") == ("NV", 512)

    def test_multi_part(self):
        assert split_defect_folder_name("V_Si_128") == ("V_Si", 128)

    def test_orientation_suffix(self):
        assert split_defect_folder_name("ClV_576_hk") == ("ClV", 576)

    def test_no_cell_size(self):
        assert split_defect_folder_name("weirdname") == ("weirdname", None)



def test_only_run_folders_directly_under_runs_are_parsed(tmp_path):
    """Copies such as old_data/85_85/OUTCAR must not replace runs/85_85/OUTCAR."""
    import pytest

    from spinlabs.parsers import parse_perturbation_directory

    def outcar(path, dzz):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            " Spin-spin contribution to zero-field splitting tensor (MHz)\n -------\n"
            "      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz\n -------\n"
            f"  {-dzz / 2:.4f}  {-dzz / 2:.4f}  {dzz:.4f}  0.0000  0.0000  0.0000\n -------\n"
        )

    tree = tmp_path / "all_bands"
    outcar(tree / "runs" / "85_85" / "OUTCAR", 2000.0)
    outcar(tree / "old_data" / "85_85" / "OUTCAR", 1000.0)  # pre-fix copy, same name
    outcar(tree / "runs" / "85_85_old" / "OUTCAR", 1500.0)
    parsed = parse_perturbation_directory(tree, order=2, amplitude=(0.025, 0.025), max_workers=1)
    assert list(parsed) == [(84, 84)]
    assert parsed[(84, 84)].zfs_tensor.to_unit("MHz").matrix[2, 2] == pytest.approx(2000.0)
