import json
from pathlib import Path
import numpy as np
import pytest

from spinlabs.models import PhononSpectrum, RawZFSData, SpinPhononCouplingData, ZFSTensor
from spinlabs.pipeline import get_unique_run_dir, run_full_pipeline


@pytest.fixture
def dummy_coupling_file(tmp_path):
    n_modes = 5
    coupling = SpinPhononCouplingData(
        order=2,
        defect="NV",
        cell_size=64,
        pert_scale=0.025,
        calc_method="all_bands",
        frequencies=np.linspace(10.0, 50.0, n_modes) * 1.60218e-22,  # in J
        frequency_unit="J",
        ground_state_zfs=ZFSTensor(matrix=np.diag([-1e-25, -1e-25, 2e-25]), unit="J"),
        V_0_0=np.array([1e-25, 2e-25, 1.5e-25, 3e-25, 2.2e-25]),
        V_p_m=np.array([2e-25, 1e-25, 2.5e-25, 1.8e-25, 3.1e-25]),
        V_0_pm=np.array([1.2e-25, 2.1e-25, 1.9e-25, 2.7e-25, 1.4e-25]),
        V2_0_0=np.ones((n_modes, n_modes)) * 1e-25,
        V2_p_m=np.ones((n_modes, n_modes)) * 2e-25,
        V2_0_pm=np.ones((n_modes, n_modes)) * 1.5e-25,
        coupling_unit="J",
        symmetries=["A1", "E", "E", "A1", "E"],
        iprs=np.full(n_modes, 0.2),
    )
    save_file = tmp_path / "test_coupling.npz"
    coupling.save(save_file)
    return save_file


def test_unique_run_dir_increments_when_existing(tmp_path):
    root = tmp_path / "runs"
    dir1 = get_unique_run_dir(output_root=root, run_name="my_experiment")
    assert dir1.exists()
    assert dir1.name == "my_experiment"

    dir2 = get_unique_run_dir(output_root=root, run_name="my_experiment")
    assert dir2.exists()
    assert dir2.name == "my_experiment_1"

    dir3 = get_unique_run_dir(output_root=root, run_name="my_experiment")
    assert dir3.exists()
    assert dir3.name == "my_experiment_2"


def test_run_full_pipeline_from_coupling_data(tmp_path, dummy_coupling_file):
    run_root = tmp_path / "test_runs"

    res = run_full_pipeline(
        coupling_file=dummy_coupling_file,
        output_root=run_root,
        run_name="pipeline_test_run",
        t_start=50.0,
        t_end=150.0,
        t_step=50.0,
        save_plots=True,
    )

    run_dir = Path(res["run_dir"])
    assert run_dir.exists()

    # Verify all expected artifacts exist in run_dir
    assert (run_dir / "spin_phonon_coupling.npz").exists()
    assert (run_dir / "transition_rates.npz").exists()
    assert (run_dir / "directional_rates.npz").exists()
    # Every initial state is solved into the same run directory.
    for state in ("ms_1", "ms_0", "ms_-1"):
        assert (run_dir / f"t1_relaxation_{state}.npz").exists()
    assert not (run_dir / "t1_relaxation.npz").exists()
    assert (run_dir / "run_info.json").exists()

    # Check run_info.json contents
    with open(run_dir / "run_info.json") as f:
        info = json.load(f)
    assert info["defect"] == "NV"
    assert info["cell_size"] == 64
    assert info["temperatures"]["count"] == 3
    assert info["init_state"] == ["ms_1", "ms_0", "ms_-1"]

    # Check T1 file contents
    t1_data = np.load(run_dir / "t1_relaxation_ms_0.npz")
    assert str(t1_data["init_state"]) == "ms_0"
    assert len(t1_data["temperatures"]) == 3
    assert len(t1_data["t1_eigenval"]) == 3
    assert len(t1_data["t1_fit"]) == 3
    assert np.all(np.isfinite(t1_data["t1_eigenval"]))

    # Check figures folder
    fig_dir = run_dir / "figures"
    assert fig_dir.exists()
    assert (fig_dir / "coupling_spectral.png").exists()
    assert (fig_dir / "coupling_spectral_1d.png").exists()
    assert (fig_dir / "coupling_spectral_2d.png").exists()
    assert (fig_dir / "t1_vs_temperature.png").exists()

    # Test plot_results script functions on this run directory
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from spinlabs.plotter import plot_run_coupling, plot_run_rates, plot_run_t1

    custom_fig_dir = tmp_path / "custom_figures"
    custom_fig_dir.mkdir()
    plot_run_coupling(run_dir, custom_fig_dir, "png", 150, False)
    plot_run_rates(run_dir, custom_fig_dir, "png", 150, False)
    plot_run_t1(run_dir, custom_fig_dir, "png", 150, False)

    assert (custom_fig_dir / "coupling_spectral.png").exists()
    assert (custom_fig_dir / "coupling_spectral_1d.png").exists()
    assert (custom_fig_dir / "coupling_spectral_2d.png").exists()
    assert (custom_fig_dir / "t1_vs_temperature.png").exists()


def test_raw_zfs_file_with_both_orders_computes_order_2(tmp_path):
    """
    Verifies that passing a single raw_zfs_file containing both 1D and 2D data:
    1. Analyzes both orders (order=[1, 2] or detected automatically) and places the run under a pert_ folder.
    2. Computes both 1D and 2D couplings via process_both_orders.
    3. Preserves calc_method='all_bands' instead of resetting to None.
    4. order=2 alone still runs, keeping 2D couplings under V2_* rather than the 1D V_* keys.
    """
    n_modes = 4
    pert_scale = 0.025
    pert_si = pert_scale * 1.60218e-22  # in J

    # Build dummy RawZFSData with both 1st and 2nd order perturbations
    gs = ZFSTensor(matrix=np.diag([-1000.0, -1000.0, 2000.0]), unit="MHz")
    first_order = {}
    for i in range(n_modes):
        first_order[i] = {
            "tensor": np.diag([-1000.0, -1000.0, 2000.0]) * 6.626e-28,
            "pert": pert_si,
            "symmetry": "A1" if i % 2 == 0 else "Ex",
            "ipr": 0.5,
        }
    second_order = {}
    for i in range(n_modes):
        for j in range(i, n_modes):
            second_order[(i, j)] = {
                "tensor": np.diag([-1000.0, -1000.0, 2000.0]) * 6.626e-28,
                "pert": (pert_si, pert_si),
                "symmetry": ("A1", "A1") if i == j else ("A1", "Ex"),
                "ipr": (0.5, 0.5),
            }

    raw = RawZFSData(
        defect="ClV",
        cell_size=128,
        pert_scale=pert_scale,
        calc_method="all_bands",
        order=1,  # Note: even if saved with order=1 in file, 2D data exists!
        ground_state_zfs=gs,
        first_order=first_order,
        second_order=second_order,
    )
    raw_file = tmp_path / "raw_zfs_data.npz"
    raw.save(raw_file)

    # Build dummy phonon spectrum
    freqs = np.linspace(10.0, 40.0, n_modes)
    syms = ["A1", "Ex", "Ey", "A1"][:n_modes]
    # Make the Ex/Ey pair degenerate so the spectrum passes e-pair completeness
    freqs[1] = freqs[2] = 30.0
    spec = PhononSpectrum(
        frequencies_mev=freqs,
        symmetries=syms,
        iprs=np.ones(n_modes) * 0.5,
        eigenvectors=np.ones((n_modes, 1, 3)),
        atom_frac_coords=np.zeros((1, 3)),
        atom_symbols=["C"],
        atomic_masses=np.array([12.0]),
        lattice=np.eye(3) * 5.0,
    )
    ph_file = tmp_path / "phonon_data.npz"
    spec.save(ph_file)

    # Test 1: User explicitly asks for both orders
    res = run_full_pipeline(
        raw_zfs_file=raw_file,
        phonon_file=ph_file,
        order=[1, 2],
        calc_method="all_bands",
        save_plots=False,
        output_root=tmp_path / "runs1",
        temperatures=[10.0],
    )
    run_dir = res["run_dir"]
    assert "pert_0.025" in str(run_dir)
    coupling_file = run_dir / "spin_phonon_coupling.npz"
    assert coupling_file.exists()
    cdata = SpinPhononCouplingData.load(coupling_file)
    assert cdata.has_second_order
    assert cdata.V2_0_0 is not None
    assert cdata.calc_method == "all_bands"

    # Test 2: User passes order=None, pipeline should detect and analyze both orders
    res2 = run_full_pipeline(
        raw_zfs_file=raw_file,
        phonon_file=ph_file,
        order=None,
        calc_method="all_bands",
        save_plots=False,
        output_root=tmp_path / "runs2",
        temperatures=[10.0],
    )
    run_dir2 = res2["run_dir"]
    assert "pert_0.025" in str(run_dir2)
    cdata2 = SpinPhononCouplingData.load(run_dir2 / "spin_phonon_coupling.npz")
    assert cdata2.has_second_order
    assert cdata2.V2_0_0 is not None

    # Test 3: order=2 only; the 2D couplings must not overwrite the 1D V_* keys
    res3 = run_full_pipeline(
        raw_zfs_file=raw_file,
        phonon_file=ph_file,
        order=2,
        calc_method="all_bands",
        save_plots=False,
        output_root=tmp_path / "runs3",
        temperatures=[10.0],
    )
    cdata3 = SpinPhononCouplingData.load(res3["run_dir"] / "spin_phonon_coupling.npz")
    assert cdata3.has_second_order
    assert np.asarray(cdata3.V_0_0).shape == (n_modes,)
    assert np.asarray(cdata3.V2_0_0).shape == (n_modes, n_modes)


def _write_mock_outcar(path: Path, d_diag=(-950.0, -950.0, 1900.0), energy=-500.123):
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


def test_pipeline_with_multiple_sim_folders(tmp_path):
    """Tests passing multiple sim_folders (1D and 2D) to pipeline, combining their datasets."""
    root = tmp_path / "NV_512"

    # Relaxed unperturbed OUTCAR in ZFS_hyp
    _write_mock_outcar(root / "ZFS_hyp" / "OUTCAR", d_diag=(-1000.0, -1000.0, 2000.0))

    # 1D perturbation folder
    sim_1d = root / "first_order" / "pert_0.1"
    _write_mock_outcar(sim_1d / "all_bands" / "runs" / "1" / "OUTCAR", d_diag=(-980.0, -980.0, 1960.0))
    _write_mock_outcar(sim_1d / "all_bands" / "runs" / "2" / "OUTCAR", d_diag=(-970.0, -970.0, 1940.0))

    # 2D perturbation folder
    sim_2d = root / "second_order" / "pert_0.1"
    _write_mock_outcar(sim_2d / "all_bands" / "runs" / "1_1" / "OUTCAR", d_diag=(-990.0, -990.0, 1980.0))
    _write_mock_outcar(sim_2d / "all_bands" / "runs" / "1_2" / "OUTCAR", d_diag=(-985.0, -985.0, 1970.0))

    spec = PhononSpectrum(
        frequencies_mev=np.array([20.0, 50.0]),
        eigenvectors=np.ones((2, 1, 3)),
        symmetries=["A1", "A1"],
        iprs=np.array([0.1, 0.1]),
        atom_frac_coords=np.zeros((1, 3)),
        atom_symbols=["C"],
        atomic_masses=np.array([12.0]),
        lattice=np.eye(3) * 5.0,
    )
    ph_file = root / "phonon_data.npz"
    spec.save(ph_file)

    res = run_full_pipeline(
        sim_folder=[sim_1d, sim_2d],
        phonon_file=ph_file,
        calc_method="all_bands",
        save_plots=False,
        output_root=tmp_path / "runs_multi_sim",
        temperatures=[10.0],
    )
    run_dir = res["run_dir"]
    assert "pert_0.1" in str(run_dir)
    raw_saved = RawZFSData.load(run_dir / "raw_zfs_data.npz")
    assert 0 in raw_saved.first_order
    assert 1 in raw_saved.first_order
    assert (0, 0) in raw_saved.second_order
    assert (0, 1) in raw_saved.second_order


def test_pipeline_with_mismatched_sim_folders_raises(tmp_path):
    """Tests that passing sim_folders with mismatched defect metadata raises ValueError."""
    root_nv = tmp_path / "NV_512"
    _write_mock_outcar(root_nv / "ZFS_hyp" / "OUTCAR")
    sim_nv = root_nv / "first_order" / "pert_0.1"
    _write_mock_outcar(sim_nv / "all_bands" / "runs" / "1" / "OUTCAR")

    root_clv = tmp_path / "ClV_128"
    _write_mock_outcar(root_clv / "ZFS_hyp" / "OUTCAR")
    sim_clv = root_clv / "second_order" / "pert_0.1"
    _write_mock_outcar(sim_clv / "all_bands" / "runs" / "1_1" / "OUTCAR")

    with pytest.raises(ValueError, match="Cannot combine RawZFSData: defect mismatch"):
        run_full_pipeline(
            sim_folder=[sim_nv, sim_clv],
            calc_method="all_bands",
            save_plots=False,
            output_root=tmp_path / "runs_mismatched",
            temperatures=[10.0],
        )


def test_pipeline_registers_one_run_for_all_init_states(tmp_path, dummy_coupling_file):
    import spinlabs.registry as registry

    res = run_full_pipeline(
        coupling_file=dummy_coupling_file,
        output_root=tmp_path / "runs",
        temperatures=[50.0, 100.0],
        init_state=["ms_0", "ms_-1"],
        save_plots=False,
    )

    assert set(res["t1_paths"]) == {"ms_0", "ms_-1"}
    rows = registry.ensure_db().execute("SELECT init_state, run_dir FROM runs").fetchall()
    assert rows == [('["ms_0", "ms_-1"]', str(Path(res["run_dir"]).resolve()))]
    # The state filter matches any state the run solved.
    assert registry.find_runs(registry.ensure_db(), [("init_state", "ms_-1")]) == [Path(rows[0][1])]


def test_input_without_perturbation_tensors_fails_before_creating_a_run(tmp_path):
    raw = RawZFSData(defect="NV", cell_size=512, pert_scale=0.05, calc_method="all_bands", order=1,
                     ground_state_zfs=ZFSTensor(matrix=np.diag([-1000.0, -1000.0, 2000.0]), unit="MHz"))
    raw_file = raw.save(tmp_path / "NV_512_raw_zfs_data_all_bands_1d.npz")

    with pytest.raises(ValueError, match="No perturbation tensors"):
        run_full_pipeline(raw_zfs_file=raw_file, output_root=tmp_path / "runs", save_plots=False)
    assert not (tmp_path / "runs").exists()
