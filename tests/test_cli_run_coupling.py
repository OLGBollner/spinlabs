"""``spinlabs run coupling`` runs only the ZFSManager stage and saves nothing."""

import numpy as np
import pytest

import spinlabs.cli as cli
import spinlabs.registry as registry
from spinlabs.models import PhononSpectrum, RawZFSData, ZFSTensor

N_MODES = 4


def _inputs(tmp_path, *, second_order=True, spin_flip=False):
    """A raw ZFS file with a linear response per mode, plus a matching phonon file.

    With ``spin_flip`` the E modes change D_xz/D_xx-D_yy (couplings that flip the
    spin), so the rates and T1 are finite; otherwise every response is A1-shaped.
    """
    gs = np.diag([-1000.0, -1000.0, 2000.0])
    syms = ["A1", "Ex", "Ey", "A1"]
    e_step = np.array([[1.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    steps = [e_step if spin_flip and syms[i] in ("Ex", "Ey") else np.diag([-1.0, -1.0, 2.0]) for i in range(N_MODES)]
    first = {
        i: {"tensor": gs + (i + 1) * steps[i], "unit": "MHz", "pert": 1e-25, "symmetry": syms[i], "ipr": 0.5}
        for i in range(N_MODES)
    }
    second = {
        (i, i): {"tensor": gs + 2.1 * (i + 1) * steps[i], "unit": "MHz", "pert": (1e-25, 1e-25),
                 "symmetry": (syms[i], syms[i]), "ipr": (0.5, 0.5)}
        for i in range(N_MODES)
    } if second_order else {}
    raw = RawZFSData(
        defect="NV", cell_size=64, pert_scale=0.1, calc_method="all_bands",
        ground_state_zfs=ZFSTensor(matrix=gs, unit="MHz"), first_order=first, second_order=second,
    )
    raw_file = tmp_path / "in" / "NV_64_raw_zfs_data_all_bands_12d.npz"
    raw_file.parent.mkdir()
    raw.save(raw_file)
    freqs = np.array([10.0, 30.0, 30.0, 40.0])  # degenerate Ex/Ey pair
    PhononSpectrum(
        frequencies_mev=freqs, symmetries=syms, iprs=np.full(N_MODES, 0.5),
        eigenvectors=np.ones((N_MODES, 1, 3)), atom_frac_coords=np.zeros((1, 3)), atom_symbols=["C"],
        atomic_masses=np.array([12.0]), lattice=np.eye(3) * 5.0,
    ).save(raw_file.parent / "phonon_data.npz")
    return raw_file


def _files(root):
    return sorted(p for p in root.rglob("*") if p.is_file() and ".spinlabs" not in p.parts)


def test_coupling_stage_prints_both_orders_and_saves_nothing(tmp_path, capsys, monkeypatch):
    raw_file = _inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    before = _files(tmp_path)

    cli.main(["run", "coupling", "--raw-zfs-file", str(raw_file), "--top", "2", "--mode", "1", "-q"])

    out = capsys.readouterr().out
    assert "NV_64  all_bands  pert 0.1" in out
    assert "4 first order, 4 second order (4 diagonal, 0 off-diagonal)" in out
    assert "First order (MHz/q):" in out and "Second order (MHz/q^2):" in out
    assert "      1  A1    10.00" in out  # --mode 1 is listed (1-based)
    assert _files(tmp_path) == before
    assert registry.ensure_db().execute("SELECT COUNT(*) FROM runs").fetchone() == (0,)


def test_coupling_stage_order_1_skips_second_order(tmp_path, capsys):
    raw_file = _inputs(tmp_path)

    cli.main(["run", "coupling", "--raw-zfs-file", str(raw_file), "--order", "1", "-q"])

    out = capsys.readouterr().out
    assert "First order" in out
    assert "Second order" not in out


def test_coupling_stage_rejects_order_2_without_second_order_data(tmp_path):
    raw_file = _inputs(tmp_path, second_order=False)

    with pytest.raises(SystemExit, match="no second-order tensors"):
        cli.main(["run", "coupling", "--raw-zfs-file", str(raw_file), "--order", "2", "-q"])


def test_coupling_stage_needs_exactly_one_input(tmp_path):
    with pytest.raises(SystemExit, match="exactly one of"):
        cli.main(["run", "coupling", "-q"])


def test_sigma_scan_reports_t1_per_sigma_and_saves_only_the_plot(tmp_path, capsys, monkeypatch):
    raw_file = _inputs(tmp_path, spin_flip=True)
    monkeypatch.chdir(tmp_path)
    before = _files(tmp_path)
    plot = tmp_path / "scan.png"

    cli.main(["run", "sigma-scan", "--raw-zfs-file", str(raw_file), "--sigma", "7.5", "1",
              "--temperature", "300", "--init-state", "0", "--plot", str(plot)])

    out = capsys.readouterr().out
    assert "== Transition rates and T1, sigma 1 meV (1/2) ==" in out and "sigma 7.5 meV (2/2)" in out
    assert "ms_0 @300K" in out and "max/min" in out
    rows = [line.split() for line in out.splitlines() if line.strip().startswith(("1 ", "7.5*"))]
    assert len(rows) == 2 and all(0 < float(r[1]) < np.inf for r in rows)
    assert _files(tmp_path) == sorted(before + [plot])
    assert registry.ensure_db().execute("SELECT COUNT(*) FROM runs").fetchone() == (0,)


def test_sigma_scan_plots_t1_against_temperature_per_sigma(tmp_path, capsys, monkeypatch):
    """--plot-temperature draws one line per sigma over the --t-range grid; the table stays at --temperature."""
    import matplotlib.pyplot as plt

    import spinlabs.cli.sigma_scan as scan

    raw_file = _inputs(tmp_path, spin_flip=True)
    figures = []
    monkeypatch.setattr(plt, "close", figures.append)  # keep the figure to inspect it
    plot = tmp_path / "t1_vs_t.png"
    cli.main(["run", "sigma-scan", "--raw-zfs-file", str(raw_file), "--sigma", "7.5", "1", "0.5",
              "--temperature", "300", "--init-state", "0", "--plot-temperature", str(plot),
              "--t-range", "50", "300", "50"])

    out = capsys.readouterr().out
    assert plot.is_file() and f"saved figure: {plot}" in out
    rows = [line.split() for line in out.splitlines() if line.strip().startswith(("0.5 ", "1 ", "7.5*"))]
    assert len(rows) == 3 and all(len(r) == 2 for r in rows)  # the grid adds no columns to the table
    panel = figures[-1].axes[0]
    assert len(panel.lines) == 3  # one line per sigma
    assert list(panel.lines[0].get_xdata()) == [50, 100, 150, 200, 250, 300]
    bar = figures[-1].axes[-1]
    assert [t.get_text() for t in bar.get_yticklabels()] == ["0.5", "1", "7.5 (default)"]
    with pytest.raises(SystemExit, match="--t-range"):
        scan._run_sigma_scan(cli.build_parser().parse_args(
            ["run", "sigma-scan", "--raw-zfs-file", str(raw_file), "--plot-temperature", str(plot),
             "--t-range", "300", "50", "50"]))


def test_sigma_scan_without_spin_flip_coupling_reports_infinite_t1(tmp_path, capsys):
    """A1-only responses give no transitions: T1 is infinite, and the plot is skipped with a note."""
    raw_file = _inputs(tmp_path)

    cli.main(["run", "sigma-scan", "--raw-zfs-file", str(raw_file), "--sigma", "1", "--init-state", "0",
              "--plot", str(tmp_path / "scan.png"), "-q"])

    out = capsys.readouterr().out
    assert "inf" in out and "warning: no finite T1 to plot" in out
    assert not (tmp_path / "scan.png").exists()
