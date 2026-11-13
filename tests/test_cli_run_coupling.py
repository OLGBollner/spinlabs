"""``spinlabs run coupling`` and ``run sigma-scan`` run single stages, save nothing, and share the production
defaults of ``spinlabs run``."""

import numpy as np
import pytest

import spinlabs.cli as cli
import spinlabs.registry as registry
from spinlabs.models import PhononSpectrum, RawZFSData, ZFSTensor

N_MODES = 4


def _inputs(tmp_path, *, second_order=True, spin_flip=False, spin=None):
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
        ground_state_zfs=ZFSTensor(matrix=gs, unit="MHz"), first_order=first, second_order=second, spin=spin,
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

    cli.main(["run", "sigma-scan", "--raw-zfs-file", str(raw_file), "--sigma", "3.5", "1",
              "--temperature", "300", "--init-state", "0", "--plot", str(plot)])

    out = capsys.readouterr().out
    assert "== Transition rates and T1, sigma 1 meV (1/2) ==" in out and "sigma 3.5 meV (2/2)" in out
    assert "ms_0 @300K" in out and "max/min" in out
    rows = [line.split() for line in out.splitlines() if line.strip().startswith(("1 ", "3.5*"))]
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
    cli.main(["run", "sigma-scan", "--raw-zfs-file", str(raw_file), "--sigma", "3.5", "1", "0.5",
              "--temperature", "300", "--init-state", "0", "--plot-temperature", str(plot),
              "--t-range", "50", "300", "50"])

    out = capsys.readouterr().out
    assert plot.is_file() and f"saved figure: {plot}" in out
    rows = [line.split() for line in out.splitlines() if line.strip().startswith(("0.5 ", "1 ", "3.5*"))]
    assert len(rows) == 3 and all(len(r) == 2 for r in rows)  # the grid adds no columns to the table
    panel = figures[-1].axes[0]
    assert len(panel.lines) == 3  # one line per sigma
    assert list(panel.lines[0].get_xdata()) == [50, 100, 150, 200, 250, 300]
    bar = figures[-1].axes[-1]
    assert [t.get_text() for t in bar.get_yticklabels()] == ["0.5", "1", "3.5 (default)"]
    with pytest.raises(SystemExit, match="--t-range"):
        scan._run_sigma_scan(cli.build_parser().parse_args(
            ["run", "sigma-scan", "--raw-zfs-file", str(raw_file), "--plot-temperature", str(plot),
             "--t-range", "300", "50", "50"]))


def test_sigma_scan_temperature_plot_stays_on_the_t_range(tmp_path, capsys, monkeypatch):
    """A --temperature outside --t-range is in the table, not on the T1-vs-T lines."""
    import matplotlib.pyplot as plt

    raw_file = _inputs(tmp_path, spin_flip=True)
    figures = []
    monkeypatch.setattr(plt, "close", figures.append)
    cli.main(["run", "sigma-scan", "--raw-zfs-file", str(raw_file), "--sigma", "3.5", "--temperature", "400",
              "--init-state", "0", "--plot-temperature", str(tmp_path / "t.png"), "--t-range", "50", "300", "50", "-q"])

    assert "ms_0 @400K" in capsys.readouterr().out
    assert list(figures[-1].axes[0].lines[0].get_xdata()) == [50, 100, 150, 200, 250, 300]


def test_missing_t1_breaks_the_plotted_line():
    """NaN, infinite and non-positive T1 become NaN, where matplotlib leaves a gap instead of a straight line."""
    import spinlabs.cli.sigma_scan as scan

    out = scan._gaps_as_nan([1.0, np.nan, np.inf, 0.0, -2.0, 3.0])
    np.testing.assert_array_equal(np.isnan(out), [False, True, True, True, True, False])
    assert out[0] == 1.0 and out[-1] == 3.0


def test_sigma_scan_without_spin_flip_coupling_reports_infinite_t1(tmp_path, capsys):
    """A1-only responses give no transitions: T1 is infinite, and the plot is skipped with a note."""
    raw_file = _inputs(tmp_path)

    cli.main(["run", "sigma-scan", "--raw-zfs-file", str(raw_file), "--sigma", "1", "--init-state", "0",
              "--plot", str(tmp_path / "scan.png"), "-q"])

    out = capsys.readouterr().out
    assert "inf" in out and "warning: no finite T1 to plot" in out
    assert not (tmp_path / "scan.png").exists()


def test_run_rejects_init_state_of_another_spin_before_the_rates(tmp_path, monkeypatch):
    """``--init-state ms_2`` on an S = 1 package fails before any rate is computed or written."""
    import spinlabs.pipeline as pipeline

    raw_file = _inputs(tmp_path, spin_flip=True)
    monkeypatch.chdir(tmp_path)

    def no_rates(*args, **kwargs):
        raise AssertionError("rates computed for an invalid initial state")

    monkeypatch.setattr(pipeline, "compute_rates", no_rates)
    with pytest.raises(ValueError, match=r"ms_2 is not a state of spin 1: ms_1, ms_0, ms_-1"):
        cli.main(["run", "--raw-zfs-file", str(raw_file), "--temperatures", "300", "--init-state", "ms_2", "-q"])
    assert not list(tmp_path.rglob("transition_rates.npz"))
    assert not list(tmp_path.rglob("directional_rates.npz"))


# -- production defaults, shared with `spinlabs run` -------------------------------------------------------------

_SHARED = ("direct", "keep_zero_gap_pairs", "first_order_raman", "exclude_inconsistent_pairs",
           "degenerate_pairs_from_symmetry", "offdiag_from_diagonal", "init_state")


def _parse(*argv):
    return cli.build_parser().parse_args(list(argv))


def test_stages_default_to_the_production_settings_of_run():
    from spinlabs.cli.production import coupling_options, rate_options
    from spinlabs.pipeline import ALL_PROCESSES, PRODUCTION_DIRECT, PRODUCTION_PROCESSES
    from spinlabs.relaxation_dynamics import INIT_STATES

    run = _parse("run", "--raw-zfs-file", "x.npz")
    scan = _parse("run", "sigma-scan", "--raw-zfs-file", "x.npz")
    coupling = _parse("run", "coupling", "--raw-zfs-file", "x.npz")
    for name in _SHARED:
        assert getattr(scan, name) == getattr(run, name), name
        if name not in ("direct", "keep_zero_gap_pairs", "first_order_raman", "init_state"):  # coupling: no rates
            assert getattr(coupling, name) == getattr(run, name), name
    assert rate_options(scan) == dict(direct=PRODUCTION_DIRECT, nogap=True, processes=PRODUCTION_PROCESSES)
    assert "second_order" not in PRODUCTION_PROCESSES and set(PRODUCTION_PROCESSES) < set(ALL_PROCESSES)
    assert coupling_options(coupling) == dict(
        exclude_inconsistent_pairs=True, degenerate_pairs_from_symmetry=False, offdiag_from_diagonal=True
    )
    assert scan.init_state == list(INIT_STATES)


def test_stage_flags_bring_back_the_old_settings():
    from spinlabs.cli.production import coupling_options, rate_options
    from spinlabs.pipeline import ALL_PROCESSES

    old = ["--direct", "gaussian", "--keep-zero-gap-pairs", "--first-order-raman",
           "--no-exclude-inconsistent-pairs", "--no-offdiag-from-diagonal"]
    scan = _parse("run", "sigma-scan", "--raw-zfs-file", "x.npz", *old)
    assert rate_options(scan) == dict(direct="gaussian", nogap=False, processes=ALL_PROCESSES)
    assert coupling_options(scan) == dict(
        exclude_inconsistent_pairs=False, degenerate_pairs_from_symmetry=False, offdiag_from_diagonal=False
    )
    coupling = _parse("run", "coupling", "--raw-zfs-file", "x.npz", "--keep-inconsistent-pairs")  # older spelling
    assert coupling.exclude_inconsistent_pairs is False
    assert _parse("run", "coupling", "--raw-zfs-file", "x.npz", "--no-offdiag-from-diagonal").offdiag_from_diagonal is False


def test_coupling_stage_excludes_inconsistent_pairs_by_default(tmp_path, capsys):
    raw_file = _inputs(tmp_path)

    cli.main(["run", "coupling", "--raw-zfs-file", str(raw_file), "-q"])
    assert "inconsistent entries excluded" in capsys.readouterr().out
    cli.main(["run", "coupling", "--raw-zfs-file", str(raw_file), "--no-exclude-inconsistent-pairs", "-q"])
    assert "inconsistent entries kept" in capsys.readouterr().out


@pytest.mark.parametrize("old", [False, True])
def test_sigma_scan_passes_the_rate_settings_to_compute_rates(tmp_path, capsys, monkeypatch, old):
    import spinlabs.pipeline as pipeline

    calls = []
    real = pipeline.compute_rates

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "compute_rates", spy)
    raw_file = _inputs(tmp_path, spin_flip=True)
    flags = ["--direct", "gaussian", "--keep-zero-gap-pairs", "--first-order-raman"] if old else []
    cli.main(["run", "sigma-scan", "--raw-zfs-file", str(raw_file), "--sigma", "3.5", "--init-state", "0", "-q",
              *flags])

    (kwargs,) = calls
    if old:
        assert kwargs["direct"] == "gaussian" and kwargs["degenerate_groups"] is None
        assert kwargs["processes"] == pipeline.ALL_PROCESSES
    else:
        assert kwargs["direct"] == pipeline.PRODUCTION_DIRECT and kwargs["spectrum"] is not None
        assert [1, 2] in kwargs["degenerate_groups"]  # the Ex/Ey doublet
        assert kwargs["processes"] == pipeline.PRODUCTION_PROCESSES
    out = capsys.readouterr().out
    assert ("direct rate gaussian, zero-gap pairs kept" if old else "direct rate debye, zero-gap pairs left out") in out


def test_sigma_scan_at_the_run_sigma_gives_the_run_t1(tmp_path, capsys):
    """With the defaults, the scan's T1 at sigma 3.5 meV is the T1 of `spinlabs run` on the same input."""
    raw_file = _inputs(tmp_path, spin_flip=True)

    cli.main(["run", "sigma-scan", "--raw-zfs-file", str(raw_file), "--sigma", "3.5", "--temperature", "300",
              "--init-state", "0", "-q"])
    (row,) = [line.split() for line in capsys.readouterr().out.splitlines() if line.strip().startswith("3.5*")]
    scan_ms = float(row[1])

    cli.main(["run", "--raw-zfs-file", str(raw_file), "--temperatures", "300", "--init-state", "0",
              "-o", str(tmp_path / "runs"), "-q"])
    (t1_file,) = (tmp_path / "runs").rglob("t1_relaxation_ms_0.npz")
    run_ms = float(np.load(t1_file)["t1_fit"][0]) * 1e3
    assert 0 < run_ms < np.inf
    assert scan_ms == pytest.approx(run_ms, rel=1e-3)  # the table prints 4 significant figures


def test_sigma_scan_takes_the_initial_states_of_the_spin(tmp_path, capsys):
    """S = 3/2: the default states are the spin's; S = 1 names are rejected, as by `spinlabs run`."""
    raw_file = _inputs(tmp_path, spin_flip=True, spin=1.5)

    cli.main(["run", "sigma-scan", "--raw-zfs-file", str(raw_file), "--sigma", "3.5", "-q"])
    out = capsys.readouterr().out
    assert "ms_pm1_2 @300K" in out and "ms_pm3_2 @300K" in out and "ms_0 @300K" not in out

    cli.main(["run", "sigma-scan", "--raw-zfs-file", str(raw_file), "--sigma", "3.5", "--init-state", "ms_pm3_2", "-q"])
    assert "ms_pm3_2 @300K" in capsys.readouterr().out

    with pytest.raises(SystemExit, match="ms_0 is not a state of spin 1.5"):
        cli.main(["run", "sigma-scan", "--raw-zfs-file", str(raw_file), "--sigma", "3.5", "--init-state", "0", "-q"])


def test_sigma_scan_rejects_states_of_another_spin(tmp_path):
    raw_file = _inputs(tmp_path, spin_flip=True)

    with pytest.raises(SystemExit, match="ms_pm3_2 is not a state of spin 1"):
        cli.main(["run", "sigma-scan", "--raw-zfs-file", str(raw_file), "--sigma", "3.5",
                  "--init-state", "ms_pm3_2", "-q"])
