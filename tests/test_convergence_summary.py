"""``spinlabs plot convergence``: one summary figure per defect by default, per-mode plots with --mode."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import spinlabs.cli as cli
import spinlabs.cli.plotting as cli_plot
import spinlabs.convergence_summary as summary
from spinlabs.convergence_summary import analyse, column_title
from spinlabs.pack_convergence import ConvergenceScan, save_convergence

D0 = np.diag([-1000.0, -1000.0, 2000.0])
STEPS = [-0.2, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.2]


def _pack(path, defect="NV", cell="512", method="all", key="test_all", modes=(5, 7), displacement="normalized"):
    """A convergence pack whose modes respond linearly (E) and quadratically (A1), with a little noise."""
    rng = np.random.default_rng(1)
    e_part = np.array([[1.0, 0.0, 0.5], [0.0, -1.0, 0.0], [0.5, 0.0, 0.0]])
    a1_part = np.diag([-1.0, -1.0, 2.0])
    tensors = np.array([
        [D0 + 40 * (m - 4) * q * e_part + 300 * q**2 * a1_part + rng.normal(0, 0.01, (3, 3)) for q in STEPS]
        for m in modes
    ])
    tensors = (tensors + tensors.transpose(0, 1, 3, 2)) / 2
    scan = ConvergenceScan(path.parent, list(modes), STEPS, tensors, np.zeros(tensors.shape[:2]), tensors.shape[0] * 8,
                           displacement=displacement)
    gs = path.parent / f"{key}_OUTCAR"
    gs.write_text(
        " Spin-spin contribution to zero-field splitting tensor (MHz)\n ---\n"
        "      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz\n ---\n"
        "  -1000.0  -1000.0  2000.0  0.0  0.0  0.0\n ---\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return save_convergence(scan, path, ground_state=gs, defect=defect, cell=cell, method=method, method_key=key)


def _plot(monkeypatch, tmp_path, argv):
    monkeypatch.chdir(tmp_path)
    args = cli.build_parser().parse_args(["plot", "convergence", *argv])
    args.func(args)
    return sorted(str(p.relative_to(tmp_path)) for p in (tmp_path / "figures").rglob("*.png"))


def test_summary_is_the_default_one_figure_per_defect(tmp_path, monkeypatch):
    nv_all = _pack(tmp_path / "NV_all.npz")
    nv_occup = _pack(tmp_path / "NV_occup.npz", method="approx", key="test_occup")
    div = _pack(tmp_path / "DiV.npz", defect="DiV", cell="128")
    columns = []
    real = cli_plot.plot_convergence_summary
    monkeypatch.setattr(cli_plot, "plot_convergence_summary", lambda packs, out, **kw: columns.append(packs) or real(packs, out, **kw))

    made = _plot(monkeypatch, tmp_path, [str(nv_occup), str(div), str(nv_all)])
    assert made == ["figures/convergence/DiV_128_normalized_convergence_summary.png",
                    "figures/convergence/NV_512_normalized_convergence_summary.png"]
    assert [nv_all, nv_occup] in columns  # all bands first, whatever the order given


def test_displacements_get_separate_summaries(tmp_path, monkeypatch):
    new = _pack(tmp_path / "NV_new.npz")
    old = _pack(tmp_path / "NV_old.npz", key="test_all_mass_weighted", displacement="mass_weighted")
    assert _plot(monkeypatch, tmp_path, [str(new), str(old)]) == [
        "figures/convergence/NV_512_mass_weighted_convergence_summary.png",
        "figures/convergence/NV_512_normalized_convergence_summary.png",
    ]


def test_mode_gives_the_per_mode_plots(tmp_path, monkeypatch):
    new = _pack(tmp_path / "NV_all.npz")
    old = _pack(tmp_path / "NV_old.npz", key="test_all_mass_weighted", displacement="mass_weighted")
    assert _plot(monkeypatch, tmp_path, [str(new), str(old), "-m", "7"]) == [
        "figures/convergence/NV_512_test_all_mass_weighted_mode7.png",  # the suffix already names it
        "figures/convergence/NV_512_test_all_normalized_mode7.png",
    ]


def test_per_mode_options_need_a_mode(tmp_path, monkeypatch):
    pack = _pack(tmp_path / "NV_all.npz")
    with pytest.raises(SystemExit, match="--quad apply to the per-mode plots; pick modes with --mode"):
        _plot(monkeypatch, tmp_path, [str(pack), "--quad"])


def test_one_output_for_several_defects_is_refused(tmp_path, monkeypatch):
    packs = [_pack(tmp_path / "NV.npz"), _pack(tmp_path / "DiV.npz", defect="DiV", cell="128")]
    with pytest.raises(SystemExit, match="-o names one figure but the files make 2 figures"):
        _plot(monkeypatch, tmp_path, [*map(str, packs), "-o", str(tmp_path / "x.png")])


def test_log_q_axis_is_ticked_at_the_run_step_sizes(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(summary.plt, "close", figures.append)  # keep the figure to inspect it
    summary.plot_convergence_summary([_pack(tmp_path / "NV.npz")], tmp_path / "out.png")
    rows = figures[0].axes[:2]
    for ax in rows:
        assert ax.get_xscale() == "log"
        assert list(ax.get_xticks()) == pytest.approx([0.025, 0.05, 0.1])  # 0.2 is the 2Q helper
    # the rows share x, so only the bottom one carries the labels
    assert [t.get_text() for t in rows[-1].get_xticklabels()] == ["0.025", "0.05", "0.1"]


def _legend_texts(fig):
    """Every legend label of the figure's first axes (the mode legend is kept as an extra artist)."""
    ax = fig.axes[0]
    legends = [a for a in ax.get_children() if isinstance(a, matplotlib.legend.Legend)]
    return [t.get_text() for legend in legends for t in legend.get_texts()]


def _plot_summary(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(summary.plt, "close", figures.append)
    summary.plot_convergence_summary([_pack(tmp_path / "NV.npz")], tmp_path / "out.png")
    return figures[0]


def test_legend_names_each_mode_with_its_symmetry(tmp_path, monkeypatch):
    from spinlabs.models import PhononSpectrum
    from spinlabs.parsers import save_phonon_npz
    from spinlabs.registry import ensure_db, register_file

    n = 9  # modes 5 and 7 are the 5th and 7th of the full spectrum
    labels = ["A1", "A1", "Ex", "Ey", "Ex", "Ey", "A1", "A2", "A1"]
    spectrum = PhononSpectrum(
        frequencies_mev=np.linspace(10.0, 90.0, n), eigenvectors=np.eye(n).reshape(n, 3, 3),
        atom_frac_coords=np.zeros((3, 3)), atom_symbols=["C"] * 3, atomic_masses=np.full(3, 12.011),
        lattice=np.eye(3) * 3.57, symmetries=labels, original_indices=np.arange(n),
    )
    phonon = tmp_path / "data" / "NV_512" / "phonon_data.npz"
    phonon.parent.mkdir(parents=True)
    save_phonon_npz(spectrum, phonon)
    register_file(ensure_db(), phonon, "phonon", defect="NV", cell_size=512)

    texts = _legend_texts(_plot_summary(tmp_path, monkeypatch))
    assert "5: Ex" in texts and "7: A1" in texts


def test_legend_falls_back_to_mode_numbers_without_phonon_data(tmp_path, monkeypatch, caplog):
    texts = _legend_texts(_plot_summary(tmp_path, monkeypatch))
    assert "5" in texts and "7" in texts
    assert "no registered phonon data for NV_512" in caplog.text


def _row(mode, lo, hi, *, snr=100.0, b=10.0, resolved=True, next_resolved=True):
    """A summary row with the SNR report fields shade_safe_windows reads, for the first order.

    The finite difference has SNR ``snr`` at each measured step (0.025 and 0.05, at +-Q).
    """
    from types import SimpleNamespace

    report = SimpleNamespace(
        b=b, c=1.0, q_lo1=lo, q_hi1=hi, b_resolved=resolved, next1_resolved=next_resolved, target=3.0,
        first_order=lambda q: (np.full(np.shape(q), 1 / snr), np.zeros(np.shape(q))),
    )
    first = np.array([[q, 10.0] for q in (-0.05, -0.025, 0.025, 0.05)])
    return mode, None, first, None, report


def test_safe_windows_are_banded_per_mode_and_shaded_where_all_overlap(caplog):
    import matplotlib.pyplot as plt

    rows = [
        _row(5, 0.02, 0.1),
        _row(7, 0.05, 0.2),
        _row(9, 0.01, 0.03, next_resolved=False),  # upper end not limited
        _row(11, 0.001, 1.0, resolved=False),  # b ~ 0: no window, no limit
        _row(13, 0.08, 0.3, snr=2.0),  # below SNR 3 at every measured step: left out
    ]
    fig, ax = plt.subplots()
    with caplog.at_level("INFO", logger="spinlabs"):
        neglected = summary.shade_safe_windows(ax, rows, 1, (0.01, 0.3), "NV_512 all bands first order")
    bands = [line for line in ax.lines if line.get_linewidth() == summary.BAND_WIDTH]
    assert [list(b.get_xdata()) for b in bands] == [[0.02, 0.1], [0.05, 0.2], [0.01, 0.3], [0.08, 0.3]]
    assert bands[-1].get_alpha() < bands[0].get_alpha()  # the left-out mode is drawn faint
    assert neglected == [13]
    # largest lower end and smallest known upper end of the modes kept: mode 13's 0.08 does not count
    assert "safe |Q| for all modes 0.05 .. 0.1" in caplog.text
    plt.close(fig)


@pytest.mark.parametrize("fraction, neglected, window", [
    (0.05, [7], "0.02 .. 0.1"),  # mode 7 is 3% of the strongest: left out
    (0.0, [], "0.08 .. 0.1"),  # cutoff off: mode 7's window counts
])
def test_relative_cutoff_leaves_weak_modes_out(caplog, fraction, neglected, window):
    import matplotlib.pyplot as plt

    rows = [_row(5, 0.02, 0.1, b=100.0), _row(7, 0.08, 0.3, b=3.0)]  # both well above the SNR target
    fig, ax = plt.subplots()
    with caplog.at_level("INFO", logger="spinlabs"):
        got = summary.shade_safe_windows(ax, rows, 1, (0.01, 0.3), "DiV_128 all bands first order", fraction)
    assert got == neglected
    assert f"safe |Q| for all modes {window}" in caplog.text
    plt.close(fig)


def test_analyse_measures_the_linear_and_quadratic_terms(tmp_path):
    rows, sigma, offset = analyse(_pack(tmp_path / "NV.npz"))
    assert [mode for mode, *_ in rows] == [5, 7]
    report = rows[1][-1]
    assert report.b_resolved and report.c_resolved
    assert sigma < 0.1 and offset < 0.1
    assert column_title(tmp_path / "NV.npz") == "NV_512 all bands"


def test_the_2q_helper_is_no_first_order_point_but_completes_the_pair_scheme(tmp_path):
    rows = analyse(_pack(tmp_path / "NV.npz"))[0]
    _mode, p, first, second, _report = rows[0]
    assert 0.2 not in np.abs(p) and 0.2 not in np.abs(first[:, 0])  # left out of the fit and first order
    assert sorted(set(np.abs(second[:, 0]))) == pytest.approx([0.025, 0.05, 0.1])  # 0.1 uses D(0.2)


def test_central_quotients_cancel_the_terms_that_split_forward_ones():
    """A cubic term and a step at Q = 0 split the forward second difference by side; central cancels both."""
    from spinlabs.convergence_summary import quotients

    p = np.array([-0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2])
    unit = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    parts = np.outer(100.0 * p + 40.0 * p**2 + 300.0 * p**3 + 0.05 * np.sign(p), unit)
    first, second = quotients(p, parts, parts[4], central=True)
    assert list(second[:, 0]) == pytest.approx([0.025, 0.05, 0.1, 0.2])  # one point per |Q|, 2Q no helper
    assert second[:, 1] == pytest.approx(80.0)  # 2c exactly: odd terms cancel
    assert first[:, 1] == pytest.approx(100.0 + 300.0 * first[:, 0] ** 2 + 0.05 / first[:, 0])
    forward = quotients(p, parts, parts[4])[1]
    # forward at -Q, +Q: 2c -+ 6 d Q (45) +- the step's 0.05/Q^2 (80)
    assert forward[np.isclose(np.abs(forward[:, 0]), 0.025), 1] == pytest.approx([115.0, 45.0])


def test_central_summary_uses_every_step_and_its_own_name(tmp_path, monkeypatch):
    pack = _pack(tmp_path / "NV_all.npz")
    rows = analyse(pack, central=True)[0]
    _mode, p, first, second, report = rows[1]
    assert report.central and 0.2 in np.abs(p)
    assert list(first[:, 0]) == list(second[:, 0]) == pytest.approx([0.025, 0.05, 0.1, 0.2])
    assert _plot(monkeypatch, tmp_path, [str(pack), "--central"]) == [
        "figures/convergence/NV_512_normalized_central_convergence_summary.png"
    ]
    with pytest.raises(SystemExit, match="--central applies to the summary figure"):
        _plot(monkeypatch, tmp_path, [str(pack), "--central", "-m", "7"])


def test_second_order_coefficients_come_from_the_fit_reaching_2q():
    """b from the runs without the 2Q helper; c and d from the runs with it (snr_report's pair)."""
    from spinlabs.convergence import snr_report

    perts = np.array([-0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2])
    unit = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    parts = np.outer(5.0 * perts + 40.0 * perts**2 + 30.0 * perts**3, unit)
    fit = ~np.isclose(np.abs(perts), 0.2)
    report = snr_report(perts[fit], parts[fit], 1e-4, "", np.zeros(5), pair=(perts, parts))
    assert report.b == pytest.approx(5.0, rel=1e-6)
    assert (report.c, report.d) == pytest.approx((40.0, 30.0), rel=1e-6)


@pytest.mark.parametrize("steps, helper", [
    ([0.025, 0.05, 0.1, 0.2], 0.2),  # 0.2 is the 2Q partner of 0.1
    ([0.025, 0.05, 0.1, 0.3], None),  # 0.3 is no double of a run step
    ([0.025, 0.05, 0.1], None),  # leaving 0.1 out would leave 2 step sizes: kept
])
def test_helper_step(steps, helper):
    from spinlabs.convergence import helper_step

    perts = np.array(sorted([-s for s in steps] + [0.0] + steps))
    assert helper_step(perts) == helper
