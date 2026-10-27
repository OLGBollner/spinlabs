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


def _pack(path, defect="NV", cell="512", method="all", key="test_all", modes=(5, 7)):
    """A convergence pack whose modes respond linearly (E) and quadratically (A1), with a little noise."""
    rng = np.random.default_rng(1)
    e_part = np.array([[1.0, 0.0, 0.5], [0.0, -1.0, 0.0], [0.5, 0.0, 0.0]])
    a1_part = np.diag([-1.0, -1.0, 2.0])
    tensors = np.array([
        [D0 + 40 * (m - 4) * q * e_part + 300 * q**2 * a1_part + rng.normal(0, 0.01, (3, 3)) for q in STEPS]
        for m in modes
    ])
    tensors = (tensors + tensors.transpose(0, 1, 3, 2)) / 2
    scan = ConvergenceScan(path.parent, list(modes), STEPS, tensors, np.zeros(tensors.shape[:2]), tensors.shape[0] * 8)
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
    assert made == ["figures/convergence/DiV_128_convergence_summary.png", "figures/convergence/NV_512_convergence_summary.png"]
    assert [nv_all, nv_occup] in columns  # all bands first, whatever the order given


def test_mode_gives_the_per_mode_plots(tmp_path, monkeypatch):
    pack = _pack(tmp_path / "NV_all.npz")
    assert _plot(monkeypatch, tmp_path, [str(pack), "-m", "7"]) == ["figures/convergence/NV_512_test_all_mode7.png"]


def test_per_mode_options_need_a_mode(tmp_path, monkeypatch):
    pack = _pack(tmp_path / "NV_all.npz")
    with pytest.raises(SystemExit, match="--quad apply to the per-mode plots; pick modes with --mode"):
        _plot(monkeypatch, tmp_path, [str(pack), "--quad"])


def test_one_output_for_several_defects_is_refused(tmp_path, monkeypatch):
    packs = [_pack(tmp_path / "NV.npz"), _pack(tmp_path / "DiV.npz", defect="DiV", cell="128")]
    with pytest.raises(SystemExit, match="-o names one figure but the files hold 2 defects"):
        _plot(monkeypatch, tmp_path, [*map(str, packs), "-o", str(tmp_path / "x.png")])


def test_log_q_axis_is_ticked_at_the_run_step_sizes(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(summary.plt, "close", figures.append)  # keep the figure to inspect it
    summary.plot_convergence_summary([_pack(tmp_path / "NV.npz")], tmp_path / "out.png")
    rows = figures[0].axes[:3]
    for ax in rows:
        assert ax.get_xscale() == "log"
        assert list(ax.get_xticks()) == pytest.approx([0.025, 0.05, 0.1, 0.2])
    # the rows share x, so only the bottom one carries the labels
    assert [t.get_text() for t in rows[-1].get_xticklabels()] == ["0.025", "0.05", "0.1", "0.2"]


def test_analyse_measures_the_linear_and_quadratic_terms(tmp_path):
    rows, sigma, offset = analyse(_pack(tmp_path / "NV.npz"))
    assert [mode for mode, *_ in rows] == [5, 7]
    report = rows[1][-1]
    assert report.b_resolved and report.c_resolved
    assert sigma < 0.1 and offset < 0.1
    assert column_title(tmp_path / "NV.npz") == "NV_512 all bands"
