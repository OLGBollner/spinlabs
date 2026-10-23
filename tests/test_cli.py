"""Tests for the spinlabs CLI."""

import numpy as np

from spinlabs.cli import build_parser


def _make_t1_npz(path, defect="NV", cell=2, method="pbe", state="ms0"):
    t = np.linspace(0, 10, 50)
    pop = np.exp(-t / 5)
    np.savez(
        path,
        temperatures=t,
        t1_fit=pop,
        t1_eigenval=pop * 0.5,
        defect=defect,
        cell_size=cell,
        calc_method=method,
        init_state=state,
    )


def test_plot_t1_single_file(tmp_path, caplog):
    npz = tmp_path / "t1_relaxation.npz"
    _make_t1_npz(npz)
    out = tmp_path / "out.png"
    parser = build_parser()
    args = parser.parse_args(["plot", "t1", str(npz), "-o", str(out)])
    with caplog.at_level("INFO", logger="spinlabs"):
        args.func(args)
    assert list(tmp_path.glob("out*.png"))
    assert "Saved T1 figure to:" in caplog.text


def test_plot_t1_multiple_files_same_figure(tmp_path):
    p1 = tmp_path / "a.npz"
    p2 = tmp_path / "b.npz"
    _make_t1_npz(p1, defect="NV", cell=2)
    _make_t1_npz(p2, defect="ClV", cell=4)
    out = tmp_path / "out.png"
    parser = build_parser()
    args = parser.parse_args(["plot", "t1", str(p1), str(p2), "-o", str(out)])
    args.func(args)
    assert list(tmp_path.glob("out*.png"))


def test_plot_t1_default_output(tmp_path, monkeypatch):
    npz = tmp_path / "t1.npz"
    _make_t1_npz(npz)
    monkeypatch.chdir(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["plot", "t1", str(npz)])
    args.func(args)
    assert list((tmp_path / "figures" / "t1").glob("*.png"))
