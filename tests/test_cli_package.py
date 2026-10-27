import json
import argparse
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
from unittest.mock import patch

import pytest
from spinlabs.cli.package import build_package_parser

from spinlabs.cli.package import _run, _run_conv
from spinlabs.models import PerturbationEntry, RawZFSData, ZFSTensor

def _raw(order=1, pert=0.025, calc_method=None, **_kw):
    kwargs = dict(
        defect="NV",
        cell_size=512,
        pert_scale=pert,
        calc_method={"all": "all_bands", "approx": "defect_band_approx"}.get(calc_method, calc_method or "all_bands"),
        order=order,
    )
    if order == 2:
        # Non-empty second order so content-derived orders resolve to [2],
        # matching real second-order data.
        kwargs["second_order"] = {
            (0, 0): PerturbationEntry(
                order=2,
                mode_indices=(0, 0),
                amplitude=(0.025, 0.025),
                zfs_tensor=ZFSTensor(matrix=np.zeros((3, 3))),
            )
        }
    else:
        kwargs["first_order"] = {
            0: PerturbationEntry(
                order=1,
                mode_indices=(0,),
                amplitude=(0.025, 0.0),
                zfs_tensor=ZFSTensor(matrix=np.zeros((3, 3))),
            )
        }
    return RawZFSData(**kwargs)


def _make_args(tmp_path, orders="first,second", pert=None, yes=True, scan=False):
    return argparse.Namespace(
        defect_folder=[tmp_path / "NV_512"],
        orders=orders,
        pert=pert,
        method=["all"],
        output_root=tmp_path / "data",
        scan=scan,
        yes=yes,
    )


def _tree(tmp_path, orders=("first", "second"), perts=(0.025,)):
    for o in orders:
        for p in perts:
            outcar = tmp_path / "NV_512" / f"{o}_order" / f"pert_{p:g}" / "defect_band_approx" / "OUTCAR"
            outcar.parent.mkdir(parents=True, exist_ok=True)
            outcar.touch()


def _patch(side_effect=_raw):
    return patch("spinlabs.cli.package.parse_zfs_simulation_dataset", side_effect=side_effect)


def _registry(monkeypatch, tmp_path):
    """Point the registry at an isolated sqlite file and return the path."""
    db = tmp_path / "registry.db"
    monkeypatch.setattr("spinlabs.registry.REGISTRY_PATH", db)
    return db


def test_default_combines_orders_into_one_file(tmp_path, monkeypatch):
    _tree(tmp_path)
    db = _registry(monkeypatch, tmp_path)
    with _patch(lambda **kw: _raw(order=kw["order"])):
        _run(_make_args(tmp_path))
    files = list((tmp_path / "data").rglob("*.npz"))
    run_dir = tmp_path / "data" / "NV_512" / "pert_0.025" / "all_bands" / date.today().strftime("%Y%m%d")
    assert files == [run_dir / "NV_512_raw_zfs_data_all_bands_12d.npz"]
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT defect, cell_size, calc_method, pert_order, pert_scale FROM files").fetchone()
    assert row == ("NV", 512, "all_bands", "[1, 2]", 0.025)


def test_single_order_saves_separate_file(tmp_path):
    _tree(tmp_path, orders=("first",))
    with _patch():
        _run(_make_args(tmp_path, orders="first"))
    files = list((tmp_path / "data").rglob("*.npz"))
    run_dir = tmp_path / "data" / "NV_512" / "pert_0.025" / "all_bands" / date.today().strftime("%Y%m%d")
    assert files == [run_dir / "NV_512_raw_zfs_data_all_bands_1d.npz"]


def test_pert_filter(tmp_path):
    _tree(tmp_path, perts=(0.01, 0.025))
    with _patch():
        _run(_make_args(tmp_path, orders="first", pert=[0.025]))
    assert len(list((tmp_path / "data").rglob("*.npz"))) == 1


def _warned(caplog, text):
    """True when a spinlabs WARNING containing ``text`` was logged."""
    return any(r.levelname == "WARNING" and text in r.getMessage() for r in caplog.records)


def test_missing_order_folder_warns(tmp_path, caplog):
    _tree(tmp_path, orders=("first",))
    with _patch():
        _run(_make_args(tmp_path))
    assert _warned(caplog, "no second order folder")
    assert len(list((tmp_path / "data").rglob("*.npz"))) == 1


@pytest.mark.parametrize("answer", ["", "y", "Yes"])
def test_prompt_yes_saves(tmp_path, monkeypatch, capsys, caplog, answer):
    _tree(tmp_path, orders=("first",))
    _registry(monkeypatch, tmp_path)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or answer)
    with _patch(), caplog.at_level("INFO", logger="spinlabs"):
        _run(_make_args(tmp_path, orders="first", yes=False))
    assert prompts == ["Save the perturbation data to file? (Y/n): "]
    # The scan table is CLI output (stdout); the save confirmation is logged.
    assert "Available perturbation data:" in capsys.readouterr().out
    assert "saved raw ZFS data:" in caplog.text
    assert len(list((tmp_path / "data").rglob("*.npz"))) == 1


@pytest.mark.parametrize("answer", ["n", "no", "N"])
def test_prompt_no_saves_nothing(tmp_path, monkeypatch, answer):
    _tree(tmp_path, orders=("first",))
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    with _patch():
        _run(_make_args(tmp_path, orders="first", yes=False))
    assert not (tmp_path / "data").exists()


def test_prompt_eof_saves_nothing(tmp_path, monkeypatch):
    """Without a terminal (input() hits EOF) nothing is written."""
    _tree(tmp_path, orders=("first",))

    def _eof(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    with _patch():
        _run(_make_args(tmp_path, orders="first", yes=False))
    assert not (tmp_path / "data").exists()


def test_scan_lists_without_prompting(tmp_path, monkeypatch, capsys):
    _tree(tmp_path)

    def _no_input(prompt):
        raise AssertionError("--scan must not prompt")

    monkeypatch.setattr("builtins.input", _no_input)
    with _patch(lambda **kw: _raw(order=kw["order"])):
        _run(_make_args(tmp_path, yes=False, scan=True))
    out = capsys.readouterr().out
    assert "NV_512" in out and "all_bands" in out and "0.025" in out
    assert not (tmp_path / "data").exists()


# ── --conv --ground-state (flag-only auto-discovery) ─────────────────────────

def _outcar(path, d_diag, energy=-10.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f" free  energy   TOTEN  =       {energy:.8f} eV\n\n"
        " Spin-spin contribution to zero-field splitting tensor (MHz)\n"
        " -------------------------------------------------------------\n"
        "      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz\n"
        " -------------------------------------------------------------\n"
        f"   {d_diag[0]:.2f}   {d_diag[1]:.2f}   {d_diag[2]:.2f}      0.00      0.00      0.00\n"
        " -------------------------------------------------------------\n",
        encoding="utf-8",
    )


def _conv_args(tmp_path, ground_state, yes=True, scan=False):
    return argparse.Namespace(
        defect_folder=tmp_path / "NV_512",
        output_root=tmp_path / "data",
        ground_state=ground_state,
        yes=yes,
        scan=scan,
    )


def _conv_tree(tmp_path):
    root = tmp_path / "NV_512" / "convergence_test_all"
    for p in (0.025, 0.05):
        _outcar(root / "runs" / f"0_pert_{p:g}" / "OUTCAR", (1.0, 1.0, 2.0))
    return root


def test_conv_ground_state_flag_auto_all(tmp_path, caplog, monkeypatch):
    _conv_tree(tmp_path)
    db = _registry(monkeypatch, tmp_path)
    _outcar(tmp_path / "NV_512" / "ZFS_hyp" / "OUTCAR", (1.0, 1.0, 4.0))
    with caplog.at_level("DEBUG", logger="spinlabs"):
        _run_conv(args=_conv_args(tmp_path, Path("__auto__")), defect_folder=tmp_path / "NV_512")
    out = next((tmp_path / "data").rglob("NV_512_*_conv_test_all.npz"))
    data = np.load(out, allow_pickle=True)
    assert data["d0"].shape == (3, 3)
    assert abs(data["d0"][2, 2] - 4.0) < 1e-9
    assert "ground state:" in caplog.text
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT defect, cell_size, calc_method FROM files").fetchone()
    assert row == ("NV", 512, "all_bands")


def test_conv_ground_state_flag_auto_occup(tmp_path):
    root = tmp_path / "NV_512" / "convergence_test_occup"
    for p in (0.025,):
        _outcar(root / "runs" / f"0_pert_{p:g}" / "OUTCAR", (1.0, 1.0, 2.0))
    _outcar(tmp_path / "NV_512" / "ZFS_occup" / "OUTCAR", (1.0, 1.0, 3.0))
    _run_conv(args=_conv_args(tmp_path, Path("__auto__")), defect_folder=tmp_path / "NV_512")
    out = next((tmp_path / "data").rglob("NV_512_*_conv_test_occup.npz"))
    data = np.load(out, allow_pickle=True)
    assert abs(data["d0"][2, 2] - 3.0) < 1e-9


def test_conv_ground_state_missing_warns(tmp_path, caplog):
    _conv_tree(tmp_path)
    _run_conv(args=_conv_args(tmp_path, Path("__auto__")), defect_folder=tmp_path / "NV_512")
    out = next((tmp_path / "data").rglob("NV_512_*_conv_*.npz"))
    data = np.load(out, allow_pickle=True)
    assert not any(k == "d0" for k in data.files)
    assert any(r.levelname == "WARNING" and "no ground-state OUTCAR" in r.getMessage() for r in caplog.records)


def test_conv_ground_state_explicit_path(tmp_path):
    _conv_tree(tmp_path)
    _outcar(tmp_path / "elsewhere" / "OUTCAR", (1.0, 1.0, 5.0))
    _run_conv(args=_conv_args(tmp_path, tmp_path / "elsewhere" / "OUTCAR"), defect_folder=tmp_path / "NV_512")
    out = next((tmp_path / "data").rglob("NV_512_*_conv_*.npz"))
    data = np.load(out, allow_pickle=True)
    assert abs(data["d0"][2, 2] - 5.0) < 1e-9


def _conv_steps_tree(tmp_path, isym="3"):
    """convergence_test_all with modes 5 and 7 at +-0.025 and +-0.05, one run unfinished; ZFS_hyp ISYM 2."""
    root = tmp_path / "NV_512" / "convergence_test_all"
    for mode in (5, 7):
        for p in (-0.05, -0.025, 0.025, 0.05):
            run = root / "runs" / f"{mode}_pert_{p:g}"
            _outcar(run / "OUTCAR", (1.0, 1.0, 2.0))
            (run / "INCAR").write_text(f"ENCUT = 520\nISYM = {isym}\n")
    (root / "runs" / "7_pert_0.05" / "OUTCAR").write_text(" running...\n")
    (root / "run_convergence.sh").write_text("DISPLACEMENT=${DISPLACEMENT:-normalized}\n")
    gs = tmp_path / "NV_512" / "ZFS_hyp"
    _outcar(gs / "OUTCAR", (1.0, 1.0, 4.0))
    (gs / "INCAR").write_text("ENCUT = 520\nISYM = 2\n")
    return root


def test_conv_scan_prints_summary_and_saves_nothing(tmp_path, capsys):
    _conv_steps_tree(tmp_path)
    _run_conv(args=_conv_args(tmp_path, Path("__auto__"), yes=False, scan=True), defect_folder=tmp_path / "NV_512")
    out = capsys.readouterr().out
    assert "Available convergence data:" in out
    row = next(line for line in out.splitlines() if line.startswith("NV_512"))
    assert row.split() == ["NV_512", "test_all", "all_bands", "5", "7", "±0.025", "±0.05", "7/8"]
    assert "1 run(s) without a ZFS tensor: 7_pert_0.05" in out
    assert str(tmp_path / "NV_512" / "ZFS_hyp" / "OUTCAR") in out
    assert "NV_512 test_all: step * u/|u|" in out and "[trees' job scripts]" in out
    # the runs' ISYM 3 against the ground state's ISYM 2 is flagged
    assert "ISYM 3 (ground state 2)" in out and "!!" in out
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("answer", ["", "y"])
def test_conv_prompt_yes_saves(tmp_path, monkeypatch, caplog, answer):
    _conv_steps_tree(tmp_path)
    _registry(monkeypatch, tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)
    with caplog.at_level("INFO", logger="spinlabs"):
        _run_conv(args=_conv_args(tmp_path, Path("__auto__"), yes=False), defect_folder=tmp_path / "NV_512")
    out = next((tmp_path / "data").rglob("NV_512_*_conv_test_all.npz"))
    assert f"saved convergence data: {out} (7/8 runs finished" in caplog.text
    data = np.load(out, allow_pickle=True)
    assert list(data["perts"]) == [-0.05, -0.025, 0.025, 0.05]
    assert np.isnan(data["tensors"][1, 3]).all() and abs(data["tensors"][0, 0, 2, 2] - 2.0) < 1e-9
    assert json.loads(str(data["metadata"]))["displacement"] == "normalized"


def test_conv_prompt_no_saves_nothing(tmp_path, monkeypatch, caplog):
    _conv_steps_tree(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    with caplog.at_level("INFO", logger="spinlabs"):
        _run_conv(args=_conv_args(tmp_path, Path("__auto__"), yes=False), defect_folder=tmp_path / "NV_512")
    assert "nothing saved" in caplog.text
    assert not (tmp_path / "data").exists()


def test_conv_folder_without_runs_is_skipped(tmp_path, capsys):
    _conv_tree(tmp_path)
    (tmp_path / "NV_512" / "convergence_test_occup" / "runs").mkdir(parents=True)
    _run_conv(args=_conv_args(tmp_path, None), defect_folder=tmp_path / "NV_512")
    assert len(list((tmp_path / "data").rglob("*.npz"))) == 1
    assert "test_occup" not in capsys.readouterr().out.split("Available convergence data:")[1]


def test_multiple_folders_and_methods(tmp_path, monkeypatch):
    _tree(tmp_path)
    db = _registry(monkeypatch, tmp_path)
    clv = tmp_path / "ClV_128"
    clv.mkdir()
    (clv / "first_order" / "pert_0.025" / "defect_band_approx").mkdir(parents=True)
    (clv / "first_order" / "pert_0.025" / "defect_band_approx" / "OUTCAR").touch()
    args = _make_args(tmp_path)
    args.defect_folder = [tmp_path / "NV_512", clv]
    args.method = ["all", "approx"]
    with _patch(lambda **kw: _raw(order=kw["order"], calc_method=kw["calc_method"])):
        _run(args)
    day = date.today().strftime("%Y%m%d")
    assert (tmp_path / "data" / "NV_512" / "pert_0.025" / "all_bands" / day / "NV_512_raw_zfs_data_all_bands_12d.npz").exists()
    assert (tmp_path / "data" / "NV_512" / "pert_0.025" / "defect_band_approx" / day / "NV_512_raw_zfs_data_defect_band_approx_12d.npz").exists()
    assert (tmp_path / "data" / "ClV_128" / "pert_0.025" / "all_bands" / day / "NV_512_raw_zfs_data_all_bands_1d.npz").exists()
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT defect, cell_size, calc_method, pert_order, pert_scale FROM files ORDER BY id").fetchall()
    assert len(rows) == 4
    assert rows[2] == ("ClV", 128, "all_bands", "[1]", 0.025)


def test_missing_pert_warns_and_skips(tmp_path, caplog):
    _tree(tmp_path, perts=(0.025,))
    args = _make_args(tmp_path)
    args.pert = [0.025, 0.1]
    with _patch():
        _run(args)
    assert _warned(caplog, "--pert 0.1 not found")
    files = list((tmp_path / "data").rglob("*.npz"))
    assert len(files) == 1


def _order_dicts(n_modes=2, *, first=True, second=True, pert_si=1e10):
    """Build (first_order, second_order) dicts for RawZFSData tests."""
    fo = (
        {
            i: {
                "tensor": np.diag([-1000.0, -1000.0, 2000.0]) * 6.626e-28,
                "pert": pert_si,
                "symmetry": "A1",
                "ipr": 0.5,
            }
            for i in range(n_modes)
        }
        if first
        else {}
    )
    so = (
        {
            (i, j): {
                "tensor": np.diag([-1000.0, -1000.0, 2000.0]) * 6.626e-28,
                "pert": (pert_si, pert_si),
                "symmetry": ("A1", "A1") if i == j else ("A1", "Ex"),
                "ipr": (0.5, 0.5),
            }
            for i in range(n_modes)
            for j in range(i, n_modes)
        }
        if second
        else {}
    )
    return fo, so


def test_register_only(tmp_path, caplog):
    from spinlabs.cli.package import _register_only
    from spinlabs.registry import ensure_db

    root = tmp_path / "data"
    day = date.today().strftime("%Y%m%d")
    d1 = root / "NV_512" / "pert_0.1" / "all_bands" / day
    d1.mkdir(parents=True)
    np.savez(
        d1 / "NV_512_raw_zfs_data_all_bands_12d.npz",
        defect="NV", cell_size=512, pert_scale=0.1, calc_method="all_bands", order=12,
        first_order=np.array({}, dtype=object),
    )
    (root / "stray.npz").write_bytes(b"")  # junk with no metadata

    _register_only(root)
    assert _warned(caplog, "stray.npz")

    conn = ensure_db()
    rows = conn.execute(
        "SELECT path, kind, defect, cell_size, calc_method, pert_order, pert_scale FROM files"
    ).fetchall()
    assert len(rows) == 1
    path, kind, defect, cell, method, order, pert = rows[0]
    assert kind == "packaged_zfs"
    assert (defect, cell, method, order, pert) == ("NV", 512, "all_bands", "[1, 2]", 0.1)
    assert path.endswith("NV_512_raw_zfs_data_all_bands_12d.npz")


def test_register_only_stale_combined_metadata(tmp_path):
    """Old files store combined data with order=2; the npz content wins over that metadata."""
    from spinlabs.models import RawZFSData
    from spinlabs.registry import ensure_db

    gs = ZFSTensor(matrix=np.diag([-1000.0, -1000.0, 2000.0]), unit="MHz")
    first_order, second_order = _order_dicts()
    raw = RawZFSData(
        defect="ClV",
        cell_size=128,
        pert_scale=0.025,
        calc_method="all_bands",
        order=2,  # stale metadata from the old packaging code
        ground_state_zfs=gs,
        first_order=first_order,
        second_order=second_order,
    )
    out = tmp_path / "ClV_128_raw_zfs_data_all_bands_12d.npz"
    raw.save(out)
    conn = ensure_db()
    parser = argparse.ArgumentParser()
    build_package_parser(parser)
    args = parser.parse_args(["--register-only", str(out.parent)])
    args.func(args)

    rows = conn.execute(
        "SELECT pert_order FROM files WHERE kind='packaged_zfs'"
    ).fetchall()
    assert json.loads(rows[0][0]) == [1, 2]


def test_register_only_orders_from_content(tmp_path):
    """2d-only file: empty first_order -> [2] regardless of stale metadata."""
    from spinlabs.registry import ensure_db

    gs = ZFSTensor(matrix=np.diag([-1000.0, -1000.0, 2000.0]), unit="MHz")
    first_order, second_order = _order_dicts(first=False)
    raw = RawZFSData(
        defect="ClV",
        cell_size=128,
        pert_scale=0.025,
        calc_method="all_bands",
        order=2,
        ground_state_zfs=gs,
        first_order={},  # explicitly empty: no 1d data in this file
        second_order=second_order,
    )
    out = tmp_path / "ClV_128_raw_zfs_data_all_bands_2d.npz"
    raw.save(out)
    conn = ensure_db()
    parser = argparse.ArgumentParser()
    build_package_parser(parser)
    args = parser.parse_args(["--register-only", str(out.parent)])
    args.func(args)
    rows = conn.execute("SELECT pert_order FROM files WHERE kind='packaged_zfs'").fetchall()
    assert json.loads(rows[0][0]) == [2]


def test_package_copies_full_phonon_spectrum(tmp_path, monkeypatch):
    """The full-spectrum phonon npz lands in data/<defect>_<cell>/ and is registered."""
    _tree(tmp_path)
    db = _registry(monkeypatch, tmp_path)
    (tmp_path / "NV_512" / "phonon_data.npz").write_bytes(b"full")
    (tmp_path / "NV_512" / "phonon_data_sym_n1022.npz").write_bytes(b"sym")
    with _patch(lambda **kw: _raw(order=kw["order"])):
        _run(_make_args(tmp_path))

    dest = tmp_path / "data" / "NV_512" / "phonon_data.npz"
    assert dest.read_bytes() == b"full"
    row = sqlite3.connect(db).execute(
        "SELECT path, defect, cell_size FROM files WHERE kind = 'phonon'"
    ).fetchone()
    assert row == (str(dest.resolve()), "NV", 512)


def test_package_finds_phonon_in_defect_data_folder(tmp_path, monkeypatch):
    """<defect>/data/phonon_data.npz (the VASP-tree location) is preferred over the defect folder."""
    _tree(tmp_path)
    _registry(monkeypatch, tmp_path)
    (tmp_path / "NV_512" / "data").mkdir()
    (tmp_path / "NV_512" / "data" / "phonon_data.npz").write_bytes(b"full")
    (tmp_path / "NV_512" / "data" / "phonon_data_sym126.npz").write_bytes(b"sym")
    with _patch(lambda **kw: _raw(order=kw["order"])):
        _run(_make_args(tmp_path))
    assert (tmp_path / "data" / "NV_512" / "phonon_data.npz").read_bytes() == b"full"


def test_package_finds_differently_named_phonon(tmp_path, monkeypatch):
    _tree(tmp_path)
    _registry(monkeypatch, tmp_path)
    (tmp_path / "NV_512" / "phonons.npz").write_bytes(b"full")
    (tmp_path / "NV_512" / "phonons_sym.npz").write_bytes(b"sym")
    with _patch(lambda **kw: _raw(order=kw["order"])):
        _run(_make_args(tmp_path))
    assert (tmp_path / "data" / "NV_512" / "phonon_data.npz").read_bytes() == b"full"


def test_package_skips_symmetry_reduced_only_phonon(tmp_path, monkeypatch, caplog):
    _tree(tmp_path)
    _registry(monkeypatch, tmp_path)
    (tmp_path / "NV_512" / "phonons_sym.npz").write_bytes(b"sym")
    with _patch(lambda **kw: _raw(order=kw["order"])):
        _run(_make_args(tmp_path))
    assert _warned(caplog, "symmetry-reduced")
    assert not (tmp_path / "data" / "NV_512" / "phonon_data.npz").exists()


def test_scan_does_not_copy_phonon(tmp_path, monkeypatch):
    _tree(tmp_path)
    _registry(monkeypatch, tmp_path)
    (tmp_path / "NV_512" / "phonon_data.npz").write_bytes(b"full")
    with _patch(lambda **kw: _raw(order=kw["order"])):
        _run(_make_args(tmp_path, scan=True))
    assert not (tmp_path / "data").exists()


def test_register_only_registers_packaged_phonon(tmp_path, monkeypatch):
    from spinlabs.cli.package import _register_only

    db = _registry(monkeypatch, tmp_path)
    dest = tmp_path / "data" / "ClV_128" / "phonon_data.npz"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"full")
    _register_only(tmp_path / "data")
    row = sqlite3.connect(db).execute("SELECT kind, defect, cell_size FROM files").fetchone()
    assert row == ("phonon", "ClV", 128)


def test_register_only_registers_convergence_files(tmp_path, monkeypatch):
    from spinlabs.cli.package import _register_only
    from spinlabs.registry import find_convergence

    db = _registry(monkeypatch, tmp_path)
    _conv_steps_tree(tmp_path)
    _run_conv(args=_conv_args(tmp_path, Path("__auto__")), defect_folder=tmp_path / "NV_512")
    packed = next((tmp_path / "data").rglob("*_conv_test_all.npz"))
    # an older pack whose metadata has no method: taken from the file name instead
    old = tmp_path / "data" / "V_Si_128" / "defect_band_approx" / "20261009" / "V_Si_128_20261009_conv_test_occup.npz"
    old.parent.mkdir(parents=True)
    np.savez(old, modes=np.arange(1), perts=np.ones(1), tensors=np.zeros((1, 1, 3, 3)),
             metadata=np.array(json.dumps({"defect": None, "cell": None})))
    sqlite3.connect(db).execute("DELETE FROM files").connection.commit()

    _register_only(tmp_path / "data")
    rows = sqlite3.connect(db).execute("SELECT kind, defect, cell_size, calc_method FROM files ORDER BY defect")
    assert rows.fetchall() == [("convergence", "NV", 512, "all_bands"), ("convergence", "V_Si", 128,
                                                                         "defect_band_approx")]
    conn = sqlite3.connect(db)
    assert find_convergence(conn, defects=["NV"]) == [packed.resolve()]


@pytest.mark.parametrize("no_mass_weighted", [False, True])
def test_no_mass_weighted_flag_is_stored(tmp_path, monkeypatch, no_mass_weighted):
    _tree(tmp_path)
    _registry(monkeypatch, tmp_path)
    args = _make_args(tmp_path)
    args.no_mass_weighted = no_mass_weighted
    with _patch(lambda **kw: _raw(order=kw["order"])):
        _run(args)
    (path,) = (tmp_path / "data").rglob("*.npz")
    assert RawZFSData.load(path).mass_weighted is (not no_mass_weighted)


def test_empty_datasets_are_not_saved(tmp_path, monkeypatch, caplog):
    """A pert folder without OUTCARs parses to no tensors; it is skipped, not saved empty."""
    _tree(tmp_path)
    _registry(monkeypatch, tmp_path)
    empty = RawZFSData(defect="NV", cell_size=512, pert_scale=0.025, calc_method="all_bands", order=1)
    with _patch(lambda **kw: empty):
        _run(_make_args(tmp_path))
    assert _warned(caplog, "no ZFS data parsed")
    assert not list((tmp_path / "data").rglob("*.npz"))


def test_scan_flags_entries_with_the_wrong_zfs_shape(tmp_path, monkeypatch, capsys):
    """The scan summary runs the D-tensor shape check against the defect's phonon labels."""
    from spinlabs.models import PhononSpectrum

    _tree(tmp_path)
    _registry(monkeypatch, tmp_path)
    PhononSpectrum(
        frequencies_mev=np.array([20.0, 40.0, 40.0]), symmetries=["A1", "Ex", "Ey"],
        eigenvectors=np.ones((3, 1, 3)) / np.sqrt(3), atom_frac_coords=np.zeros((1, 3)), atom_symbols=["C"],
        atomic_masses=np.array([12.011]), lattice=np.eye(3) * 3.57,
    ).save(tmp_path / "NV_512" / "phonon_data.npz")
    gs = np.diag([-1000.0, -1000.0, 2000.0])
    raw = RawZFSData(
        defect="NV", cell_size=512, pert_scale=0.025, calc_method="all_bands", order=1,
        ground_state_zfs=ZFSTensor(matrix=gs, unit="MHz"),
        first_order={0: {"tensor": gs + np.diag([5.0, -15.0, 10.0]), "unit": "MHz"}},  # A1 mode, D_xx != D_yy
    )
    with _patch(lambda **kw: raw):
        args = _make_args(tmp_path, orders="first", scan=True)
        _run(args)
    out = capsys.readouterr().out
    assert "D-tensor shape check" in out
    assert "NV_512 all_bands pert 0.025: 1 suspicious run folder(s): 1" in out
