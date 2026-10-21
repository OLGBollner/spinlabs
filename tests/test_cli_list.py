"""``spinlabs list``: the latest packages, runs and plots per configuration, and what is out of date."""

import os

from spinlabs.cli import main
from spinlabs.registry import ensure_db, finish_run, register_file, register_run


def _package(conn, tmp_path, date: str, orders=(1, 2)):
    path = tmp_path / "data" / "NV_512" / "pert_0.025" / "all_bands" / date / "NV_512_raw_zfs_data_all_bands_12d.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    register_file(conn, path, "packaged_zfs", defect="NV", cell_size=512, calc_method="all_bands",
                  pert_order=list(orders), pert_scale=0.025)
    return path


def _run(conn, tmp_path, name: str, package, started: str):
    run_dir = tmp_path / "runs" / "NV_512" / "pert_0.025" / "all_bands" / name
    run_dir.mkdir(parents=True)
    run_id = register_run(conn, run_dir=run_dir, zfs_files=[str(package)], defect="NV", cell=512,
                          method="all_bands", order=[1, 2], pert=0.025, git_hash="abc1234")
    finish_run(conn, run_id, "done")
    conn.execute("UPDATE runs SET started_at = ? WHERE id = ?", (started, run_id))
    conn.commit()
    return run_dir


def _figure(tmp_path, kind: str, run_name: str):
    path = tmp_path / "figures" / kind / "NV_512" / "pert_0.025" / "all_bands" / run_name / "plot.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")


def test_run_from_an_older_package_is_marked(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    conn = ensure_db()
    old = _package(conn, tmp_path, "20261015")
    _run(conn, tmp_path, "20261015_120000", old, "2026-10-15T10:00:00+00:00")
    _package(conn, tmp_path, "20261016")
    main(["list", "--runs"])
    out = capsys.readouterr().out
    assert "OLDER package" in out and "1 run(s) analysed an older package" in out


def test_plots_of_an_older_run_are_named(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    conn = ensure_db()
    pkg = _package(conn, tmp_path, "20261016")
    _run(conn, tmp_path, "20261016_100000", pkg, "2026-10-16T08:00:00+00:00")
    _run(conn, tmp_path, "20261020_100000", pkg, "2026-10-20T08:00:00+00:00")
    _figure(tmp_path, "coupling", "20261016_100000")
    main(["list", "--plot"])
    out = capsys.readouterr().out
    assert "20261020_100000  none; last plotted from older run 20261016_100000: coupling" in out
    _figure(tmp_path, "t1", "20261020_100000")
    main(["list", "--plot"])
    assert "t1 (1," in capsys.readouterr().out


def test_default_lists_packages_runs_and_plots(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    conn = ensure_db()
    pkg = _package(conn, tmp_path, "20261016")
    os.utime(pkg, (1_760_000_000, 1_760_000_000))
    _run(conn, tmp_path, "20261016_100000", pkg, "2026-10-16T08:00:00+00:00")
    main(["list"])
    out = capsys.readouterr().out
    for title in ("Packaged data", "Runs (latest", "Plots of the latest runs"):
        assert title in out
    assert "newest package" in out and "data/NV_512/pert_0.025/all_bands/20261016" in out


def test_unregistered_package_on_disk_is_reported(tmp_path, capsys, monkeypatch):
    from spinlabs.registry import unregistered_packages

    monkeypatch.chdir(tmp_path)
    conn = ensure_db()
    _package(conn, tmp_path, "20261020")
    copied = tmp_path / "data" / "NV_512" / "pert_0.025" / "all_bands" / "20261021" / "NV_512_raw_zfs_data_all_bands_12d.npz"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(b"x")  # copied in, never registered
    assert unregistered_packages(conn) == [copied]
    main(["list", "--package"])
    out = capsys.readouterr().out
    assert "1 packaged file(s) on disk are not in the registry" in out
    assert f"spinlabs package --register-only {tmp_path / 'data' / 'NV_512'}" in out
    main(["list", "--package", "--defect", "ClV"])
    assert "not in the registry" not in capsys.readouterr().out
