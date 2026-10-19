"""Regression tests for PR #50 review fixes.

Covers:
- find_packaged combined-order matching (order=12 must match [1, 2] rows).
- calc_method filtering in find_packaged.
- CLI `run --method all` normalization to 'all_bands'.
- Pipeline must not downgrade a packaged_zfs row to raw_zfs after a run.
- --order 2/12 without second-order data raises instead of silently degrading.
"""
import argparse
import numpy as np
import pytest

from spinlabs.models import PhononSpectrum, RawZFSData, ZFSTensor
from spinlabs.pipeline import run_full_pipeline
from spinlabs.registry import ensure_db, find_packaged, register_file


def _register(conn, path, *, kind="packaged_zfs", orders, method="all_bands", pert=0.025):
    path.write_bytes(b"placeholder")  # file must exist for find_packaged
    return register_file(
        conn,
        path,
        kind,
        defect="NV",
        cell_size=512,
        calc_method=method,
        pert_order=orders,
        pert_scale=pert,
    )


def test_find_packaged_combined_order_matches_both_rows(tmp_path):
    """order=12 (combined) must match rows storing [1] and [1, 2]; order=1 both too."""
    conn = ensure_db()
    f1d = tmp_path / "NV_512_raw_zfs_data_all_bands_1d.npz"
    f2d = tmp_path / "NV_512_raw_zfs_data_all_bands_2d.npz"
    fcomb = tmp_path / "NV_512_raw_zfs_data_all_bands_12d.npz"
    _register(conn, f1d, orders=[1])
    _register(conn, f2d, orders=[2])
    _register(conn, fcomb, orders=[1, 2])

    combined = find_packaged(conn, pert_order=12)
    assert {p.name for p in combined} == {fcomb.name}

    only_first = find_packaged(conn, pert_order=1)
    assert {p.name for p in only_first} == {f1d.name, fcomb.name}

    only_second = find_packaged(conn, pert_order=2)
    assert {p.name for p in only_second} == {f2d.name, fcomb.name}


def test_find_packaged_filters_by_calc_method(tmp_path):
    conn = ensure_db()
    f_all = tmp_path / "NV_512_raw_zfs_data_all_bands_1d.npz"
    f_approx = tmp_path / "NV_512_raw_zfs_data_defect_band_approx_1d.npz"
    _register(conn, f_all, orders=[1], method="all_bands")
    _register(conn, f_approx, orders=[1], method="defect_band_approx")

    hits = find_packaged(conn, calc_method="all_bands")
    assert [p.name for p in hits] == [f_all.name]


def test_find_packaged_latest_keeps_newest_package_per_configuration(tmp_path):
    conn = ensure_db()
    old, new = tmp_path / "20261015", tmp_path / "20261016"
    old.mkdir(), new.mkdir()
    name = "NV_512_raw_zfs_data_all_bands_12d.npz"
    _register(conn, old / name, orders=[1, 2])
    _register(conn, new / name, orders=[1, 2])
    other = tmp_path / "NV_512_raw_zfs_data_all_bands_12d_pert01.npz"
    _register(conn, other, orders=[1, 2], pert=0.1)  # another configuration

    assert set(find_packaged(conn, latest=True)) == {new / name, other}
    # Re-registering the old package (e.g. package --register-only) does not make it newest.
    _register(conn, old / name, orders=[1, 2])
    assert set(find_packaged(conn, latest=True)) == {new / name, other}
    assert len(find_packaged(conn)) == 3


def test_find_packaged_returns_already_analyzed_data(tmp_path):
    from spinlabs.registry import finish_run, register_run

    conn = ensure_db()
    path = tmp_path / "NV_512_raw_zfs_data_all_bands_12d.npz"
    _register(conn, path, orders=[1, 2])
    run_id = register_run(conn, run_dir=tmp_path / "run", zfs_files=[str(path.resolve())], method="all_bands")
    finish_run(conn, run_id, "done")

    assert find_packaged(conn, latest=True) == [path]


def test_run_query_normalizes_method(tmp_path, monkeypatch):
    """`spinlabs run --query --method all` must query calc_method='all_bands'."""
    from spinlabs.cli import run as run_mod

    captured = {}

    def fake_find(conn, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(run_mod, "find_packaged", fake_find, raising=False)
    monkeypatch.setattr(
        "spinlabs.registry.find_packaged", fake_find, raising=False
    )

    args = argparse.Namespace(
        defect=["NV"],
        cell_size=[512],
        method=["all"],
        pert_scale=None,
        order=None,
        output_root=str(tmp_path / "runs"),
    )
    run_mod._run_query(args)
    assert captured["calc_method"] == "all_bands"


def _make_raw_zfs(tmp_path, second=True, name="NV_512_raw_zfs_data_all_bands_12d.npz"):
    n_modes = 4
    pert_si = 0.025 * 1.60218e-22
    gs = ZFSTensor(matrix=np.diag([-1000.0, -1000.0, 2000.0]), unit="MHz")
    first_order = {
        i: {
            "tensor": np.diag([-1000.0, -1000.0, 2000.0]) * 6.626e-28,
            "pert": pert_si,
            "symmetry": "A1" if i % 2 == 0 else "Ex",
            "ipr": 0.5,
        }
        for i in range(n_modes)
    }
    second_order = (
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
    raw = RawZFSData(
        defect="NV",
        cell_size=512,
        pert_scale=0.025,
        calc_method="all_bands",
        order=1,
        ground_state_zfs=gs,
        first_order=first_order,
        second_order=second_order,
    )
    f = tmp_path / name
    raw.save(f)
    return f


def _make_phonon(tmp_path):
    n = 4
    freqs = np.linspace(10.0, 40.0, n)
    freqs[1] = freqs[2] = 30.0  # degenerate pair for e-pair completeness
    spec = PhononSpectrum(
        frequencies_mev=freqs,
        symmetries=["A1", "Ex", "Ey", "A1"],
        iprs=np.ones(n) * 0.5,
        eigenvectors=np.ones((n, 1, 3)),
        atom_frac_coords=np.zeros((1, 3)),
        atom_symbols=["C"],
        atomic_masses=np.array([12.0]),
        lattice=np.eye(3) * 5.0,
    )
    f = tmp_path / "phonon_data.npz"
    spec.save(f)
    return f


def test_pipeline_does_not_downgrade_packaged_row(tmp_path):
    """After a run over a packaged file, the registry row must stay packaged_zfs
    with its pert metadata intact, and the run must be registered."""
    conn = ensure_db()
    raw_file = _make_raw_zfs(tmp_path)
    ph_file = _make_phonon(tmp_path)
    register_file(
        conn,
        raw_file,
        "packaged_zfs",
        defect="NV",
        cell_size=512,
        calc_method="all_bands",
        pert_order=[1, 2],
        pert_scale=0.025,
    )

    run_full_pipeline(
        raw_zfs_file=raw_file,
        phonon_file=ph_file,
        calc_method="all_bands",
        save_plots=False,
        output_root=tmp_path / "runs",
        temperatures=[10.0],
    )

    row = conn.execute(
        "SELECT kind, defect, cell_size, calc_method, pert_order, pert_scale "
        "FROM files WHERE path = ?",
        (str(raw_file.resolve()),),
    ).fetchone()
    assert row is not None
    kind, defect, cell, method, orders, pert = row
    assert kind == "packaged_zfs"
    assert (defect, cell, method, pert) == ("NV", 512, "all_bands", 0.025)
    import json as _json

    assert _json.loads(orders) == [1, 2]


def test_pipeline_still_registers_unregistered_raw_file(tmp_path):
    """A raw file not already in the registry gets a raw_zfs row as before."""
    conn = ensure_db()
    raw_file = _make_raw_zfs(tmp_path, second=False, name="NV_512_raw_zfs_data_all_bands_1d.npz")
    ph_file = _make_phonon(tmp_path)

    run_full_pipeline(
        raw_zfs_file=raw_file,
        phonon_file=ph_file,
        calc_method="all_bands",
        save_plots=False,
        output_root=tmp_path / "runs",
        temperatures=[10.0],
    )

    row = conn.execute(
        "SELECT kind FROM files WHERE path = ?", (str(raw_file.resolve()),)
    ).fetchone()
    assert row is not None and row[0] == "raw_zfs"


def test_order_2_without_second_order_raises(tmp_path):
    """--order 2 on 1d-only input must fail loudly, not silently degrade to order 1."""
    raw_file = _make_raw_zfs(tmp_path, second=False, name="NV_512_raw_zfs_data_all_bands_1d.npz")
    ph_file = _make_phonon(tmp_path)

    with pytest.raises(ValueError, match="no second-order"):
        run_full_pipeline(
            raw_zfs_file=raw_file,
            phonon_file=ph_file,
            order=2,
            calc_method="all_bands",
            save_plots=False,
            output_root=tmp_path / "runs",
            temperatures=[10.0],
        )


def test_find_runs_order_matching(tmp_path):
    """Regression (SPI-19): --order filtering must not silently drop every run.

    ``order`` is a JSON list of the orders the run analyzed; the filter is a
    subset match (a --order 1 query also finds a combined [1, 2] run).
    """
    import spinlabs.registry as registry

    conn = registry.ensure_db()
    d1 = tmp_path / "NV_512_pert_0.025_1"
    d2 = tmp_path / "NV_512_pert_0.025_2"
    registry.register_run(conn, run_dir=d1, defect="NV", cell=512, order=1,
                          pert=0.025, method="all_bands",
                          npz_files=["t1_relaxation.npz"])
    registry.register_run(conn, run_dir=d2, defect="NV", cell=512, order=2,
                          pert=0.025, method="all_bands",
                          npz_files=["t1_relaxation.npz"])
    conn.execute("UPDATE runs SET status='done'")
    conn.commit()

    assert [p.name for p in registry.find_runs(conn, [("order", "1")])] == [d1.name]
    assert [p.name for p in registry.find_runs(conn, [("order", "2")])] == [d2.name]
    assert registry.find_runs(conn, [("order", "12")]) == []
    # A run registered with the combined order 12 must be found by --order 12.
    d12 = tmp_path / "NV_512_pert_0.025_12"
    registry.register_run(conn, run_dir=d12, defect="NV", cell=512, order=12,
                          pert=0.025, method="all_bands",
                          npz_files=["t1_relaxation.npz"])
    conn.execute("UPDATE runs SET status='done' WHERE run_dir LIKE '%_12'")
    conn.commit()
    assert [p.name for p in registry.find_runs(conn, [("order", "12")])] == [d12.name]
    # Subset semantics: --order 1 also finds the combined run.
    assert d12.name in [p.name for p in registry.find_runs(conn, [("order", "1")])]
    # A legacy CSV-style row (scalar orders rendered as '1+2') keeps matching
    # via the migration fallback.
    conn.execute(
        'insert into runs (run_dir, status, defect, cell, "order", started_at)'
        " values (?, 'done', 'NV-', 64, '1+2', datetime('now'))",
        (str(tmp_path / "NV_64_pert_0.025"),),
    )
    conn.commit()
    assert str(tmp_path / "NV_64_pert_0.025") in [
        str(p) for p in registry.find_runs(conn, [("order", "12")])
    ]


def test_find_runs_latest_picks_most_recent(tmp_path):
    """Regression (SPI-19): --latest must return the newest matching run.

    started_at differs between the two runs (the common case), so the old
    ``ORDER BY started_at, id DESC`` picked the OLDEST one. Also check the
    started_at tie, broken by the higher (later-registered) id.
    """
    import spinlabs.registry as registry

    conn = registry.ensure_db()
    d1 = tmp_path / "NV_512_pert_0.025_1"
    d2 = tmp_path / "NV_512_pert_0.025_2"
    for d in (d1, d2):
        registry.register_run(conn, run_dir=d, defect="NV", cell=512, order=1,
                              pert=0.025, method="all_bands",
                              npz_files=["t1_relaxation.npz"])
    conn.execute("UPDATE runs SET status='done', started_at=datetime('now', '-1 day') WHERE id = 1")
    conn.commit()

    conn.execute("UPDATE runs SET status='done' WHERE id = 2")
    conn.commit()

    latest = registry.find_runs(conn, [("defect", "NV")], latest=True)
    assert [p.name for p in latest] == [d2.name]

    conn.execute("UPDATE runs SET started_at = datetime('now') WHERE id IN (1, 2)")
    conn.commit()
    conn.commit()
    latest_tie = registry.find_runs(conn, [("defect", "NV")], latest=True)
    assert [p.name for p in latest_tie] == [d2.name]


def test_parse_orders_expands_legacy_and_rejects_unknown():
    from spinlabs.registry import parse_orders

    assert parse_orders(12) == [1, 2]
    assert parse_orders([2, 1, 2]) == [1, 2]
    assert parse_orders("1+2") == [1, 2]
    with pytest.raises(ValueError, match="unknown perturbation order 3"):
        parse_orders([3])


def test_register_run_rejects_unknown_order(tmp_path):
    """register_run must not store an order outside {1, 2}."""
    import spinlabs.registry as registry

    conn = registry.ensure_db()
    with pytest.raises(ValueError, match="unknown perturbation order 3"):
        registry.register_run(conn, run_dir=tmp_path / "r", order=[3])


def test_find_runs_empty_order_filter_raises(tmp_path):
    """An empty order filter must fail loudly, not silently match every run."""
    import spinlabs.registry as registry

    conn = registry.ensure_db()
    with pytest.raises(SystemExit, match="empty --order"):
        registry.find_runs(conn, [("order", [])])


def test_find_runs_npz_files_matches_single_name(tmp_path):
    """npz_files matches when any stored name equals the filter value."""
    import spinlabs.registry as registry

    conn = registry.ensure_db()
    d = tmp_path / "NV_512_pert_0.025"
    registry.register_run(conn, run_dir=d, npz_files=["a.npz", "b.npz"])
    conn.execute("UPDATE runs SET status='done'")
    conn.commit()
    assert [p.name for p in registry.find_runs(conn, [("npz_files", "B.npz")])] == [d.name]
    assert registry.find_runs(conn, [("npz_files", "a.npz+b.npz")]) == []


def test_find_runs_repeated_field_ors_values(tmp_path):
    """Repeating a filter field matches any of its values; distinct fields AND."""
    import spinlabs.registry as registry

    conn = registry.ensure_db()
    for defect, pert in [("NV", 0.025), ("ClV", 0.1), ("DiV", 0.025), ("NV", 0.1)]:
        registry.register_run(conn, run_dir=tmp_path / f"{defect}_{pert}", defect=defect, cell=128,
                              order=1, pert=pert, method="all_bands", npz_files=["t1_relaxation.npz"])
    conn.execute("UPDATE runs SET status='done'")
    conn.commit()

    def names(f):
        return sorted(p.name for p in registry.find_runs(conn, f))

    assert names([("defect", "NV"), ("defect", "ClV")]) == ["ClV_0.1", "NV_0.025", "NV_0.1"]
    assert names([("defect", "NV"), ("defect", "DiV"), ("pert", 0.025)]) == ["DiV_0.025", "NV_0.025"]
    assert names([("pert", 0.1), ("pert", 0.025), ("defect", "ClV")]) == ["ClV_0.1"]
