"""``spinlabs plot t1 --query`` resolves runs from the registry.

It shares the ``--defect/--cell-size/--method/--order/--pert-scale`` filter
flags with ``run --query``. Rows are written through :func:`register_run` and
read back through :func:`find_runs` so the tests exercise the exact format the
pipeline produces (``order`` as a JSON list such as ``[1, 2]``).
"""


from pathlib import Path

import numpy as np
import pytest

import spinlabs.cli as cli
import spinlabs.cli.plotting as cli_plot
import spinlabs.registry as registry


def _parse(argv):
    parser = cli.build_parser()
    return parser.parse_args(["plot", "t1", *argv])


@pytest.fixture
def registry_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / "registry"))
    conn = registry.ensure_db()
    run_dir = tmp_path / "runs" / "NV_512_pert_0.025"
    run_dir.mkdir(parents=True)
    T = np.linspace(100.0, 500.0, 5)
    np.savez(
        run_dir / "t1_relaxation.npz",
        temperatures=T,
        t1_fit=T,
        t1_eigenval=T,
        defect="NV",
        cell_size="512",
        calc_method="all_bands",
        init_state="ms_1",
        times=T,
        P=T,
        rates=T,
        coupling=T,
        modes=np.arange(3),
        freqs=T[:3],
        couplings=T[:3],
    )
    run_id = registry.register_run(
        conn,
        run_dir=run_dir,
        method="all_bands",
        init_state="ms_1",
        defect="NV",
        cell=512,
        order=[1, 2],
        pert=0.025,
        npz_files=[run_dir / "t1_relaxation.npz"],
    )
    registry.finish_run(conn, run_id, "done")
    return conn, run_dir


def _run_cli(monkeypatch, argv, tmp_path):
    """Execute the t1 subcommand with plotting stubbed; return the resolved inputs."""
    saved = {}
    monkeypatch.setattr(cli_plot, "plot_t1", lambda paths, **k: saved.update(paths=paths) or (None, None))
    cli_plot._run_t1(_parse(argv))
    return saved["paths"]


def test_query_resolves_single_run(registry_db, monkeypatch, tmp_path):
    conn, run_dir = registry_db
    paths = _run_cli(
        monkeypatch,
        [
            "--query", "--defect", "NV", "--cell-size", "512", "--method", "all",
            "--order", "1", "2", "--pert-scale", "0.025", "-o", str(tmp_path / "out.png"),
        ],
        tmp_path,
    )
    assert paths == [run_dir / "t1_relaxation.npz"]


def test_query_no_match_is_error(registry_db, monkeypatch, tmp_path):
    conn, _ = registry_db
    with pytest.raises(SystemExit, match="no completed runs"):
        _run_cli(
            monkeypatch,
            ["--query", "--defect", "ClV", "-o", str(tmp_path / "out.png")],
            tmp_path,
        )


def test_query_and_inputs_are_mutually_exclusive(monkeypatch):
    with pytest.raises(SystemExit, match="not both"):
        _run_cli(monkeypatch, ["--query", "some.npz"], None)



def test_order_column_is_quoted(registry_db):
    conn, run_dir = registry_db
    dirs = registry.find_runs(conn, [("order", [1, 2])])
    assert dirs == [run_dir]


def _add_run(conn, run_dir, *, order=1, status="done", started_at=None, **kw):
    run_id = registry.register_run(
        conn, run_dir=run_dir, defect="NV", cell=512, order=order, pert=0.025,
        method="all_bands", init_state="ms_1", **kw,
    )
    if status != "running":
        registry.finish_run(conn, run_id, status)
    if started_at is not None:
        conn.execute("UPDATE runs SET started_at = ? WHERE id = ?", (started_at, run_id))
        conn.commit()
    return run_id


def test_order_12_roundtrip(tmp_path, monkeypatch):
    """--order 12 (combined-order convention) must match the row register_run wrote."""
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / "registry"))
    conn = registry.ensure_db()
    _add_run(conn, tmp_path / "r_a", order=1)
    _add_run(conn, tmp_path / "r_c", order=12)
    dirs = registry.find_runs(conn, [("order", "12")])
    assert [d.name for d in dirs] == ["r_c"]
    assert registry.find_runs(conn, [("order", "1")])[0].name == "r_a"


def test_order_list_roundtrip(tmp_path, monkeypatch):
    """A pipeline-registered order list ([1, 2]) must be queryable as '1 2'."""
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / "registry"))
    conn = registry.ensure_db()
    _add_run(conn, tmp_path / "r_c", order=[1, 2])
    assert registry.find_runs(conn, [("order", "1 2")])[0].name == "r_c"
    assert registry.find_runs(conn, [("order", "12")])[0].name == "r_c"


def test_latest_returns_newest(tmp_path, monkeypatch):
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / "registry"))
    conn = registry.ensure_db()
    _add_run(conn, tmp_path / "r_oct01", started_at="2026-10-01 12:00:00")
    _add_run(conn, tmp_path / "r_oct05", started_at="2026-10-05 12:00:00")
    _add_run(conn, tmp_path / "r_oct10", started_at="2026-10-10 12:00:00")
    dirs = registry.find_runs(conn, [("defect", "NV")], latest=True)
    assert [d.name for d in dirs] == ["r_oct10"]
    # Non-latest queries keep returning all matching runs (oldest first).
    assert [d.name for d in registry.find_runs(conn, [("defect", "NV")])] == [
        "r_oct01", "r_oct05", "r_oct10",
    ]


def test_latest_keeps_newest_per_configuration(tmp_path, monkeypatch):
    """--latest over several pert-scales/defects returns the newest run of each, not one run."""
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / "registry"))
    conn = registry.ensure_db()
    runs = [
        ("nv_01_old", "NV", 0.1, "2026-10-01 12:00:00"),
        ("nv_01_new", "NV", 0.1, "2026-10-05 12:00:00"),
        ("nv_0025_new", "NV", 0.025, "2026-10-03 12:00:00"),
        ("nv_0025_old", "NV", 0.025, "2026-10-02 12:00:00"),
        ("clv_01", "ClV", 0.1, "2026-10-09 12:00:00"),
    ]
    for name, defect, pert, started_at in runs:
        run_id = registry.register_run(
            conn,
            run_dir=tmp_path / name,
            defect=defect,
            cell=512,
            order=1,
            pert=pert,
            method="all_bands",
            init_state="ms_0",
        )
        registry.finish_run(conn, run_id, "done")
        conn.execute("UPDATE runs SET started_at = ? WHERE id = ?", (started_at, run_id))
    conn.commit()
    filters = [("defect", "NV"), ("pert", "0.1"), ("pert", "0.025")]
    dirs = registry.find_runs(conn, filters, latest=True)
    assert sorted(d.name for d in dirs) == ["nv_0025_new", "nv_01_new"]


def test_latest_ignores_running_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / "registry"))
    conn = registry.ensure_db()
    _add_run(conn, tmp_path / "r_done", started_at="2026-10-01 12:00:00")
    _add_run(conn, tmp_path / "r_running", status="running", started_at="2026-10-11 12:00:00")
    dirs = registry.find_runs(conn, [("defect", "NV")], latest=True)
    assert [d.name for d in dirs] == ["r_done"]


def test_no_match_invalid_order_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / "registry"))
    conn = registry.ensure_db()
    _add_run(conn, tmp_path / "r_a", order=1)
    with pytest.raises(SystemExit):
        registry.find_runs(conn, [("order", "3")])


@pytest.mark.parametrize("subcmd", ["coupling", "rates"])
def test_plot_run_subcommand_query(subcmd, registry_db, tmp_path, monkeypatch, capsys):
    """plot coupling/rates --query resolve registry runs like plot t1."""
    monkeypatch.chdir(tmp_path)
    from spinlabs.cli import build_parser

    called = []

    def fake_plot(run_dir, out_dir, fmt, dpi, show, title=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        called.append(run_dir)
        return []

    import spinlabs.cli.plotting as plotting
    monkeypatch.setattr(plotting, "plot_run_coupling", fake_plot)
    monkeypatch.setattr(plotting, "plot_run_rates", fake_plot)
    args = build_parser().parse_args(["plot", subcmd, "--query", "--defect", "NV", "--cell-size", "512"])
    args.func(args)
    assert len(called) == 1
    assert called[0].name == "NV_512_pert_0.025"
    assert (tmp_path / "figures" / subcmd / "NV_512" / "pert_0.025" / "all_bands" / "NV_512_pert_0.025").is_dir()


@pytest.mark.parametrize("subcmd", ["coupling", "rates"])
def test_plot_run_subcommand_latest_plots_each_configuration(subcmd, tmp_path, monkeypatch):
    """--latest plots the newest run of every matching configuration, each into its own figures."""
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / "registry"))
    monkeypatch.chdir(tmp_path)
    conn = registry.ensure_db()
    runs = [
        ("p01_old", 0.1, "2026-10-01 12:00:00"),
        ("p01_new", 0.1, "2026-10-05 12:00:00"),
        ("p0025_new", 0.025, "2026-10-03 12:00:00"),
    ]
    for name, pert, started_at in runs:
        run_id = registry.register_run(
            conn,
            run_dir=tmp_path / "runs" / name,
            defect="NV",
            cell=512,
            order=1,
            pert=pert,
            method="all_bands",
            init_state="ms_0",
        )
        registry.finish_run(conn, run_id, "done")
        conn.execute("UPDATE runs SET started_at = ? WHERE id = ?", (started_at, run_id))
    conn.commit()

    from spinlabs.cli import build_parser
    import spinlabs.cli.plotting as plotting

    titles = {}

    def fake_plot(run_dir, out_dir, fmt, dpi, show, title=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        titles[run_dir.name] = title
        return []

    monkeypatch.setattr(plotting, f"plot_run_{subcmd}", fake_plot)
    args = build_parser().parse_args(
        ["plot", subcmd, "--query", "--defect", "NV", "--pert-scale", "0.1", "0.025", "--latest"]
    )
    args.func(args)
    assert sorted(titles) == ["p0025_new", "p01_new"]
    assert titles["p01_new"] == "NV 512, all bands, pert 0.1 (p01_new)"
    root = tmp_path / "figures" / subcmd / "NV_512"
    assert (root / "pert_0.1" / "all_bands" / "p01_new").is_dir()
    assert (root / "pert_0.025" / "all_bands" / "p0025_new").is_dir()


@pytest.mark.parametrize("subcmd", ["coupling", "rates", "t1"])
def test_plot_query_no_match_errors(subcmd, registry_db, monkeypatch):
    from spinlabs.cli import build_parser

    args = build_parser().parse_args(["plot", subcmd, "--query", "--defect", "ClV"])
    with pytest.raises(SystemExit, match="no completed runs"):
        args.func(args)


@pytest.mark.parametrize("subcmd", ["coupling", "rates"])
def test_plot_path_and_query_conflict(subcmd, registry_db):
    from spinlabs.cli import build_parser

    args = build_parser().parse_args(["plot", subcmd, "some_dir", "--query", "--defect", "NV"])
    with pytest.raises(SystemExit, match="not both"):
        args.func(args)


def test_plot_query_multi_valued_filters_become_repeated_fields():
    from spinlabs.cli import build_parser
    from spinlabs.cli.plotting import _query_filters

    args = build_parser().parse_args(
        ["plot", "rates", "--query", "--defect", "NV", "ClV", "--pert-scale", "0.1", "0.025", "--method", "all"]
    )
    assert _query_filters(args) == [
        ("defect", "NV"), ("defect", "ClV"), ("method", "all_bands"), ("pert", 0.1), ("pert", 0.025),
    ]


@pytest.mark.parametrize("given,expected", [("0", "ms_0"), ("1", "ms_1"), ("-1", "ms_-1"), ("+1", "ms_1"), ("ms_0", "ms_0"), ("ms-1", "ms_-1")])
def test_plot_t1_init_state_accepts_shorthand(given, expected):
    from spinlabs.cli import build_parser

    args = build_parser().parse_args(["plot", "t1", "--query", "--init-state", given])
    assert args.init_state == [expected]


def test_plot_t1_init_state_filters_query_and_curves(tmp_path):
    import numpy as np
    from spinlabs.cli import build_parser
    from spinlabs.cli.plotting import _query_filters
    from spinlabs.plotter import plot_t1

    args = build_parser().parse_args(["plot", "t1", "--query", "--defect", "ClV", "--init-state", "0", "-1"])
    assert ("init_state", "ms_0") in _query_filters(args) and ("init_state", "ms_-1") in _query_filters(args)

    for state in ("ms_1", "ms_0", "ms_-1"):
        np.savez(tmp_path / f"t1_relaxation_{state}.npz", temperatures=[10.0, 20.0], t1_fit=[1.0, 0.5],
                 defect="ClV", cell_size=128, calc_method="all_bands", init_state=state)
    _, ax = plot_t1([tmp_path], init_states=["ms_0"])
    assert len(ax.get_lines()) == 1


def test_run_figure_subdir_puts_run_name_last(tmp_path):
    """Figure folders mirror the run layout with the run (timestamp) folder as the leaf."""
    from spinlabs.cli.plotting import _run_figure_subdir

    run = tmp_path / "runs" / "NV_64" / "pert_0.1" / "all_bands" / "20261015_094009"
    assert _run_figure_subdir(run, ("NV", 64, "all_bands", 0.1)) == Path("NV_64/pert_0.1/all_bands/20261015_094009")
    # Unregistered runs fall back to their own path layout, or just their name.
    assert _run_figure_subdir(run, None) == Path("NV_64/pert_0.1/all_bands/20261015_094009")
    assert _run_figure_subdir(tmp_path / "old_data", None) == Path("old_data")


def test_combined_figure_subdir_joins_differing_levels(tmp_path, monkeypatch):
    """T1 figures of several runs share one folder: common levels kept, differing ones joined with '+'."""
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / "registry"))
    from spinlabs.cli.plotting import _combined_figure_subdir

    runs = tmp_path / "runs"
    a = runs / "NV_64" / "pert_0.1" / "all_bands" / "20261015_094009"
    b = runs / "NV_512" / "pert_0.1" / "all_bands" / "20261016_071501"
    assert _combined_figure_subdir([a]) == Path("NV_64/pert_0.1/all_bands/20261015_094009")
    assert _combined_figure_subdir([a, b]) == Path("NV_64+NV_512/pert_0.1/all_bands/20261016_071501")
    # Loose npz folders outside the run layout add no level.
    assert _combined_figure_subdir([tmp_path]) == Path()
    assert _combined_figure_subdir([tmp_path, a]) == Path("NV_64/pert_0.1/all_bands/20261015_094009")


def test_latest_prefers_the_run_with_more_orders(tmp_path, monkeypatch):
    """A newer first-order-only run never hides a run with second-order data of the same configuration."""
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / "registry"))
    conn = registry.ensure_db()
    _add_run(conn, tmp_path / "r_1d_old", order=1, started_at="2026-10-01 12:00:00")
    _add_run(conn, tmp_path / "r_12d", order=[1, 2], started_at="2026-10-05 12:00:00")
    _add_run(conn, tmp_path / "r_1d_new", order=1, started_at="2026-10-09 12:00:00")
    assert [d.name for d in registry.find_runs(conn, [("defect", "NV")], latest=True)] == ["r_12d"]
    # Equally complete runs: the newest wins.
    _add_run(conn, tmp_path / "r_12d_new", order=[1, 2], started_at="2026-10-10 12:00:00")
    assert [d.name for d in registry.find_runs(conn, [("defect", "NV")], latest=True)] == ["r_12d_new"]
    # Asking for first order only still shows the more complete run, which holds first order too.
    assert [d.name for d in registry.find_runs(conn, [("order", "1")], latest=True)] == ["r_12d_new"]


def test_latest_keeps_a_first_order_run_when_nothing_more_complete_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / "registry"))
    conn = registry.ensure_db()
    _add_run(conn, tmp_path / "r_1d_old", order=1, started_at="2026-10-01 12:00:00")
    _add_run(conn, tmp_path / "r_1d_new", order=1, started_at="2026-10-09 12:00:00")
    assert [d.name for d in registry.find_runs(conn, [("defect", "NV")], latest=True)] == ["r_1d_new"]
