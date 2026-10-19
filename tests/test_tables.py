"""``spinlabs table t1`` builds a LaTeX T1 table from the latest registry runs."""

import numpy as np
import pytest

import spinlabs.cli as cli
import spinlabs.registry as registry
from spinlabs.tables import _fmt_ms, collect_t1_rows, format_t1_table, t1_at


def _add_run(conn, tmp_path, name, *, t1_s, init_state="ms_0", method="all_bands", defect="NV",
             cell=512, pert=0.1, legacy=False, order=(1, 2)):
    """Register a completed run whose T1 is ``t1_s`` seconds at every temperature."""
    run_dir = tmp_path / "runs" / f"{defect}_{cell}" / method / f"pert_{pert}" / name
    run_dir.mkdir(parents=True)
    temps = np.array([0.0, 100.0, 200.0, 301.0])
    np.savez(
        run_dir / "t1_relaxation.npz",
        defect=defect,
        cell_size=cell,
        calc_method=method,
        init_state=init_state,
        temperatures=temps,
        t1_fit=np.full_like(temps, t1_s),
    )
    meta = {} if legacy else {"defect": defect, "cell": cell, "pert": pert, "order": list(order)}
    run_id = registry.register_run(conn, run_dir=run_dir, method=method, init_state=init_state, **meta)
    registry.finish_run(conn, run_id, "done")
    return run_dir


def test_t1_at_interpolates_log_linearly(tmp_path):
    path = tmp_path / "t1.npz"
    np.savez(path, temperatures=np.array([100.0, 300.0]), t1_fit=np.array([1.0, 100.0]))
    assert t1_at(path, 200.0) == pytest.approx(10.0)
    assert np.isnan(t1_at(path, 400.0))


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(1.352, "1350"), (0.09208, "92.1"), (0.007534, "7.53"), (0.013, "13.0"), (0.09996, "100"),
     (2106.937, r"$2.11\times10^{6}$"), (float("nan"), "---")],
)
def test_fmt_ms_three_significant_figures(seconds, text):
    assert _fmt_ms(seconds) == text


def test_latest_run_wins_per_state_method_and_pert(tmp_path):
    conn = registry.ensure_db()
    _add_run(conn, tmp_path, "a", t1_s=1.0)
    _add_run(conn, tmp_path, "b", t1_s=2.0)  # newer ms_0 run replaces "a"
    _add_run(conn, tmp_path, "c", t1_s=3.0, init_state="ms_1")
    _add_run(conn, tmp_path, "d", t1_s=4.0, method="defect_band_approx")
    _add_run(conn, tmp_path, "e", t1_s=5.0, cell=64, pert=0.025, legacy=True)

    rows = collect_t1_rows(conn)

    assert [(r.label, r.pert) for r in rows] == [
        ("NV 512", 0.1), ("NV 512 (spin corrected)", 0.1), ("NV 64", 0.025),
    ]
    assert rows[0].t1 == {"ms_0": pytest.approx(2.0), "ms_1": pytest.approx(3.0)}
    assert rows[2].t1 == {"ms_0": pytest.approx(5.0)}  # metadata from npz + path


def test_table_layout_and_pert_column():
    from spinlabs.tables import T1Row

    rows = [T1Row("NV", 512, "all_bands", 0.1, {"ms_1": 1.352, "ms_0": 4.118, "ms_-1": 1.352})]
    tex = format_t1_table(rows)
    assert r"\multicolumn{3}{c}{$T_1$ (ms)}" in tex
    assert r"NV 512 & 1350 & 4120 & 1350 \\" in tex
    assert "Pert" not in tex.split(r"\caption")[0]

    rows.append(T1Row("NV", 64, "defect_band_approx", 0.025, {"ms_0": 0.007534}))
    tex = format_t1_table(rows)
    assert r"NV 64 (spin corrected) & 0.025 & --- & 7.53 & --- \\" in tex


def test_cli_writes_filtered_table(tmp_path):
    conn = registry.ensure_db()
    _add_run(conn, tmp_path, "a", t1_s=1.0)
    _add_run(conn, tmp_path, "b", t1_s=2.0, defect="DiV", cell=128)
    out = tmp_path / "t1.tex"

    cli.main(["table", "t1", "--defect", "NV", "-o", str(out)])

    tex = out.read_text()
    assert r"NV 512 & --- & 1000 & --- \\" in tex
    assert "DiV" not in tex


def test_cli_errors_without_runs():
    with pytest.raises(SystemExit, match="no completed runs"):
        cli.main(["table", "t1"])


def _d(defect, cell, column, d_mhz):
    return {"defect": defect, "column": column, "cell": cell,
            "d_mhz": d_mhz, "source": f"{defect}_{cell}_{column}"}


def test_defect_table_appends_t1_per_cell_with_superscripts():
    from spinlabs.tables import T1Row, merge_rows, render_defect_table

    d_rows = [
        _d("NV", "512", "corrected", 2945.556), _d("NV", "64", "corrected", 2832.941),
        _d("NV", "512", "contaminated", 3045.316), _d("ClV", "128", "corrected", 1439.75),
    ]
    t1_rows = [
        T1Row("NV", 512, "defect_band_approx", None, {"ms_1": 0.01172, "ms_0": 0.008297, "ms_-1": 0.01172}),
        T1Row("NV", 64, "defect_band_approx", None, {"ms_0": 0.007534}),
        T1Row("NV", 512, "all_bands", None, {"ms_0": 4.118}),
    ]
    tex = render_defect_table(merge_rows(d_rows, t1_rows))

    # 4H-SiC block before diamond; letters in order of first appearance.
    assert tex.index("4H-SiC") < tex.index("Diamond")
    assert (
        r"NV & --- & --- & 2950$^{(b)}$, 2830$^{(c)}$ & 3050$^{(b)}$ & 11.7$^{(b)}$ & "
        r"8.30$^{(b)}$, 7.53$^{(c)}$ & 11.7$^{(b)}$ & --- & 4120$^{(b)}$ & --- \\" in tex
    )
    assert r"ClV & --- & --- & 1440$^{(a)}$ & --- &" in tex
    assert "(a) cell size 128; (b) cell size 512; (c) cell size 64" in tex
    assert r"\cmidrule(lr){6-8}" in tex


def test_defect_table_without_t1_is_d_only():
    from spinlabs.tables import merge_rows, render_defect_table

    tex = render_defect_table(merge_rows([_d("V_Si", "128", "corrected", 3.249)]), include_t1=False)
    assert r"V$_\mathrm{Si}$ & --- & --- & 3.25$^{(a)}$ & --- \\" in tex
    assert "T_{1" not in tex


def test_cli_defect_table_reads_t1_from_registry(tmp_path):
    conn = registry.ensure_db()
    _add_run(conn, tmp_path, "a", t1_s=1.0, pert=0.025)
    _add_run(conn, tmp_path, "b", t1_s=2.0, pert=0.1)  # newest run wins across pert-scales
    out = tmp_path / "defects.tex"

    cli.main(["table", "defects", "-o", str(out)])

    assert r"NV & --- & --- & --- & --- & --- & --- & --- & --- & 2000$^{(a)}$ & --- \\" in out.read_text()


def test_t1_rows_read_every_state_file_of_one_run(tmp_path):
    conn = registry.ensure_db()
    run_dir = tmp_path / "runs" / "NV_64" / "all_bands" / "pert_0.1" / "run"
    run_dir.mkdir(parents=True)
    temps = np.array([0.0, 301.0])
    for state, t1_s in (("ms_1", 0.02), ("ms_0", 0.04), ("ms_-1", 0.02)):
        np.savez(run_dir / f"t1_relaxation_{state}.npz", defect="NV", cell_size=64, calc_method="all_bands",
                 init_state=state, temperatures=temps, t1_fit=np.full_like(temps, t1_s))
    run_id = registry.register_run(conn, run_dir=run_dir, method="all_bands", init_state=["ms_1", "ms_0", "ms_-1"],
                                   defect="NV", cell=64, pert=0.1)
    registry.finish_run(conn, run_id, "done")

    (row,) = collect_t1_rows(conn)

    assert row.t1 == {"ms_1": pytest.approx(0.02), "ms_0": pytest.approx(0.04), "ms_-1": pytest.approx(0.02)}
    assert r"NV 64 & 20.0 & 40.0 & 20.0 \\" in format_t1_table([row])


def _add_stateless_run(conn, tmp_path, name, init_state):
    """A run whose t1_relaxation.npz lacks the init_state key."""
    run_dir = tmp_path / "runs" / "NV_64" / "pert_0.1" / "all_bands" / name
    run_dir.mkdir(parents=True)
    temps = np.array([0.0, 301.0])
    np.savez(run_dir / "t1_relaxation.npz", defect="NV", cell_size=64, calc_method="all_bands",
             temperatures=temps, t1_fit=np.full_like(temps, 0.05))
    run_id = registry.register_run(conn, run_dir=run_dir, method="all_bands", init_state=init_state,
                                   defect="NV", cell=64, pert=0.1)
    registry.finish_run(conn, run_id, "done")


def test_t1_file_without_state_takes_the_runs_single_state(tmp_path):
    conn = registry.ensure_db()
    _add_stateless_run(conn, tmp_path, "run", ["ms_1"])

    (row,) = collect_t1_rows(conn)

    assert row.t1 == {"ms_1": pytest.approx(0.05)}


def test_t1_file_without_state_is_skipped_when_run_solved_several(tmp_path):
    conn = registry.ensure_db()
    _add_stateless_run(conn, tmp_path, "run", ["ms_1", "ms_0"])

    assert collect_t1_rows(conn) == []


def test_merged_rows_prefer_runs_with_second_order_and_report_the_choice(tmp_path, caplog):
    conn = registry.ensure_db()
    _add_run(conn, tmp_path, "full", t1_s=0.005, pert=0.1)
    _add_run(conn, tmp_path, "first_only", t1_s=5000.0, pert=0.025, order=(1,))  # newer, but no 2nd order

    (row,) = collect_t1_rows(conn, by_pert=False)

    assert row.t1 == {"ms_0": pytest.approx(0.005)}
    assert "runs at pert 0.025 orders [1], pert 0.1 orders [1, 2]; using pert 0.1 orders [1, 2]" in caplog.text


def test_newer_complete_run_still_wins(tmp_path, caplog):
    conn = registry.ensure_db()
    _add_run(conn, tmp_path, "old", t1_s=0.005, pert=0.1)
    _add_run(conn, tmp_path, "new", t1_s=0.002, pert=0.025)

    (row,) = collect_t1_rows(conn, by_pert=False)

    assert row.t1 == {"ms_0": pytest.approx(0.002)}
    assert "using pert 0.025 orders [1, 2]" in caplog.text


def test_reruns_of_one_configuration_are_not_reported(tmp_path, caplog):
    conn = registry.ensure_db()
    _add_run(conn, tmp_path, "a", t1_s=0.005)
    _add_run(conn, tmp_path, "b", t1_s=0.004)

    (row,) = collect_t1_rows(conn, by_pert=False)

    assert row.t1 == {"ms_0": pytest.approx(0.004)}
    assert "runs at" not in caplog.text
