"""``spinlabs table t1`` builds a LaTeX T1 table from the latest registry runs."""

import numpy as np
import pytest

import spinlabs.cli as cli
import spinlabs.registry as registry
from spinlabs.tables import _fmt_ms, collect_outcar_rows, collect_t1_rows, format_t1_table, t1_at

ZFS_BLOCK = (
    " Spin-spin contribution to zero-field splitting tensor (MHz)\n"
    " -------------------------------------------------------------\n"
    "      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz\n"
    " -------------------------------------------------------------\n"
    "   -100.00   -100.00    200.00      0.00      0.00      0.00\n"
    " -------------------------------------------------------------\n"
)


def test_outcar_rows_scale_the_corrected_d_by_the_spin_from_nupdown(tmp_path):
    for name, nupdown in (("V_Si_128", 3), ("NV_64", 2)):
        for sub in ("ZFS_hyp", "ZFS_occup"):
            run = tmp_path / name / sub
            run.mkdir(parents=True)
            (run / "OUTCAR").write_text(ZFS_BLOCK)
            (run / "INCAR").write_text(f"NUPDOWN = {nupdown}\n")
    d = {(r["defect"], r["column"]): r["d_mhz"] for r in collect_outcar_rows(tmp_path)}
    # D = 3/2 D_zz = 300 MHz before the correction
    assert d[("V_Si", "contaminated")] == pytest.approx(300.0)
    assert d[("V_Si", "corrected")] == pytest.approx(300.0 * 2 / 3)
    assert d[("NV", "corrected")] == pytest.approx(300.0 * 3 / 2)


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
        r"NV & --- & 2870~\cite{loubser_electron_1978} & $2950^{(b)}$, $2830^{(c)}$ & $3050^{(b)}$ & "
        r"$8.30^{(b)}$, $7.53^{(c)}$ & $11.7^{(b)}$ & $4120^{(b)}$ & --- \\" in tex
    )
    assert r"ClV & kk & --- & $1440^{(a)}$ & --- &" in tex
    assert "(a) cell size 128; (b) cell size 512; (c) cell size 64" in tex
    assert r"\cmidrule(lr){6-7} \cmidrule(lr){8-9}" in tex
    assert r"& Pumped & Other & Pumped & Other \\" in tex
    assert r"optically pumped initial state ($m_s=0$ for $S=1$)" in tex
    assert r"\begin{table*}" in tex and r"$T_{1,\text{UE}}$" in tex


def test_defect_table_without_t1_is_d_only():
    from spinlabs.tables import merge_rows, render_defect_table

    tex = render_defect_table(merge_rows([_d("V_Si", "128", "corrected", 3.249)]), include_t1=False)
    assert r"V$_{\mathrm{Si}}$ & k & 35.0~\cite{ivady_identification_2017} & $3.25^{(a)}$ & --- \\" in tex
    assert "T_{1" not in tex


def test_defect_table_layout_matches_the_paper():
    """Caption first, UE before AE, DiV shown as VV, label tab:zfs_d, a single-column table."""
    from spinlabs.tables import merge_rows, render_defect_table

    rows = [_d("DiV", "128", "corrected", 1502.7), _d("DiV", "128", "contaminated", 1775.9),
            _d("NV", "512", "corrected", 2945.6)]
    lines = render_defect_table(merge_rows(rows), include_t1=False).splitlines()
    assert lines[0] == r"\begin{table}[htbp]"
    assert lines[1] == (r"  \caption{Zero-field splitting parameter $D$ of the treated point defects. "
                        r"The superscripts refer to different defect supercells: (a) cell size 128; (b) cell size 512.}")
    assert lines[2] == r"  \centering"
    assert lines[5] == r"    Defect & Configuration & $D_\text{experiment}$ & $D_\text{UE}$ & $D_\text{AE}$ \\"
    assert r"    VV & kk & 1305~\cite{falk_polytype_2013} & $1500^{(a)}$ & $1780^{(a)}$ \\" in lines
    assert r"    NV & --- & 2870~\cite{loubser_electron_1978} & $2950^{(b)}$ & --- \\" in lines
    assert lines[-2:] == [r"  \label{tab:zfs_d}", r"\end{table}"]


def test_cli_defect_table_reads_t1_from_registry(tmp_path):
    conn = registry.ensure_db()
    _add_run(conn, tmp_path, "a", t1_s=1.0, pert=0.025)
    _add_run(conn, tmp_path, "b", t1_s=2.0, pert=0.1)  # newest run wins across pert-scales
    out = tmp_path / "defects.tex"

    cli.main(["table", "defects", "-o", str(out)])

    assert r"NV & --- & 2870~\cite{loubser_electron_1978} & --- & --- & --- & --- & $2000^{(a)}$ & --- \\" in out.read_text()


def test_cli_defect_table_shows_spin_3_2_t1(tmp_path):
    conn = registry.ensure_db()
    _add_run(conn, tmp_path, "p", t1_s=0.04, defect="V_Si", cell=128, init_state="ms_pm1_2")
    _add_run(conn, tmp_path, "o", t1_s=0.02, defect="V_Si", cell=128, init_state="ms_pm3_2")
    out = tmp_path / "defects.tex"

    cli.main(["table", "defects", "-o", str(out)])

    assert r"V$_{\mathrm{Si}}$ & k & 35.0~\cite{ivady_identification_2017} & --- & --- & --- & --- & " \
           r"$40.0^{(a)}$ & $20.0^{(a)}$ \\" in out.read_text()


def test_defect_table_shows_spin_3_2_t1_under_pumped_and_other():
    from spinlabs.tables import T1Row, merge_rows, render_defect_table

    t1_rows = [T1Row("V_Si", 128, "defect_band_approx", None, {"ms_pm1_2": 0.0412, "ms_pm3_2": 0.0205}),
               T1Row("NV", 512, "defect_band_approx", None, {"ms_0": 0.008297, "ms_1": 0.01172})]
    tex = render_defect_table(merge_rows([_d("V_Si", "128", "corrected", 3.249)], t1_rows))

    assert (r"V$_{\mathrm{Si}}$ & k & 35.0~\cite{ivady_identification_2017} & $3.25^{(b)}$ & --- & "
            r"$41.2^{(b)}$ & $20.5^{(b)}$ & --- & --- \\" in tex)
    assert r"$m_s=\pm 1/2$ for $S=3/2$" in tex and r"$m_s=\pm 3/2$ for $S=3/2$" in tex


def test_defect_table_other_shows_both_s1_states_only_when_they_differ():
    from spinlabs.tables import T1Row, merge_rows, render_defect_table

    t1_rows = [T1Row("NV", 512, "all_bands", None, {"ms_1": 0.01172, "ms_0": 0.008297, "ms_-1": 0.0135}),
               T1Row("NV", 64, "all_bands", None, {"ms_1": 0.01, "ms_0": 0.007534, "ms_-1": 0.01}),
               T1Row("DiV", 128, "all_bands", None, {"ms_-1": 0.002})]
    tex = render_defect_table(merge_rows([], t1_rows))

    assert r"& $8.30^{(b)}$, $7.53^{(c)}$ & $11.7 / 13.5^{(b)}$, $10.0^{(c)}$ \\" in tex
    assert r"VV & kk & 1305~\cite{falk_polytype_2013} & --- & --- & --- & --- & --- & $2.00^{(a)}$ \\" in tex  # only ms_-1 solved


def test_defect_table_warns_for_a_spin_without_a_state_mapping(caplog):
    from spinlabs.tables import T1Row, merge_rows, render_defect_table

    t1_rows = [T1Row("X", 64, "all_bands", None, {"ms_0": 1.0, "ms_pm1": 2.0, "ms_pm2": 3.0})]
    tex = render_defect_table(merge_rows([], t1_rows))

    assert r"X & --- & --- & --- & --- & --- & --- & --- & --- \\" in tex
    assert "no T1 state mapping for spin S = 2.0" in caplog.text


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


def test_defect_table_takes_d_from_the_package_of_the_chosen_t1_run(tmp_path, caplog):
    """D and T1 describe the same data: an older package with another ground state is not used."""
    from spinlabs.models import RawZFSData, ZFSTensor

    def package(date, d_mhz, pert):
        gs = np.diag([-d_mhz / 3, -d_mhz / 3, 2 * d_mhz / 3])
        raw = RawZFSData(defect="NV", cell_size=64, pert_scale=pert, calc_method="all_bands", order=1,
                         ground_state_zfs=ZFSTensor(matrix=gs, unit="MHz"),
                         first_order={0: {"tensor": gs, "unit": "MHz"}})
        path = raw.save(tmp_path / "data" / "NV_64" / f"pert_{pert}" / "all_bands" / date / "NV_64_raw.npz")
        registry.register_file(conn, path, "packaged_zfs", defect="NV", cell_size=64, calc_method="all_bands",
                               pert_order=[1], pert_scale=pert)
        return path

    conn = registry.ensure_db()
    new = package("20261021", 3000.0, 0.1)
    package("20261016", 3100.0, -0.025)  # older ground state, registered later
    run_dir = _add_run(conn, tmp_path, "r", t1_s=1.0, cell=64, pert=0.1)
    conn.execute("UPDATE runs SET zfs_files = ? WHERE run_dir = ?", (f'["{new}"]', str(run_dir)))
    conn.commit()
    out = tmp_path / "defects.tex"
    cli.main(["table", "defects", "-o", str(out)])
    assert "$3000^{(a)}$" in out.read_text() and "3100" not in out.read_text()
