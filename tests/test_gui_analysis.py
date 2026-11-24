"""Headless tests for the GUI's analysis, catalogue, state and plot builders."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import spinlabs.registry as registry
from spinlabs.gui.analysis import PackSession
from spinlabs.gui.catalog import Catalog
from spinlabs.gui.plots import (
    ChannelColumnPlot,
    NormColumnPlot,
    build_channel_summary_figure,
    build_mode_figure,
    build_norm_summary_figure,
    message_figure,
)
from spinlabs.gui.state import GuiState, TabState, load_state, save_state
from spinlabs.pack_convergence import ConvergenceScan, save_convergence

D0 = np.diag([-950.0, -1000.0, 1950.0])
OUTCAR = (
    " Spin-spin contribution to zero-field splitting tensor (MHz)\n ---\n"
    "      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz\n ---\n"
    "   -950.0  -1000.0  1950.0  0.0  0.0  0.0\n ---\n"
)


def _save(
    path: Path,
    *,
    modes=(5, 7),
    perts=(-0.2, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.2),
    tensors=None,
    ground=True,
    method="AO",
    key="test_AO",
    displacement="normalized",
    cell="512",
    spin_sublevel=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    if tensors is None:
        tensors = np.array([[D0 + q * (m - 4) * np.eye(3) for q in perts] for m in modes])
    gs = None
    if ground:
        gs = path.parent / "gs_OUTCAR"
        gs.write_text(OUTCAR)
    scan = ConvergenceScan(
        path.parent, list(modes), list(perts), np.asarray(tensors, dtype=float),
        np.zeros((len(modes), len(perts))), len(modes) * len(perts), displacement=displacement,
    )
    return save_convergence(
        scan, path, ground_state=gs, defect="NV", cell=cell, method=method, method_key=key,
        spin_sublevel=spin_sublevel,
    )


def test_availability_handles_more_modes_than_steps(tmp_path):
    modes = tuple(range(9))
    perts = (-0.1, -0.05, 0.05, 0.1)
    tensors = np.array([[D0 + q * np.eye(3) for q in perts] for _ in modes])
    tensors[0, 1] = np.nan  # an unfinished run at -0.05
    pack = _save(tmp_path / "NV_512.npz", modes=modes, perts=perts, tensors=tensors)
    session = PackSession(pack, "AO", "normalized")
    availability = session.availability()
    assert availability.unfinished == [-0.05]
    assert availability.has_q0 and availability.has_plus and availability.has_minus
    assert availability.modes == list(modes)


def test_modes_fall_back_to_numbers_without_phonon_data(tmp_path, monkeypatch):
    pack = _save(tmp_path / "NV_512.npz")
    monkeypatch.setattr("spinlabs.gui.analysis.mode_details", lambda *a, **k: {})
    session = PackSession(pack, "AO", "normalized")
    entries = session.modes()
    assert [e.mode for e in entries] == [5, 7]
    assert all(e.symmetry is None and e.frequency is None for e in entries)
    assert session.mode_warnings


def test_modes_use_registered_phonon_details(tmp_path, monkeypatch):
    pack = _save(tmp_path / "NV_512.npz")
    monkeypatch.setattr(
        "spinlabs.gui.analysis.mode_details",
        lambda *a, **k: {5: ("Ex", 10.0), 7: ("A1", 20.0)},
    )
    entries = PackSession(pack, "AO", "normalized").modes()
    assert [(e.mode, e.symmetry, e.frequency) for e in entries] == [(5, "Ex", 10.0), (7, "A1", 20.0)]


def test_norm_without_q0_reports_a_warning_instead_of_raising(tmp_path):
    pack = _save(tmp_path / "NV_512.npz", perts=(-0.1, 0.1), ground=False)
    session = PackSession(pack, "AO", "normalized")
    result = session.norm({5, 7})
    assert not result.rows
    assert any("no Q=0" in warning for warning in result.warnings)


def test_plot_builders_make_figures(tmp_path):
    pack = _save(tmp_path / "NV_512.npz")
    session = PackSession(pack, "AO", "normalized")
    modes = {entry.mode: entry for entry in session.modes()}
    forward = session.norm(set(modes), central=False)
    central = session.norm(set(modes), central=True)
    norm = build_norm_summary_figure([NormColumnPlot("AO", modes, forward, central)])
    assert len(norm.axes) == 2
    # Central includes the 2Q helper step, so the axis must not clip it.
    assert norm.axes[0].get_xlim()[1] > 0.2
    channels = session.channels(set(modes))
    channel_fig = build_channel_summary_figure([ChannelColumnPlot("AO", modes, channels)])
    assert len(channel_fig.axes) == 2
    mode_fig = build_mode_figure(session, 5)
    assert mode_fig.figure is not None
    assert mode_fig.report is not None
    assert mode_fig.extra_lines == []
    quad_fig = build_mode_figure(session, 5, quad=True)
    assert quad_fig.extra_lines
    forward_only = build_mode_figure(session, 5, schemes=("forward",))
    assert forward_only.report is not None
    assert len(forward_only.figure.axes) == 6
    central_only = build_mode_figure(session, 5, schemes=("central",))
    assert central_only.report is not None
    assert len(central_only.figure.axes) == 6
    no_scheme = build_mode_figure(session, 5, schemes=())
    assert no_scheme.report is None
    assert len(no_scheme.figure.axes) == 4
    assert message_figure("empty").axes


def _legend_texts(figure):
    texts = []
    for ax in figure.axes:
        legend = ax.get_legend()
        if legend is not None:
            texts.extend(item.get_text() for item in legend.get_texts())
    return texts


def _mode_colours(figure):
    colours = {}
    for ax in figure.axes:
        for legend in ax.get_children():
            if not isinstance(legend, matplotlib.legend.Legend):
                continue
            for text, handle in zip(legend.get_texts(), legend.legend_handles):
                label = text.get_text()
                if ":" in label or label.isdigit():
                    colours[label.split(":")[0]] = handle.get_color()
    return colours


def test_mode_colour_survives_filtering(tmp_path):
    pack = _save(tmp_path / "NV_512.npz")
    session = PackSession(pack, "AO", "normalized")
    modes = {entry.mode: entry for entry in session.modes()}
    everything = build_norm_summary_figure(
        [NormColumnPlot("AO", modes, session.norm(set(modes), central=False), None)]
    )
    only_one = build_norm_summary_figure(
        [NormColumnPlot("AO", modes, session.norm({7}, central=False), None)]
    )
    all_colours = _mode_colours(everything)
    one_colours = _mode_colours(only_one)
    assert one_colours["7"] == all_colours["7"]


def test_legends_show_only_the_enabled_scheme(tmp_path):
    pack = _save(tmp_path / "NV_512.npz")
    session = PackSession(pack, "AO", "normalized")
    modes = {entry.mode: entry for entry in session.modes()}
    channels = session.channels(set(modes))
    forward_channel = build_channel_summary_figure(
        [ChannelColumnPlot("AO", modes, channels)], schemes=("forward",)
    )
    texts = _legend_texts(forward_channel)
    assert not any("forward" in text.lower() or "central" in text.lower() for text in texts)
    forward = session.norm(set(modes), central=False)
    norm = build_norm_summary_figure([NormColumnPlot("AO", modes, forward, None)])
    texts = _legend_texts(norm)
    assert "+Q" in texts
    assert not any("central" in text.lower() for text in texts)
    central = session.norm(set(modes), central=True)
    norm = build_norm_summary_figure([NormColumnPlot("AO", modes, None, central)])
    texts = _legend_texts(norm)
    assert "+Q" not in texts
    assert any("central" in text.lower() for text in texts)


def test_empty_mode_selection_is_empty_not_all(tmp_path):
    pack = _save(tmp_path / "NV_512.npz")
    session = PackSession(pack, "AO", "normalized")
    assert session.norm(set()).rows == []
    assert session.norm(set()).warnings
    assert session.channels(set()).modes == []


def test_channels_without_q0_report_a_warning(tmp_path):
    pack = _save(tmp_path / "NV_512.npz", perts=(-0.1, 0.1), ground=False)
    session = PackSession(pack, "AO", "normalized")
    analysis = session.channels({5})
    assert analysis.modes == []
    assert any("no Q=0" in warning for warning in analysis.warnings)


def test_catalog_groups_dates_methods_and_latest(tmp_path):
    conn = registry.ensure_db()
    older = _save(tmp_path / "packs" / "20260101" / "NV_512_20260101_conv_test_AO.npz")
    newer = _save(tmp_path / "packs" / "20260201" / "NV_512_20260201_conv_test_AO.npz")
    uo = _save(tmp_path / "uo" / "NV_512_20260115_conv_test_UO.npz", method="UO", key="test_UO")
    for path, method in ((older, "AO"), (newer, "AO"), (uo, "UO")):
        registry.register_file(conn, path, "convergence", defect="NV", cell_size=512, calc_method=method)
    catalog = Catalog(conn)
    assert catalog.defects() == ["NV"]
    assert catalog.methods() == ["AO", "UO"]
    groups = catalog.groups(latest_only=False)
    assert len(groups) == 1
    group = groups[0]
    assert len(group.schemes) == 2
    ao = next(s for s in group.schemes if s.method == "AO")
    assert [item.date for item in ao.dates] == ["20260201", "20260101"]
    latest = catalog.groups(latest_only=True)[0]
    assert len(latest.schemes) == 2
    assert len(next(s for s in latest.schemes if s.method == "AO").dates) == 1
    only_uo = catalog.groups(method="UO")[0]
    assert [s.method for s in only_uo.schemes] == ["UO"]
    assert catalog.group_for_path(older) is not None


def test_catalog_groups_dated_decont_sets(tmp_path):
    conn = registry.ensure_db()
    for date in ("20260101", "20260201"):
        folder = tmp_path / "decont" / date
        folder.mkdir(parents=True)
        path = _save(
            folder / f"NV_64_{date}_conv_test_decont_ms_1.npz", method="full_decont", key="test_decont",
            cell="64", spin_sublevel="ms_1",
        )
        if date == "20260101":
            registry.register_file(
                conn, path, "convergence_sublevel", defect="NV", cell_size=64, calc_method="full_decont",
                spin_sublevel="ms_1",
            )
        else:
            registry.register_file(
                conn, path, "convergence_sublevel", defect="NV", cell_size=64, calc_method="full_decont",
                spin_sublevel="ms_1",
            )
    group = Catalog(conn).groups()[0]
    assert group.schemes[0].label == "full decont"
    assert [item.date for item in group.schemes[0].dates] == ["20260201", "20260101"]


def test_catalog_folds_a_decont_sublevel_folder(tmp_path):
    conn = registry.ensure_db()
    folder = tmp_path / "decont"
    folder.mkdir()
    path = _save(folder / "NV_64_20260101_conv_test_decont_ms_1.npz", method="full_decont", key="test_decont", cell="64")
    registry.register_file(
        conn, path, "convergence_sublevel", defect="NV", cell_size=64, calc_method="full_decont",
        spin_sublevel="ms_1",
    )
    groups = Catalog(conn).groups()
    assert len(groups) == 1
    assert groups[0].cell == "64"
    assert len(groups[0].schemes) == 1
    assert groups[0].schemes[0].label == "full decont"


def test_catalog_legacy_suffixes_are_separate_schemes(tmp_path):
    conn = registry.ensure_db()
    ao = _save(tmp_path / "legacy" / "NV_512_conv_all.npz", method=None, key=None)
    uo = _save(tmp_path / "legacy" / "NV_512_conv_occup.npz", method=None, key=None)
    for path in (ao, uo):
        registry.register_file(conn, path, "convergence", defect="NV", cell_size=512)
    group = Catalog(conn).groups()[0]
    assert sorted(s.label for s in group.schemes) == ["AO", "UO"]


def test_catalog_marks_a_missing_q0_anchor(tmp_path):
    conn = registry.ensure_db()
    path = _save(tmp_path / "NV_512_20260101_conv_test_AO.npz", perts=(-0.1, 0.1), ground=False)
    registry.register_file(conn, path, "convergence", defect="NV", cell_size=512, calc_method="AO")
    group = Catalog(conn).groups()[0]
    assert group.schemes[0].has_q0 is False


def test_catalog_synthetic_group_parses_v_si_name(tmp_path):
    path = tmp_path / "V_Si_128_20260101_conv_test_AO.npz"
    path.write_bytes(b"not an npz")
    catalog = Catalog()
    group = catalog.synthetic_group(path)
    assert group.defect == "V_Si"
    assert group.cell == "128"


def test_state_round_trip(tmp_path):
    path = tmp_path / "gui.json"
    state = GuiState(
        geometry="1200x700",
        filters={"defect": "NV"},
        tabs=[TabState(group_key="NV|512|normalized", visible_modes=[5, 7], active_mode=7)],
    )
    save_state(state, path)
    loaded = load_state(path)
    assert loaded.geometry == "1200x700"
    assert loaded.filters["defect"] == "NV"
    assert loaded.tabs[0].active_mode == 7
    assert loaded.tabs[0].visible_modes == [5, 7]


def test_corrupt_state_is_a_clean_default(tmp_path):
    path = tmp_path / "gui.json"
    path.write_text("{not json")
    state = load_state(path)
    assert state.tabs == []
    assert state.geometry == "1500x950"
