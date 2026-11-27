"""Headless tests of the GUI's comparison views: the production data layer and its figure builders."""

import json
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import spinlabs.registry as registry
from spinlabs import pipeline
from spinlabs.gui import production_plots as pplots
from spinlabs.gui.production import (
    CHANNEL_KEYS,
    PROCESSES,
    T1_STATES,
    ProductionPackage,
    ProductionSession,
    Settings,
    initial_state,
    production_packages,
    rate_direction,
)
from spinlabs.gui.state import GuiState, load_state, save_state
from spinlabs.transition_rate import TransitionRate
from spinlabs.utils import MathUtils
from test_unified_pipeline import _phonons, _raw, _save

SETTINGS = Settings(direct="gaussian", t_min=50.0, t_max=300.0, t_step=50.0)


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "registry.db")
    path = _save(tmp_path, _raw(2000.0))
    package = ProductionPackage(path=path, defect="NV", cell=64, method="AO", step=0.1)
    return ProductionSession(package, cache_dir=tmp_path / "cache", phonon_file=_phonons(tmp_path / "phonon_data.npz"))


def _manager(session):
    manager, _ = pipeline.coupling_manager(raw_zfs_file=session.package.path, phonon_file=session.phonon_file)
    return manager


def test_couplings_are_the_production_couplings_and_are_cached_on_disk(session, tmp_path):
    c = session.get("couplings", SETTINGS)
    manager = _manager(session)
    base = tmp_path / "reference"
    manager.process_both_orders(output_filename=str(base))
    from spinlabs.models import SpinPhononCouplingData

    ref = SpinPhononCouplingData.load(f"{base}.npz").to_unit("MHz").frequencies_to_unit("meV")
    for channel, (first, second) in CHANNEL_KEYS.items():
        assert np.array_equal(c.first[channel], np.abs(getattr(ref, first)))
        assert np.array_equal(c.second[channel], getattr(ref, second))
    assert c.deg_groups == [list(map(int, g)) for g in manager.spectrum.deg_groups]
    fresh = ProductionSession(session.package, cache_dir=session.cache_dir, phonon_file=None)  # no phonon: disk only
    again = fresh.get("couplings", SETTINGS)
    assert all(np.array_equal(again.second[k], c.second[k]) for k in c.second)


def test_spectral_functions_are_what_the_rates_integrate(session):
    c = session.get("couplings", SETTINGS)
    spectra = session.get("spectral", SETTINGS)
    maps = session.get("spectral_map", SETTINGS)
    zero_gap = TransitionRate.symmetry_zero_gap_pairs(len(c.frequencies), c.deg_groups)
    weights = TransitionRate.pair_weights(np.where(zero_gap, 0.0, c.second["d1"]))
    _x, _y, density = MathUtils.get_2d_spectral_density(c.frequencies, weights, min(0.5, SETTINGS.sigma / 4),
                                                        SETTINGS.sigma)
    assert np.array_equal(maps["d1"][2], density)
    assert np.array_equal(spectra["two"]["d1"][1], np.diag(density))
    omega, j1 = spectra["one"]["d1"]
    assert np.trapezoid(j1, omega) == pytest.approx(np.sum(c.first["d1"] ** 2), rel=1e-3)  # sum |V|^2


def test_rates_and_t1_are_rate_series_and_t1_details(session):
    c = session.get("couplings", SETTINGS)
    rates = session.get("rates", SETTINGS)
    series = pipeline.rate_series(c.coupling_path, SETTINGS.temperatures, sigma=SETTINGS.sigma, has_second_order=True,
                                  degenerate_groups=c.deg_groups, direct="gaussian")
    for process, by_direction in series.rates.items():
        for direction, values in by_direction.items():
            assert np.array_equal(rates.rates[process][direction], values)
    for which, expected in zip(T1_STATES, ("ms_0", "ms_1", "ms_-1")):
        temps, t1, state = session.get("t1", SETTINGS, which)
        assert state == expected
        ref = pipeline.t1_details(temps, rates.total(pipeline.PRODUCTION_PROCESSES), expected, 1.0)[0]
        assert np.array_equal(t1, ref, equal_nan=True)
    assert rates.total(PROCESSES["total"])[rate_direction(1.0, "d1")][-1] > 0


def test_a_magnetic_field_reaches_the_rates_and_splits_the_mirror_transitions(session):
    field = Settings(direct="gaussian", t_min=50.0, t_max=300.0, t_step=50.0, b_field_mt=60.0)
    assert field.rate_key() != SETTINGS.rate_key() and field.coupling_key() == SETTINGS.coupling_key()
    c = session.get("couplings", field)
    rates = session.get("rates", field)
    series = pipeline.rate_series(c.coupling_path, field.temperatures, sigma=field.sigma, has_second_order=True,
                                  degenerate_groups=c.deg_groups, direct="gaussian", b_field_mt=60.0)
    for process, by_direction in series.rates.items():
        for direction, values in by_direction.items():
            assert np.array_equal(rates.rates[process][direction], values)
    total = rates.total(PROCESSES["total"])
    assert not np.allclose(total[rate_direction(1.0, "d1")], total[rate_direction(1.0, "d1", -1)])
    plain = session.get("rates", SETTINGS).total(PROCESSES["total"])
    assert np.allclose(plain[rate_direction(1.0, "d1")], plain[rate_direction(1.0, "d1", -1)])
    fig = pplots.build_rates_figure([("A", "#1f77b4", rates)], ["d1", "d2"], {"total": PROCESSES["total"]},
                                    mirror=True)
    assert len(fig.axes) == 4 and "-1→0" in fig.axes[2].get_title()  # the -S row


def test_the_worker_process_computes_what_the_session_computes(session):
    from spinlabs.gui.production import stop_worker, worker_executor

    executor = worker_executor()
    try:
        remote = ProductionSession(session.package, cache_dir=session.cache_dir.parent / "remote_cache",
                                   phonon_file=session.phonon_file)
        assert remote.get("t1", SETTINGS, T1_STATES[0], executor=executor) is None  # queued in the process
        assert remote.status("t1", SETTINGS, T1_STATES[0]) == "pending" and remote.busy()
        executor.submit(int).result(timeout=300)  # jobs run in order: the T1 job has finished
        assert remote.status("t1", SETTINGS, T1_STATES[0]) == "ready" and not remote.busy()
        temps, t1, state = remote.peek("t1", SETTINGS, T1_STATES[0])
        ref = session.get("t1", SETTINGS, T1_STATES[0])
        assert state == ref[2] and np.array_equal(t1, ref[1], equal_nan=True)
        assert remote.status("rates", SETTINGS) == "missing"  # the GUI side only holds what it asked for
        assert (session.cache_dir.parent / "remote_cache").is_dir()  # the process wrote the disk cache
    finally:
        stop_worker(executor)


def test_the_debye_direct_rate_falls_back_with_a_warning(session, monkeypatch):
    def no_fit(self, spectrum):
        raise ValueError("no folded acoustic modes")

    monkeypatch.setattr(TransitionRate, "set_debye_from_spectrum", no_fit)
    rates = session.get("rates", Settings(t_min=50.0, t_max=100.0, t_step=50.0))  # production: Debye
    assert rates.direct == "gaussian" and "Debye direct rate not available" in rates.warnings[0]


def test_queued_work_finishes_and_a_failure_is_reported(session, tmp_path):
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert session.get("rates", SETTINGS, executor=executor) is None  # queued
        executor.submit(lambda: None).result()  # the queue is processed in order
    assert session.peek("rates", SETTINGS) is not None and not session.busy()
    assert session.status("rates", SETTINGS) == "ready" and session.status("t1", SETTINGS, "x") == "missing"
    broken = ProductionSession(ProductionPackage(path=tmp_path / "missing.npz", defect="X"), cache_dir=tmp_path / "c")
    with ThreadPoolExecutor(max_workers=1) as executor:
        broken.get("couplings", SETTINGS, executor=executor)
    assert broken.status("couplings", SETTINGS) == "failed" and broken.errors()
    with pytest.raises(RuntimeError):
        broken.get("couplings", SETTINGS)


def test_production_packages_list_files_and_decont_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "registry.db")
    single = tmp_path / "NV_64" / "pert_0.1" / "AO" / "20261101" / "pack.npz"
    folder = tmp_path / "NV_64" / "pert_0.2" / "full_decont" / "20261119"
    single.parent.mkdir(parents=True)
    folder.mkdir(parents=True)
    first_only = tmp_path / "NV_64" / "pert_0.05" / "AO" / "20261016" / "NV_64_raw_zfs_data_all_bands_1d.npz"
    first_only.parent.mkdir(parents=True)
    single.write_bytes(b"x")
    first_only.write_bytes(b"z")
    conn = registry.ensure_db()
    registry.register_file(conn, single, "packaged_zfs", defect="NV", cell_size=64, calc_method="AO", pert_scale=0.1,
                           pert_order=[1, 2])
    registry.register_file(conn, first_only, "packaged_zfs", defect="NV", cell_size=64, calc_method="AO",
                           pert_scale=0.05, pert_order=[1])
    for level in ("ms_1", "ms_0_127"):
        f = folder / f"{level}.npz"
        f.write_bytes(b"y")
        registry.register_file(conn, f, "packaged_zfs_sublevel", defect="NV", cell_size=64,
                               calc_method="full_decont", pert_scale=0.2, spin_sublevel=level, pert_order=[1, 2])
    packages = {p.path: p for p in production_packages(conn)}
    assert set(packages) == {single, folder}  # the first-order-only package is left out
    assert packages[single].date == "20261101" and packages[folder].method == "full_decont"
    every = {p.path: p for p in production_packages(conn, complete_only=False)}
    assert set(every) == {single, folder, first_only} and not every[first_only].complete


def test_packages_are_grouped_by_setup_latest_first(tmp_path, monkeypatch):
    from spinlabs.gui.production import production_setups, version_labels

    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "registry.db")
    conn = registry.ensure_db()

    def register(rel, method="AO", step=0.1, cell=64):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rel.encode())
        registry.register_file(conn, path, "packaged_zfs", defect="NV", cell_size=cell, calc_method=method,
                               pert_scale=step, pert_order=[1, 2])
        return path

    older = register("NV_64/pert_0.1/AO/20261101/a.npz")
    legacy = register("NV_64/pert_0.1/all_bands/20261020/b.npz", method="all_bands")  # same setup: AO
    newest = register("NV_64/pert_0.1/AO/20261015/c.npz")  # registered last: the latest, as --latest picks it
    other = register("NV_64/pert_0.05/AO/20261101/d.npz", step=0.05)
    setups = production_setups(production_packages(conn))
    assert len(setups) == 2
    versions = next(v for v in setups.values() if v[0].step == 0.1)
    assert [p.path for p in versions] == [newest, legacy, older]
    assert registry.find_packaged(conn, defect="NV", cell_size=64, calc_method="AO", pert_scale=0.1,
                                  latest=True) == [versions[0].path]
    assert version_labels(versions) == ["20261015 (latest)", "20261020", "20261101"]
    assert [p.path for v in setups.values() if v[0].step == 0.05 for p in v] == [other]
    same_day = register("NV_64/pert_0.1/AO/20261015/e.npz")
    versions = next(v for v in production_setups(production_packages(conn)).values() if v[0].step == 0.1)
    assert version_labels(versions)[:2] == ["20261015 · e.npz (latest)", "20261015 · c.npz"]
    assert versions[0].path == same_day


def test_a_package_without_second_order_is_refused(tmp_path, monkeypatch):
    from spinlabs.gui.production import package_for_path

    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "registry.db")
    raw = _raw(2000.0)
    raw.second_order = {}
    path = tmp_path / "NV_64_raw_zfs_data_all_bands_1d.npz"
    raw.save(path)
    with pytest.raises(ValueError, match="first-order data only"):
        package_for_path(path)
    assert package_for_path(_save(tmp_path, _raw(2000.0))).defect == "NV"  # both orders: accepted


def test_settings_validation_and_state_round_trip(tmp_path):
    with pytest.raises(ValueError):
        Settings(pairs="everything")
    with pytest.raises(ValueError):
        Settings(sigma=0)
    s = Settings(sigma=2.0, t_min=10, t_max=20, t_step=5)
    assert Settings.from_dict(s.to_dict()) == s
    assert Settings.from_dict({"sigma": -1}) == Settings()  # an invalid saved value falls back to the defaults
    comparison = {"key": "comparison:3", "title": "Comparison 3", "entries": ["/x/a.npz"], "settings": s.to_dict()}
    path = save_state(GuiState(comparisons=[comparison]), tmp_path / "gui.json")
    assert load_state(path).comparisons == [comparison]
    assert json.loads(path.read_text())["comparisons"][0]["settings"]["sigma"] == 2.0


def test_states_and_directions_for_both_spins():
    assert [initial_state(1.5, w) for w in T1_STATES] == ["ms_pm1_2", "ms_pm3_2", "ms_pm3_2"]  # doublets
    assert [initial_state(1.0, w) for w in T1_STATES] == ["ms_0", "ms_1", "ms_-1"]
    assert rate_direction(1.0, "d1") == "1_to_0" and rate_direction(1.0, "d2") == "1_to_-1"
    assert rate_direction(1.0, "d1", -1) == "-1_to_0" and rate_direction(1.0, "d2", -1) == "-1_to_1"
    assert rate_direction(1.5, "d1") == "1.5_to_0.5" and rate_direction(1.5, "d2") == "1.5_to_-0.5"


def test_figure_builders_draw_every_entry(session):
    c = session.get("couplings", SETTINGS)
    entries = [("A", "#1f77b4", c), ("B", "#d62728", c)]
    fig = pplots.build_coupling_figure(entries, ["d1", "d2"])  # as the spectral functions: a column per channel
    assert len(fig.axes) == 4 and all(len(ax.collections) == 2 and not ax.lines for ax in fig.axes)
    # one vertical line per mode with a coupling, from just below the smallest one (log scale) up to |V|
    values = np.abs(np.asarray(c.first["d1"], float))
    segments = fig.axes[0].collections[0].get_segments()
    assert len(segments) == np.count_nonzero(values > 0)
    assert all(seg[0, 0] == seg[1, 0] for seg in segments)  # vertical
    assert np.allclose(sorted(seg[1, 1] for seg in segments), np.sort(values[values > 0]))
    assert all(seg[0, 1] == fig.axes[0].get_ylim()[0] > 0 for seg in segments)
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["A", "B"]  # entry order
    linear = pplots.build_coupling_figure(entries, ["d1"], log=False)
    assert linear.axes[0].get_ylim() == pytest.approx((0, 1.05 * values.max())) and all(seg[0, 1] == 0 for seg in linear.axes[0].collections[0].get_segments())
    top, bottom = fig.axes[0], fig.axes[2]  # first order above, the diagonal second order below
    assert "first order" in top.get_title() and "second order" in bottom.get_title()
    spectra = session.get("spectral", SETTINGS)
    fig = pplots.build_spectral_figure([("A", "#1f77b4", spectra)], ["d1"], SETTINGS.sigma)
    assert len(fig.axes) == 2
    fig = pplots.build_map_figure("A", session.get("spectral_map", SETTINGS), ["d1", "d2"], SETTINGS.sigma)
    assert any(ax.collections for ax in fig.axes)
    rates = session.get("rates", SETTINGS)
    fig = pplots.build_rates_figure([("A", "#1f77b4", rates)], ["d0", "d1", "d2"],
                                    {"total": PROCESSES["total"], "direct": PROCESSES["direct"]})
    assert len(fig.axes) == 2 and len(fig.axes[0].lines) == 2  # d0 drives no transitions: no panel
    t1 = session.get("t1", SETTINGS, T1_STATES[0])
    fig = pplots.build_t1_figure([("A", "#1f77b4", t1), ("B", "#d62728", t1)], ratio=True)
    assert len(fig.axes) == 2
    assert np.allclose(fig.axes[1].lines[0].get_ydata(), 1.0)  # B / A of the same data


def test_garbage_is_collected_on_the_tk_thread_only():
    import gc

    from spinlabs.gui.collector import MainThreadCollector

    class Root:
        def __init__(self):
            self.scheduled = []

        def after(self, ms, fn):
            self.scheduled.append(fn)

    was_enabled = gc.isenabled()
    try:
        root = Root()
        collector = MainThreadCollector(root)
        assert not gc.isenabled() and root.scheduled  # automatic collection off, a poll on the event loop instead
        gc.collect()
        assert MainThreadCollector.collect() is None  # below the threshold: nothing to do
        junk = []
        for _ in range(gc.get_threshold()[0] + 10):
            a, b = [], []
            a.append(b)
            b.append(a)
            junk.append(a)
        del junk, a, b
        assert MainThreadCollector.collect() == 0
        root.scheduled.pop()()  # the tick reschedules itself
        assert root.scheduled and collector.root is root
    finally:
        if was_enabled:
            gc.enable()


def test_experimental_data_for_the_rates_and_t1_views(tmp_path, monkeypatch):
    from spinlabs.experimental import ExperimentalSeries, t1_from_rates
    from spinlabs.gui.production import experiment_rates, experiment_t1, experimental_series

    folder = tmp_path / "experimental_data" / "NV"
    folder.mkdir(parents=True)
    (folder / "Paper_2000.csv").write_text("Paper et al. (2000)\nK,Hz\nS1,\nX,Y\n100,2.0\n50,0.5\n300,0.0\n")
    (folder / "Rates_2001.csv").write_text(
        "K,kHz\nomega,,gamma,\nX,Y,X,Y\n100,0.01,100,0.02\n200,0.1,200,0.3\n")
    package = tmp_path / "data" / "NV_64" / "pack.npz"
    package.parent.mkdir(parents=True)
    monkeypatch.chdir(package.parent)  # found next to the package's tree, not only in the working directory
    series = experimental_series("NV", near=package)
    assert experimental_series("ClV", near=package) == []
    relax = next(s for s in series if s.kind == "relaxation")
    assert np.allclose(relax.x, [50, 100]) and np.allclose(relax.y, [2.0, 0.5])  # 1/T1 (Hz) -> T1 (s); zero dropped
    assert relax.legend == "Paper et al. (2000) S1"
    (omega,), (gamma,) = experiment_rates(series, "d1"), experiment_rates(series, "d2")
    assert omega.kind == "omega" and np.allclose(omega.y, [10.0, 100.0]) and gamma.kind == "gamma"
    assert experiment_rates(series, "d0") == []
    lowest = experiment_t1(series, T1_STATES[0])
    assert [s.kind for s, _x, _y in lowest] == ["relaxation", "relaxation"]  # measured data, and from Omega/gamma
    ref = t1_from_rates(ExperimentalSeries("omega", omega.x, omega.y, "K", "Hz", "Rates_2001", "omega"),
                        ExperimentalSeries("gamma", gamma.x, gamma.y, "K", "Hz", "Rates_2001", "gamma"), "ms_1")
    (derived, x, y), = experiment_t1(series, T1_STATES[1])  # m_s = +1: no measured relaxation data
    assert np.allclose(x, ref.x) and np.allclose(y, ref.y) and "ms_1" in derived.label
    fig = pplots.build_t1_figure([("A", "#1f77b4", (np.array([50.0, 100.0]), np.array([1.0, 0.1]), "ms_0"))],
                                 experiment=[(s.legend, "o", None, x, y) for s, x, y in lowest])
    assert sum(1 for line in fig.axes[0].lines if line.get_label().startswith("exp.:")) == 2


def test_a_chosen_experimental_data_folder_replaces_the_search(tmp_path, monkeypatch):
    from spinlabs.gui.production import experiment_folder, experimental_series

    found = tmp_path / "repo" / "experimental_data" / "NV"
    found.mkdir(parents=True)
    (found / "Near_2000.csv").write_text("K,Hz\nS1,\nX,Y\n100,2.0\n")
    chosen = tmp_path / "elsewhere" / "NV"
    chosen.mkdir(parents=True)
    (chosen / "Chosen_2001.csv").write_text("K,Hz\nS1,\nX,Y\n100,4.0\n200,8.0\n")
    package = tmp_path / "repo" / "data" / "NV_64" / "pack.npz"
    monkeypatch.chdir(tmp_path)
    assert [s.source for s in experimental_series("NV", near=package)] == ["Near_2000"]
    assert [s.source for s in experimental_series("NV", near=package, root=tmp_path / "elsewhere")] == ["Chosen_2001"]
    assert experiment_folder("ClV", near=package, root=tmp_path / "elsewhere") is None  # nothing else searched
    path = save_state(GuiState(experiment_dir=str(tmp_path / "elsewhere")), tmp_path / "gui.json")
    assert load_state(path).experiment_dir == str(tmp_path / "elsewhere")
    assert load_state(save_state(GuiState(), tmp_path / "default.json")).experiment_dir is None


def test_a_dead_worker_process_is_restarted(session):
    import os
    from concurrent.futures import BrokenExecutor

    from spinlabs.gui.production import stop_worker, worker_executor

    worker = worker_executor()
    try:
        with pytest.raises(BrokenExecutor):
            worker.submit(os._exit, 1).result(timeout=120)  # the process dies, as when the OOM killer takes it
        assert worker.submit(int, "7").result(timeout=120) == 7 and worker.restarts == 1  # a fresh process
    finally:
        stop_worker(worker)


def test_a_job_that_kills_the_worker_is_retried_once_then_reported(session):
    from concurrent.futures import Future
    from concurrent.futures.process import BrokenProcessPool

    class Dying:
        def submit(self, *args, **kwargs):
            future = Future()
            future.set_exception(BrokenProcessPool("killed"))
            return future

    assert session.get("rates", SETTINGS, executor=Dying()) is None
    assert session.status("rates", SETTINGS) == "missing"  # not failed: the next draw queues it again
    session.get("rates", SETTINGS, executor=Dying())
    assert session.status("rates", SETTINGS) == "failed" and "died twice" in session.errors(SETTINGS)[0]


def test_the_cache_follows_the_code_and_the_phonon_spectrum(session, monkeypatch):
    import spinlabs.gui.production as production

    folder = session._folder("couplings", SETTINGS.coupling_key())
    assert production.code_fingerprint() == production.code_fingerprint()  # stable
    monkeypatch.setattr(production, "_CODE_FINGERPRINT", "changed code")
    assert session._folder("couplings", SETTINGS.coupling_key()) != folder
    monkeypatch.undo()
    other = ProductionSession(session.package, cache_dir=session.cache_dir, phonon_file=session.phonon_file.parent / "x.npz")
    assert other._folder("couplings", SETTINGS.coupling_key()) != folder


def test_errors_and_memory_follow_the_current_settings(session):
    import spinlabs.gui.production as production

    good, bad = SETTINGS, Settings(direct="gaussian", t_min=50.0, t_max=300.0, t_step=50.0, sigma=2.0)
    key = ("rates", session._settings_key("rates", bad), ())
    session._errors[key] = "ValueError: no rate"
    assert session.errors(bad) == ["ValueError: no rate"] and session.errors(good) == []
    assert session.errors() == ["ValueError: no rate"]
    for sigma in (1.0, 2.0, 3.0, 4.0, 5.0):
        session.get("spectral_map", Settings(direct="gaussian", sigma=sigma))
    kept = [k for k in session._results if k[0] == "spectral_map"]
    assert len(kept) == production.KEEP_RESULTS and kept[-1][1][-1] == 5.0  # the latest ones


def test_a_decont_set_with_a_first_order_only_file_is_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "registry.db")
    folder = tmp_path / "NV_64" / "pert_0.2" / "full_decont" / "20261119"
    folder.mkdir(parents=True)
    conn = registry.ensure_db()
    for level, orders in (("ms_1", [1, 2]), ("ms_0_127", [1, 2]), ("ms_-1", [1])):
        f = folder / f"{level}.npz"
        f.write_bytes(level.encode())
        registry.register_file(conn, f, "packaged_zfs_sublevel", defect="NV", cell_size=64,
                               calc_method="full_decont", pert_scale=0.2, spin_sublevel=level, pert_order=orders)
    assert production_packages(conn) == []
    (package,) = production_packages(conn, complete_only=False)
    assert package.path == folder and not package.complete
