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
    for which, expected in zip(T1_STATES, ("ms_0", "ms_1")):
        temps, t1, state = session.get("t1", SETTINGS, which)
        assert state == expected
        ref = pipeline.t1_details(temps, rates.total(pipeline.PRODUCTION_PROCESSES), expected, 1.0)[0]
        assert np.array_equal(t1, ref, equal_nan=True)
    assert rates.total(PROCESSES["total"])[rate_direction(1.0, "d1")][-1] > 0


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
    broken = ProductionSession(ProductionPackage(path=tmp_path / "missing.npz", defect="X"), cache_dir=tmp_path / "c")
    with ThreadPoolExecutor(max_workers=1) as executor:
        broken.get("couplings", SETTINGS, executor=executor)
    with pytest.raises(RuntimeError):
        broken.get("couplings", SETTINGS)
    assert broken.errors()


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
    assert [initial_state(1.5, w) for w in T1_STATES] == ["ms_pm1_2", "ms_pm3_2"]
    assert rate_direction(1.0, "d1") == "1_to_0" and rate_direction(1.0, "d2") == "1_to_-1"
    assert rate_direction(1.5, "d1") == "1.5_to_0.5" and rate_direction(1.5, "d2") == "1.5_to_-0.5"


def test_figure_builders_draw_every_entry(session):
    c = session.get("couplings", SETTINGS)
    entries = [("A", "#1f77b4", c), ("B", "#d62728", c)]
    fig = pplots.build_coupling_figure(entries, ["d1", "d2"])  # as the spectral functions: a column per channel
    assert len(fig.axes) == 4 and all(len(ax.lines) == 2 for ax in fig.axes)
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
