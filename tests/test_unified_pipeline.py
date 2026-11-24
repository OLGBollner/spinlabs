"""One loading, coupling, rate and T1 path for every method (spinlabs.pipeline): build_manager / coupling_manager,
rate_series and t1_details. AO, UO and the full decontamination differ only in how their D tensors are loaded."""

import numpy as np
import pytest

from spinlabs import pipeline
from spinlabs.cli.production import add_coupling_args, coupling_options
from spinlabs.models import PhononSpectrum, RawZFSData, ZFSTensor

SYMS = ["A1", "Ex", "Ey", "A1"]
FREQS = np.array([10.0, 30.0, 30.0, 40.0])  # a degenerate Ex/Ey pair
E_STEP = np.array([[1.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])  # spin-flip couplings, finite T1
A_STEP = np.diag([-1.0, -1.0, 2.0])


def _raw(gs_dzz, scale=1.0, *, calc_method="AO", sublevel=None, pairs=True):
    gs = np.diag([-gs_dzz / 2, -gs_dzz / 2, gs_dzz])
    steps = [scale * (E_STEP if s in ("Ex", "Ey") else A_STEP) for s in SYMS]
    first = {i: {"tensor": gs + (i + 1) * steps[i], "unit": "MHz", "pert": 1e-25, "symmetry": SYMS[i], "ipr": 0.5}
             for i in range(len(SYMS))}
    second = {(i, i): {"tensor": gs + 2.1 * (i + 1) * steps[i], "unit": "MHz", "pert": (1e-25, 1e-25),
                       "symmetry": (SYMS[i], SYMS[i]), "ipr": (0.5, 0.5)} for i in range(len(SYMS))}
    if pairs:
        second[(0, 3)] = {"tensor": gs + 0.7 * A_STEP, "unit": "MHz", "pert": (1e-25, 1e-25),
                          "symmetry": ("A1", "A1"), "ipr": (0.5, 0.5)}
    raw = RawZFSData(defect="NV", cell_size=64, pert_scale=0.1, calc_method=calc_method, spin=1.0,
                     ground_state_zfs=ZFSTensor(matrix=gs, unit="MHz"), first_order=first, second_order=second)
    raw.spin_sublevel = sublevel
    return raw


def _phonons(path):
    PhononSpectrum(
        frequencies_mev=FREQS, symmetries=SYMS, iprs=np.full(len(SYMS), 0.5),
        eigenvectors=np.ones((len(SYMS), 1, 3)), atom_frac_coords=np.zeros((1, 3)), atom_symbols=["C"],
        atomic_masses=np.array([12.0]), lattice=np.eye(3) * 5.0,
    ).save(path)
    return path


def _couplings(manager, tmp_path, name):
    base = tmp_path / name
    manager.process_both_orders(output_filename=str(base))
    with np.load(f"{base}.npz", allow_pickle=True) as d:
        return {k: np.asarray(d[k]) for k in ("V_0_0", "V_p_m", "V_0_pm", "V2_0_0", "V2_p_m", "V2_0_pm")}


def test_build_manager_defaults_are_the_production_options(tmp_path):
    m = pipeline.build_manager(_raw(2000.0), None)
    for key, value in pipeline.PRODUCTION_COUPLING_OPTIONS.items():
        assert getattr(m, key) == value, key
    assert m.pairs == "sqrt" and m.calc_method == "AO"


def test_the_cli_flags_default_to_the_same_options():
    import argparse

    ap = argparse.ArgumentParser()
    add_coupling_args(ap.add_argument_group())
    assert coupling_options(ap.parse_args([])) == pipeline.PRODUCTION_COUPLING_OPTIONS


def test_pair_modes_pick_the_runs_and_the_fill():
    diag = pipeline.build_manager(_raw(2000.0), None, pairs="diagonal")
    assert all(i == j for i, j in diag.zfs_tensors_2d) and not diag.offdiag_from_diagonal
    fill = pipeline.build_manager(_raw(2000.0), None, pairs="diagonal-sqrt")
    assert all(i == j for i, j in fill.zfs_tensors_2d) and fill.offdiag_from_diagonal
    computed = pipeline.build_manager(_raw(2000.0), None, pairs="computed")
    assert (0, 3) in computed.zfs_tensors_2d and not computed.offdiag_from_diagonal


def test_conflicting_or_unknown_options_are_refused():
    with pytest.raises(ValueError, match="offdiag_from_diagonal"):
        pipeline.build_manager(_raw(2000.0), None, pairs="computed", offdiag_from_diagonal=True)
    with pytest.raises(ValueError, match="pair treatment"):
        pipeline.build_manager(_raw(2000.0), None, pairs="everything")
    with pytest.raises(TypeError, match="coupling option"):
        pipeline.build_manager(_raw(2000.0), None, mirror=False)


def test_a_full_decont_folder_goes_through_the_same_code_as_one_combined_file(tmp_path):
    """The sublevel set is combined on load; from there it is the same analysis as a package of the combined D."""
    folder = tmp_path / "full_decont"
    folder.mkdir()
    sets = {"ms_1": _raw(2104.5, 1.0, calc_method="full_decont", sublevel="ms_1"),
            "ms_0_127": _raw(-1126.7, 0.6, calc_method="full_decont", sublevel="ms_0_127"),
            "ms_0_128": _raw(-1126.7, 0.4, calc_method="full_decont", sublevel="ms_0_128")}
    for name, raw in sets.items():
        raw.save(folder / f"NV_64_raw_zfs_data_full_decont_{name}_12d.npz")
    phonons = _phonons(tmp_path / "phonon_data.npz")
    from_folder, _ = pipeline.coupling_manager(raw_zfs_file=folder, phonon_file=phonons)
    assert from_folder.calc_method == "full_decont"

    combined = RawZFSData.decontaminated({n: RawZFSData.load(folder / f"NV_64_raw_zfs_data_full_decont_{n}_12d.npz")
                                          for n in sets})
    single = tmp_path / "combined" / "NV_64_raw_zfs_data_full_decont_12d.npz"
    single.parent.mkdir()
    combined.save(single)
    from_file, _ = pipeline.coupling_manager(raw_zfs_file=single, phonon_file=phonons)

    a, b = _couplings(from_folder, tmp_path, "folder"), _couplings(from_file, tmp_path, "file")
    for key in a:
        assert np.allclose(a[key], b[key], rtol=1e-12, atol=0), key
    assert np.any(a["V2_0_pm"] > 0)  # the comparison is not of zeros


def test_compute_rates_and_solve_t1_are_rate_series_and_t1_details(tmp_path):
    phonons = _phonons(tmp_path / "phonon_data.npz")
    m, _ = pipeline.coupling_manager(raw_zfs_file=_save(tmp_path, _raw(2000.0)), phonon_file=phonons)
    base = tmp_path / "c"
    m.process_both_orders(output_filename=str(base))
    temps = [50.0, 150.0, 300.0]
    rs = pipeline.rate_series(f"{base}.npz", temps, sigma=3.5, has_second_order=True, direct="gaussian",
                              degenerate_groups=m.spectrum.deg_groups)
    valid, totals, directional = pipeline.compute_rates(
        f"{base}.npz", temps, sigma=3.5, has_second_order=True, direct="gaussian",
        degenerate_groups=m.spectrum.deg_groups, processes=pipeline.PRODUCTION_PROCESSES)
    assert np.allclose(valid, rs.temperatures)
    for d, v in rs.directional(pipeline.PRODUCTION_PROCESSES).items():
        assert np.allclose(directional[d], v, rtol=1e-14, atol=0)
    for p, by_class in rs.totals.items():
        for c, v in by_class.items():
            assert np.allclose(totals[p][c], v, rtol=1e-14, atol=0)
    assert rs.directional()["0_to_1"][-1] > 0

    path = tmp_path / "directional.npz"
    np.savez(path, temperatures=valid, **directional)
    t1_fit, t1_eig = pipeline.solve_t1(path, "ms_0")
    t1, eig, src = pipeline.t1_details(valid, directional, "ms_0")
    assert np.allclose(t1_fit, t1) and np.allclose(t1_eig, eig) and len(src) == len(temps)


def _save(tmp_path, raw):
    path = tmp_path / "in" / "NV_64_raw_zfs_data_all_bands_12d.npz"
    path.parent.mkdir(exist_ok=True)
    raw.save(path)
    return path


@pytest.mark.parametrize(
    "folders, method",
    [
        ("NV_64/second_order/pert_0.1/AO", "AO"),
        ("NV_64/second_order/pert_0.1/defect_band_approx", "UO"),
        (["NV_64/first_order/pert_0.1/UO", "NV_64/second_order/pert_0.1/UO"], "UO"),
    ],
)
def test_sim_folder_method_is_read_from_the_folder_names(folders, method):
    from spinlabs.pipeline import sim_folder_method

    assert sim_folder_method(folders) == method


@pytest.mark.parametrize("folders", ["NV_64/second_order/pert_0.1/runs",
                                     ["NV_64/second_order/pert_0.1/AO", "NV_64/second_order/pert_0.1/UO"]])
def test_sim_folder_without_a_method_is_an_error(folders):
    from spinlabs.pipeline import coupling_manager, sim_folder_method

    with pytest.raises(ValueError, match="pass calc_method"):
        sim_folder_method(folders)
    with pytest.raises(ValueError, match="pass calc_method"):
        coupling_manager(sim_folder=folders)


def test_build_manager_leaves_the_callers_raw_data_alone():
    raw = _raw(2000.0)
    raw.displacement = "unweighted"
    m = pipeline.build_manager(raw, None, displacement="mass_weighted")
    assert raw.displacement == "unweighted" and m.raw_data.displacement == "mass_weighted"


def _unclassifiable(monkeypatch):
    from spinlabs import symmetry
    from spinlabs.models import DefectCentreError

    def classify(*args, **kwargs):
        raise DefectCentreError("no single defect centre (find_defect found: none)")

    monkeypatch.setattr(symmetry, "classify_and_pair", classify)


def test_a_manager_refuses_modes_it_cannot_label(tmp_path, monkeypatch):
    spectrum = PhononSpectrum.load(_phonons(tmp_path / "ph.npz"))
    spectrum.symmetries = None
    _unclassifiable(monkeypatch)
    with pytest.raises(ValueError, match="no symmetry labels and cannot be classified"):
        pipeline.build_manager(_raw(2000.0), spectrum)


def test_a_manager_falls_back_on_stored_labels_without_a_defect_centre(tmp_path, monkeypatch):
    spectrum = PhononSpectrum.load(_phonons(tmp_path / "ph.npz"))
    _unclassifiable(monkeypatch)
    m = pipeline.build_manager(_raw(2000.0), spectrum)
    assert [str(s) for s in m.spectrum.symmetries] == SYMS
