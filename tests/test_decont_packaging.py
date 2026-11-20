"""Packaging full-decontamination runs: one npz per spin sublevel, in one folder, kept apart in the registry."""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from spinlabs.cli.package import _run, _run_conv
from spinlabs.decont import (
    decontaminate_conv_set,
    high_spin,
    load_conv_sublevel_set,
    load_decontaminated,
    load_sublevel_set,
    sublevel_info,
)
from spinlabs.parsers import parse_decont_dataset

INCAR = """ENCUT = 520
ALGO = Normal
LDMATRIX = .TRUE.
NUPDOWN = 2
NBANDS = 8
DOCCUP = 4*1.0 4*0.0
DOCCDO = 2*1.0 6*0.0
"""
SUBLEVELS = ["ms_0_3", "ms_0_4", "ms_1"]
OFFSET = {"ms_1": 0.0, "ms_0_3": 100.0, "ms_0_4": 200.0}  # tells the sublevels' tensors apart
STEP = 0.025
DAY = date.today().strftime("%Y%m%d")


def _outcar(path: Path, dzz: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        " free  energy   TOTEN  =       -10.00000000 eV\n\n"
        " Spin-spin contribution to zero-field splitting tensor (MHz)\n -------\n"
        "      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz\n -------\n"
        f"  {-dzz / 2:.4f}  {-dzz / 2:.4f}  {dzz:.4f}  0.0000  0.0000  0.0000\n -------\n"
        " Elapsed time (sec):  100.000\n"
    )


def _defect(tmp_path: Path) -> tuple[Path, list[int]]:
    """NV_64 with --full-decont trees (first, second order, convergence), every run finished; returns
    the defect folder and the tree's 1-based modes."""
    from spinlabs.models import PhononSpectrum
    from spinlabs.parsers import save_phonon_npz
    from spinlabs.vasp.modes import mode_numbers
    from spinlabs.vasp.trees import build_perturbation_tree

    defect = tmp_path / "NV_64"
    (defect / "data").mkdir(parents=True)
    save_phonon_npz(PhononSpectrum(
        frequencies_mev=np.array([10.0, 20.0, 30.0]), eigenvectors=np.zeros((3, 1, 3)),
        atom_frac_coords=np.zeros((1, 3)), atom_symbols=["C"], atomic_masses=np.array([12.0]), lattice=np.eye(3),
        symmetries=["A1", "Ex", "A2"], original_indices=np.arange(3),
    ), defect / "data" / "phonon_data.npz")
    for stage in ("ZFS_hyp", "ZFS_occup"):
        (defect / stage).mkdir()
        (defect / stage / "INCAR").write_text(INCAR)
        for name in ("POSCAR", "KPOINTS", "POTCAR"):
            (defect / stage / name).write_text(name + "\n")
        _outcar(defect / stage / "OUTCAR", 2000.0)
    assert build_perturbation_tree(defect, [STEP], defect / "data" / "phonon_data.npz", displacement="normalized",
                                   conv_modes=[1], conv_perts=[STEP], array_jobs=1, vasp_binary=Path("/bin/true"),
                                   decont=True) == []
    modes = mode_numbers(next((defect / "data").glob("phonon_data_sym*.npz")))
    for lv in SUBLEVELS:
        _outcar(defect / "ZFS_full" / lv / "OUTCAR", 2000.0 + OFFSET[lv])
        for m in modes:
            _outcar(defect / "first_order" / f"pert_{STEP}" / "full_decont" / "runs" / str(m) / lv / "OUTCAR",
                    2000.0 + OFFSET[lv] + m)
            _outcar(defect / "second_order" / f"pert_{STEP}" / "full_decont" / "runs" / f"{m}_{m}" / lv / "OUTCAR",
                    2000.0 + OFFSET[lv] + 2 * m)
        for p in (-STEP, STEP):
            _outcar(defect / "convergence_test_decont" / "runs" / f"1_pert_{p:g}" / lv / "OUTCAR",
                    2000.0 + OFFSET[lv] + p)
    return defect, modes


def _args(tmp_path, defect, method=None):
    return argparse.Namespace(defect_folder=[defect], orders="first,second", pert=None, method=method,
                              output_root=tmp_path / "data", scan=False, yes=True)


def test_sublevel_names():
    assert sublevel_info("ms_1") == (1.0, None)
    assert sublevel_info("ms_0_127") == (0.0, 127)
    assert sublevel_info("ms_0.5_6") == (0.5, 6)
    assert high_spin(["ms_0_127", "ms_1", "ms_0_128"]) == "ms_1"
    with pytest.raises(ValueError):
        sublevel_info("ms1")


def test_one_dataset_per_sublevel_with_its_ground_state_from_zfs_full(tmp_path):
    defect, modes = _defect(tmp_path)
    sets = parse_decont_dataset(defect / "first_order" / f"pert_{STEP}", order=1)
    assert sorted(sets) == SUBLEVELS
    for lv, raw in sets.items():
        assert raw.spin_sublevel == lv and raw.calc_method == "full_decont" and raw.spin == 1.0
        assert raw.ground_state_zfs.to_unit("MHz").matrix[2, 2] == pytest.approx(2000.0 + OFFSET[lv])
        assert sorted(raw.first_order) == [m - 1 for m in modes]
        m = modes[0]
        assert raw.first_order[m - 1].zfs_tensor.to_unit("MHz").matrix[2, 2] == pytest.approx(2000.0 + OFFSET[lv] + m)


def test_a_run_missing_a_sublevel_is_left_out_of_every_sublevel(tmp_path):
    defect, modes = _defect(tmp_path)
    (defect / "first_order" / f"pert_{STEP}" / "full_decont" / "runs" / str(modes[0]) / "ms_0_4" / "OUTCAR").unlink()
    sets = parse_decont_dataset(defect / "first_order" / f"pert_{STEP}", order=1)
    for raw in sets.values():
        assert sorted(raw.first_order) == [m - 1 for m in modes[1:]]


def test_without_the_ground_state_the_set_is_refused(tmp_path):
    defect, _ = _defect(tmp_path)
    (defect / "ZFS_full" / "ms_0_3" / "OUTCAR").unlink()
    with pytest.raises(ValueError, match="ms_0_3"):
        parse_decont_dataset(defect / "first_order" / f"pert_{STEP}", order=1)


def test_package_writes_one_file_per_sublevel_into_one_folder(tmp_path):
    from spinlabs.registry import ensure_db, find_packaged, find_sublevel_sets, unregistered_packages

    defect, modes = _defect(tmp_path)
    _run(_args(tmp_path, defect))  # default methods: only the decont trees exist
    folder = tmp_path / "data" / "NV_64" / f"pert_{STEP:g}" / "full_decont" / DAY
    assert sorted(p.name for p in folder.iterdir()) == [
        f"NV_64_raw_zfs_data_full_decont_{lv}_12d.npz" for lv in SUBLEVELS]
    # tagged with the alignment of the phonon file the trees' job scripts displace along (classified only here)
    for f in folder.iterdir():
        assert bool(np.load(f, allow_pickle=True)["aligned"]) is False
    assert not (tmp_path / "data" / "NV_64" / f"pert_{STEP:g}" / "all_bands").exists()

    loaded = load_sublevel_set(folder)
    assert list(loaded) == ["ms_1", "ms_0_3", "ms_0_4"]  # high spin first
    for lv, raw in loaded.items():
        assert raw.spin_sublevel == lv
        assert sorted(raw.first_order) == [m - 1 for m in modes]
        assert sorted(raw.second_order) == [(m - 1, m - 1) for m in modes]
        assert raw.displacement == "normalized"
        assert raw.ground_state_zfs.to_unit("MHz").matrix[2, 2] == pytest.approx(2000.0 + OFFSET[lv])

    conn = ensure_db()
    assert find_packaged(conn) == [folder.resolve()]  # the set, as one package: never a single sublevel
    sets = find_sublevel_sets(conn, defect="NV", cell_size=64, latest=True)
    assert len(sets) == 1 and sorted(sets[0]) == SUBLEVELS
    assert {p.parent for p in sets[0].values()} == {folder.resolve()}
    assert unregistered_packages(conn, [tmp_path / "data"]) == []

    # --register-only puts them back the same way
    conn.execute("DELETE FROM files")
    conn.commit()
    from spinlabs.cli.package import _register_only

    _register_only(tmp_path / "data")
    assert find_packaged(conn) == [folder.resolve()]
    assert sorted(find_sublevel_sets(conn)[0]) == SUBLEVELS


def test_convergence_is_packaged_per_sublevel_with_its_d0(tmp_path):
    from spinlabs.registry import ensure_db, find_convergence, find_sublevel_sets

    defect, _ = _defect(tmp_path)
    args = argparse.Namespace(defect_folder=defect, output_root=tmp_path / "data", ground_state=None, yes=True,
                              scan=False)
    _run_conv(args, defect)
    folder = tmp_path / "data" / "NV_64" / "full_decont" / DAY
    assert sorted(p.name for p in folder.iterdir()) == [
        f"NV_64_{DAY}_conv_test_decont_{lv}.npz" for lv in SUBLEVELS]
    conv = load_conv_sublevel_set(folder)
    for lv, data in conv.items():
        assert data["metadata"]["spin_sublevel"] == lv
        assert data["d0"][2, 2] == pytest.approx(2000.0 + OFFSET[lv])
        assert list(data["perts"]) == [-STEP, STEP]
        assert data["tensors"][0, 1, 2, 2] == pytest.approx(2000.0 + OFFSET[lv] + STEP)
    conn = ensure_db()
    assert find_convergence(conn) == [folder.resolve()]
    assert sorted(find_sublevel_sets(conn, kind="convergence_sublevel")[0]) == SUBLEVELS


def test_an_old_registry_gets_the_spin_sublevel_column(tmp_path, monkeypatch):
    import sqlite3

    import spinlabs.registry as reg

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, kind TEXT NOT NULL, defect TEXT,"
        " cell_size INTEGER, calc_method TEXT, pert_order TEXT, pert_scale REAL, sha256 TEXT,"
        " registered_at TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO files (path, kind, registered_at) VALUES ('/x.npz', 'packaged_zfs', 'now')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(reg, "REGISTRY_PATH", db)
    conn = reg.ensure_db()
    assert "spin_sublevel" in {row[1] for row in conn.execute("PRAGMA table_info(files)")}
    assert conn.execute("SELECT path, spin_sublevel FROM files").fetchall() == [("/x.npz", None)]


def test_a_packaged_set_and_its_convergence_pack_are_decontaminated(tmp_path):
    defect, modes = _defect(tmp_path)
    _run(_args(tmp_path, defect))
    raw = load_decontaminated(tmp_path / "data" / "NV_64" / f"pert_{STEP:g}" / "full_decont" / DAY)
    # S = 1: (D(ms_1) - mean D(ms_0)) / 2 along the axis
    gs = (2000.0 + OFFSET["ms_1"] - (2000.0 + (OFFSET["ms_0_3"] + OFFSET["ms_0_4"]) / 2)) / 2
    assert raw.calc_method == "full_decont" and raw.spin_sublevel is None
    assert raw.ground_state_zfs.to_unit("MHz").matrix[2, 2] == pytest.approx(gs)
    assert sorted(raw.first_order) == [m - 1 for m in modes] and sorted(raw.second_order) == [(m - 1, m - 1) for m in modes]

    args = argparse.Namespace(defect_folder=defect, output_root=tmp_path / "data", ground_state=None, yes=True,
                              scan=False)
    _run_conv(args, defect)
    conv = decontaminate_conv_set(load_conv_sublevel_set(tmp_path / "data" / "NV_64" / "full_decont" / DAY))
    assert conv["d0"][2, 2] == pytest.approx(gs)
    assert conv["tensors"].shape == (1, 2, 3, 3)


def test_a_set_is_queried_and_loaded_as_one_decontaminated_dataset(tmp_path):
    from spinlabs.models import RawZFSData
    from spinlabs.registry import ensure_db, find_packaged, package_digest, package_info

    defect, modes = _defect(tmp_path)
    _run(_args(tmp_path, defect))
    folder = (tmp_path / "data" / "NV_64" / f"pert_{STEP:g}" / "full_decont" / DAY).resolve()
    conn = ensure_db()
    assert find_packaged(conn, calc_method="full_decont", latest=True) == [folder]
    assert find_packaged(conn, calc_method="full_decont", pert_order=[1, 2]) == [folder]
    assert find_packaged(conn, calc_method="all_bands") == []
    info = package_info(conn, folder)
    assert info["calc_method"] == "full_decont" and info["pert_order"] == "[1, 2]" and info["defect"] == "NV"

    raw = RawZFSData.load(folder)  # what --query hands the pipeline
    assert raw.calc_method == "full_decont" and raw.spin_sublevel is None
    gs = (OFFSET["ms_1"] - (OFFSET["ms_0_3"] + OFFSET["ms_0_4"]) / 2) / 2
    assert raw.ground_state_zfs.to_unit("MHz").matrix[2, 2] == pytest.approx(gs)
    assert RawZFSData.load(sorted(folder.glob("*.npz"))).ground_state_zfs.to_unit("MHz").matrix[2, 2] == pytest.approx(gs)
    single = RawZFSData.load(next(folder.glob("*ms_1_*.npz")))  # one file is still that sublevel
    assert single.spin_sublevel == "ms_1"

    digest = package_digest(conn, folder)
    assert len(digest) == 64 and digest == package_digest(conn, folder)

    from spinlabs.cli.listing import package_table
    from spinlabs.cli.run import _infer_method_from_file

    assert _infer_method_from_file(folder) == "full_decont"
    assert "full_decont" in package_table(conn)


def test_the_d_table_shows_the_decontaminated_column_only_with_data(tmp_path):
    from spinlabs.tables import collect_d_rows, merge_rows, render_defect_table

    defect, _ = _defect(tmp_path)
    _run(_args(tmp_path, defect))
    rows = collect_d_rows([tmp_path / "data"])
    assert [r["column"] for r in rows] == ["decontaminated"]  # one row for the set, not one per sublevel
    gs = (OFFSET["ms_1"] - (OFFSET["ms_0_3"] + OFFSET["ms_0_4"]) / 2) / 2
    assert rows[0]["d_mhz"] == pytest.approx(1.5 * gs)
    tex = render_defect_table(merge_rows(rows, []), include_t1=False)
    assert r"$D_\text{FD}$" in tex and r"$D_\text{AO}$" in tex
    plain = render_defect_table(merge_rows([{**rows[0], "column": "contaminated"}], []), include_t1=False)
    assert "FD" not in plain


def test_a_convergence_set_loads_decontaminated(tmp_path):
    from spinlabs.convergence import load_convergence
    from spinlabs.convergence_summary import column_title

    defect, _ = _defect(tmp_path)
    args = argparse.Namespace(defect_folder=defect, output_root=tmp_path / "data", ground_state=None, yes=True,
                              scan=False)
    _run_conv(args, defect)
    folder = tmp_path / "data" / "NV_64" / "full_decont" / DAY
    conv = load_convergence(folder)
    gs = (OFFSET["ms_1"] - (OFFSET["ms_0_3"] + OFFSET["ms_0_4"]) / 2) / 2
    assert conv.q0[2, 2] == pytest.approx(gs) and sorted(conv.perts) == [-STEP, 0.0, STEP]
    assert column_title(folder) == "NV_64 full decont"
