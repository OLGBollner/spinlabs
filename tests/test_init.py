"""spinlabs init tree / init zfs: a new defect's folders and its ground-state ZFS runs."""

import re
from pathlib import Path

import pytest

from spinlabs.cli import main
from spinlabs.vasp.init import init_tree, init_zfs

# NCORE = 128: two band groups on the template's 2 x 128 ranks, so the 6 bands of _relaxed are kept
INCAR = "ENCUT = 520\nALGO = Normal\nISMEAR = 0\nIBRION = 2\nNSW = 50\nISYM = 3\nNELECT = 8\nISPIN = 2\nNCORE = 128\n"


def _tag(text, tag):
    found = re.findall(rf"^\s*{tag}\s*=\s*(.+?)\s*(?:[!#].*)?$", text, re.M)
    return found[-1] if found else None


def _inputs(tmp_path):
    inp = tmp_path / "inputs"
    (inp / "data").mkdir(parents=True)
    for name, text in (("INCAR", INCAR), ("POSCAR", "POSCAR\n"), ("KPOINTS", "KPOINTS\n"), ("POTCAR", "POTCAR\n"),
                       ("run_vasp.sh", "#!/bin/sh\n"), ("phonopy.yaml", "phonons\n")):
        (inp / name).write_text(text)
    (inp / "data" / "notes.txt").write_text("notes\n")
    return inp


def _relaxed(defect: Path):
    """A finished relaxation: 6 bands, up 1..5 occupied, down 1..3 (unpaired 4 and 5), Gamma only."""
    relax = defect / "relax"
    (relax / "CONTCAR").write_text("CONTCAR\n")
    (relax / "CHGCAR").write_text("CHGCAR\n")
    (relax / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n")
    (relax / "OUTCAR").write_text(" Elapsed time (sec):  600.000\n")
    rows = "".join(f"  {b}  {b:.1f}  {b:.1f}  {1.0 if b <= 5 else 0.0}  {1.0 if b <= 3 else 0.0}\n" for b in range(1, 7))
    (relax / "EIGENVAL").write_text("x\nx\nx\nx\nx\n 8 1 6\n\n 0 0 0 1\n" + rows)
    return relax


def test_tree_puts_the_inputs_in_relax_and_the_phonons_in_data(tmp_path):
    out = init_tree("NV_8", _inputs(tmp_path), parent=tmp_path)
    assert sorted(p.name for p in (out / "relax").iterdir()) == ["INCAR", "KPOINTS", "POSCAR", "POTCAR"]
    assert sorted(p.name for p in (out / "data").iterdir()) == ["notes.txt", "phonopy.yaml"]
    for zfs in ("ZFS_AO", "ZFS_UO"):
        assert sorted(p.name for p in (out / zfs).iterdir()) == ["INCAR", "KPOINTS", "POSCAR", "POTCAR"]
    with pytest.raises(FileExistsError):
        init_tree("NV_8", tmp_path / "inputs", parent=tmp_path)
    init_tree("NV_8", tmp_path / "inputs", parent=tmp_path, force=True)


def test_tree_finds_the_inputs_in_subfolders_the_shallowest_first(tmp_path):
    inp = tmp_path / "inputs"
    (inp / "vasp" / "old").mkdir(parents=True)
    (inp / "phonons").mkdir()
    for name in ("INCAR", "KPOINTS", "POTCAR"):
        (inp / "vasp" / name).write_text(f"{name}\n")
        (inp / "vasp" / "old" / name).write_text("old\n")
    (inp / "vasp" / "structure.vasp").write_text("structure\n")
    (inp / "phonons" / "phonopy.yaml").write_text("phonons\n")
    out = init_tree("NV_8", inp, parent=tmp_path)
    assert sorted(p.name for p in (out / "relax").iterdir()) == ["INCAR", "KPOINTS", "POSCAR", "POTCAR"]
    assert (out / "relax" / "INCAR").read_text() == "INCAR\n"  # not the deeper old/INCAR
    assert (out / "relax" / "POSCAR").read_text() == "structure\n"
    assert (out / "data" / "phonopy.yaml").is_file()


def test_tree_needs_the_vasp_inputs(tmp_path):
    inp = _inputs(tmp_path)
    (inp / "POTCAR").unlink()
    with pytest.raises(FileNotFoundError, match="POTCAR"):
        init_tree("NV_8", inp, parent=tmp_path)


def test_zfs_ao_and_uo_are_one_self_consistent_zfs_run_each(tmp_path):
    out = init_tree("NV_8", _inputs(tmp_path), parent=tmp_path)
    relax = _relaxed(out)
    assert init_zfs(relax, "all", vasp_binary="/bin/true") == []
    assert init_zfs(relax, "unpaired", vasp_binary="/bin/true") == []
    ao, uo = (out / "ZFS_AO" / "INCAR").read_text(), (out / "ZFS_UO" / "INCAR").read_text()
    for incar in (ao, uo):
        assert (_tag(incar, "ALGO"), _tag(incar, "NSW"), _tag(incar, "IBRION")) == ("Normal", "0", "-1")
        assert (_tag(incar, "ISYM"), _tag(incar, "LDMATRIX"), _tag(incar, "NUPDOWN")) == ("0", ".TRUE.", "2")
        assert _tag(incar, "ISMEAR") == "0" and _tag(incar, "FERWE") is None  # left as in the relaxation
    assert (_tag(ao, "DOCCUP"), _tag(ao, "DOCCDO")) == ("5*1.0 1*0.0", "3*1.0 3*0.0")
    assert (_tag(uo, "DOCCUP"), _tag(uo, "DOCCDO")) == ("3*0.0 2*1.0 1*0.0", "6*0.0")  # only the unpaired orbitals
    for zfs, job in (("ZFS_AO", "NV_8_ground_state_AO"), ("ZFS_UO", "NV_8_ground_state_UO")):
        assert (out / zfs / "POSCAR").read_text() == "CONTCAR\n" and (out / zfs / "CHGCAR").is_file()
        script = (out / zfs / "run_ground_state.sh").read_text()
        assert f"#SBATCH -J {job}" in script and _tag(script, "binary") == str(Path("/bin/true").resolve())  # resolved: /bin may link to /usr/bin


def test_zfs_decont_builds_zfs_full_from_the_relaxation(tmp_path):
    out = init_tree("NV_8", _inputs(tmp_path), parent=tmp_path)
    assert init_zfs(_relaxed(out), "decont") == []
    full = out / "ZFS_full"
    assert sorted(p.name for p in full.iterdir() if p.is_dir()) == ["ms_0_4", "ms_0_5", "ms_1"]
    scf = (full / "INCAR").read_text()
    assert _tag(scf, "LWAVE") == ".TRUE." and _tag(scf, "LDMATRIX") == ".FALSE." and _tag(scf, "ISYM") == "0"
    assert _tag((full / "ms_0_5" / "INCAR").read_text(), "ALGO") == "None"
    assert (full / "run_ground_state.sh").is_file()


def test_a_filled_folder_is_kept_and_algo_none_is_corrected(tmp_path):
    out = init_tree("NV_8", _inputs(tmp_path), parent=tmp_path)
    relax = _relaxed(out)
    (relax / "INCAR").write_text(INCAR.replace("ALGO = Normal", "ALGO = None"))
    init_zfs(relax, "AO")
    assert _tag((out / "ZFS_AO" / "INCAR").read_text(), "ALGO") == "Normal"  # never a non-self-consistent ZFS run
    (out / "ZFS_AO" / "OUTCAR").write_text("ran\n")
    init_zfs(relax, "AO")
    assert (out / "ZFS_AO" / "OUTCAR").is_file()  # kept
    init_zfs(relax, "AO", force=True)
    assert not (out / "ZFS_AO" / "OUTCAR").exists()


def test_cli(tmp_path, capsys):
    main(["init", "tree", "NV_8", str(_inputs(tmp_path)), "-o", str(tmp_path)])
    assert "created" in capsys.readouterr().out
    _relaxed(tmp_path / "NV_8")
    main(["init", "zfs", str(tmp_path / "NV_8" / "relax"), "--method", "UO"])
    assert (tmp_path / "NV_8" / "ZFS_UO" / "run_ground_state.sh").is_file()
    with pytest.raises(SystemExit):
        main(["init", "zfs", str(tmp_path / "NV_8" / "relax"), "--method", "approx"])


def test_cli_resolves_relative_and_home_paths(tmp_path, monkeypatch, capsys):
    """Paths given relative to the working directory or with ~ end up absolute, in the tree and in the job scripts."""
    home = tmp_path / "home"
    (home / "bin").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    _inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    main(["init", "tree", "NV_8", "inputs", "-o", "."])
    assert f"created {tmp_path / 'NV_8'};" in capsys.readouterr().out
    _relaxed(tmp_path / "NV_8")
    monkeypatch.chdir(tmp_path / "NV_8")
    main(["init", "zfs", "./relax/", "--method", "all", "--vasp-binary", "../bin/vasp_std"])
    main(["init", "zfs", "relax", "--method", "decont", "--vasp-binary", "~/bin/vasp_std"])
    assert _tag((tmp_path / "NV_8" / "ZFS_AO" / "run_ground_state.sh").read_text(), "binary") == str(
        tmp_path / "bin" / "vasp_std")
    assert _tag((tmp_path / "NV_8" / "ZFS_full" / "run_ground_state.sh").read_text(), "binary") == str(
        home / "bin" / "vasp_std")


def test_init_zfs_resolves_a_relative_binary(tmp_path, monkeypatch):
    init_tree("NV_8", _inputs(tmp_path), parent=tmp_path)
    relax = _relaxed(tmp_path / "NV_8")
    monkeypatch.chdir(tmp_path)
    assert init_zfs("NV_8/relax", "UO", vasp_binary="bin/vasp_std") == []
    assert relax.is_dir()
    assert _tag((tmp_path / "NV_8" / "ZFS_UO" / "run_ground_state.sh").read_text(), "binary") == str(
        tmp_path / "bin" / "vasp_std")


# -- NBANDS fitted to the ground-state job, so VASP keeps it and DOCCUP/DOCCDO fit -----------------------


def test_zfs_rounds_nbands_up_to_the_band_groups_of_the_job(tmp_path):
    out = init_tree("NV_8", _inputs(tmp_path), parent=tmp_path)
    relax = _relaxed(out)
    (relax / "INCAR").write_text(INCAR.replace("NCORE = 128", "NCORE = 32"))  # 256 ranks / 32 = 8 band groups
    assert init_zfs(relax, "all") == [] and init_zfs(relax, "unpaired") == []
    ao, uo = (out / "ZFS_AO" / "INCAR").read_text(), (out / "ZFS_UO" / "INCAR").read_text()
    assert _tag(ao, "NBANDS") == "8" and _tag(uo, "NBANDS") == "8"  # the relaxation's 6, up to a multiple of 8
    assert (_tag(ao, "DOCCUP"), _tag(ao, "DOCCDO")) == ("5*1.0 3*0.0", "3*1.0 5*0.0")  # the extra bands empty
    assert (_tag(uo, "DOCCUP"), _tag(uo, "DOCCDO")) == ("3*0.0 2*1.0 3*0.0", "8*0.0")


def test_zfs_decont_sublevels_follow_the_rounded_nbands(tmp_path):
    from spinlabs.vasp.occupations import nbands_problems

    out = init_tree("NV_8", _inputs(tmp_path), parent=tmp_path)
    relax = _relaxed(out)
    (relax / "INCAR").write_text(INCAR.replace("NCORE = 128", "NCORE = 32"))
    assert init_zfs(relax, "decont") == []
    for incar in [out / "ZFS_full" / "INCAR", *sorted((out / "ZFS_full").glob("ms_*/INCAR"))]:
        text = incar.read_text()
        assert _tag(text, "NBANDS") == "8", incar
        assert nbands_problems(text, 256) == [], incar


def test_zfs_replaces_a_finer_relaxation_grid_by_gamma(tmp_path, caplog):
    out = init_tree("NV_8", _inputs(tmp_path), parent=tmp_path)
    relax = _relaxed(out)
    (relax / "KPOINTS").write_text("relax grid\n0\nGamma\n3 3 3\n0 0 0\n")
    assert init_zfs(relax, "all") == []
    kpoints = (out / "ZFS_AO" / "KPOINTS").read_text().split("\n")
    assert kpoints[1:4] == ["0", "Gamma", "1 1 1"]
    assert "Gamma 3 3 3, not Gamma 1 1 1" in caplog.text


# -- HSE06 (spinlabs init zfs --hse) ---------------------------------------------------------------------


def test_hse_incar_sets_the_hybrid_tags_in_place_and_appends_the_rest():
    from spinlabs.vasp.hse import HSE_HEADER, HSE_TAGS, hse_incar, is_hse

    text = hse_incar("ENCUT = 520\nALGO = Normal  # PBE\nNCORE = 32\n")
    assert _tag(text, "ALGO") == "All" and "ALGO = All  # PBE" in text  # in place, its comment kept
    assert _tag(text, "NCORE") == "1" and text.count("NCORE") == 1
    for tag, value in HSE_TAGS.items():
        assert _tag(text, tag) == value, tag
    assert HSE_HEADER in text and _tag(text, "ENCUT") == "520"
    assert hse_incar(text) == text  # a second pass changes nothing
    assert is_hse(text) and not is_hse("ENCUT = 520\n") and not is_hse("LHFCALC = .TRUE.\nLHFCALC = .FALSE.\n")


def test_zfs_hse_fits_nbands_to_ncore_1(tmp_path):
    """NCORE = 1 makes every MPI rank a band group: NBANDS goes up to a multiple of the job's 256 ranks, and the
    occupations follow, so VASP keeps NBANDS and reads DOCCUP/DOCCDO."""
    from spinlabs.vasp.occupations import nbands_problems

    out = init_tree("NV_8", _inputs(tmp_path), parent=tmp_path)
    relax = _relaxed(out)
    assert init_zfs(relax, "all", hse=True) == []
    incar = (out / "ZFS_AO" / "INCAR").read_text()
    assert (_tag(incar, "LHFCALC"), _tag(incar, "HFSCREEN"), _tag(incar, "NCORE")) == (".TRUE.", "0.2", "1")
    assert (_tag(incar, "ALGO"), _tag(incar, "PREC"), _tag(incar, "PRECFOCK")) == ("All", "Accurate", "Fast")
    assert _tag(incar, "LDMATRIX") == ".TRUE." and _tag(incar, "NSW") == "0"  # still the ZFS run
    assert _tag(incar, "NBANDS") == "256" and _tag(incar, "DOCCUP") == "5*1.0 251*0.0"
    assert nbands_problems(incar, 256) == []
    assert init_zfs(relax, "unpaired") == []  # without --hse: PBE, as before
    assert _tag((out / "ZFS_UO" / "INCAR").read_text(), "LHFCALC") is None


def test_cli_zfs_hse(tmp_path):
    out = init_tree("NV_8", _inputs(tmp_path), parent=tmp_path)
    main(["init", "zfs", str(_relaxed(out)), "--method", "all", "--hse"])
    assert _tag((out / "ZFS_AO" / "INCAR").read_text(), "LHFCALC") == ".TRUE."


def test_trees_of_an_hse_ground_state_keep_their_wavecar(tmp_path):
    """The trees copy the ground state's INCAR: an HSE06 one gets KEEP_WAVECAR=1 in its job scripts, a PBE one
    keeps the default (removed)."""
    from test_perturbation_tree import _save, _tiny  # the tree tests' tiny spectrum
    from spinlabs.vasp import build_perturbation_tree
    from spinlabs.vasp.hse import hse_incar

    defect = tmp_path / "NV_8"
    (defect / "data").mkdir(parents=True)
    _save(_tiny([10.0, 20.0, 30.0], ["A1", "Ex", "A2"]), defect / "data" / "phonon_data.npz")
    for stage, incar in (("ZFS_AO", hse_incar("ENCUT = 520\n")), ("ZFS_UO", "ENCUT = 520\n")):
        d = defect / stage
        d.mkdir()
        (d / "INCAR").write_text(incar)
        for name in ("POSCAR", "KPOINTS", "POTCAR"):
            (d / name).write_text(name + "\n")
        (d / "OUTCAR").write_text("Elapsed time (sec):   100.000\n")
    assert build_perturbation_tree(out=defect, pert=[0.05], displacement="normalized", conv_modes=None,
                                   phonon=defect / "data" / "phonon_data.npz", array_jobs=1, force=True,
                                   vasp_binary=Path("/bin/true")) == []
    for basis, keep in (("AO", "${KEEP_WAVECAR:-1}"), ("UO", "${KEEP_WAVECAR:-0}")):
        tree = defect / "first_order" / "pert_0.05" / basis
        assert _tag((tree / "run_perturbation_first_order.sh").read_text(), "KEEP_WAVECAR") == keep, basis
    hse = (defect / "first_order" / "pert_0.05" / "AO" / "input" / "INCAR").read_text()
    assert _tag(hse, "LHFCALC") == ".TRUE." and _tag(hse, "ICHARG") == "1"  # from the CHGCAR, as PBE runs
