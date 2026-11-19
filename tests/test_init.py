"""spinlabs init tree / init zfs: a new defect's folders and its ground-state ZFS runs."""

import re
from pathlib import Path

import pytest

from spinlabs.cli import main
from spinlabs.vasp.init import init_tree, init_zfs

INCAR = "ENCUT = 520\nALGO = Normal\nISMEAR = 0\nIBRION = 2\nNSW = 50\nISYM = 3\nNELECT = 8\nISPIN = 2\n"


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
        assert f"#SBATCH -J {job}" in script and re.search(r"^binary=/bin/true$", script, re.M)


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
