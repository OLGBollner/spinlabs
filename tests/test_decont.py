"""Full spin decontamination (Biktagirov): spin sublevels, their INCARs and the --full-decont trees."""

import re
from pathlib import Path

import numpy as np
import pytest

from spinlabs.vasp.occupations import expand_occupations, format_occupations, parse_occupations, spin_sublevels
from spinlabs.vasp.trees import decont_scf_incar, decont_sublevel_incar

NV_UP = [1.0] * 128 + [0.0] * 128  # 125 bulk bands + a1 (126) + ex, ey (127, 128), NBANDS = 256
NV_DN = [1.0] * 126 + [0.0] * 130


def _tag(incar: str, tag: str) -> str | None:
    vals = re.findall(rf"^\s*{tag}\s*=\s*(.+?)\s*(?:!|#|$)", incar, re.M | re.I)
    return vals[-1] if vals else None


def test_occupations_round_trip():
    assert expand_occupations("2*1.0 1*0.0 0.5") == [1.0, 1.0, 0.0, 0.5]
    assert format_occupations(NV_UP) == "128*1.0 128*0.0"
    assert expand_occupations(format_occupations(NV_DN, (126, 128))) == NV_DN


def test_nv_triplet_gives_ms1_and_the_two_spin_flips():
    levels = {lv.name: lv for lv in spin_sublevels(NV_UP, NV_DN)}
    assert list(levels) == ["ms_1", "ms_0_127", "ms_0_128"]
    assert levels["ms_1"].nupdown == 2
    assert levels["ms_1"].doccup() == "125*1.0 1*1.0 1*1.0 1*1.0 128*0.0"
    # the singlet determinant written out by hand for NV
    assert levels["ms_0_128"].nupdown == 0
    assert levels["ms_0_128"].doccup() == "125*1.0 1*1.0 1*1.0 1*0.0 128*0.0"
    assert levels["ms_0_128"].doccdo() == "125*1.0 1*1.0 1*0.0 1*1.0 128*0.0"
    assert levels["ms_0_127"].doccup() == "125*1.0 1*1.0 1*0.0 1*1.0 128*0.0"
    assert levels["ms_0_127"].doccdo() == "125*1.0 1*1.0 1*1.0 1*0.0 128*0.0"
    for level in levels.values():  # electrons are conserved
        assert sum(level.up) + sum(level.dn) == sum(NV_UP) + sum(NV_DN)


def test_quartet_gives_ms_three_halves_and_three_flips():
    up, dn = [1.0] * 7 + [0.0] * 3, [1.0] * 4 + [0.0] * 6
    levels = spin_sublevels(up, dn)
    assert [lv.name for lv in levels] == ["ms_1.5", "ms_0.5_5", "ms_0.5_6", "ms_0.5_7"]
    assert [lv.nupdown for lv in levels] == [3, 1, 1, 1]


@pytest.mark.parametrize(
    "up, dn, message",
    [([1.0, 1.0, 0.0], [1.0, 1.0, 0.0], "no unpaired"), ([1.0, 0.0], [1.0, 1.0], "down channel only")],
)
def test_no_high_spin_state_is_refused(up, dn, message):
    with pytest.raises(ValueError, match=message):
        spin_sublevels(up, dn)


ZFS_INCAR = """ENCUT = 520
ALGO = Normal
ISTART = 2
ICHARG = 1
ISYM = 3
LDMATRIX = .TRUE.
NUPDOWN = 2
NBANDS = 256
DOCCUP = 128*1.0 128*0.0
DOCCDO = 126*1.0 130*0.0
"""


def test_scf_incar_writes_the_wavecar_and_skips_ldmatrix():
    incar = decont_scf_incar(ZFS_INCAR)
    for tag, value in (("LDMATRIX", ".FALSE."), ("LWAVE", ".TRUE."), ("ALGO", "Normal"), ("ISTART", "0"),
                       ("ICHARG", "1"), ("DOCCUP", "128*1.0 128*0.0"), ("NUPDOWN", "2")):
        assert _tag(incar, tag) == value, tag


def test_sublevel_incar_reads_the_wavecar_with_algo_none():
    level = spin_sublevels(*parse_occupations(ZFS_INCAR))[2]
    incar = decont_sublevel_incar(ZFS_INCAR, level)
    for tag, value in (("ALGO", "None"), ("ISTART", "1"), ("ICHARG", "0"), ("LDMATRIX", ".TRUE."), ("NUPDOWN", "0"),
                       ("LWAVE", ".FALSE."), ("LCHARG", ".FALSE."), ("NSW", "0"), ("IBRION", "-1"),
                       ("DOCCUP", "125*1.0 1*1.0 1*1.0 1*0.0 128*0.0"),
                       ("DOCCDO", "125*1.0 1*1.0 1*0.0 1*1.0 128*0.0"), ("ENCUT", "520")):
        assert _tag(incar, tag) == value, tag
    assert len(re.findall(r"^\s*ALGO\s*=", incar, re.M)) == 1


def _defect(tmp_path: Path) -> Path:
    from spinlabs.models import PhononSpectrum
    from spinlabs.parsers import save_phonon_npz

    defect = tmp_path / "NV_64"
    (defect / "data").mkdir(parents=True)
    n = 3
    save_phonon_npz(PhononSpectrum(
        frequencies_mev=np.array([10.0, 20.0, 30.0]), eigenvectors=np.zeros((n, 1, 3)),
        atom_frac_coords=np.zeros((1, 3)), atom_symbols=["C"], atomic_masses=np.array([12.0]), lattice=np.eye(3),
        symmetries=["A1", "Ex", "A2"], original_indices=np.arange(n),
    ), defect / "data" / "phonon_data.npz")
    for stage in ("ZFS_hyp", "ZFS_occup"):
        d = defect / stage
        d.mkdir()
        (d / "INCAR").write_text(ZFS_INCAR)
        for name in ("POSCAR", "KPOINTS", "POTCAR"):
            (d / name).write_text(name + "\n")
        (d / "OUTCAR").write_text("Elapsed time (sec):   100.000\n")
    return defect


def test_full_decont_tree_has_the_scf_input_and_one_incar_per_sublevel(tmp_path):
    from spinlabs.vasp import build_perturbation_tree

    defect = _defect(tmp_path)
    failures = build_perturbation_tree(
        defect, [0.025], defect / "data" / "phonon_data.npz", displacement="normalized", conv_modes=[1],
        array_jobs=1, vasp_binary=Path("/bin/true"), decont=True,
    )
    assert failures == []
    assert sorted(p.relative_to(defect).as_posix() for p in defect.glob("*/pert_*/*")) == [
        "first_order/pert_0.025/full_decont", "second_order/pert_0.025/full_decont"]
    assert not (defect / "convergence_test_all").exists()
    for tree in (defect / "first_order" / "pert_0.025" / "full_decont", defect / "convergence_test_decont"):
        inp = tree / "input"
        assert _tag((inp / "INCAR").read_text(), "LWAVE") == ".TRUE."
        assert _tag((inp / "INCAR").read_text(), "ISYM") == "0"
        assert _tag((inp / "sublevels" / "ms_1" / "INCAR").read_text(), "ISYM") == "0"
        assert sorted(p.name for p in (inp / "sublevels").iterdir()) == ["ms_0_127", "ms_0_128", "ms_1"]
        assert _tag((inp / "sublevels" / "ms_0_128" / "INCAR").read_text(), "ALGO") == "None"

    from spinlabs.vasp.verify import is_decont_tree, verify_setup

    tree = defect / "first_order" / "pert_0.025" / "full_decont"
    assert is_decont_tree(tree)
    assert [p for p in verify_setup(tree, "run_perturbation_first_order.sh") if "INCAR" in p] == []
    assert (defect / "ZFS_full").is_dir()  # the ground state, at the defect level


def test_verify_flags_a_sublevel_incar_without_algo_none(tmp_path):
    from spinlabs.vasp import build_perturbation_tree
    from spinlabs.vasp.verify import verify_setup

    defect = _defect(tmp_path)
    build_perturbation_tree(defect, [0.025], defect / "data" / "phonon_data.npz", displacement="normalized",
                            conv_modes=None, array_jobs=1, vasp_binary=Path("/bin/true"), decont=True)
    tree = defect / "first_order" / "pert_0.025" / "full_decont"
    level = tree / "input" / "sublevels" / "ms_1" / "INCAR"
    level.write_text(level.read_text().replace("ALGO = None", "ALGO = Normal"))
    problems = verify_setup(tree, "run_perturbation_first_order.sh")
    assert any(p.startswith("FAIL") and "ms_1" in p and "ALGO" in p for p in problems)


def test_default_tree_is_unchanged_by_the_decont_option(tmp_path):
    from spinlabs.vasp import build_perturbation_tree

    defect = _defect(tmp_path)
    build_perturbation_tree(defect, [0.025], defect / "data" / "phonon_data.npz", displacement="normalized",
                            conv_modes=None, array_jobs=1, vasp_binary=Path("/bin/true"))
    inp = defect / "first_order" / "pert_0.025" / "all_bands" / "input"
    assert not (inp / "sublevels").exists()
    assert _tag((inp / "INCAR").read_text(), "LDMATRIX") == ".TRUE."
