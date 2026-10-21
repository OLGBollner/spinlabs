"""Tests for phonon resolution in create_perturbation_tree."""

import re

from pathlib import Path

import numpy as np
import pytest

from spinlabs.vasp import build_zfs_tree, default_phonon, resolve_sym_phonon, sbatch_time


@pytest.fixture
def defect(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    return tmp_path, data


def _save(spec, path):
    from spinlabs.parsers import save_phonon_npz

    save_phonon_npz(spec, path)


def _tiny(freqs, syms):
    """Minimal PhononSpectrum with preset labels (bypasses classification)."""
    from spinlabs.models import PhononSpectrum

    n = len(freqs)
    return PhononSpectrum(
        frequencies_mev=np.asarray(freqs, dtype=float),
        eigenvectors=np.zeros((n, 1, 3)),
        atom_frac_coords=np.zeros((1, 3)),
        atom_symbols=["C"],
        atomic_masses=np.array([12.0]),
        lattice=np.eye(3),
        symmetries=list(syms),
        original_indices=np.arange(n),
    )


def test_finds_existing_sym_file(defect):
    out, data = defect
    spec = _tiny([10.0, 20.0, 30.0], ["A1", "Ex", "A2"])
    _save(spec, data / "phonon_data.npz")
    _save(spec, data / "phonon_data_sym3.npz")
    phonon = resolve_sym_phonon(out, data / "phonon_data.npz")
    assert phonon == data / "phonon_data_sym3.npz"


def test_creates_sym_file_when_missing(defect):
    out, data = defect
    spec = _tiny([10.0, 20.0, 30.0], ["A1", "Ex", "A2"])
    _save(spec, data / "phonon_data.npz")
    phonon = resolve_sym_phonon(out, data / "phonon_data.npz")
    assert phonon.is_file()
    assert phonon.name == "phonon_data_sym3.npz"
    assert phonon.parent == data


def test_default_phonon_in_data_folder(defect):
    out, data = defect
    _save(_tiny([10.0, 20.0, 30.0], ["A1", "Ex", "A2"]), data / "phonon_data.npz")

    phonon = default_phonon(out)
    assert phonon == data / "phonon_data.npz"


def test_sbatch_time_rounds_up():
    assert sbatch_time(0.1) == "0:01:00"
    assert sbatch_time(3661) == "1:02:00"


def test_build_zfs_tree(tmp_path):
    inp = tmp_path / "NV_512"
    (inp / "data").mkdir(parents=True)
    (inp / "template" / "relax").mkdir(parents=True)
    (inp / "data" / "POSCAR").write_text("poscar\n")
    (inp / "template" / "relax" / "INCAR").write_text("INCAR\n")

    out = build_zfs_tree(inp, out=tmp_path / "tree")
    assert (out / "data" / "POSCAR").is_file()
    for stage in ("relaxation_data", "ZFS_hyp", "ZFS_occup"):
        assert (out / stage / "POSCAR").is_file()
        assert (out / stage / "INCAR").is_file()


def test_build_zfs_tree_refuses_overwrite(tmp_path):
    inp = tmp_path / "NV_512"
    (inp / "data").mkdir(parents=True)
    (inp / "template" / "relax").mkdir(parents=True)
    (inp / "data" / "POSCAR").write_text("poscar\n")
    (inp / "template" / "relax" / "INCAR").write_text("INCAR\n")
    build_zfs_tree(inp, out=tmp_path / "tree")
    with pytest.raises(FileExistsError):
        build_zfs_tree(inp, out=tmp_path / "tree")


def test_patch_restart_incar_rewrites_flags():
    from spinlabs.vasp.trees import patch_restart_incar

    def tag(text, name):
        vals = re.findall(rf"^\s*{name}\s*=\s*(.+?)\s*(?:!|#|$)", text, re.M | re.I)
        return vals[-1] if vals else None

    incar = "ISYM = 3\nISTART = 2\nICHARG = 2\nLWAVE = .FALSE.\nLCHARG = .FALSE.\n"
    out = patch_restart_incar(incar)
    assert tag(out, "ISTART") == "0"
    assert tag(out, "ICHARG") == "1"
    assert tag(out, "LWAVE") == ".TRUE."
    assert tag(out, "LCHARG") == ".TRUE."
    assert out.count("ISTART") == 1  # old value replaced, not duplicated


def test_patch_restart_incar_appends_missing():
    from spinlabs.vasp.trees import patch_restart_incar

    out = patch_restart_incar("ENCUT = 520\n")
    for name in ("ISTART", "ICHARG", "LWAVE", "LCHARG"):
        assert re.search(rf"^\s*{name}\s*=", out, re.M | re.I), name
    assert re.search(r"^\s*ENCUT\s*=", out, re.M)


def test_forbid_algo_none_corrects_and_warns():
    from spinlabs.vasp.trees import forbid_algo_none

    incar = "ALGO = None\nENCUT = 520\n"
    out, warning = forbid_algo_none(incar)
    assert re.search(r"^\s*ALGO\s*=\s*Normal\b", out, re.M | re.I)
    assert out.count("ALGO") == 1
    assert warning is not None and "ALGO=Normal" in warning


def test_forbid_algo_none_leaves_scf_incar_alone():
    from spinlabs.vasp.trees import forbid_algo_none

    out, warning = forbid_algo_none("ALGO = Normal\nENCUT = 520\n")
    assert warning is None
    assert out == "ALGO = Normal\nENCUT = 520\n"
    out, warning = forbid_algo_none("ENCUT = 520\n")
    assert warning is None
    assert out == "ENCUT = 520\n"


def test_perturbation_input_incar_is_restart_ready(tmp_path):
    from spinlabs.vasp import build_perturbation_tree

    defect = tmp_path / "NV_512"
    (defect / "data").mkdir(parents=True)
    _save(_tiny([10.0, 20.0, 30.0], ["A1", "Ex", "A2"]), defect / "data" / "phonon_data.npz")
    for stage in ("ZFS_hyp", "ZFS_occup"):
        d = defect / stage
        d.mkdir()
        (d / "INCAR").write_text("ENCUT = 520\nISTART = 2\nICHARG = 2\n")
        for name in ("POSCAR", "KPOINTS", "POTCAR"):
            (d / name).write_text(name + "\n")
        (d / "OUTCAR").write_text("Elapsed time (sec):   100.000\n")
    failures = build_perturbation_tree(
        out=defect,
        pert=[0.0001],
        phonon=defect / "data" / "phonon_data.npz",
        scripts_dir=Path("scripts/cluster"),
        array_jobs=1,
        force=True,
        vasp_binary=Path("/bin/true"),
    )
    assert failures == []
    incar = (defect / "first_order" / "pert_0.0001" / "all_bands" / "input" / "INCAR").read_text()
    for name, value in (("ISTART", "0"), ("ICHARG", "1"), ("LWAVE", ".TRUE."), ("LCHARG", ".TRUE.")):
        vals = re.findall(rf"^\s*{name}\s*=\s*(.+?)\s*(?:!|#|$)", incar, re.M | re.I)
        assert vals, name
        assert vals[-1] == value, name


def test_forbid_sym_ldmatrix_corrects_isym_12():
    from spinlabs.vasp.trees import forbid_sym_ldmatrix

    for isym in ("1", "2"):
        out, warning = forbid_sym_ldmatrix(f"ISYM = {isym}\nLDMATRIX = .TRUE.\n")
        assert re.search(r"^\s*ISYM\s*=\s*3\b", out, re.M | re.I)
        assert warning is not None and f"ISYM={isym}" in warning


def test_forbid_sym_ldmatrix_leaves_valid_incars_alone():
    from spinlabs.vasp.trees import forbid_sym_ldmatrix

    cases = (
        "ISYM = 3\nLDMATRIX = .TRUE.\n",  # already safe
        "LDMATRIX = .TRUE.\n",  # no ISYM tag
        "ISYM = 2\nENCUT = 520\n",
    )  # no LDMATRIX
    for incar in cases:
        out, warning = forbid_sym_ldmatrix(incar)
        assert warning is None and out == incar


def test_perturbation_input_incar_corrects_algo_none(tmp_path, caplog):
    """ALGO=None in the ZFS source INCAR is corrected to Normal with a log."""
    from spinlabs.vasp import build_perturbation_tree

    defect = tmp_path / "NV_512"
    (defect / "data").mkdir(parents=True)
    _save(_tiny([10.0, 20.0, 30.0], ["A1", "Ex", "A2"]), defect / "data" / "phonon_data.npz")
    for stage in ("ZFS_hyp", "ZFS_occup"):
        d = defect / stage
        d.mkdir()
        (d / "INCAR").write_text("ENCUT = 520\nALGO = None\n")
        for name in ("POSCAR", "KPOINTS", "POTCAR"):
            (d / name).write_text(name + "\n")
        (d / "OUTCAR").write_text("Elapsed time (sec):   100.000\n")
    failures = build_perturbation_tree(
        out=defect,
        pert=[0.0001],
        phonon=defect / "data" / "phonon_data.npz",
        scripts_dir=Path("scripts/cluster"),
        array_jobs=1,
        force=True,
        vasp_binary=Path("/bin/true"),
    )
    assert failures == []
    incar = (defect / "first_order" / "pert_0.0001" / "all_bands" / "input" / "INCAR").read_text()
    assert re.search(r"^\s*ALGO\s*=\s*Normal\b", incar, re.M | re.I)
    assert "ALGO=None" not in incar
    out = caplog.text
    assert "WARN" in out and "ALGO=Normal" in out


def test_perturbation_input_incar_corrects_isym_with_ldmatrix(tmp_path, caplog):
    """ISYM=1/2 with LDMATRIX in the source INCAR is corrected to 3."""
    from spinlabs.vasp import build_perturbation_tree

    defect = tmp_path / "NV_512"
    (defect / "data").mkdir(parents=True)
    _save(_tiny([10.0, 20.0, 30.0], ["A1", "Ex", "A2"]), defect / "data" / "phonon_data.npz")
    for stage in ("ZFS_hyp", "ZFS_occup"):
        d = defect / stage
        d.mkdir()
        (d / "INCAR").write_text("ENCUT = 520\nISYM = 2\nLDMATRIX = .TRUE.\n")
        for name in ("POSCAR", "KPOINTS", "POTCAR"):
            (d / name).write_text(name + "\n")
        (d / "OUTCAR").write_text("Elapsed time (sec):   100.000\n")
    failures = build_perturbation_tree(
        out=defect,
        pert=[0.0001],
        phonon=defect / "data" / "phonon_data.npz",
        scripts_dir=Path("scripts/cluster"),
        array_jobs=1,
        force=True,
        vasp_binary=Path("/bin/true"),
    )
    assert failures == []
    incar = (defect / "first_order" / "pert_0.0001" / "all_bands" / "input" / "INCAR").read_text()
    assert re.search(r"^\s*ISYM\s*=\s*3\b", incar, re.M | re.I)
    out = caplog.text
    assert "WARN" in out and "ISYM=2" in out and "LDMATRIX" in out


def test_forbid_relaxation_corrects_ibrion():
    from spinlabs.vasp.trees import forbid_relaxation

    for incar in ("IBRION = 2\nNSW = 100\nLDMATRIX = .TRUE.\n", "NSW = 10\nLDMATRIX = .TRUE.\n"):
        out, warning = forbid_relaxation(incar)
        assert "IBRION = -1" in out.replace("IBRION=-1", "IBRION = -1") or "-1" in out
        assert warning is not None and "IBRION=-1" in warning


def test_forbid_relaxation_leaves_static_incars_alone():
    from spinlabs.vasp.trees import forbid_relaxation

    for incar in (
        "IBRION = -1\nLDMATRIX = .TRUE.\n",
        "LDMATRIX = .TRUE.\n",
        "IBRION = -1\nNSW = 0\nLDMATRIX = .TRUE.\n",
        "IBRION = 2\nNSW = 100\n",
    ):  # no LDMATRIX -> untouched
        out, warning = forbid_relaxation(incar)
        assert out == incar and warning is None


# ---------- verify_setup (pre-sbatch INCAR + script checks) ----------

REPO = Path(__file__).resolve().parents[1]  # this checkout, which holds the runner


def _verify_folder(
    tmp_path,
    script="DEFECT=NV_512\nBINARY={bin}\n"
    "PHONON_PATH={phon}\nSPINLABS_REPO={repo}\nPERT=0.025\n",
    incar="LDMATRIX = .TRUE.\nALGO = Normal\nIBRION = -1\n",
):
    from spinlabs.vasp.verify import verify_setup

    phon = tmp_path / "phon.npz"
    phon.write_bytes(b"x")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "INCAR").write_text(incar)
    (tmp_path / "run_x.sh").write_text(script.format(bin="/bin/true", phon=str(tmp_path / "phon.npz"), repo=REPO))
    return verify_setup(tmp_path, "run_x.sh")


def test_verify_setup_passes_clean_folder(tmp_path):
    problems = _verify_folder(tmp_path)
    assert [p for p in problems if p.startswith("FAIL")] == []
    assert problems == []


def test_verify_setup_flags_algo_none(tmp_path):
    problems = _verify_folder(tmp_path, incar="LDMATRIX = .TRUE.\nALGO = None\n")
    assert any(p.startswith("FAIL") and "ALGO=None" in p for p in problems)


def test_verify_setup_flags_leftover_placeholder(tmp_path):
    problems = _verify_folder(
        tmp_path, script="DEFECT=<defect>\nBINARY=/bin/true\nPHONON_PATH=/tmp/phon.npz\nPERT=0.025\n"
    )
    assert any("placeholder" in p for p in problems)


def test_verify_setup_flags_missing_phonon_path(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "INCAR").write_text("LDMATRIX = .TRUE.\n")
    (tmp_path / "run_x.sh").write_text("BINARY=/bin/true\nPHONON_PATH=/no/such/phon.npz\n")
    from spinlabs.vasp.verify import verify_setup

    problems = verify_setup(tmp_path, "run_x.sh")
    assert any(p.startswith("FAIL") and "PHONON_PATH" in p for p in problems)


def test_verify_setup_warns_on_gamma_binary(tmp_path):
    problems = _verify_folder(
        tmp_path, script="BINARY=/opt/vasp/vasp.5.4.4_gam\nPHONON_PATH=" + str(tmp_path / "phon.npz") + "\nPERT=0.025\n"
    )
    assert any(p.startswith("WARNING") and "gamma-only" in p for p in problems)


def test_verify_setup_lowercase_vars_and_preflight_lines(tmp_path):
    # Cluster-style script: lowercase config vars, preflight fail/case lines
    # that legitimately contain '<...>' placeholders must not be flagged.
    script = (
        "#!/bin/sh\n"
        "# config block\n"
        "DEFECT=NV_512\n"
        "binary=/bin/true\n"
        "PHONON_PATH={phon}\n"
        "SPINLABS_REPO={repo}\n"
        "PERT=0.025\n"
        'fail() {{ echo "FATAL: $*"; exit 1; }}\n'
        "case $DEFECT in *'<defect>'*) fail \"DEFECT placeholder\" ;; esac\n"
        "case $binary in *'<path_to>'*) fail \"BINARY placeholder\" ;; esac\n"
        '[ -x "$binary" ] || fail "VASP binary missing: $binary"\n'
    )
    problems = _verify_folder(tmp_path, script=script)
    assert problems == []


def test_verify_setup_flags_real_unfilled_placeholder(tmp_path):
    problems = _verify_folder(
        tmp_path, script="DEFECT=<defect>\nbinary=/bin/true\nPHONON_PATH=/tmp/phon.npz\nPERT=0.025\n"
    )
    assert any(":1 placeholder not filled: <defect>" in p for p in problems)


def test_verify_setup_flags_repo_without_runner(tmp_path):
    other = tmp_path / "not_spinlabs"
    other.mkdir()
    problems = _verify_folder(
        tmp_path,
        script="BINARY={bin}\nPHONON_PATH={phon}\nSPINLABS_REPO=" + str(other) + "\nPERT=0.025\n",
    )
    assert any(p.startswith("FAIL") and "SPINLABS_REPO has no scripts/cluster/perturbation_runner.sh" in p for p in problems)


def test_verify_setup_requires_spinlabs_repo(tmp_path):
    problems = _verify_folder(tmp_path, script="BINARY={bin}\nPHONON_PATH={phon}\nPERT=0.025\n")
    assert any(p.startswith("FAIL") and "SPINLABS_REPO not set" in p for p in problems)
