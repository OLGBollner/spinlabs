"""The displacement convention through the cluster scripts: tree creation, structure scripts, script updates."""

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "cluster"
ENV = {"PYTHONPATH": str(REPO / "src"), "PATH": os.environ["PATH"]}
STEP = 0.025

# The convergence job script used on the cluster before run_convergence.sh (DiV_128).
OLD_CONVERGENCE = """#!/bin/sh
#SBATCH -A naiss2025-3-15
#SBATCH -J DiV_128_convergence
#SBATCH -t 1:00:00
#SBATCH -a 0-4

# ---- config (override via environment, e.g. DEFECT=NV_512 sbatch ...) ----
DEFECT=DiV_128
binary=/cfs/vasp_binary/vasp_std
create_struct=/cfs/exjobb/scripts/cluster/create_phonon_struct.py
PHONON_PATH=/cfs/DiV_128/data/phonon_data_sym252.npz
PERTS="-0.1 -0.05 -0.025 0.025 0.05 0.1"
MODES="13 62 231 378"
# --------------------------------------------------------------------------
PERT_ID=0
for p in $PERTS; do :; done
"""


def _run(*args, cwd=None):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, cwd=cwd, env=ENV)


def _config(path: Path, key: str) -> str | None:
    match = re.search(rf"^{key}=(.*?)(\s+#.*)?$", path.read_text(), re.M)
    return match.group(1) if match else None


def test_old_convergence_script_becomes_run_convergence_keeping_its_runs(tmp_path):
    folder = tmp_path / "DiV_128" / "convergence_test_all"
    folder.mkdir(parents=True)
    old = folder / "run_perturbation_first_order.sh"
    old.write_text(OLD_CONVERGENCE)
    r = _run("sh", SCRIPTS / "update_tree_scripts.sh", tmp_path / "DiV_128")
    assert r.returncode == 0, r.stdout + r.stderr
    new = folder / "run_convergence.sh"
    assert new.is_file() and not old.exists() and (folder / "run_perturbation_first_order.sh.bak").is_file()
    assert _config(new, "MODES") == '"13 62 231 378"'
    assert _config(new, "PERTS") == '"-0.1 -0.05 -0.025 0.025 0.05 0.1"'
    assert _config(new, "binary") == "/cfs/vasp_binary/vasp_std"
    assert _config(new, "PHONON_PATH") == "/cfs/DiV_128/data/phonon_data_sym252.npz"
    assert _config(new, "DISPLACEMENT") == "mass_weighted"  # what its runs used
    assert "#SBATCH -J DiV_128_convergence" in new.read_text()
    assert 'exec sh "$SPINLABS_CODE/$RUNNER" "$ORDER"' in new.read_text() and "ORDER=convergence" in new.read_text()


@pytest.mark.parametrize("config, flag, expected", [
    ("MASS_WEIGHTED=0\n", [], "unweighted"),
    ("MASS_WEIGHTED=${MASS_WEIGHTED:-1}\n", [], "mass_weighted"),
    # the default form as NV_64's first-order scripts have it: its value, 0, counts
    ("MASS_WEIGHTED=${MASS_WEIGHTED:-0}  # 1: displace by Q*e/sqrt(m); 0: by Q*e\n", [], "unweighted"),
    ("", [], "mass_weighted"),  # from before both settings
    ("", ["--no-mass-weighted"], "unweighted"),
    ("DISPLACEMENT=normalized\n", ["--no-mass-weighted"], "normalized"),  # the script's own setting wins
])
def test_updated_tree_scripts_keep_the_displacement_their_runs_used(tmp_path, config, flag, expected):
    tree = tmp_path / "SiV" / "first_order" / "pert_0.1" / "all_bands"
    tree.mkdir(parents=True)
    script = tree / "run_perturbation_first_order.sh"
    script.write_text(f"#!/bin/sh\nDEFECT=SiV\nPERT=0.1\n{config}")
    r = _run("sh", SCRIPTS / "update_tree_scripts.sh", *flag, tmp_path / "SiV")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _config(script, "DISPLACEMENT") == expected
    assert _config(script, "PERT") == "0.1" and "MASS_WEIGHTED=" not in script.read_text()


def _defect(tmp_path):
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.vasp.inputs import Poscar

    from spinlabs.models import PhononSpectrum
    from spinlabs.parsers import save_phonon_npz

    defect = tmp_path / "SiV_3"
    (defect / "data").mkdir(parents=True)
    ref = Structure(Lattice.cubic(3.57), ["C", "C", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0]])
    q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(9, 9)))
    save_phonon_npz(PhononSpectrum(
        frequencies_mev=np.linspace(10.0, 90.0, 9), eigenvectors=q.T.reshape(9, 3, 3),
        atom_frac_coords=np.array([s.frac_coords for s in ref]), atom_symbols=["C", "C", "Si"],
        atomic_masses=np.array([12.011, 12.011, 28.0855]), lattice=np.eye(3) * 3.57,
        symmetries=["A1"] * 9, original_indices=np.arange(9),
    ), defect / "data" / "phonon_data.npz")
    for stage in ("ZFS_hyp", "ZFS_occup"):
        d = defect / stage
        d.mkdir()
        (d / "INCAR").write_text("ENCUT = 520\n")
        Poscar(ref).write_file(d / "POSCAR")
        for name in ("KPOINTS", "POTCAR"):
            (d / name).write_text(name + "\n")
        (d / "OUTCAR").write_text("Elapsed time (sec):   100.000\n")
    return defect


def test_tree_creation_requires_a_step_and_convergence_modes(tmp_path):
    defect = _defect(tmp_path)
    create = SCRIPTS / "create_perturbation_tree.py"
    no_step = _run("python", create, defect, "--no-convergence", cwd=tmp_path)
    assert no_step.returncode != 0 and "--pert" in no_step.stderr
    no_modes = _run("python", create, defect, "--pert", STEP, cwd=tmp_path)
    assert no_modes.returncode != 0 and "--conv-modes" in no_modes.stderr
    made = _run("python", create, defect, "--pert", STEP, "--conv-modes", 2, 7, "--vasp-binary", "/bin/true",
                cwd=tmp_path)
    assert made.returncode == 0, made.stdout + made.stderr
    assert _config(defect / "first_order" / f"pert_{STEP}" / "all_bands" / "run_perturbation_first_order.sh",
                   "DISPLACEMENT") == "normalized"
    assert (defect / "convergence_test_all" / "run_convergence.sh").is_file()


def _fake_slurm(tmp_path: Path) -> Path:
    """srun that writes a complete ZFS block, squeue that reports no jobs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    zfs = (" Spin-spin contribution to zero-field splitting tensor (MHz)\n ---\n"
           "      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz\n ---\n"
           "  -1.0  -1.0  2.0  0.0  0.0  0.0\n ---\n")
    (bin_dir / "srun").write_text(f"#!/bin/sh\nprintf '%s' '{zfs}' > OUTCAR\n")
    (bin_dir / "squeue").write_text("#!/bin/sh\nexit 0\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    return bin_dir


@pytest.mark.parametrize("perts", ["0.025 0.05", "-0.05 -0.025 0.0 0.025 0.05 0.05"])
def test_convergence_runner_runs_each_step_size_at_both_signs(tmp_path, perts):
    """A list of sizes and an older signed list (with 0 and a repeat) give the same runs."""
    from spinlabs.vasp.trees import build_convergence_tree

    defect = _defect(tmp_path)
    phonon = defect / "data" / "phonon_data.npz"
    assert build_convergence_tree(defect, phonon, [2, 7], [STEP], displacement="normalized",
                                  vasp_binary=Path("/bin/true")) == []
    folder = defect / "convergence_test_all"
    env = {**ENV, "PATH": f"{_fake_slurm(tmp_path)}:{ENV['PATH']}", "SPINLABS_CODE": str(REPO), "DEFECT": "SiV_3",
           "binary": "/bin/true", "PHONON_PATH": str(phonon), "MODES": "2 7", "PERTS": perts,
           "DISPLACEMENT": "normalized", "SLURM_ARRAY_TASK_ID": "0", "SLURM_ARRAY_TASK_COUNT": "1",
           "SLURM_ARRAY_JOB_ID": "99"}
    r = subprocess.run(["sh", str(SCRIPTS / "perturbation_runner.sh"), "convergence"], cwd=folder, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "steps -0.025 0.025 -0.05 0.05" in r.stdout
    assert sorted(p.name for p in (folder / "runs").iterdir()) == sorted(
        f"{m}_pert_{s}" for m in (2, 7) for s in ("-0.025", "0.025", "-0.05", "0.05"))


def test_convergence_runner_refuses_a_non_numeric_step(tmp_path):
    folder = tmp_path / "conv"
    (folder / "input").mkdir(parents=True)
    env = {**ENV, "SPINLABS_CODE": str(REPO), "DEFECT": "SiV_3", "binary": "/bin/true",
           "PHONON_PATH": "/nonexistent.npz", "MODES": "2", "PERTS": "0.025 abc", "DISPLACEMENT": "normalized"}
    r = subprocess.run(["sh", str(SCRIPTS / "perturbation_runner.sh"), "convergence"], cwd=folder, env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "PERTS must be numbers, got: 'abc'" in r.stderr


def test_normalized_structure_passes_only_its_own_check(tmp_path):
    defect = _defect(tmp_path)
    poscar, phonons = defect / "ZFS_hyp" / "POSCAR", defect / "data" / "phonon_data.npz"
    made = _run("python", SCRIPTS / "create_phonon_struct.py", "-o", tmp_path / "POSCAR_n", poscar, phonons, 9, STEP,
                "--displacement", "normalized")
    assert made.returncode == 0 and "total displacement 0.025000" in made.stdout, made.stdout + made.stderr
    check = ["python", SCRIPTS / "check_perturbation.py", tmp_path / "POSCAR_n", poscar, phonons, 9, "--pert", STEP]
    assert _run(*check, "--displacement", "normalized").returncode == 0
    assert _run(*check, "--displacement", "mass_weighted").returncode == 1
    # a negative step (convergence runs) is a positional value, not an option
    neg = _run("python", SCRIPTS / "create_phonon_struct.py", "-o", tmp_path / "POSCAR_m", poscar, phonons, 9, -STEP,
               "--displacement", "normalized")
    assert neg.returncode == 0, neg.stdout + neg.stderr
    assert _run("python", SCRIPTS / "check_perturbation.py", tmp_path / "POSCAR_m", poscar, phonons, 9, "--pert", -STEP,
                "--displacement", "normalized").returncode == 0


def test_add_convergence_to_an_existing_defect_takes_its_trees_displacement(tmp_path):
    defect = _defect(tmp_path)
    tree = defect / "first_order" / "pert_0.1" / "all_bands"
    tree.mkdir(parents=True)
    (tree / "run_perturbation_first_order.sh").write_text("binary=/opt/vasp_std\nMASS_WEIGHTED=${MASS_WEIGHTED:-1}\n")
    r = _run("python", SCRIPTS / "add_convergence_tree.py", defect, "--modes", 2, 7, "--pert", 0.1, cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    script = defect / "convergence_test_occup" / "run_convergence.sh"
    assert _config(script, "DISPLACEMENT") == "mass_weighted"
    assert _config(script, "binary") == "/opt/vasp_std"
    assert _config(script, "PERTS") == '"0.05 0.1 0.2"'
