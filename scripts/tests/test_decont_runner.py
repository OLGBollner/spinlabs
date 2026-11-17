"""The perturbation runner in a --full-decont tree: SCF -> WAVECAR, then the ALGO=None ZFS of each spin sublevel."""

import os
import subprocess
from pathlib import Path

from test_displacement_scripts import ENV, REPO, SCRIPTS, STEP, _defect

INCAR = """ENCUT = 520
ALGO = Normal
LDMATRIX = .TRUE.
NUPDOWN = 2
NBANDS = 8
DOCCUP = 4*1.0 4*0.0
DOCCDO = 2*1.0 6*0.0
"""
SUBLEVELS = ["ms_0_3", "ms_0_4", "ms_1"]
RUNS = sorted(f"{m}_pert_{s}" for m in (2, 7) for s in ("-0.025", "0.025"))
ZFS = (" Spin-spin contribution to zero-field splitting tensor (MHz)\n ---\n"
       "      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz\n ---\n"
       "  -1.0  -1.0  2.0  0.0  0.0  0.0\n ---\n")


def _fake_slurm(tmp_path: Path) -> Path:
    """srun as a fake VASP: an ALGO=None run writes a ZFS block (none in the folder named by $FAIL_IN),
    the SCF writes WAVECAR, CHGCAR and a converged OSZICAR. Every call is logged to $SRUN_LOG."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "srun").write_text(f"""#!/bin/sh
here=$(basename "$(pwd)")
if grep -q '^ALGO = None' INCAR; then
  echo "zfs $here" >> "$SRUN_LOG"
  [ -L WAVECAR ] && [ -f WAVECAR ] || exit 3
  if [ "$here" = "${{FAIL_IN:-}}" ]; then echo 'no block' > OUTCAR; else printf '%s' '{ZFS}' > OUTCAR; fi
  echo ' Elapsed time (sec):  10.0' >> OUTCAR
else
  echo "scf $here" >> "$SRUN_LOG"
  echo wave > WAVECAR; echo chg > CHGCAR
  printf 'DAV:   1    -0.1E+02   -0.1E-05\\n   1 F= -10.0\\n' > OSZICAR
  echo ' Elapsed time (sec):  100.0' > OUTCAR
fi
""")
    (bin_dir / "squeue").write_text("#!/bin/sh\nexit 0\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    return bin_dir


def _tree(tmp_path: Path) -> Path:
    from spinlabs.vasp.trees import build_convergence_tree

    defect = _defect(tmp_path)
    (defect / "ZFS_hyp" / "INCAR").write_text(INCAR)
    assert build_convergence_tree(defect, defect / "data" / "phonon_data.npz", [2, 7], [STEP],
                                  displacement="normalized", vasp_binary=Path("/bin/true"), decont=True) == []
    return defect / "convergence_test_decont"


def _run(tmp_path: Path, folder: Path, job: str = "99", bin_dir: Path | None = None, **extra_env):
    env = {**ENV, "PATH": f"{bin_dir or _fake_slurm(tmp_path)}:{ENV['PATH']}", "SPINLABS_CODE": str(REPO), "DEFECT": "SiV_3",
           "binary": "/bin/true", "PHONON_PATH": str(folder.parent / "data" / "phonon_data.npz"), "MODES": "2 7",
           "PERTS": str(STEP), "DISPLACEMENT": "normalized", "SLURM_ARRAY_TASK_ID": "0",
           "SLURM_ARRAY_TASK_COUNT": "1", "SLURM_ARRAY_JOB_ID": job, "SRUN_LOG": str(tmp_path / "srun.log"),
           **extra_env}
    return subprocess.run(["sh", str(SCRIPTS / "perturbation_runner.sh"), "convergence"], cwd=folder, env=env,
                          capture_output=True, text=True)


def test_each_run_does_the_scf_then_every_sublevel_from_its_wavecar(tmp_path):
    folder = _tree(tmp_path)
    r = _run(tmp_path, folder)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "full spin decontamination" in r.stdout
    assert sorted(p.name for p in (folder / "runs").iterdir() if (p / ".done").is_file()) == RUNS
    ground = folder.parent / "ZFS_full"  # the defect's ground state, next to ZFS_hyp/
    for d in [ground] + [folder / "runs" / run for run in RUNS]:
        assert (d / "WAVECAR").is_file() and not (d / "WAVECAR").is_symlink()  # kept
        assert not (d / "CHGCAR").exists()
        for level in SUBLEVELS:
            assert (d / level / ".done").is_file()
            assert os.readlink(d / level / "WAVECAR") == "../WAVECAR"
            assert "ALGO = None" in (d / level / "INCAR").read_text()
            assert (d / level / "POSCAR").read_text() == (d / "POSCAR").read_text()
    # the ground state keeps the unperturbed structure, the runs are displaced
    assert (ground / "POSCAR").read_text() == (folder / "input" / "POSCAR").read_text()
    assert (folder / "runs" / "2_pert_0.025" / "POSCAR").read_text() != (folder / "input" / "POSCAR").read_text()
    assert not (folder / "runs" / "ref").exists()
    log = (tmp_path / "srun.log").read_text().split("\n")
    assert log[0] == "scf ZFS_full"  # first, before the runs
    assert sum(line.startswith("scf") for line in log) == len(RUNS) + 1
    assert sum(line.startswith("zfs") for line in log) == (len(RUNS) + 1) * len(SUBLEVELS)


def test_the_ground_state_runs_once_per_defect(tmp_path):
    """A second decontamination tree of the same defect skips the finished ZFS_full."""
    from spinlabs.vasp.trees import build_perturbation_tree

    folder = _tree(tmp_path)
    assert _run(tmp_path, folder).returncode == 0
    defect = folder.parent
    assert build_perturbation_tree(defect, [STEP], defect / "data" / "phonon_data.npz", displacement="normalized",
                                   conv_modes=None, array_jobs=1, vasp_binary=Path("/bin/true"), decont=True) == []
    tree = defect / "first_order" / f"pert_{STEP}" / "full_decont"
    (tmp_path / "srun.log").unlink()
    env = {**ENV, "PATH": f"{_fake_slurm(tmp_path)}:{ENV['PATH']}", "SPINLABS_CODE": str(REPO), "DEFECT": "SiV_3",
           "binary": "/bin/true", "PHONON_PATH": str(next((defect / "data").glob("phonon_data_sym*.npz"))),
           "PERT": str(STEP), "DISPLACEMENT": "normalized", "SLURM_ARRAY_TASK_ID": "0", "SLURM_ARRAY_TASK_COUNT": "1",
           "SLURM_ARRAY_JOB_ID": "200", "SRUN_LOG": str(tmp_path / "srun.log")}
    r = subprocess.run(["sh", str(SCRIPTS / "perturbation_runner.sh"), "first_order"], cwd=tree, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ground state in ../../../ZFS_full" in r.stdout
    assert "skipping ground state ZFS_full" in r.stdout
    assert "ZFS_full" not in (tmp_path / "srun.log").read_text()


def test_a_restart_reruns_only_the_sublevels_left(tmp_path):
    folder = _tree(tmp_path)
    r = _run(tmp_path, folder, FAIL_IN="ms_0_4")
    assert r.returncode == 0, r.stdout + r.stderr
    run = folder.parent / "ZFS_full"
    assert not (run / ".done").exists() and (run / ".scf_done").is_file()
    assert (run / "ms_1" / ".done").is_file() and not (run / "ms_0_4" / ".done").exists()
    assert (run / "CHGCAR").is_file()  # removed only once the run is done

    (tmp_path / "srun.log").unlink()
    r = _run(tmp_path, folder, job="100")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (run / ".done").is_file()
    log = sorted((tmp_path / "srun.log").read_text().split())
    assert log == sorted(["zfs", "ms_0_4"] * (len(RUNS) + 1))  # no SCF, only the failed sublevel


def test_an_unconverged_scf_runs_no_sublevel(tmp_path):
    folder = _tree(tmp_path)
    bin_dir = _fake_slurm(tmp_path)
    srun = bin_dir / "srun"
    srun.write_text(srun.read_text().replace("DAV:   1    -0.1E+02   -0.1E-05", "DAV:  60    -0.1E+02   -0.1E+01"))
    r = _run(tmp_path, folder, bin_dir=bin_dir)
    assert r.returncode == 0, r.stdout + r.stderr
    run = folder.parent / "ZFS_full"
    assert not (run / ".scf_done").exists() and not (run / "ms_1").exists()
    assert "SCF did not converge" in (run / "CHECK").read_text()
