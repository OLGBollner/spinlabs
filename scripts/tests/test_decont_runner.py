"""The perturbation runner in a --full-decont tree: SCF -> WAVECAR, then the ALGO=None ZFS of each spin sublevel."""

import os
import subprocess
from pathlib import Path

import numpy as np

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


def _scf_wavecar(path: Path, nb: int = 8, nplw: int = 40) -> None:
    """The WAVECAR the fake SCF writes: one k-point, complex64; the unpaired pair (bands 3, 4 of the
    INCAR above) degenerate in both channels, the down pair rotated 70 deg against the up pair."""
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(nplw, nb)) + 1j * rng.normal(size=(nplw, nb)))
    up = q.T
    down = up.copy()
    t = np.radians(70.0)
    down[2:4] = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]) @ up[2:4]
    eig = [0.0, 1.0, 5.0, 5.0, 8.0, 9.0, 10.0, 11.0]
    recl = nplw * 8 + 8 * (4 + 3 * nb)
    records = [np.array([recl, 2, 45200], dtype=np.float64), np.array([1, nb, 520.0, *np.eye(3).ravel(), 0.0])]
    for coeffs in (up, down):
        records.append(np.array([nplw, 0, 0, 0, *[x for b in range(nb) for x in (eig[b], 0.0, 1.0)]]))
        records.extend(np.asarray(c, dtype=np.complex64) for c in coeffs)
    with open(path, "wb") as f:
        for rec in records:
            raw = rec.tobytes()
            f.write(raw + b"\0" * (recl - len(raw)))


def _fake_slurm(tmp_path: Path) -> Path:
    """srun as a fake VASP: an ALGO=None run writes a ZFS block (none in the folder named by $FAIL_IN),
    the SCF writes WAVECAR, CHGCAR and a converged OSZICAR. Every call is logged to $SRUN_LOG.
    In the folder named by $END_AT the job passes its time limit: squeue then reports it COMPLETING
    (from $STATE_FILE) and srun is refused, as on Dardel."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _scf_wavecar(bin_dir / "WAVECAR.scf")
    (bin_dir / "srun").write_text(f"""#!/bin/sh
here=$(basename "$(pwd)")
if [ -n "${{END_AT:-}}" ] && [ "$here" = "$END_AT" ]; then
  echo COMPLETING > "$STATE_FILE"
  echo "srun: error: Unable to create step for job 1: Job/step already completing or completed" >&2
  exit 1
fi
if grep -q '^ALGO = None' INCAR; then
  echo "zfs $here" >> "$SRUN_LOG"
  [ -L WAVECAR ] && [ -f WAVECAR ] || exit 3
  if [ "$here" = "${{FAIL_IN:-}}" ]; then echo 'no block' > OUTCAR; else printf '%s' '{ZFS}' > OUTCAR; fi
  echo ' Elapsed time (sec):  10.0' >> OUTCAR
else
  echo "scf $here" >> "$SRUN_LOG"
  cp "{bin_dir}/WAVECAR.scf" WAVECAR; echo chg > CHGCAR
  printf 'DAV:   1    -0.1E+02   -0.1E-05\\n   1 F= -10.0\\n' > OSZICAR
  echo ' Elapsed time (sec):  100.0' > OUTCAR
fi
""")
    (bin_dir / "squeue").write_text(
        '#!/bin/sh\ncase "$*" in *%T*) cat "${STATE_FILE:-/dev/null}" 2>/dev/null;; *%L*) echo UNLIMITED;; *) exit 0;; esac\n'
    )
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
    for d in [folder / "runs" / run for run in RUNS]:
        assert (d / "WAVECAR").is_file() and not (d / "WAVECAR").is_symlink()  # kept
        assert (d / ".wavecar_aligned").is_file()
        assert not (d / "CHGCAR").exists()
        for level in SUBLEVELS:
            assert (d / level / ".done").is_file()
            assert os.readlink(d / level / "WAVECAR") == "../WAVECAR"
            assert "ALGO = None" in (d / level / "INCAR").read_text()
            assert (d / level / "POSCAR").read_text() == (d / "POSCAR").read_text()
    assert (folder / "runs" / "2_pert_0.025" / "POSCAR").read_text() != (folder / "input" / "POSCAR").read_text()
    # the ground state is its own job: the tree's job leaves ZFS_full as it was made
    ground = folder.parent / "ZFS_full"
    assert not (ground / ".attempted").exists() and not (ground / "OUTCAR").exists()
    # every SCF's down e pair was rotated onto the up pair before the sublevels ran
    assert r.stdout.count("aligned down bands [3, 4] to up: rotation 70.0 -> 0.00 deg") == len(RUNS)
    from spinlabs.vasp.wavecar import pair_overlap

    assert np.allclose(np.abs(pair_overlap(folder / "runs" / RUNS[0] / "WAVECAR", (3, 4))), np.eye(2), atol=1e-5)
    log = (tmp_path / "srun.log").read_text().split("\n")
    assert sum(line.startswith("scf") for line in log) == len(RUNS)
    assert sum(line.startswith("zfs") for line in log) == len(RUNS) * len(SUBLEVELS)


def _run_ground_state(tmp_path: Path, defect: Path, job: str = "300", bin_dir: Path | None = None, **extra_env):
    env = {**ENV, "PATH": f"{bin_dir or _fake_slurm(tmp_path)}:{ENV['PATH']}", "SPINLABS_CODE": str(REPO),
           "DEFECT": defect.name, "binary": "/bin/true", "SLURM_JOB_ID": job, "SRUN_LOG": str(tmp_path / "srun.log"),
           **extra_env}
    return subprocess.run(["sh", str(SCRIPTS / "perturbation_runner.sh"), "ground_state"], cwd=defect / "ZFS_full",
                          env=env, capture_output=True, text=True)


def test_the_ground_state_folder_is_ready_to_submit_on_its_own(tmp_path):
    folder = _tree(tmp_path)
    ground = folder.parent / "ZFS_full"
    assert sorted(p.name for p in ground.iterdir()) == sorted(
        ["INCAR", "POSCAR", "KPOINTS", "POTCAR", "run_ground_state.sh", *SUBLEVELS])
    script = (ground / "run_ground_state.sh").read_text()
    assert "ORDER=ground_state" in script and "#SBATCH -J SiV_3_ground_state" in script
    assert "#SBATCH -a" not in script

    r = _run_ground_state(tmp_path, folder.parent)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (ground / ".done").is_file() and (ground / "WAVECAR").is_file()
    assert (ground / "CHGCAR").is_file()  # kept in the ground state (the runs drop theirs)
    assert (ground / ".wavecar_aligned").is_file() and "rotation 70.0 -> 0.00 deg" in r.stdout
    for level in SUBLEVELS:
        assert (ground / level / ".done").is_file()
        assert os.readlink(ground / level / "WAVECAR") == "../WAVECAR"
        assert (ground / level / "POSCAR").read_text() == (folder / "input" / "POSCAR").read_text()
    assert sorted((tmp_path / "srun.log").read_text().split("\n")[:-1]) == sorted(
        ["scf ZFS_full"] + [f"zfs {lv}" for lv in SUBLEVELS])

    # a resubmission of a finished ground state does nothing
    (tmp_path / "srun.log").unlink()
    r = _run_ground_state(tmp_path, folder.parent, job="301")
    assert r.returncode == 0 and "already done" in r.stdout
    assert not (tmp_path / "srun.log").exists()


def test_a_restart_reruns_only_the_sublevels_left(tmp_path):
    folder = _tree(tmp_path)
    ground = folder.parent / "ZFS_full"
    r = _run_ground_state(tmp_path, folder.parent, FAIL_IN="ms_0_4")
    assert r.returncode == 1  # a sublevel without its ZFS block fails the job
    assert not (ground / ".done").exists() and (ground / ".scf_done").is_file()
    assert (ground / "ms_1" / ".done").is_file() and not (ground / "ms_0_4" / ".done").exists()

    (tmp_path / "srun.log").unlink()
    r = _run(tmp_path, folder, FAIL_IN="ms_0_4")
    assert r.returncode == 0, r.stdout + r.stderr
    run = folder / "runs" / RUNS[0]
    assert not (run / ".done").exists() and (run / ".scf_done").is_file()
    assert (run / "CHGCAR").is_file()  # removed only once the run is done

    (tmp_path / "srun.log").unlink()
    assert _run_ground_state(tmp_path, folder.parent, job="302").returncode == 0
    r = _run(tmp_path, folder, job="100")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (ground / ".done").is_file() and (run / ".done").is_file()
    log = sorted((tmp_path / "srun.log").read_text().split())
    assert log == sorted(["zfs", "ms_0_4"] * (len(RUNS) + 1))  # no SCF, only the failed sublevel


def test_an_unconverged_scf_runs_no_sublevel(tmp_path):
    folder = _tree(tmp_path)
    bin_dir = _fake_slurm(tmp_path)
    srun = bin_dir / "srun"
    srun.write_text(srun.read_text().replace("DAV:   1    -0.1E+02   -0.1E-05", "DAV:  60    -0.1E+02   -0.1E+01"))
    r = _run(tmp_path, folder, bin_dir=bin_dir)
    assert r.returncode == 0, r.stdout + r.stderr
    run = folder / "runs" / RUNS[0]
    assert not (run / ".scf_done").exists() and not (run / "ms_1").exists()
    assert "SCF did not converge" in (run / "CHECK").read_text()
    r = _run_ground_state(tmp_path, folder.parent, bin_dir=bin_dir)
    assert r.returncode == 1
    ground = folder.parent / "ZFS_full"
    assert not (ground / ".scf_done").exists() and not (ground / "ms_1" / "OUTCAR").exists()


def test_a_wavecar_that_cannot_be_aligned_stops_the_sublevels(tmp_path):
    folder = _tree(tmp_path)
    bin_dir = _fake_slurm(tmp_path)
    (bin_dir / "WAVECAR.scf").write_bytes(b"not a wavecar")
    r = _run(tmp_path, folder, bin_dir=bin_dir)
    assert r.returncode == 0, r.stdout + r.stderr
    run = folder / "runs" / RUNS[0]
    assert (run / ".scf_done").is_file() and not (run / ".wavecar_aligned").exists()
    assert not (run / "ms_1").exists() and "WAVECAR alignment failed" in (run / "CHECK").read_text()


def _job_env(tmp_path: Path, end_at: str) -> dict:
    """A SLURM job that passes its time limit when srun is called in the folder ``end_at``."""
    (tmp_path / "state").write_text("RUNNING\n")
    return {"SLURM_JOB_ID": "27684478", "STATE_FILE": str(tmp_path / "state"), "END_AT": end_at}


def test_a_job_that_ends_in_a_sublevel_stops_without_failures(tmp_path):
    """Dardel job 27684478: past its time limit every srun was refused, and the runner went on to new runs."""
    folder = _tree(tmp_path)
    r = _run(tmp_path, folder, **_job_env(tmp_path, "ms_0_4"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "sublevel ms_0_4 not run -- the job is COMPLETING" in r.stdout
    assert "2 sublevel(s) left -- the job ended" in r.stdout and "has no ZFS block" not in r.stdout
    assert "VASP exited" not in r.stdout and "stopped: job 27684478 is COMPLETING" in r.stdout
    first = folder / "runs" / RUNS[0]
    assert (first / "ms_0_3" / ".done").is_file() and not (first / "ms_0_4" / ".done").exists()
    assert not (first / ".done").exists() and not (first / ".attempted").exists()  # not counted as failed
    assert [p.name for p in (folder / "runs").iterdir()] == [RUNS[0]]  # nothing else started
    assert "scf" not in (tmp_path / "srun.log").read_text().split("\n")[1:]

    # the next submission runs the sublevels left from the run's WAVECAR, then the rest
    (tmp_path / "srun.log").unlink()
    r = _run(tmp_path, folder, job="100")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (first / ".done").is_file()
    assert (tmp_path / "srun.log").read_text().split("\n")[:2] == ["zfs ms_0_4", "zfs ms_1"]


def test_a_job_that_ends_in_the_scf_starts_no_sublevel_and_nothing_else(tmp_path):
    folder = _tree(tmp_path)
    r = _run(tmp_path, folder, **_job_env(tmp_path, RUNS[0]))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "mode 2 step -0.025 (SCF) not run -- the job is COMPLETING" in r.stdout
    assert "VASP exited" not in r.stdout
    first = folder / "runs" / RUNS[0]
    assert not (first / ".scf_done").exists() and not (first / "ms_1").exists()
    assert not (first / ".attempted").exists()
    assert [p.name for p in (folder / "runs").iterdir()] == [RUNS[0]]


def test_the_ground_state_job_ends_cleanly_and_resumes(tmp_path):
    folder = _tree(tmp_path)
    ground = folder.parent / "ZFS_full"
    r = _run_ground_state(tmp_path, folder.parent, **_job_env(tmp_path, "ms_0_4"))
    assert r.returncode == 0, r.stdout + r.stderr  # a job that ended is not a failure
    assert (ground / ".scf_done").is_file() and not (ground / ".done").exists()
    (tmp_path / "srun.log").unlink()
    assert _run_ground_state(tmp_path, folder.parent, job="301").returncode == 0
    assert (ground / ".done").is_file()
    assert sorted((tmp_path / "srun.log").read_text().split("\n")[:-1]) == ["zfs ms_0_4", "zfs ms_1"]
