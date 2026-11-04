"""The perturbation runner stops when its SLURM job ends instead of stamping every run as attempted.

On Dardel (job 27178920, 2026-11-03) a task past its time limit got no TERM: each srun failed at once
with "Job/step already completing or completed" and the loop marked 58 runs it never started as
attempted. The runner's own functions are run here with stub python, srun and squeue.
"""

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "cluster" / "perturbation_runner.sh"


def _runner_functions() -> tuple[str, str, str]:
    """Lock, job state, run_one, visit and walk from the runner, without its preflight and main loop."""
    text = RUNNER.read_text()
    body = text[text.index("# ---- run-directory lock ----"):text.index("mkdir -p runs")]
    loop = text[text.index("visit() {"):text.index("PASS=own")]
    final = text[text.index('if [ "$STOP" = yes ]; then'):text.index('if [ "$VASP_FAILED" -gt 0 ]')]
    return body, loop, final


def _stubs(tmp_path: Path, srun_body: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {
        # create_struct and check_perturbation both succeed
        "python": '#!/bin/sh\ncase "$*" in *check_perturbation*) echo "perturbation OK";; *) echo Success;; esac\n',
        # the job's state is read from a file the test (or srun) writes; the lock code asks with -o %i
        "squeue": '#!/bin/sh\ncase "$*" in *%T*) cat "$STATE_FILE";; *) exit 0;; esac\n',
        "srun": "#!/bin/sh\n" + srun_body,
    }
    for name, text in stubs.items():
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    return bin_dir


def _run(tmp_path: Path, srun_body: str, prev_attempt: dict | None = None, state: str = "RUNNING"):
    body, loop, final = _runner_functions()
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "INCAR").write_text("x\n")
    (tmp_path / "state").write_text(f"{state}\n")
    for run, tag in (prev_attempt or {}).items():
        (tmp_path / "runs" / run).mkdir(parents=True)
        (tmp_path / "runs" / run / ".attempted").write_text(f"{tag}\n")
    bin_dir = _stubs(tmp_path, srun_body)
    script = (
        "set -eu\nHEAD_DIR=$(pwd)\nID=$SLURM_ARRAY_TASK_ID\nNJOBS=1\nINPUT_DIR=input\n"
        "create_struct=create.py\nSCRIPTS=scripts\nPHONON_PATH=phon.npz\nPERT=0.05\nDISP_FLAG=\nbinary=vasp\n"
        "ZFS_REGEX_PATTERN=ZFSBLOCK\nORDER=first_order\nMODES='1 2 3'\nVASP_FAILED=0\nmkdir -p runs\n"
        + body + loop + "PASS=own\nwalk\n" + final + 'echo "VASP_FAILED=$VASP_FAILED STOP=$STOP"\n'
    )
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "SLURM_JOB_ID": "27178920",
           "SLURM_ARRAY_JOB_ID": "27178059", "SLURM_ARRAY_TASK_ID": "0", "STATE_FILE": str(tmp_path / "state")}
    return subprocess.run(["sh", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)


def test_a_job_that_ends_stops_the_task_and_leaves_its_runs_unmarked(tmp_path):
    # srun is refused because the job has just passed its time limit
    r = _run(tmp_path, 'echo COMPLETING > "$STATE_FILE"\necho "srun: error: Unable to create step" >&2\nexit 1\n',
             prev_attempt={"1": "26000000"})
    assert r.returncode == 0, r.stderr
    assert "mode 1 not run -- the job is COMPLETING" in r.stdout
    assert "job 27178920 is COMPLETING" in r.stdout and "VASP_FAILED=0 STOP=yes" in r.stdout
    assert "starting job for mode 2" not in r.stdout  # nothing started after the refusal
    assert (tmp_path / "runs/1/.attempted").read_text() == "26000000\n"  # its earlier marker is back
    assert not (tmp_path / "runs/2").exists() and not (tmp_path / "runs/1/.lock").exists()


def test_a_job_already_past_its_end_starts_nothing(tmp_path):
    r = _run(tmp_path, "exit 0\n", state="COMPLETING")
    assert r.returncode == 0, r.stderr
    assert "starting job" not in r.stdout and "VASP_FAILED=0 STOP=yes" in r.stdout
    assert not any((tmp_path / "runs").iterdir())  # no run directory marked or filled


def test_a_vasp_crash_in_a_running_job_still_counts_and_the_task_goes_on(tmp_path):
    r = _run(tmp_path, "exit 1\n")  # the job stays RUNNING
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("VASP exited with 1") == 3 and "VASP_FAILED=3 STOP=no" in r.stdout
    assert all((tmp_path / f"runs/{m}/.attempted").read_text() == "27178059\n" for m in (1, 2, 3))


def test_without_an_answer_from_squeue_the_job_counts_as_running(tmp_path):
    r = _run(tmp_path, 'echo "" > "$STATE_FILE"\nexit 1\n')  # squeue prints nothing
    assert "VASP_FAILED=3 STOP=no" in r.stdout
