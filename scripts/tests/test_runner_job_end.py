"""The perturbation runner around the end of its SLURM job, and restarts from a run's own CHGCAR.

On Dardel (job 27178920, 2026-11-03) a task past its time limit got no TERM: each srun failed at once
with "Job/step already completing or completed" and the loop marked 58 runs it never started as
attempted. The runner now stops when the job no longer runs, or has less time left than a run takes,
and restarts an unfinished run from its own CHGCAR when that is complete. The runner's own functions
are run here with stub python, srun, squeue and date.
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
        # create_struct and check_perturbation succeed; a CHGCAR is complete if it says COMPLETE
        "python": (
            '#!/bin/sh\ncase "$*" in\n'
            '  *spinlabs.vasp.restart*) eval "f=\\${$#}"; grep -q COMPLETE "$f" && echo complete && exit 0;'
            ' echo "density cut short"; exit 1;;\n'
            '  *check_perturbation*) echo "perturbation OK";;\n  *) echo Success;;\nesac\n'
        ),
        # the job's state and time left are read from files the test (or srun) writes;
        # the lock code asks with -o %i
        "squeue": '#!/bin/sh\ncase "$*" in *%T*) cat "$STATE_FILE";; *%L*) cat "$LEFT_FILE";; *) exit 0;; esac\n',
        # a clock that moves 300 s per call: a run (from its start to its end) takes 300 s
        "date": '#!/bin/sh\nt=$(cat "$CLOCK_FILE")\necho "$t"\necho $((t + 300)) > "$CLOCK_FILE"\n',
        "srun": "#!/bin/sh\n" + srun_body,
    }
    for name, text in stubs.items():
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    return bin_dir


def _run(tmp_path: Path, srun_body: str, prev_attempt: dict | None = None, state: str = "RUNNING",
         left: str = "UNLIMITED", own_chgcar: dict | None = None, run_seconds: str | None = None):
    body, loop, final = _runner_functions()
    (tmp_path / "input").mkdir(parents=True)
    (tmp_path / "input" / "INCAR").write_text("x\n")
    (tmp_path / "input" / "CHGCAR").write_text("input density COMPLETE\n")
    (tmp_path / "state").write_text(f"{state}\n")
    (tmp_path / "left").write_text(f"{left}\n")
    (tmp_path / "clock").write_text("1000\n")
    for run, tag in (prev_attempt or {}).items():
        (tmp_path / "runs" / run).mkdir(parents=True, exist_ok=True)
        (tmp_path / "runs" / run / ".attempted").write_text(f"{tag}\n")
    for run, text in (own_chgcar or {}).items():
        (tmp_path / "runs" / run).mkdir(parents=True, exist_ok=True)
        (tmp_path / "runs" / run / "CHGCAR").write_text(text)
    bin_dir = _stubs(tmp_path, srun_body)
    script = (
        "set -eu\nHEAD_DIR=$(pwd)\nID=$SLURM_ARRAY_TASK_ID\nNJOBS=1\nINPUT_DIR=input\n"
        "create_struct=create.py\nSCRIPTS=scripts\nPHONON_PATH=phon.npz\nPERT=0.05\nDISP_FLAG=\nbinary=vasp\n"
        "ZFS_REGEX_PATTERN=ZFSBLOCK\nORDER=first_order\nMODES='1 2 3'\nVASP_FAILED=0\nmkdir -p runs\n"
        + body + loop + "PASS=own\nwalk\n" + final + 'echo "VASP_FAILED=$VASP_FAILED STOP=$STOP"\n'
    )
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "SLURM_JOB_ID": "27178920",
           "SLURM_ARRAY_JOB_ID": "27178059", "SLURM_ARRAY_TASK_ID": "0", "STATE_FILE": str(tmp_path / "state"),
           "LEFT_FILE": str(tmp_path / "left"), "CLOCK_FILE": str(tmp_path / "clock")}
    env.pop("RUN_SECONDS", None)
    if run_seconds is not None:
        env["RUN_SECONDS"] = run_seconds
    return subprocess.run(["sh", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)


# ---- the job ends ----

def test_a_job_that_ends_stops_the_task_and_leaves_its_runs_unmarked(tmp_path):
    # srun is refused because the job has just passed its time limit
    r = _run(tmp_path, 'echo COMPLETING > "$STATE_FILE"\necho "srun: error: Unable to create step" >&2\nexit 1\n',
             prev_attempt={"1": "26000000"})
    assert r.returncode == 0, r.stderr
    assert "mode 1 not run -- the job is COMPLETING" in r.stdout
    assert "stopped: job 27178920 is COMPLETING" in r.stdout and "VASP_FAILED=0 STOP=yes" in r.stdout
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
    r = _run(tmp_path, 'echo "" > "$STATE_FILE"\nexit 1\n', left="")  # squeue prints nothing
    assert "VASP_FAILED=3 STOP=no" in r.stdout


# ---- too little time left for a run ----

def test_no_run_starts_with_less_time_left_than_a_run_takes(tmp_path):
    r = _run(tmp_path, "exit 0\n", left="1:30", run_seconds="120")  # 90 s left, runs take 120 s
    assert r.returncode == 0, r.stderr
    assert "starting job" not in r.stdout
    assert "stopped: 90 s left in job 27178920, a run takes up to 120 s" in r.stdout
    assert not any((tmp_path / "runs").iterdir())


def test_time_left_formats(tmp_path):
    for k, (left, starts) in enumerate((("1-00:00:00", 3), ("2:05:00", 3), ("UNLIMITED", 3), ("119", 0))):
        r = _run(tmp_path / str(k), "exit 0\n", left=left, run_seconds="120")
        assert r.stdout.count("starting job") == starts, (left, r.stdout)


def test_the_task_learns_how_long_a_run_takes(tmp_path):
    # no estimate: the first run starts; it takes 300 s, and 240 s are left for the next
    r = _run(tmp_path, "exit 0\n", left="4:00")
    assert r.stdout.count("starting job") == 1
    assert "stopped: 240 s left in job 27178920, a run takes up to 300 s" in r.stdout


# ---- restarts from the run's own CHGCAR ----

def test_a_complete_own_chgcar_is_kept_and_an_incomplete_one_replaced(tmp_path):
    r = _run(tmp_path, "exit 0\n", own_chgcar={"1": "own density COMPLETE\n", "2": "own density, cut\n"})
    assert r.returncode == 0, r.stderr
    assert "mode 1 restarts from its own CHGCAR" in r.stdout
    assert "mode 2: its own CHGCAR is not usable (density cut short) -- starting from the input CHGCAR" in r.stdout
    assert (tmp_path / "runs/1/CHGCAR").read_text() == "own density COMPLETE\n"  # kept: no ZFS block, no .done
    assert (tmp_path / "runs/2/CHGCAR").read_text() == "input density COMPLETE\n"
    assert (tmp_path / "runs/3/CHGCAR").read_text() == "input density COMPLETE\n"
    assert (tmp_path / "runs/1/INCAR").exists()  # the rest of input/ still goes in


def test_a_restart_from_its_own_chgcar_that_crashes_drops_it(tmp_path):
    r = _run(tmp_path, "exit 1\n", own_chgcar={"1": "own density COMPLETE\n"})
    assert "mode 1 crashed after restarting from its own CHGCAR -- dropped" in r.stdout
    assert not (tmp_path / "runs/1/CHGCAR").exists()  # the next attempt copies the input one
