"""The perturbation runner locks each run directory so two arrays cannot work in it at once."""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "cluster"
TEMPLATES = REPO / "src" / "spinlabs" / "vasp" / "templates"  # the per-tree job scripts
RUNNER = SCRIPTS / "perturbation_runner.sh"
WRAPPERS = ("run_perturbation_first_order.sh", "run_perturbation_second_order.sh")


def _lock_block() -> str:
    text = RUNNER.read_text()
    match = re.search(r"# ---- run-directory lock ----\n.*?# ---- end lock ----\n", text, re.S)
    assert match, "no lock block in the runner"
    return match.group(0)


def _sh(tmp_path: Path, body: str, *, running_jobs: str | None, job: str = "100_3") -> subprocess.CompletedProcess:
    """Run the lock block plus ``body`` under sh, as array task ``job``, with a fake squeue."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    squeue = bin_dir / "squeue"
    if running_jobs is None:
        squeue.write_text("#!/bin/sh\nexit 1\n")  # controller unreachable
    else:
        squeue.write_text(f"#!/bin/sh\nprintf '%s' '{running_jobs}'\n")
    squeue.chmod(0o755)
    array_job, task = job.split("_")
    script = "set -eu\nHEAD_DIR=$(pwd)\nID=$SLURM_ARRAY_TASK_ID\n" + _lock_block() + body
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "SLURM_ARRAY_JOB_ID": array_job,
           "SLURM_ARRAY_TASK_ID": task}
    return subprocess.run(["sh", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)


def _hold(tmp_path: Path, job: str) -> None:
    """Leave runs/12_34 locked by ``job``, as a task that is still running (or was killed) would."""
    lock = tmp_path / "runs" / "12_34" / ".lock"
    lock.mkdir(parents=True)
    (lock / "owner").write_text(f"job={job}\nhost=nid001\nsince=2026-10-19 10:00:00\n")


@pytest.mark.parametrize("path", [RUNNER, *(TEMPLATES / w for w in WRAPPERS)], ids=lambda p: p.name)
def test_job_scripts_parse(path):
    assert subprocess.run(["sh", "-n", str(path)], capture_output=True).returncode == 0


def test_runner_takes_the_lock_before_touching_a_run_directory():
    text = RUNNER.read_text()
    run_one = text[text.index("run_one() {"):text.index("mkdir -p runs")]
    assert run_one.index('acquire_lock "$TARGET_DIR"') < run_one.index('cp "$INPUT_DIR"/*')
    # Released after the run, once back in the tree's top folder.
    assert run_one.rindex("release_lock") > run_one.rindex('cd "$HEAD_DIR"')
    assert "mkdir -p $TARGET_DIR" not in run_one  # the lock creates the directory


@pytest.mark.parametrize("name", WRAPPERS)
def test_wrappers_hold_no_loop_logic(name):
    """Per-tree scripts only configure and start the runner, so a fix reaches every tree."""
    text = (TEMPLATES / name).read_text()
    assert "acquire_lock" not in text and "srun" not in text
    assert 'exec sh "$SPINLABS_CODE/$RUNNER" "$ORDER"' in text


def test_free_directory_is_locked_with_owner_and_released_on_exit(tmp_path):
    r = _sh(tmp_path, 'acquire_lock runs/12_34\ncat "$HELD_LOCK/owner"\n', running_jobs="")
    assert r.returncode == 0, r.stderr
    assert "job=100_3" in r.stdout
    assert not (tmp_path / "runs" / "12_34" / ".lock").exists()  # EXIT trap released it
    assert (tmp_path / "runs" / "12_34").is_dir()


def test_directory_held_by_a_running_job_is_skipped(tmp_path):
    _hold(tmp_path, "200_7")
    r = _sh(tmp_path, 'if acquire_lock runs/12_34; then echo GOT; else echo SKIPPED; fi\n', running_jobs="200_7\n200_8\n")
    assert "SKIPPED" in r.stdout and "GOT" not in r.stdout
    assert "locked by running job: job=200_7" in r.stdout
    assert (tmp_path / "runs" / "12_34" / ".lock" / "owner").read_text().startswith("job=200_7")


def test_stale_lock_of_a_finished_job_is_taken_over(tmp_path):
    _hold(tmp_path, "200_7")
    r = _sh(tmp_path, 'acquire_lock runs/12_34\ncat "$HELD_LOCK/owner"\n', running_jobs="300_1\n")
    assert r.returncode == 0, r.stderr
    assert "took over stale lock on runs/12_34" in r.stdout
    assert "job=100_3" in r.stdout
    assert not list((tmp_path / "runs" / "12_34").glob(".lock.*"))  # no takeover or stale leftovers


def test_a_stale_lock_retaken_meanwhile_is_left_to_its_new_owner(tmp_path):
    """Two tasks found the same stale lock; the other one replaced it first. This task must not
    remove the other's fresh lock (the race of the old rename-then-mkdir takeover)."""
    _hold(tmp_path, "200_7")
    # Just as this task starts its takeover, task 100_4 has already replaced the stale lock.
    retaken = (
        'mkdir() { if [ "$1" = "$HEAD_DIR/runs/12_34/.lock.takeover" ]; then '
        "printf 'job=100_4\\n' > \"$HEAD_DIR/runs/12_34/.lock/owner\"; fi; command mkdir \"$@\"; }\n"
    )
    r = _sh(tmp_path, retaken + 'if acquire_lock runs/12_34; then echo GOT; else echo SKIPPED; fi\n',
            running_jobs="300_1\n")
    assert "SKIPPED" in r.stdout and "GOT" not in r.stdout, r.stdout + r.stderr
    assert (tmp_path / "runs" / "12_34" / ".lock" / "owner").read_text().startswith("job=100_4")
    assert not (tmp_path / "runs" / "12_34" / ".lock.takeover").exists()


def test_a_takeover_in_progress_is_not_joined(tmp_path):
    _hold(tmp_path, "200_7")
    (tmp_path / "runs" / "12_34" / ".lock.takeover").mkdir()
    r = _sh(tmp_path, 'if acquire_lock runs/12_34; then echo GOT; else echo SKIPPED; fi\n', running_jobs="300_1\n")
    assert "SKIPPED" in r.stdout and "another task is taking over" in r.stdout
    assert (tmp_path / "runs" / "12_34" / ".lock" / "owner").read_text().startswith("job=200_7")


def test_lock_is_kept_when_squeue_cannot_be_asked(tmp_path):
    _hold(tmp_path, "200_7")
    r = _sh(tmp_path, 'if acquire_lock runs/12_34; then echo GOT; else echo SKIPPED; fi\n', running_jobs=None)
    assert "SKIPPED" in r.stdout
    assert "squeue is unavailable" in r.stdout


def test_lock_path_survives_a_cd_into_the_run_directory(tmp_path):
    """The loop cds into the run directory; release must still find the lock."""
    r = _sh(tmp_path, 'acquire_lock runs/12_34\ncd runs/12_34\nrelease_lock\n[ ! -e .lock ] && echo RELEASED\n',
            running_jobs="")
    assert "RELEASED" in r.stdout, r.stderr
