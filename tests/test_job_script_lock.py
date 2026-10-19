"""The perturbation job scripts lock each run directory so two arrays cannot work in it at once."""

import os
import re
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "cluster"
NAMES = ("run_perturbation_first_order.sh", "run_perturbation_second_order.sh")


def _lock_block(name: str) -> str:
    text = (SCRIPTS / name).read_text()
    match = re.search(r"# ---- run-directory lock ----\n.*?# ---- end lock ----\n", text, re.S)
    assert match, f"no lock block in {name}"
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
    script = "set -eu\nHEAD_DIR=$(pwd)\nID=$SLURM_ARRAY_TASK_ID\n" + _lock_block(NAMES[0]) + body
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "SLURM_ARRAY_JOB_ID": array_job,
           "SLURM_ARRAY_TASK_ID": task}
    return subprocess.run(["sh", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)


def _hold(tmp_path: Path, job: str) -> None:
    """Leave runs/12_34 locked by ``job``, as a task that is still running (or was killed) would."""
    lock = tmp_path / "runs" / "12_34" / ".lock"
    lock.mkdir(parents=True)
    (lock / "owner").write_text(f"job={job}\nhost=nid001\nsince=2026-10-19 10:00:00\n")


def test_both_scripts_carry_the_same_lock_block():
    assert _lock_block(NAMES[0]) == _lock_block(NAMES[1])


@pytest.mark.parametrize("name", NAMES)
def test_scripts_take_the_lock_before_touching_a_run_directory(name):
    text = (SCRIPTS / name).read_text()
    loop = text[text.index("PAIR_ID=0"):]
    assert loop.index('acquire_lock "$TARGET_DIR"') < loop.index('cp "$INPUT_DIR"/*')
    # Released after the run, once back in the tree's top folder.
    assert loop.rindex("release_lock") > loop.rindex('cd "$HEAD_DIR"')
    assert 'mkdir -p $TARGET_DIR' not in loop  # the lock creates the directory


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
    assert not list((tmp_path / "runs" / "12_34").glob(".lock.stale.*"))


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
