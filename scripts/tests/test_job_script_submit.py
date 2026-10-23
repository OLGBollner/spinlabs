"""Per-tree job scripts submit a snapshot of the repo and run the loop from it.

`sh run_perturbation_*.sh` on a login node snapshots SPINLABS_REPO at its commit into
.spinlabs-code/<commit>/ and submits itself with that path; the SLURM job then runs the
runner from the snapshot, so one submission runs one version of the code.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TEMPLATES = REPO / "src" / "spinlabs" / "vasp" / "templates"  # the per-tree job scripts


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


def _repo(tmp_path: Path, version: str) -> Path:
    """A git repo whose runner prints its version, argv and the config it was given."""
    repo = tmp_path / "spinlabs"
    (repo / "scripts" / "cluster").mkdir(parents=True)
    (repo / "src" / "spinlabs").mkdir(parents=True)
    (repo / "src" / "spinlabs" / "__init__.py").write_text("")
    _write_runner(repo, version)
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "v")
    return repo


def _write_runner(repo: Path, version: str) -> None:
    (repo / "scripts" / "cluster" / "perturbation_runner.sh").write_text(
        f'echo "runner {version} order=$1 defect=$DEFECT binary=$binary pert=$PERT pair=${{PAIR_MODE:-}}"\n'
        'echo "pythonpath=$PYTHONPATH"\n'
    )


def _tree(tmp_path: Path, repo: Path, order: str = "second_order") -> Path:
    """An all_bands folder with the wrapper and its config filled, plus stand-ins for the cluster tools."""
    tree = tmp_path / "tree"
    tree.mkdir()
    script = (TEMPLATES / f"run_perturbation_{order}.sh").read_text()
    for placeholder, value in (("<defect>", "NV_64"), ("<vasp_binary>", "/bin/true"),
                               ("<path_to_phonon_data.npz>", "/data/phon.npz"), ("<pert_scale>", "0.1"),
                               ("<pair_mode>", "all"), ("<path_to_spinlabs_repo>", str(repo))):
        script = script.replace(placeholder, value)
    (tree / f"run_{order}.sh").write_text(script)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {tmp_path}/sbatch_args\n')
    (bin_dir / "ml").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "activate").write_text("true\n")  # sourced by `source activate <env>`
    for tool in ("sbatch", "ml"):
        (bin_dir / tool).chmod(0o755)
    return tree


def _run(tree: Path, script: str, *args: str, **env: str) -> subprocess.CompletedProcess:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_") and k != "SPINLABS_CODE"}
    full = {**clean, "PATH": f"{tree.parent / 'bin'}:{os.environ['PATH']}", **env}
    # bash: the job part sources the conda env with `source`, as on the cluster.
    return subprocess.run(["bash", script, *args], cwd=tree, env=full, capture_output=True, text=True)


@pytest.fixture
def setup(tmp_path):
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("needs git and bash")
    repo = _repo(tmp_path, "v1")
    return repo, _tree(tmp_path, repo)


def test_submit_snapshots_the_commit_and_submits_itself(setup, tmp_path):
    repo, tree = setup
    commit = _git(repo, "rev-parse", "--short=12", "HEAD")

    r = _run(tree, "run_second_order.sh", "--array=0-3")

    assert r.returncode == 0, r.stderr
    code = tree / ".spinlabs-code" / commit
    assert (code / "scripts" / "cluster" / "perturbation_runner.sh").is_file()
    assert (code / "src" / "spinlabs" / "__init__.py").is_file()
    args = (tmp_path / "sbatch_args").read_text().splitlines()
    assert args == [f"--export=ALL,SPINLABS_CODE={code}", "--array=0-3", "run_second_order.sh"]
    assert f"spinlabs commit {commit}" in r.stdout


def test_second_submit_reuses_the_snapshot(setup):
    repo, tree = setup
    _run(tree, "run_second_order.sh")
    r = _run(tree, "run_second_order.sh")
    assert r.returncode == 0, r.stderr
    assert [p.name for p in (tree / ".spinlabs-code").iterdir()] == [_git(repo, "rev-parse", "--short=12", "HEAD")]


def test_uncommitted_changes_warn_and_stay_out_of_the_snapshot(setup):
    repo, tree = setup
    _write_runner(repo, "dirty")

    r = _run(tree, "run_second_order.sh")

    assert "WARNING" in r.stderr and "uncommitted changes" in r.stderr
    (code,) = (tree / ".spinlabs-code").iterdir()
    assert "runner v1" in (code / "scripts" / "cluster" / "perturbation_runner.sh").read_text()


def test_job_runs_the_snapshot_even_after_the_repo_moved_on(setup):
    repo, tree = setup
    _run(tree, "run_second_order.sh")
    (code,) = (tree / ".spinlabs-code").iterdir()
    _write_runner(repo, "v2")  # pulled/committed while the job waits in the queue
    _git(repo, "commit", "-qam", "v2")

    r = _run(tree, "run_second_order.sh", SLURM_JOB_ID="42", SLURM_ARRAY_TASK_ID="0", SPINLABS_CODE=str(code))

    assert r.returncode == 0, r.stderr
    assert "runner v1 order=second_order defect=NV_64 binary=/bin/true pert=0.1 pair=all" in r.stdout
    assert f"pythonpath={code}/src" in r.stdout


def test_plain_sbatch_without_a_snapshot_is_refused(setup):
    _, tree = setup
    r = _run(tree, "run_second_order.sh", SLURM_JOB_ID="42", SLURM_ARRAY_TASK_ID="3")
    assert r.returncode == 1
    assert "no code snapshot for this job -- submit with: sh run_second_order.sh" in r.stderr


def test_first_order_wrapper_starts_the_first_order_loop(tmp_path):
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("needs git and bash")
    repo = _repo(tmp_path, "v1")
    tree = _tree(tmp_path, repo, order="first_order")
    _run(tree, "run_first_order.sh")
    (code,) = (tree / ".spinlabs-code").iterdir()
    r = _run(tree, "run_first_order.sh", SLURM_JOB_ID="7", SPINLABS_CODE=str(code))
    assert "runner v1 order=first_order defect=NV_64" in r.stdout, r.stderr


def test_checkout_without_the_runner_is_refused_at_submission(tmp_path):
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("needs git and bash")
    repo = _repo(tmp_path, "v1")
    # An older checkout: scripts/cluster/ exists, but without the runner.
    (repo / "scripts" / "cluster" / "create_phonon_struct.py").write_text("")
    _git(repo, "add", ".")
    _git(repo, "rm", "-q", "scripts/cluster/perturbation_runner.sh")
    _git(repo, "commit", "-qm", "old checkout")
    tree = _tree(tmp_path, repo)

    r = _run(tree, "run_second_order.sh")

    assert r.returncode == 1
    assert "has no scripts/cluster/perturbation_runner.sh -- update the checkout" in r.stderr
    assert not (tmp_path / "sbatch_args").exists()
