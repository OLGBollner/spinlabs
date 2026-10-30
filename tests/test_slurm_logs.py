"""Why perturbation runs did not finish, from the tree's SLURM logs (spinlabs.vasp.slurm_logs)."""

import numpy as np
import pytest

from spinlabs.vasp.slurm_logs import run_label, summarize, unfinished_reasons

# The end of a task the scheduler killed, as Dardel writes it (VASP output runs into srun's lines).
TIME_LIMIT_END = (
    " isp1:  2  isp2:  1  IDO:    33488896\n"
    "[2026-10-28T19:53:28.267] error: *** STEP 26997662.12 ON nid001141 CANCELLED AT 2026-10-28T19:53:28 "
    "DUE TO TIME LIMIT ***\n"
    "[2026-10-28T19:53:28.268] error: *** JOB 26997662 ON nid001141 CANCELLED AT 2026-10-28T19:53:28 "
    "DUE TO TIME LIMIT ***\n"
    "srun: error: nid001271: task 153: Terminated\n"
    "srun: Terminating StepId=26997662.12\n"
    " isp1:  2  isp2:  2srun: error: nid001141: tasks 0-40,42-45,47-51,53-98,100-127: Terminated\n"
    "srun: Force Terminated StepId=26997662.12\n"
)


def _run(tree, name, tag=None):
    run = tree / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    if tag is not None:
        (run / ".attempted").write_text(f"{tag}\n")
    return run


@pytest.mark.parametrize(
    "run, label", [("4", "mode 4"), ("4_7", "pair 4, 7"), ("13_pert_-0.05", "mode 13 step -0.05")]
)
def test_run_label_matches_the_runner(run, label):
    assert run_label(run) == label


def test_reasons_from_one_task_log(tmp_path):
    (tmp_path / "slurm-500_0.out").write_text(
        "0\n10\nPreflight OK: first_order, 10 modes, binary=/bin/vasp\n"
        "0: starting job for mode 4\nSuccessfully created perturbed structure\n0: Done! mode 4\n"
        "0: removing excess VASP results (WAVECAR, CHGCAR, CHG)\n"
        "0: starting job for mode 5\nSuccessfully created perturbed structure\n0: Done! mode 5\n"
        "0: mode 5 did not finish -- keeping CHGCAR/CHG so it can resume\n"
        "0: starting job for mode 6\nSuccessfully created perturbed structure\n"
        "0: VASP exited with 137 for mode 6\n0: mode 6 did not finish -- keeping CHGCAR/CHG so it can resume\n"
        "0: starting job for mode 7\nFailed to create structure.\nShutting down job 0\nTraceback ...\n"
    )
    (tmp_path / "slurm-500_1.out").write_text(
        "1: starting job for mode 8\n1: mode 8: perturbed POSCAR is not the intended displacement -- VASP not started\n"
        "1: starting job for mode 9\nSuccessfully created perturbed structure\n" + TIME_LIMIT_END
    )
    for run in ("5", "6", "7", "8", "9"):
        _run(tmp_path, run, tag=500)
    # mode 5 stopped at NELM = 60 with |dE| above EDIFF
    (tmp_path / "runs" / "5" / "INCAR").write_text("NELM = 60\nEDIFF = 1e-6\n")
    (tmp_path / "runs" / "5" / "OSZICAR").write_text("DAV:  60    -0.1E+04   -0.3E-03\n")
    _run(tmp_path, "11", tag=777)  # tried by a job whose log is gone

    reasons = unfinished_reasons(tmp_path, ["5", "6", "7", "8", "9", "10", "11"])
    assert reasons == {
        "5": "SCF did not converge (job 500)",
        "6": "VASP exited with 137 (job 500)",
        "7": "creating the perturbed structure failed (job 500)",
        "8": "perturbed POSCAR failed the displacement check (job 500)",
        "9": "cancelled due to time limit (job 500)",
        "10": "not started",
        "11": "attempted (job 777), no log found",
    }


def test_did_not_finish_without_oszicar_and_a_run_still_going(tmp_path):
    (tmp_path / "slurm-600_2.out").write_text(
        "2: starting job for pair 4, 7\n2: Done! pair 4, 7\n2: pair 4, 7 did not finish -- keeping CHGCAR/CHG\n"
        "2: starting job for pair 5, 5\nSuccessfully created perturbed structure\n isp1:  2\n"
    )
    reasons = unfinished_reasons(tmp_path, ["4_7", "5_5"])
    assert reasons == {
        "4_7": "VASP ended without a ZFS block (job 600)",
        "5_5": "still running, or the log ends mid-run (job 600)",
    }


def test_the_job_in_attempted_is_preferred(tmp_path):
    # job 700 ran out of time on mode 9; a later job 800 crashed on it, and .attempted says 700
    (tmp_path / "slurm-700_0.out").write_text("0: starting job for mode 9\n" + TIME_LIMIT_END)
    (tmp_path / "slurm-800_3.out").write_text("3: starting job for mode 9\n3: VASP exited with 1 for mode 9\n")
    _run(tmp_path, "9", tag=700)
    assert unfinished_reasons(tmp_path, ["9"]) == {"9": "cancelled due to time limit (job 700)"}
    # without .attempted the latest job counts
    (tmp_path / "runs" / "9" / ".attempted").unlink()
    assert unfinished_reasons(tmp_path, ["9"]) == {"9": "VASP exited with 1 (job 800)"}


def test_cancelled_without_a_reason_and_convergence_labels(tmp_path):
    (tmp_path / "slurm-900_0.out").write_text(
        "0: starting job for mode 13 step -0.05\n"
        "slurmstepd: error: *** JOB 900 ON nid1 CANCELLED AT 2026-10-29T10:00:00 ***\n"
    )
    assert unfinished_reasons(tmp_path, ["13_pert_-0.05"]) == {"13_pert_-0.05": "cancelled (job 900)"}


def test_summarize_groups_by_reason_most_runs_first():
    reasons = {str(m): "cancelled due to time limit (job 1)" for m in range(1444, 1454)}
    reasons |= {"1500": "VASP exited with 137 (job 1)", "99": "not started", "100": "not started"}
    assert summarize(reasons, max_listed=3) == [
        "cancelled due to time limit (job 1): 10 runs: 1444, 1445, 1446, ...",
        "not started: 2 runs: 99, 100",
        "VASP exited with 137 (job 1): 1 run: 1500",
    ]


def test_verify_prints_why_runs_did_not_finish(tmp_path, capsys):
    from spinlabs.vasp.verify import offer_rescale, unfinished_runs

    phon = tmp_path / "phon.npz"
    np.savez(phon, idx=np.array([3, 4, 6]), freqs=np.ones(3))
    (tmp_path / "run_x.sh").write_text(f"#SBATCH -t 1:00:00\n#SBATCH -a 0-1\nPHONON_PATH={phon}\nORDER=first_order\n")
    done = _run(tmp_path, "4", tag=500)
    (done / ".done").touch()
    (done / "OUTCAR").write_text(" Elapsed time (sec):  600.0\n")
    _run(tmp_path, "5", tag=500)
    (tmp_path / "slurm-500_0.out").write_text(
        "0: starting job for mode 4\n0: Done! mode 4\n0: starting job for mode 5\n" + TIME_LIMIT_END
    )

    assert unfinished_runs(tmp_path) == {"5": "cancelled due to time limit (job 500)", "7": "not started"}
    offer_rescale(tmp_path, "run_x.sh", ask=lambda q: "n")
    out = capsys.readouterr().out
    assert "1/3 runs done (2 left; 1 attempted but failed)." in out
    assert "  cancelled due to time limit (job 500): 1 run: 5\n" in out
    assert "  not started: 1 run: 7\n" in out


def test_unfinished_runs_needs_one_job_script(tmp_path):
    from spinlabs.vasp.verify import unfinished_runs

    assert unfinished_runs(tmp_path) is None


def test_package_prints_unfinished_runs(capsys):
    from spinlabs.cli.package import _print_settings_of

    trees = {
        "first_order": {
            "n_runs": 3,
            "runs": {"ISYM": {"3": 3}},
            "unfinished": {"5": "cancelled due to time limit (job 500)", "7": "not started"},
        }
    }
    _print_settings_of("NV_512 all_bands pert 0.05", {"ground_state": None, "trees": trees, "problems": []})
    out = capsys.readouterr().out
    assert "!! first_order: 2 run(s) not finished (from the tree's slurm-*.out logs):" in out
    assert "cancelled due to time limit (job 500): 1 run: 5" in out
