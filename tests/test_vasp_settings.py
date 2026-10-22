"""VASP settings of a dataset: the ground state and every run must match where it matters.

ClV_576's ground state ran with ISYM = 3 and its perturbation runs with ISYM = 0, which
left D0 off its runs by about the size of the second-order changes.
"""

import argparse
from unittest.mock import patch

import numpy as np
import pytest

from spinlabs.cli.package import _run
from spinlabs.models import RawZFSData, ZFSTensor
from spinlabs.vasp.settings import compare_settings, normalize_value, read_incar, summary_line, tree_settings

GROUND = """ISTART = 0              # do not use WAVECAR
ICHARG = 1              # try to use POTCAR
LWAVE  = .TRUE.         # wavecar
ISYM = 3                # soft symmetrization
EDIFF =   1E-10        # stopping criterion
NCORE = 32
"""
RUN = """ISTART = 0 # do not use WAVECAR
ICHARG = 1 # try to use POTCAR
LWAVE  = .TRUE. # wavecar
ISYM = 0                # no symmetry
EDIFF =   1E-6        # stopping criterion
NCORE = 16
"""


def _tree(root, incars: list[str], input_incar: str | None = None):
    basis = root / "first_order" / "pert_0.025" / "all_bands"
    for i, text in enumerate(incars, start=1):
        (basis / "runs" / str(i)).mkdir(parents=True)
        (basis / "runs" / str(i) / "INCAR").write_text(text)
    (basis / "input").mkdir(parents=True, exist_ok=True)
    (basis / "input" / "INCAR").write_text(input_incar if input_incar is not None else incars[0])
    return basis


@pytest.mark.parametrize("raw, expected", [
    ("1E-6", "1e-06"), ("1e-06", "1e-06"), ("0.1E-05", "1e-06"), (" 3 ", "3"),
    (".TRUE.", "T"), ("Normal", "NORMAL"), ("400.0", "400"),
])
def test_values_compare_by_meaning(raw, expected):
    assert normalize_value(raw) == expected


def test_incar_comments_and_statements(tmp_path):
    path = tmp_path / "INCAR"
    path.write_text("ISYM = 0 ! no symmetry\nEDIFF = 1E-8; ALGO = Normal  # both on one line\n# ENCUT = 500\n")
    assert read_incar(path) == {"ISYM": "0", "EDIFF": "1e-08", "ALGO": "NORMAL"}


def test_ground_state_vs_runs_reports_isym_and_ediff_but_not_ncore(tmp_path):
    (tmp_path / "GROUND").write_text(GROUND)
    ground = read_incar(tmp_path / "GROUND")
    trees = {"first_order": tree_settings(_tree(tmp_path, [RUN, RUN]))}
    problems = compare_settings(ground, trees)
    assert any("ISYM = 0 in the runs but 3 in the ground state" in p for p in problems)
    assert any("EDIFF = 1e-06 in the runs but 1e-10 in the ground state" in p for p in problems)
    assert not any("NCORE" in p or "LWAVE" in p for p in problems)
    assert summary_line(ground, trees).startswith("ISYM 0 (ground state 3), EDIFF 1e-06 (ground state 1e-10)")


def test_runs_with_mixed_settings_and_a_changed_input(tmp_path):
    later = RUN.replace("ISYM = 0", "ISYM = 3")
    trees = {"first_order": tree_settings(_tree(tmp_path, [RUN, RUN, later], input_incar=later))}
    problems = compare_settings(read_incar_text(tmp_path, RUN), trees)
    assert any("ISYM differs between runs: 0 (2), 3 (1)" in p for p in problems)
    assert any("input/INCAR now has ISYM = 3, but the runs ran with 0, 3" in p for p in problems)


def test_matching_settings_report_nothing(tmp_path):
    trees = {"first_order": tree_settings(_tree(tmp_path, [RUN, RUN]))}
    assert compare_settings(read_incar_text(tmp_path, RUN), trees) == []


def read_incar_text(tmp_path, text):
    path = tmp_path / "GS_INCAR"
    path.write_text(text)
    return read_incar(path)


def test_package_scan_prints_and_stores_the_settings(tmp_path, monkeypatch, capsys):
    """package --scan lists the settings per dataset, and the packaged file keeps them."""
    monkeypatch.setattr("spinlabs.registry.REGISTRY_PATH", tmp_path / "registry.db")
    defect = tmp_path / "ClV_576"
    _tree(defect, [RUN, RUN])
    (defect / "ZFS_hyp").mkdir()
    (defect / "ZFS_hyp" / "INCAR").write_text(GROUND)
    gs = np.diag([-500.0, -500.0, 1000.0])
    raw = RawZFSData(defect="ClV", cell_size=576, pert_scale=0.025, calc_method="all_bands", order=1,
                     ground_state_zfs=ZFSTensor(matrix=gs, unit="MHz"),
                     first_order={0: {"tensor": gs + 1.0, "unit": "MHz"}})
    args = argparse.Namespace(defect_folder=[defect], orders="first", pert=None, method=["all"],
                              output_root=tmp_path / "data", scan=False, yes=True)
    with patch("spinlabs.cli.package.parse_zfs_simulation_dataset", side_effect=lambda **kw: raw):
        _run(args)
    out = capsys.readouterr().out
    assert "VASP settings (ground state vs the runs differenced against it):" in out
    assert "ClV_576 all_bands pert 0.025: 2 difference(s); ISYM 0 (ground state 3)" in out
    assert "!! first_order: ISYM = 0 in the runs but 3 in the ground state" in out
    saved = sorted((tmp_path / "data").rglob("*.npz"))
    stored = RawZFSData.load([p for p in saved if "raw_zfs" in p.name][0]).vasp_settings
    assert stored["ground_state"]["ISYM"] == "3"
    assert stored["trees"]["first_order"]["runs"]["ISYM"] == {"0": 2}
    assert len(stored["problems"]) == 2


def _with(text: str, **changes) -> str:
    lines = [l for l in text.splitlines() if l.split("=")[0].strip().upper() not in changes]  # noqa: E741
    return "\n".join(lines + [f"{k} = {v}" for k, v in changes.items()]) + "\n"


def test_output_relax_and_start_keys_are_kept_apart(tmp_path):
    from spinlabs.vasp.settings import start_differences

    ground = read_incar_text(tmp_path, _with(RUN, IBRION=1, NSW=0, LHYPERFINE=".TRUE.", ISTART=1))
    trees = {"first_order": tree_settings(_tree(tmp_path, [_with(RUN, IBRION=-1)] * 2))}
    problems = compare_settings(ground, trees)
    assert problems == []  # IBRION without ionic steps, hyperfine output: no effect on D
    assert start_differences(ground, trees) == ["first_order: ISTART = 0 in the runs but 1 in the ground state"]


def test_ionic_motion_is_judged_per_run_not_across_runs(tmp_path):
    # Neither run moves the ions; NSW of the second with IBRION of the first would.
    static = [_with(RUN, NSW=0, IBRION=0), _with(RUN, NSW=10, IBRION=-1)]
    trees = {"first_order": tree_settings(_tree(tmp_path, static))}
    assert trees["first_order"]["moving_ions"] == []
    assert not any("move the ions" in p for p in compare_settings(None, trees))


def test_runs_that_move_the_ions_are_named(tmp_path):
    incars = [_with(RUN, NSW=0), _with(RUN, NSW=20, IBRION=2), _with(RUN, NSW=0)]
    trees = {"first_order": tree_settings(_tree(tmp_path, incars))}
    assert [p for p in compare_settings(None, trees) if "move the ions" in p] == [
        "first_order: 1 of 3 runs move the ions (NSW > 0 with IBRION != -1): runs 2"
    ]


def test_a_relaxing_ground_state_and_non_scf_settings_are_flagged(tmp_path):
    ground = read_incar_text(tmp_path, _with(RUN, IBRION=1, NSW=50, ALGO="None", ICHARG=11))
    trees = {"first_order": tree_settings(_tree(tmp_path, [_with(RUN, ALGO="Normal")] * 2))}
    problems = compare_settings(ground, trees)
    assert any(p.startswith("ground state moves the ions (NSW = 50, IBRION = 1)") for p in problems)
    assert any("ALGO = NORMAL in the runs but NONE" in p and "no self-consistency" in p for p in problems)
    assert any("ICHARG = 1 in the runs but 11" in p and "fixed charge density" in p for p in problems)


def test_mixed_runs_name_the_minority_folders(tmp_path):
    other = RUN.replace("ISYM = 0", "ISYM = 3")
    trees = {"first_order": tree_settings(_tree(tmp_path, [other, RUN, other, other]))}
    problems = compare_settings(read_incar_text(tmp_path, other), trees)
    assert any("ISYM differs between runs: 3 (3), 0 (1) -- 0: runs 2" in p for p in problems)


# ---- SCF convergence ----

def _oszicar(path, n_steps: int, last_de: float):
    lines = ["       N       E                     dE             d eps       ncg     rms          rms(c)"]
    for n in range(1, n_steps + 1):
        de = last_de if n == n_steps else 1e-2 / n
        lines.append(f"DAV: {n:3d}    -0.123456789012E+04   {de: .5E}   -0.12345E-03  1234   0.123E-02")
    lines.append("   1 F= -.12345678E+04 E0= -.12345678E+04  d E =-.123456E-07  mag=     2.0000")
    path.write_text("\n".join(lines) + "\n")


def _scf_run(tmp_path, name, incar_extra: str, n_steps=None, last_de=1e-8, outcar=None):
    run = tmp_path / name
    run.mkdir(parents=True)
    (run / "INCAR").write_text(RUN + incar_extra)
    if n_steps is not None:
        _oszicar(run / "OSZICAR", n_steps, last_de)
    if outcar is not None:
        (run / "OUTCAR").write_text(outcar)
    return run


def test_scf_status_from_oszicar(tmp_path):
    from spinlabs.vasp.settings import scf_problem, scf_status

    ok = scf_status(_scf_run(tmp_path, "ok", "NELM = 60\n", n_steps=23, last_de=5e-7))
    assert ok["converged"] and ok["steps"] == 23 and scf_problem(ok) is None
    stuck = scf_status(_scf_run(tmp_path, "stuck", "NELM = 40\n", n_steps=40, last_de=3e-4))
    assert stuck["converged"] is False
    assert "stopped at NELM = 40 steps with |dE| = 3.0e-04 > EDIFF = 1e-06" in scf_problem(stuck)
    # converging on the very last allowed step still counts as converged
    assert scf_status(_scf_run(tmp_path, "last", "NELM = 40\n", n_steps=40, last_de=2e-7))["converged"]
    nonscf = scf_status(_scf_run(tmp_path, "none", "ALGO = None\n", n_steps=1, last_de=1e-3))
    assert "ALGO = None" in scf_problem(nonscf)


def test_scf_status_from_outcar_without_oszicar(tmp_path):
    from spinlabs.vasp.settings import scf_status

    done = "Iteration    1(  12)\n aborting loop because EDIFF is reached\n General timing and accounting\n"
    assert scf_status(_scf_run(tmp_path, "a", "NELM = 60\n", outcar=done))["converged"] is True
    stuck = "".join(f"Iteration    1({n:4d})\n" for n in range(1, 61)) + " General timing and accounting\n"
    assert scf_status(_scf_run(tmp_path, "b", "NELM = 60\n", outcar=stuck))["converged"] is False
    assert scf_status(_scf_run(tmp_path, "c", "", outcar="no scf section\n"))["converged"] is None


def test_unconverged_runs_are_named_in_the_settings_check(tmp_path):
    basis = _tree(tmp_path, [RUN + "NELM = 40\n"] * 3)
    for name, steps, de in (("1", 20, 1e-7), ("2", 40, 1e-3), ("3", 18, 1e-7)):
        _oszicar(basis / "runs" / name / "OSZICAR", steps, de)
    trees = {"second_order": tree_settings(basis)}
    assert trees["second_order"]["unconverged"] == ["2"]
    problems = compare_settings(read_incar_text(tmp_path, RUN + "NELM = 40\n"), trees)
    assert "second_order: 1 of 3 runs did not converge (NELM reached with |dE| > EDIFF): runs 2" in problems
