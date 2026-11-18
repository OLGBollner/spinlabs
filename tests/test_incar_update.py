"""spinlabs update: set INCAR tags consistently across every run under a folder."""

import pytest

from spinlabs.cli import main
from spinlabs.vasp.incar_update import apply_updates, find_incars, parse_settings, plan_updates, summary


def _tree(tmp_path):
    """A defect with a ground state, a tree input, a sublevel input, a finished run and two snapshots."""
    files = {
        "ZFS_hyp/INCAR": "ENCUT = 520\nEDIFF = 1E-6   # tight\nISYM = 3\n",
        "first_order/pert_0.025/full_decont/input/INCAR": "EDIFF = 1e-06\nISYM = 0\n",
        "first_order/pert_0.025/full_decont/input/sublevels/ms_1/INCAR": "EDIFF=1E-6\n",
        "first_order/pert_0.025/full_decont/runs/27/INCAR": "EDIFF = 1E-6\nISYM = 0\n",
        "first_order/pert_0.025/full_decont/.spinlabs-code/abc/INCAR": "EDIFF = 1E-6\n",
        "data/INCAR": "EDIFF = 1E-6\n",
    }
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    (tmp_path / "first_order/pert_0.025/full_decont/runs/27/.done").touch()
    return tmp_path


def test_only_tags_that_must_agree_may_be_set():
    assert parse_settings(["ediff=1E-7", "ISYM=0"]) == {"EDIFF": "1E-7", "ISYM": "0"}
    with pytest.raises(ValueError, match="NUPDOWN: set per run"):
        parse_settings(["NUPDOWN=2"])
    with pytest.raises(ValueError, match="ALGO: set per run step"):
        parse_settings(["ALGO=Normal"])
    with pytest.raises(ValueError, match="NBANDS: fixed by the system"):
        parse_settings(["NBANDS=512"])
    with pytest.raises(ValueError, match="not on the list"):
        parse_settings(["FOO=1"])
    with pytest.raises(ValueError, match="TAG=VALUE"):
        parse_settings(["EDIFF"])


def test_every_incar_but_snapshots_and_data(tmp_path):
    root = _tree(tmp_path)
    found = {p.relative_to(root).as_posix() for p in find_incars(root)}
    assert found == {"ZFS_hyp/INCAR", "first_order/pert_0.025/full_decont/input/INCAR",
                     "first_order/pert_0.025/full_decont/input/sublevels/ms_1/INCAR",
                     "first_order/pert_0.025/full_decont/runs/27/INCAR"}


def test_replaces_keeping_comments_appends_missing_and_skips_equal_values(tmp_path):
    root = _tree(tmp_path)
    changes = plan_updates(root, {"EDIFF": "1E-7", "ISYM": "0"})
    isym = {c.path.relative_to(root).as_posix(): c.old for c in changes if c.tag == "ISYM"}
    assert isym == {"ZFS_hyp/INCAR": "3", "first_order/pert_0.025/full_decont/input/sublevels/ms_1/INCAR": None}
    assert len([c for c in changes if c.tag == "EDIFF"]) == 4  # 1E-6 and 1e-06 count as one value
    assert any("finished run" in line for line in summary(changes, root))
    apply_updates(changes)
    assert (root / "ZFS_hyp/INCAR").read_text() == "ENCUT = 520\nEDIFF = 1E-7   # tight\nISYM = 0\n"
    assert (root / "first_order/pert_0.025/full_decont/input/sublevels/ms_1/INCAR").read_text() == "EDIFF=1E-7\nISYM = 0\n"
    assert (root / "data/INCAR").read_text() == "EDIFF = 1E-6\n"  # never touched
    assert plan_updates(root, {"EDIFF": "1e-07", "ISYM": "0"}) == []  # a second pass changes nothing


def test_skip_done_leaves_finished_runs_out(tmp_path):
    root = _tree(tmp_path)
    paths = {c.path for c in plan_updates(root, {"EDIFF": "1E-7"}, skip_done=True)}
    assert root / "first_order/pert_0.025/full_decont/runs/27/INCAR" not in paths and len(paths) == 3


def test_cli_dry_run_writes_nothing_and_yes_writes(tmp_path, capsys):
    root = _tree(tmp_path)
    main(["update", str(root), "--set", "EDIFF=1E-7", "--dry-run"])
    assert "dry run: nothing written" in capsys.readouterr().out
    assert "1E-6" in (root / "ZFS_hyp/INCAR").read_text()
    main(["update", str(root), "--set", "EDIFF=1E-7", "--yes", "--skip-done"])
    assert "wrote 3 INCAR(s)" in capsys.readouterr().out
    assert "1E-7" in (root / "ZFS_hyp/INCAR").read_text()
    assert "1E-6" in (root / "first_order/pert_0.025/full_decont/runs/27/INCAR").read_text()
    with pytest.raises(SystemExit, match="NUPDOWN"):
        main(["update", str(root), "--set", "NUPDOWN=2"])
