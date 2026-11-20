"""spinlabs.vasp.mode_selection and ``spinlabs init convergence``: a varied set of modes for the convergence runs."""

import re

import numpy as np
import pytest

from spinlabs.cli import main
from spinlabs.models import PhononSpectrum
from spinlabs.parsers import save_phonon_npz
from spinlabs.vasp.mode_selection import (
    ModeInfo,
    format_selection,
    given_modes,
    mode_properties,
    select_convergence_modes,
)

NAN = float("nan")


def _modes():
    """Two translations, a local A1 mode, bulk-like modes of each irrep over 20-160 meV."""
    rows = [
        (1, 0.1, "A1", 0.010, 0.10, True),  # translation
        (2, 0.1, "Ex", 0.010, 0.10, True),  # translation
        (3, 20.0, "Ex", 0.020, 0.30, False),  # lowest
        (5, 40.0, "A2", 0.015, 0.20, False),
        (6, 45.0, "A2", 0.030, 0.55, False),  # the most defect-centred A2
        (8, 60.0, "A1", 0.020, 0.25, False),
        (9, 75.0, "A1", 0.130, 0.85, False),  # local mode: largest IPR and P_2bond
        (12, 90.0, "Ex", 0.008, 0.05, False),  # most delocalized
        (14, 110.0, "Ex", 0.025, 0.40, False),
        (15, 130.0, "A1", 0.022, 0.35, False),
        (17, 160.0, "Ex", 0.021, 0.45, False),  # highest
    ]
    return [ModeInfo(n, f, s, ipr, p, 3.0, t) for n, f, s, ipr, p, t in rows]


def _reasons(selection):
    return {s.mode.number: s.reasons for s in selection}


def test_the_selection_spans_localization_frequency_and_irrep():
    selection = select_convergence_modes(_modes(), n=6)
    reasons = _reasons(selection)
    assert list(reasons) == sorted(reasons)  # by mode number
    assert reasons[9] == ("most localized (largest IPR)", "most defect-centred (largest P_2bond)")
    assert reasons[3] == ("lowest frequency",)
    assert reasons[17] == ("highest frequency",)
    assert reasons[6] == ("most defect-centred A2",)
    assert reasons[12] == ("most delocalized (smallest IPR)",)
    assert len(selection) == 6 and 1 not in reasons and 2 not in reasons  # translations never
    assert {s.mode.irrep for s in selection} == {"A1", "A2", "Ex"}


def test_n_limits_the_picks_in_priority_order_and_never_exceeds_the_pool():
    assert sorted(_reasons(select_convergence_modes(_modes(), n=3))) == [3, 9, 17]
    assert len(select_convergence_modes(_modes(), n=50)) == 9  # every vibrational mode
    with pytest.raises(ValueError):
        select_convergence_modes(_modes(), n=0)
    with pytest.raises(ValueError, match="no vibrational modes"):
        select_convergence_modes(_modes()[:2])


def test_without_a_defect_centre_the_ipr_ranks_nearness():
    modes = [ModeInfo(m.number, m.freq_mev, m.irrep, m.ipr, NAN, NAN, m.translation) for m in _modes()]
    reasons = _reasons(select_convergence_modes(modes, n=6))
    assert reasons[6] == ("most defect-centred A2",)  # the A2 of larger IPR
    assert all("P_2bond" not in why for r in reasons.values() for why in r if "gap" not in why)
    assert "nan" in format_selection(select_convergence_modes(modes, n=2))


def test_given_modes_keep_their_order_and_reject_unknown_numbers():
    assert [s.mode.number for s in given_modes(_modes(), [9, 3, 9])] == [9, 3]
    with pytest.raises(ValueError, match=r"\[4\]"):
        given_modes(_modes(), [4])


# ---- the CLI ----

def _defect(tmp_path):
    """A defect folder with ZFS_AO/ZFS_UO inputs and a 3-atom, 9-mode labelled phonon file."""
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(9, 9)))
    spectrum = PhononSpectrum(
        frequencies_mev=np.linspace(10.0, 90.0, 9), eigenvectors=q.T.reshape(9, 3, 3),
        atom_frac_coords=np.array([[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0]]),
        atom_symbols=["C", "C", "Si"], atomic_masses=np.array([12.011, 12.011, 28.0855]),
        lattice=np.eye(3) * 5.0, symmetries=["A1", "Ex", "A2"] * 3,
    )
    defect = tmp_path / "SiV_3"
    (defect / "data").mkdir(parents=True)
    save_phonon_npz(spectrum, defect / "data" / "phonon_data.npz")
    for stage in ("ZFS_AO", "ZFS_UO"):
        (defect / stage).mkdir()
        for name in ("INCAR", "POSCAR", "KPOINTS", "POTCAR"):
            (defect / stage / name).write_text(name + "\n")
        (defect / stage / "OUTCAR").write_text("Elapsed time (sec):   100.000\n")
    return defect


@pytest.fixture
def defect(tmp_path, monkeypatch):
    # a random 3-atom cell has no point group to symmetrize by: use the labelled file as it is
    monkeypatch.setattr("spinlabs.vasp.trees.resolve_sym_phonon", lambda out, phonon: phonon)
    return _defect(tmp_path)


def _answer(monkeypatch, text):
    asked = []

    def fake_input(question):
        asked.append(question)
        if text is None:
            raise EOFError
        return text

    monkeypatch.setattr("builtins.input", fake_input)
    return asked


def test_mode_properties_reads_every_mode(defect):
    modes = mode_properties(defect / "data" / "phonon_data.npz")
    assert [m.number for m in modes] == list(range(1, 10))
    assert [m.irrep for m in modes[:3]] == ["A1", "Ex", "A2"]
    assert all(0 < m.ipr <= 1 for m in modes)


@pytest.mark.parametrize("answer", ["n", "no", None])
def test_cli_shows_the_modes_and_creates_nothing_unless_confirmed(defect, monkeypatch, capsys, answer):
    asked = _answer(monkeypatch, answer)
    main(["init", "convergence", str(defect), "--pert", "0.025", "-n", "4"])
    out = capsys.readouterr().out
    assert asked == ["create convergence runs of modes? [Y/n]: "]
    assert "most localized (largest IPR)" in out and "no runs created" in out
    assert re.search(r"^modes: +(\d+ ){3}\d+$", out, re.M)
    assert not list(defect.glob("convergence_test_*"))


@pytest.mark.parametrize("answer", ["", "y", "Yes"])
def test_cli_creates_the_runs_of_the_shown_modes_once_confirmed(defect, monkeypatch, capsys, answer):
    _answer(monkeypatch, answer)
    main(["init", "convergence", str(defect), "--pert", "0.025", "-n", "4", "--method", "unpaired"])
    shown = re.search(r"^modes: +(.*)$", capsys.readouterr().out, re.M).group(1)
    assert [p.name for p in defect.glob("convergence_test_*")] == ["convergence_test_UO"]
    script = (defect / "convergence_test_UO" / "run_convergence.sh").read_text()
    assert f'MODES="{shown}"' in script
    assert 'PERTS="0.0125 0.025 0.05 0.1"' in script
    assert "DISPLACEMENT=normalized" in script  # no trees: normalized, as for new trees


def test_cli_yes_skips_the_question_and_given_modes_are_used(defect, monkeypatch):
    asked = _answer(monkeypatch, "n")
    main(["init", "convergence", str(defect), "--steps", "0.05", "--modes", "4", "7", "-y",
          "--displacement", "mass_weighted"])
    assert asked == []
    for folder in ("convergence_test_AO", "convergence_test_UO"):
        script = (defect / folder / "run_convergence.sh").read_text()
        assert 'MODES="4 7"' in script and 'PERTS="0.05"' in script and "DISPLACEMENT=mass_weighted" in script


def test_cli_takes_the_step_from_the_trees_or_asks_for_one(defect, monkeypatch):
    _answer(monkeypatch, "n")
    with pytest.raises(SystemExit, match="pass --pert or --steps"):
        main(["init", "convergence", str(defect)])
    (defect / "first_order" / "pert_0.1" / "AO").mkdir(parents=True)
    _answer(monkeypatch, "")
    main(["init", "convergence", str(defect), "-n", "2", "--method", "all"])
    assert 'PERTS="0.05 0.1 0.2 0.4"' in (defect / "convergence_test_AO" / "run_convergence.sh").read_text()


def test_cli_rejects_a_mode_not_in_the_phonon_file(defect, monkeypatch):
    _answer(monkeypatch, "")
    with pytest.raises(SystemExit, match=r"modes \[42\]"):
        main(["init", "convergence", str(defect), "--pert", "0.025", "--modes", "42"])
