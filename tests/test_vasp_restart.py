"""Whether a run's own CHGCAR is complete enough to restart from (spinlabs.vasp.restart)."""

import pytest

from spinlabs.vasp.restart import chgcar_problem, main

HEADER_VASP5 = """NV test
1.0
 3.5 0.0 0.0
 0.0 3.5 0.0
 0.0 0.0 3.5
 C N
 1 1
Direct
 0.0 0.0 0.0
 0.5 0.5 0.5

 2 2 3
"""
VALUES = " 0.1 0.2 0.3 0.4 0.5\n 0.6 0.7 0.8 0.9 1.0\n 1.1 1.2\n"  # 2 * 2 * 3 = 12 values


def _write(tmp_path, text):
    path = tmp_path / "CHGCAR"
    path.write_text(text)
    return path


def test_a_complete_chgcar_is_usable(tmp_path):
    assert chgcar_problem(_write(tmp_path, HEADER_VASP5 + VALUES + "augmentation occupancies 1 2\n")) is None


def test_vasp4_header_and_selective_dynamics(tmp_path):
    vasp4 = HEADER_VASP5.replace(" C N\n", "").replace("Direct\n", "Selective dynamics\nDirect\n")
    assert chgcar_problem(_write(tmp_path, vasp4 + VALUES)) is None


@pytest.mark.parametrize(
    "text, problem",
    [
        ("", "empty"),
        ("NV test\n1.0\n", "header cut short"),
        (HEADER_VASP5.split(" 0.0 0.0 0.0")[0], "positions cut short"),
        (HEADER_VASP5.replace(" 2 2 3\n", ""), "no grid line"),
        (HEADER_VASP5 + " 0.1 0.2 0.3\n", "density cut short (3 of 12 values)"),
    ],
)
def test_incomplete_chgcars_say_why(tmp_path, text, problem):
    assert chgcar_problem(_write(tmp_path, text)) == problem


def test_cli_exit_codes(tmp_path, capsys):
    assert main([str(_write(tmp_path, HEADER_VASP5 + VALUES))]) == 0
    assert capsys.readouterr().out.strip() == "complete"
    assert main([str(tmp_path / "missing")]) == 1
    assert "unreadable" in capsys.readouterr().out
