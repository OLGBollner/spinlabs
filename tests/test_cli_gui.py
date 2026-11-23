"""``spinlabs gui`` parser wiring and the missing-Tk error."""

import importlib.util

import pytest

import spinlabs.cli as cli
from spinlabs.cli.gui import _run_gui


def test_gui_parser_accepts_paths(tmp_path):
    args = cli.build_parser().parse_args(["gui", str(tmp_path / "pack.npz"), str(tmp_path / "set")])
    assert args.paths[0].name == "pack.npz"
    assert args.paths[1].name == "set"
    assert args.func is _run_gui


def test_gui_parser_works_without_a_gui(tmp_path):
    # --help must not import tkinter or matplotlib.
    with pytest.raises(SystemExit) as exit_info:
        cli.build_parser().parse_args(["gui", "--help"])
    assert exit_info.value.code == 0


@pytest.mark.skipif(importlib.util.find_spec("tkinter") is not None, reason="Tkinter is installed")
def test_missing_tkinter_is_a_clear_error(tmp_path):
    args = cli.build_parser().parse_args(["gui"])
    with pytest.raises(SystemExit, match="Tkinter is not available"):
        _run_gui(args)
