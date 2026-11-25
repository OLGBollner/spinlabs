"""``spinlabs gui``: launch the interactive convergence viewer."""

from __future__ import annotations

import argparse

from spinlabs.utils import resolve_path


def build_gui_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "paths",
        nargs="*",
        type=resolve_path,
        help="Convergence .npz files or full-decontamination folders to open (optional)",
    )
    parser.set_defaults(func=_run_gui)


def _run_gui(args: argparse.Namespace) -> None:
    try:
        from spinlabs.gui.app import main
    except ModuleNotFoundError as exc:
        if exc.name == "tkinter":
            raise SystemExit(
                "error: Tkinter is not available in this Python. Install it with your package manager "
                "(e.g. 'sudo apt install python3-tk' on Debian/Ubuntu, or use a Python build that includes Tk)."
            ) from exc
        raise
    except ImportError as exc:  # a Tk build missing its shared library
        if "tkinter" in str(exc).lower() or "tk" in str(exc).lower():
            raise SystemExit(f"error: cannot load Tkinter: {exc}") from exc
        raise
    code = main(list(args.paths))
    if code:
        raise SystemExit(code)
