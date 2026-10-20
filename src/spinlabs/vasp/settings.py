"""VASP settings of a perturbation dataset: ground state vs the runs that are differenced against it.

Every finite difference is D(run) - D0, so D0 must come from a calculation with the same
physics settings as the runs: a ground state with ISYM = 3 against runs with ISYM = 0 left
ClV_576's D0 about 0.002-0.006 MHz off its runs, which is comparable to its second-order
changes. The runner copies ``input/`` into every run folder before VASP starts, so each
run's own INCAR is the one it ran with; a run without one falls back to its OUTCAR.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

# Keys that change how a calculation runs or what it writes, not its result.
IGNORED_KEYS = frozenset({
    "NCORE", "NPAR", "KPAR", "NSIM", "LPLANE", "LSCALU", "LSCALAPACK", "NBLOCK", "KBLOCK",
    "LWAVE", "LCHARG", "LVTOT", "LVHAR", "LELF", "LORBIT", "NWRITE", "SYSTEM", "LREAL_COMPAT",
})
# Shown for every dataset even when they agree, so a scan states the settings it relied on.
KEY_SETTINGS = ("ISYM", "EDIFF", "ALGO", "ICHARG", "ISTART", "ENCUT", "ISMEAR", "NUPDOWN")
# OUTCAR fallback when a run folder has no INCAR: VASP's parameter echo.
_OUTCAR_KEYS = {
    "ISYM": r"^\s*ISYM\s*=\s*(-?\d+)",
    "EDIFF": r"^\s*EDIFF\s*=\s*([-+0-9.Ee]+)",
    "ENCUT": r"^\s*ENCUT\s*=\s*([-+0-9.Ee]+)",
    "ISMEAR": r"^\s*ISMEAR\s*=\s*(-?\d+)",
    "NUPDOWN": r"^\s*NUPDOWN\s*=\s*([-+0-9.Ee]+)",
}
_BOOLS = {".TRUE.": "T", "TRUE": "T", "T": "T", ".FALSE.": "F", "FALSE": "F", "F": "F"}


def normalize_value(value: str) -> str:
    """Canonical text of an INCAR value: numbers by value (1E-6 = 1e-06), booleans as T/F, case-blind."""
    text = value.strip().strip("'\"").strip()
    upper = text.upper()
    if upper in _BOOLS:
        return _BOOLS[upper]
    try:
        number = float(text.replace("D", "E").replace("d", "e"))
    except ValueError:
        return " ".join(upper.split())
    return f"{int(number)}" if number.is_integer() and "." not in text and "E" not in upper else f"{number:g}"


def read_incar(path: str | Path) -> dict[str, str]:
    """KEY -> normalized value of an INCAR; comments (# or !) and ';'-separated statements handled."""
    settings: dict[str, str] = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        line = re.split(r"[#!]", line, maxsplit=1)[0]
        for statement in line.split(";"):
            if "=" not in statement:
                continue
            key, value = statement.split("=", 1)
            key = key.strip().upper()
            if key:
                settings[key] = normalize_value(value)
    return settings


def outcar_settings(path: str | Path) -> dict[str, str]:
    """The settings VASP echoes in an OUTCAR's parameter section (a fallback for runs without INCAR)."""
    head = Path(path).read_text(errors="replace")[:400_000]  # the echo sits well before the SCF
    out = {}
    for key, pattern in _OUTCAR_KEYS.items():
        match = re.search(pattern, head, re.M)
        if match:
            out[key] = normalize_value(match.group(1))
    return out


def run_settings(run_dir: Path) -> dict[str, str] | None:
    """Settings of one run folder: its INCAR, else its OUTCAR's echo, else None."""
    if (run_dir / "INCAR").is_file():
        return read_incar(run_dir / "INCAR")
    if (run_dir / "OUTCAR").is_file():
        return outcar_settings(run_dir / "OUTCAR")
    return None


def tree_settings(basis_dir: str | Path) -> dict:
    """Settings of a tree's input and of every run: {"input": {...}, "runs": {KEY: {value: n}}, "n_runs": n}."""
    basis_dir = Path(basis_dir)
    counts: dict[str, Counter] = {}
    n_runs = 0
    for run_dir in sorted(p for p in (basis_dir / "runs").glob("*") if p.is_dir()):
        settings = run_settings(run_dir)
        if settings is None:
            continue
        n_runs += 1
        for key, value in settings.items():
            counts.setdefault(key, Counter())[value] += 1
    incar = basis_dir / "input" / "INCAR"
    return {
        "input": read_incar(incar) if incar.is_file() else {},
        "runs": {key: dict(counter) for key, counter in counts.items()},
        "n_runs": n_runs,
    }


def compare_settings(ground: dict[str, str] | None, trees: dict[str, dict]) -> list[str]:
    """Problems: run settings that differ from the ground state, vary between runs, or from the tree input.

    ``trees`` maps a tree label (e.g. 'first_order') to :func:`tree_settings`. Keys in
    IGNORED_KEYS are skipped; a key set in only one of two INCARs counts as a difference
    (VASP then uses its default on the other side).
    """
    problems = []
    for label, tree in trees.items():
        runs = {k: v for k, v in tree["runs"].items() if k not in IGNORED_KEYS}
        for key in sorted(runs):
            values = runs[key]
            n_with = sum(values.values())
            if len(values) > 1 or n_with < tree["n_runs"]:
                spread = ", ".join(f"{v} ({n})" for v, n in sorted(values.items(), key=lambda kv: -kv[1]))
                unset = tree["n_runs"] - n_with
                problems.append(
                    f"{label}: {key} differs between runs: {spread}" + (f", unset ({unset})" if unset else "")
                )
        if ground is None:
            continue
        keys = (set(ground) | set(runs)) - IGNORED_KEYS
        for key in sorted(keys):
            g = ground.get(key)
            run_values = set(runs.get(key, {}))
            if tree["n_runs"] and sum(runs.get(key, {}).values()) < tree["n_runs"]:
                run_values.add(None)
            if run_values and run_values != {g}:
                shown = ", ".join("unset" if v is None else v for v in sorted(run_values, key=str))
                problems.append(f"{label}: {key} = {shown} in the runs but {'unset' if g is None else g} in the ground state")
        tree_input = {k: v for k, v in tree["input"].items() if k not in IGNORED_KEYS}
        for key in sorted(set(tree_input) & set(runs)):
            if set(runs[key]) != {tree_input[key]}:
                problems.append(
                    f"{label}: input/INCAR now has {key} = {tree_input[key]}, but the runs ran with "
                    + ", ".join(sorted(runs[key]))
                )
    return problems


def summary_line(ground: dict[str, str] | None, trees: dict[str, dict]) -> str:
    """'ISYM 0, EDIFF 1e-06, ...' of the runs (and ground state when it differs), for the key settings."""
    parts = []
    for key in KEY_SETTINGS:
        run_values = sorted({v for tree in trees.values() for v in tree["runs"].get(key, {})})
        g = ground.get(key) if ground else None
        if not run_values and g is None:
            continue
        text = f"{key} {'/'.join(run_values) or 'unset'}"
        if ground is not None and run_values != [g]:
            text += f" (ground state {g or 'unset'})"
        parts.append(text)
    return ", ".join(parts)
