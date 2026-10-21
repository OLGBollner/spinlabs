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
    "LAECHG", "NEDOS", "EMIN", "EMAX",
    "LHYPERFINE", "NGYROMAG",  # hyperfine output: computed after the SCF, the spin density is the same
})
# Keys that only set the starting guess: they change the result by at most the SCF tolerance
# (ICHARG >= 10, a fixed charge density, is the exception and counts as a real difference).
START_KEYS = frozenset({"ISTART", "ICHARG", "INIWAV"})
# Ionic-motion keys: they matter only when the ions move (NSW > 0 and IBRION != -1).
RELAX_KEYS = frozenset({"IBRION", "NSW", "POTIM", "ISIF", "EDIFFG", "NFREE", "IOPT", "SMASS"})
# Shown for every dataset even when they agree, so a scan states the settings it relied on.
KEY_SETTINGS = ("ISYM", "EDIFF", "ALGO", "ICHARG", "ISTART", "ENCUT", "ISMEAR", "NUPDOWN")
MAX_LISTED = 12  # run folders named for a minority setting
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


# --------------------------------------------------------------------------- #
# SCF convergence
# --------------------------------------------------------------------------- #
_OSZICAR_STEP = re.compile(r"^\s*(DAV|RMM|CG|SDA|EDD|ORT|DIA):\s+(\d+)\s+\S+\s+(\S+)", re.M)
DEFAULT_NELM = 60
DEFAULT_EDIFF = 1e-4


def scf_status(run_dir: str | Path, settings: dict[str, str] | None = None) -> dict:
    """Whether a run's first electronic minimisation reached EDIFF before NELM.

    Read from the OSZICAR (one line per SCF step; small, so cheap for whole trees):
    a run that used all NELM steps with |dE| still above EDIFF did not converge, and
    its ZFS stays close to the starting density. Without an OSZICAR the OUTCAR
    decides: "aborting loop because EDIFF is reached" means converged, NELM iterations
    without it means not. ``converged`` is None when neither file says. ALGO = None runs do not iterate and are marked ``non_scf``.
    """
    run_dir = Path(run_dir)
    if settings is None:
        settings = read_incar(run_dir / "INCAR") if (run_dir / "INCAR").is_file() else {}
    try:
        nelm = int(float(settings.get("NELM", DEFAULT_NELM)))
        ediff = float(settings.get("EDIFF", DEFAULT_EDIFF))
    except ValueError:
        nelm, ediff = DEFAULT_NELM, DEFAULT_EDIFF
    status = {"steps": None, "nelm": nelm, "ediff": ediff, "last_de": None, "converged": None,
              "non_scf": settings.get("ALGO") == "NONE"}
    oszicar = run_dir / "OSZICAR"
    if oszicar.is_file():
        text = oszicar.read_text(errors="replace")
        first = text.split(" F=", 1)[0]  # the first ionic step (NSW = 0: the only one)
        steps = _OSZICAR_STEP.findall(first)
        if steps:
            status["steps"] = int(steps[-1][1])
            try:
                status["last_de"] = abs(float(steps[-1][2]))
            except ValueError:
                pass
            hit_limit = status["steps"] >= nelm
            above = status["last_de"] is None or status["last_de"] > ediff
            status["converged"] = not (hit_limit and above)
            return status
    outcar = run_dir / "OUTCAR"
    if outcar.is_file():
        text = outcar.read_text(errors="replace")
        # "Iteration    1(  37)": SCF step 37 of the first ionic step
        steps = [int(n) for n in re.findall(r"Iteration\s+1\(\s*(\d+)\)", text)]
        if steps:
            status["steps"] = max(steps)
        if "aborting loop because EDIFF is reached" in text:
            status["converged"] = True
        elif status["steps"] is not None and status["steps"] >= nelm:
            status["converged"] = False
    return status


def scf_problem(status: dict) -> str | None:
    """Readable reason a run's SCF is not to be trusted, or None."""
    if status["non_scf"]:
        return "not self-consistent: ALGO = None (one diagonalisation of the starting density)"
    if status["converged"] is False:
        if status["steps"] is not None:
            de = f"{status['last_de']:.1e}" if status["last_de"] is not None else "?"
            return (f"SCF did not converge: stopped at NELM = {status['nelm']} steps with |dE| = {de} "
                    f"> EDIFF = {status['ediff']:g} (the ZFS stays near the starting density)")
        return f"SCF did not converge: {status['nelm']} iterations (NELM) without reaching EDIFF"
    return None


def tree_settings(basis_dir: str | Path) -> dict:
    """Settings of a tree's input and runs.

    ``{"input": {...}, "runs": {KEY: {value: n}}, "n_runs": n, "folders": {KEY: {value: [run]}}}``;
    ``folders`` names the run folders behind each value of a key the runs disagree on.
    """
    basis_dir = Path(basis_dir)
    counts: dict[str, Counter] = {}
    names: dict[str, dict[str, list[str]]] = {}
    run_names = []
    unconverged, unknown = [], 0
    for run_dir in sorted(p for p in (basis_dir / "runs").glob("*") if p.is_dir()):
        settings = run_settings(run_dir)
        if settings is None:
            continue
        run_names.append(run_dir.name)
        scf = scf_status(run_dir, settings)
        if scf["converged"] is False and not scf["non_scf"]:
            unconverged.append(run_dir.name)
        elif scf["converged"] is None and not scf["non_scf"]:
            unknown += 1
        for key, value in settings.items():
            counts.setdefault(key, Counter())[value] += 1
            names.setdefault(key, {}).setdefault(value, []).append(run_dir.name)
    folders = {}
    for key, by_value in names.items():
        if len(by_value) > 1 or sum(len(v) for v in by_value.values()) < len(run_names):
            listed = {name for v in by_value.values() for name in v}
            unset = [n for n in run_names if n not in listed]
            folders[key] = {**by_value, **({"unset": unset} if unset else {})}
    incar = basis_dir / "input" / "INCAR"
    return {
        "input": read_incar(incar) if incar.is_file() else {},
        "runs": {key: dict(counter) for key, counter in counts.items()},
        "n_runs": len(run_names),
        "folders": folders,
        "unconverged": unconverged,
        "scf_unknown": unknown,
    }


def moves_ions(settings: dict[str, str]) -> bool:
    """Whether a calculation with these settings moves the ions (NSW > 0 and IBRION != -1)."""
    try:
        nsw = float(settings.get("NSW", "0"))
    except ValueError:
        return False
    if nsw <= 0:
        return False
    return settings.get("IBRION", "0") != "-1"  # VASP's IBRION default is 0 (MD) when NSW > 0


def _is_start_only(key: str, values) -> bool:
    if key not in START_KEYS:
        return False
    if key == "ICHARG":
        return all(v is None or not v.lstrip("-").isdigit() or int(v) < 10 for v in values)
    return True


def _note(key: str, values) -> str:
    if key == "ALGO" and "NONE" in values:
        return " (ALGO = NONE: no self-consistency)"
    if key == "ICHARG" and any(v and v.isdigit() and int(v) >= 10 for v in values):
        return " (ICHARG >= 10: fixed charge density, no self-consistency)"
    return ""


def _minority(tree: dict, key: str, majority: str | None) -> str:
    """'runs 12, 15, ...' behind the values of ``key`` other than ``majority``."""
    by_value = tree.get("folders", {}).get(key, {})
    out = []
    for value, runs in sorted(by_value.items(), key=lambda kv: len(kv[1])):
        if value == majority:
            continue
        shown = ", ".join(sorted(runs, key=_run_key)[:MAX_LISTED]) + (", ..." if len(runs) > MAX_LISTED else "")
        out.append(f"{value}: runs {shown}")
    return "; ".join(out)


def _run_key(name: str):
    return tuple(int(p) if p.isdigit() else 0 for p in name.split("_"))


def compare_settings(ground: dict[str, str] | None, trees: dict[str, dict]) -> list[str]:
    """Differences that can change the result: settings that differ from the ground state,
    vary between runs, or no longer match the tree's input.

    ``trees`` maps a tree label (e.g. 'first_order') to :func:`tree_settings`. Output and
    starting-point keys are left out (see :func:`start_differences`); ionic keys count only
    as "the ground state (or a run) moves the ions". A key set in only one of two INCARs
    counts as a difference (VASP then uses its default on the other side).
    """
    return _compare(ground, trees, start=False)


def start_differences(ground: dict[str, str] | None, trees: dict[str, dict]) -> list[str]:
    """The same comparison for starting-point keys (ISTART, ICHARG < 10, INIWAV) only."""
    return _compare(ground, trees, start=True)


def _compare(ground, trees, *, start: bool) -> list[str]:
    problems = []
    if not start and ground is not None and moves_ions(ground):
        problems.append(
            f"ground state moves the ions (NSW = {ground.get('NSW')}, IBRION = {ground.get('IBRION', '0')}): "
            "D0 comes from a relaxed geometry, not the one the runs are displaced from"
        )
    for label, tree in trees.items():
        n_runs = tree["n_runs"]
        bad = tree.get("unconverged", [])
        if not start and bad:
            shown = ", ".join(sorted(bad, key=_run_key)[:MAX_LISTED]) + (", ..." if len(bad) > MAX_LISTED else "")
            problems.append(
                f"{label}: {len(bad)} of {n_runs} runs did not converge (NELM reached with |dE| > EDIFF): runs {shown}"
            )
        runs = {k: v for k, v in tree["runs"].items() if k not in IGNORED_KEYS and k not in RELAX_KEYS}
        if not start and any(moves_ions({"NSW": nsw, "IBRION": ib}) for nsw in tree["runs"].get("NSW", {})
                             for ib in tree["runs"].get("IBRION", {"0": 1})):
            problems.append(f"{label}: some runs move the ions (NSW > 0 with IBRION != -1)")

        def wanted(key, values):
            return _is_start_only(key, values) == start

        for key in sorted(runs):
            values = runs[key]
            n_with = sum(values.values())
            if (len(values) > 1 or n_with < n_runs) and wanted(key, list(values) + [None] * (n_with < n_runs)):
                spread = ", ".join(f"{v} ({n})" for v, n in sorted(values.items(), key=lambda kv: -kv[1]))
                unset = n_runs - n_with
                majority = max(values, key=values.get)
                problems.append(
                    f"{label}: {key} differs between runs: {spread}" + (f", unset ({unset})" if unset else "")
                    + _note(key, values) + (f" -- {_minority(tree, key, majority)}" if not start else "")
                )
        tree_input = {k: v for k, v in tree["input"].items() if k not in IGNORED_KEYS and k not in RELAX_KEYS}
        if ground is not None:
            g_keys = {k for k in ground if k not in IGNORED_KEYS and k not in RELAX_KEYS}
            for key in sorted(g_keys | set(runs)):
                g = ground.get(key)
                run_values = set(runs.get(key, {}))
                if n_runs and sum(runs.get(key, {}).values()) < n_runs:
                    run_values.add(None)
                if run_values and run_values != {g} and wanted(key, run_values | {g}):
                    shown = ", ".join("unset" if v is None else v for v in sorted(run_values, key=str))
                    problems.append(
                        f"{label}: {key} = {shown} in the runs but {'unset' if g is None else g} in the ground state"
                        + _note(key, run_values | {g})
                    )
        for key in sorted(set(tree_input) & set(runs)):
            if set(runs[key]) != {tree_input[key]} and wanted(key, set(runs[key]) | {tree_input[key]}):
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
