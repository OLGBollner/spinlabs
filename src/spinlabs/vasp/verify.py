"""Static verification of a generated perturbation-run folder.

Checks the INCAR and the prefilled SLURM script in a run folder so that
problems are caught *before* sbatch queues the job, not hours later in
the scheduler.  The checks mirror the preflight in the run templates but
run locally at generation time.  It also counts the finished runs and
offers to fit the script's time limit to the runs that are left
(:func:`main`, which a tree's verify_perturbation_setup.py calls).
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from spinlabs.vasp.modes import mode_numbers
from spinlabs.vasp.trees import read_outcar_time, sbatch_time, step_sizes

PLACEHOLDER = re.compile(r"<[^<>\s]+>")
RUNNER = "scripts/cluster/perturbation_runner.sh"  # the loop the run scripts start
# Lines that legitimately contain '<...>': comments and the script's own
# preflight checks (fail messages, case patterns).
SKIP_LINE = re.compile(r"^\s*#|^\s*fail\b|^\s*case \$", re.IGNORECASE)
TIME_MARGIN = 1.2  # new time limit = runs per array task x median run time x this
TIMING_SAMPLE = 30  # finished runs whose OUTCAR times give the median


def _read(path: Path) -> str:
    return path.read_text() if path.is_file() else ""


def _tag_value(incar: str, tag: str) -> str | None:
    m = re.search(rf"^\s*{tag}\s*=\s*(\S+)", incar, re.MULTILINE | re.IGNORECASE)
    return m.group(1) if m else None


def script_config(text: str) -> dict[str, str]:
    """The script's VAR=value assignments, keys upper-cased (scripts may use binary=, BINARY=, ...).

    A ``${VAR:-default}`` value gives its default, the value the job runs with; quotes and a
    trailing comment are dropped. The first assignment of a name wins.
    """
    config: dict[str, str] = {}
    for line in text.splitlines():
        if SKIP_LINE.match(line):
            continue
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if not m:
            continue
        val = re.sub(r"\s+#.*$", "", m.group(2).strip())
        default = re.fullmatch(r"\$\{\w+:?-(.*)\}", val)
        config.setdefault(m.group(1).upper(), (default.group(1) if default else val).strip("'\""))
    return config


def verify_setup(folder: Path, script_name: str) -> list[str]:
    """Verify the folder's INCAR and SLURM script.

    Returns a list of problem descriptions (empty when everything is
    consistent).  Path problems are FAILs; INCAR policy issues that only
    risk slow or unusual runs are WARNINGs.
    """
    problems: list[str] = []
    folder = folder.resolve()

    # --- SLURM script: no leftover template placeholders -------------------
    script = folder / script_name
    if not script.is_file():
        problems.append(f"FAIL: missing run script {script}")
    else:
        text = _read(script)
        for line_no, line in enumerate(text.splitlines(), 1):
            if SKIP_LINE.match(line):
                continue
            for ph in PLACEHOLDER.findall(line):
                problems.append(f"FAIL: {script_name}:{line_no} placeholder not filled: {ph}")
        # path existence for the referenced files
        assigns = script_config(text)
        # The structure scripts come from the code snapshot of SPINLABS_REPO; older
        # self-contained scripts name them directly (checked only when present).
        required = ("BINARY", "PHONON_PATH", "SPINLABS_REPO")
        for key in (*required, "CREATE_STRUCT", "GET_N_MODES"):
            if key not in assigns:
                if key in required:
                    problems.append(f"FAIL: {script_name}: {key} not set")
                continue
            val = assigns[key]
            if not val or PLACEHOLDER.search(val):
                continue  # placeholder already reported above
            if not Path(val).exists():
                problems.append(f"FAIL: {script_name}: {key} path missing: {val}")
            elif key == "SPINLABS_REPO" and not (Path(val) / RUNNER).is_file():
                problems.append(f"FAIL: {script_name}: SPINLABS_REPO has no {RUNNER}: {val}")

    # --- INCAR: policy checks ---------------------------------------------
    incar = folder / "input" / "INCAR"
    if not incar.is_file():
        problems.append(f"FAIL: missing {incar}")
    else:
        itext = _read(incar)
        if (_tag_value(itext, "LDMATRIX") or "").lower() != ".true.":
            problems.append("FAIL: INCAR LDMATRIX is not .TRUE. -- no ZFS tensor")
        if (_tag_value(itext, "ALGO") or "").lower() == "none":
            problems.append("FAIL: INCAR ALGO=None is not self-consistent -- eigenvalues/occupations are meaningless")
        ibrion = _tag_value(itext, "IBRION")
        nsw = int(_tag_value(itext, "NSW") or "0")
        if ibrion is not None and int(ibrion) >= 0 and nsw > 0:
            problems.append(f"FAIL: INCAR relaxes ions (IBRION={ibrion}, NSW>0) -- ZFS must be static")
        isym = _tag_value(itext, "ISYM")
        if isym is not None and int(isym) in (1, 2) and (_tag_value(itext, "LDMATRIX") or "").lower() == ".true.":
            problems.append("FAIL: INCAR ISYM=1/2 together with LDMATRIX -- use ISYM=3")
        gam = "gam" in (_tag_value(_read(script), "BINARY") or "").lower()
        if gam:
            problems.append(
                "WARNING: the SLURM script uses a gamma-only "
                "VASP build; 5.4.4 gamma-only builds produced "
                "broken D tensors with NCORE>1 (issue #33)"
            )
        if (ncore := _tag_value(itext, "NCORE")) is not None and gam:
            problems.append(f"WARNING: gamma-only build with NCORE={ncore} -- untested combination for ZFS")
    return problems


# --------------------------------------------------------------------------- #
# progress of the runs and the time limit for the ones left
# --------------------------------------------------------------------------- #
_ARRAY = re.compile(r"^(#SBATCH\s+(?:-a\s+|--array[= ]))(\S+)", re.M)
_TIME = re.compile(r"^(#SBATCH\s+(?:-t\s+|--time[= ]))(\S+)", re.M)


def expected_runs(config: dict[str, str], order: str) -> list[str]:
    """Run folder names (below runs/) the job script's runner works through, in its order.

    Mirrors the walk in scripts/cluster/perturbation_runner.sh: first order ``<i>``, second
    order ``<i>_<j>`` (PAIR_MODE diag or all, i <= j), convergence ``<i>_pert_<step>`` for each
    of MODES at -p and +p for every step size p in PERTS.
    """
    if order == "convergence":
        modes = config["MODES"].split()
        steps = [f"{s:g}" for p in step_sizes([float(x) for x in config["PERTS"].split()]) for s in (-p, p)]
        return [f"{i}_pert_{p}" for p in steps for i in modes]
    modes = mode_numbers(Path(config["PHONON_PATH"]))
    if order == "first_order":
        return [str(i) for i in modes]
    pair_mode = config.get("PAIR_MODE")
    if pair_mode not in ("diag", "all"):
        raise ValueError(f"PAIR_MODE must be diag or all, got {pair_mode!r}")
    if pair_mode == "diag":
        return [f"{i}_{i}" for i in modes]
    return [f"{i}_{j}" for i in modes for j in modes if i <= j]


@dataclass
class RunProgress:
    """Which of a tree's runs are finished (``.done``) and which failed (``.attempted``, no ``.done``)."""

    expected: list[str]
    done: list[str]
    failed: list[str]

    @property
    def left(self) -> int:
        return len(self.expected) - len(self.done)


def run_progress(folder: Path, expected: list[str]) -> RunProgress:
    """The finished and failed runs among ``expected`` in folder/runs/."""
    runs = Path(folder) / "runs"
    done = [r for r in expected if (runs / r / ".done").is_file()]
    finished = set(done)
    failed = [r for r in expected if r not in finished and (runs / r / ".attempted").is_file()]
    return RunProgress(expected, done, failed)


def median_run_seconds(folder: Path, done: list[str], sample: int = TIMING_SAMPLE) -> float | None:
    """Median VASP wall time (s) of up to ``sample`` finished runs, spread evenly over ``done``."""
    if not done:
        return None
    step = max(1, len(done) // sample)
    times = [t for r in done[::step][:sample] if (t := read_outcar_time(Path(folder) / "runs" / r)) is not None]
    return statistics.median(times) if times else None


def array_tasks(spec: str) -> tuple[int, int]:
    """(number of tasks, how many run at once) of an sbatch --array spec such as 0-9, 1,3,5-7 or 0-31%8."""
    indices, _, throttle = spec.partition("%")
    n = 0
    for part in indices.split(","):
        rng, _, stride = part.partition(":")
        lo, _, hi = rng.partition("-")
        n += len(range(int(lo), int(hi or lo) + 1, int(stride or 1)))
    return n, min(n, int(throttle)) if throttle else n


def rescaled_limits(text: str, n_left: int, seconds_per_run: float) -> tuple[str, str]:
    """(time limit, array spec) for the runs left, keeping the script's array size.

    The runner shares runs between the tasks, so the busiest task does ceil(left / tasks
    running at once) of them; the limit is that x seconds_per_run x TIME_MARGIN. An array with
    more tasks than runs left shrinks to one task per run.
    """
    m = _ARRAY.search(text)
    if m is None:
        raise ValueError("the script has no #SBATCH -a/--array line")
    spec = m.group(2)
    n_tasks, at_once = array_tasks(spec)
    if n_tasks > n_left:
        throttle = spec.partition("%")[2]
        spec = f"0-{n_left - 1}" + (f"%{throttle}" if throttle else "")
        n_tasks, at_once = array_tasks(spec)
    seconds = math.ceil(n_left / at_once) * seconds_per_run * TIME_MARGIN
    return sbatch_time(seconds), spec


def set_limits(text: str, time: str, array: str) -> str:
    """The script with its #SBATCH time limit and array spec replaced (-t/--time, -a/--array)."""
    return _ARRAY.sub(lambda m: m.group(1) + array, _TIME.sub(lambda m: m.group(1) + time, text))


def _script_order(script_name: str, config: dict[str, str]) -> str:
    """first_order, second_order or convergence: the script's ORDER, else its file name."""
    order = config.get("ORDER", "")
    if order in ("first_order", "second_order", "convergence"):
        return order
    if script_name == "run_convergence.sh":
        return "convergence"
    return "second_order" if "second_order" in script_name else "first_order"


def offer_rescale(folder: Path, script_name: str, ask=None) -> None:
    """Print how many runs are done and offer to fit the script's time limit to the ones left.

    ``ask`` (default: input) gets the question and returns the answer (empty or y/yes: update);
    no stdin is a no.
    """
    ask = ask or input
    script = Path(folder) / script_name
    text = _read(script)
    config = script_config(text)
    try:
        expected = expected_runs(config, _script_order(script_name, config))
    except (KeyError, ValueError, OSError) as error:
        print(f"Cannot count the runs: {error}")
        return
    progress = run_progress(folder, expected)
    failed = f"; {len(progress.failed)} attempted but failed" if progress.failed else ""
    print(f"{len(progress.done)}/{len(expected)} runs done ({progress.left} left{failed}).")
    if progress.left == 0:
        print("All runs are done; nothing to update.")
        return
    per_run = median_run_seconds(folder, progress.done)
    if per_run is None:
        print("No finished run with an OUTCAR wall time yet; cannot estimate a new time limit.")
        return
    try:
        time, array = rescaled_limits(text, progress.left, per_run)
    except ValueError as error:
        print(f"Cannot update the limits: {error}")
        return
    old_time = m.group(2) if (m := _TIME.search(text)) else "unset"
    old_array = _ARRAY.search(text).group(2)
    print(
        f"Median run time {per_run / 60:.1f} min -> -t {old_time} -> {time}, "
        f"-a {old_array} -> {array} ({TIME_MARGIN:g}x margin)"
    )
    try:
        answer = ask(f"Update {script_name} with new arrays and time allocations? [Y/n]: ")
    except EOFError:
        answer = "n"
    if answer.strip().lower() not in ("", "y", "yes"):
        print(f"{script_name} left as it is.")
        return
    script.write_text(set_limits(text, time, array))
    print(f"Updated {script_name}: -t {time}, -a {array}")


def main(folder: Path, argv: list[str] | None = None) -> int:
    """verify_perturbation_setup.py in a tree folder: check the setup, then the runs' progress.

    Exit 1 when a setup check FAILs.
    """
    ap = argparse.ArgumentParser(description="Verify the INCAR and SLURM script of this run folder")
    ap.add_argument("--script", default=None, help="run script name (default: the single run_*.sh here)")
    args = ap.parse_args(argv)

    folder = Path(folder).resolve()
    script = args.script
    if script is None:
        candidates = sorted(folder.glob("run_*.sh"))
        if len(candidates) != 1:
            ap.exit(1, "Error: cannot decide which run script to check; pass --script\n")
        script = candidates[0].name

    problems = verify_setup(folder, script)
    for p in problems:
        print(p)
    if not problems:
        print("OK: setup looks consistent -- ready for sbatch")
    offer_rescale(folder, script)
    return 1 if any(p.startswith("FAIL") for p in problems) else 0
