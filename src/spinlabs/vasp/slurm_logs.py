"""Why a perturbation run did not finish, read from its tree's SLURM logs.

The job scripts set no ``-o``, so every array task writes ``slurm-<jobid>_<task>.out`` into
the tree folder. In it scripts/cluster/perturbation_runner.sh prints
``<task>: starting job for <label>`` before each run (``mode 4``, ``pair 4, 7``,
``mode 13 step -0.05``) and one line on how it ended: ``Done!``, ``VASP exited with N``,
``... did not finish``, ``Failed to create structure.`` or ``... not the intended
displacement``. A run that the scheduler killed has none of them; the log then ends with
SLURM's own line, e.g.

    [2026-10-28T19:53:28.268] error: *** JOB 26997662 ON nid001141 CANCELLED AT 2026-10-28T19:53:28 DUE TO TIME LIMIT ***

which VASP output may run into without a line break, so it is searched for anywhere in a line.
A run's ``.attempted`` names the array job that tried it last; its log is preferred.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from spinlabs.vasp.settings import scf_problem, scf_status

LOG_GLOB = "slurm-*.out"
MAX_LISTED = 8  # runs named per reason in a summary line

_LOG_NAME = re.compile(r"slurm-(\d+)(?:_(\d+))?\.out$")
_START = re.compile(r"starting job for (.+?)\s*$")
_CANCELLED = re.compile(r"\*\*\* JOB (\S+) ON \S+ CANCELLED AT \S+(?: DUE TO (.+?))? \*\*\*")
_OOM = re.compile(r"oom[-_]kill|out of memory", re.IGNORECASE)
_VASP_EXIT = re.compile(r"VASP exited with (\d+)")


def run_label(run: str) -> str:
    """The runner's label for a run folder name: 4 -> mode 4, 4_7 -> pair 4, 7, 13_pert_-0.05 -> mode 13 step -0.05."""
    if "_pert_" in run:
        mode, step = run.split("_pert_", 1)
        return f"mode {mode} step {step}"
    if "_" in run:
        i, j = run.split("_", 1)
        return f"pair {i}, {j}"
    return f"mode {run}"


@dataclass
class _Attempt:
    """One run's stretch of one task's log: from its 'starting job for' line to the task's next run."""

    log: Path
    job: str  # array job id from the file name
    lines: list[str]
    last: bool  # the last run in the log: a SLURM kill ends this one
    killed: str | None  # the log's 'CANCELLED ... DUE TO ...' reason, if any


def _read_logs(folder: Path) -> dict[str, list[_Attempt]]:
    """Every run label's attempts in folder's SLURM logs."""
    attempts: dict[str, list[_Attempt]] = {}
    for log in sorted(Path(folder).glob(LOG_GLOB)):
        name = _LOG_NAME.search(log.name)
        job = name.group(1) if name else ""
        lines = log.read_text(errors="replace").splitlines()
        killed = None
        for line in lines:
            if m := _CANCELLED.search(line):
                killed = f"cancelled due to {m.group(2).lower()}" if m.group(2) else "cancelled"
        starts = [(i, m.group(1)) for i, line in enumerate(lines) if (m := _START.search(line))]
        for k, (i, label) in enumerate(starts):
            end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
            attempts.setdefault(label, []).append(
                _Attempt(log, job, lines[i + 1 : end], last=k + 1 == len(starts), killed=killed)
            )
    return attempts


def _attempt_reason(attempt: _Attempt, run_dir: Path) -> str:
    """How one attempt ended, in words."""
    text = "\n".join(attempt.lines)
    if "Failed to create structure" in text:
        return "creating the perturbed structure failed"
    if "not the intended displacement" in text:
        return "perturbed POSCAR failed the displacement check"
    if m := _VASP_EXIT.search(text):
        return f"VASP exited with {m.group(1)}"
    if "did not finish" in text:
        scf = scf_problem(scf_status(run_dir)) if run_dir.is_dir() else None
        return scf.split(":")[0] if scf else "VASP ended without a ZFS block"
    if "Done!" in text:
        return "VASP finished but the run has no .done"
    if _OOM.search(text):
        return "out of memory"
    if attempt.last and attempt.killed:
        return attempt.killed
    if attempt.last:
        return "still running, or the log ends mid-run"
    return "no outcome in the log"


def unfinished_reasons(folder: Path, runs: list[str]) -> dict[str, str]:
    """Why each of ``runs`` (folder names below runs/, none of them finished) did not finish.

    A run without folder or ``.attempted`` that no log mentions was not started. Otherwise its
    last attempt in the logs decides, preferring the array job its ``.attempted`` names; the
    reason ends with that job's id.
    """
    folder = Path(folder)
    attempts = _read_logs(folder)
    reasons = {}
    for run in runs:
        run_dir = folder / "runs" / run
        tag_file = run_dir / ".attempted"
        tag = tag_file.read_text().strip() if tag_file.is_file() else None
        found = attempts.get(run_label(run), [])
        if not found:
            reasons[run] = f"attempted (job {tag}), no log found" if tag else "not started"
            continue
        attempt = max(found, key=lambda a: (a.job == tag, int(a.job or 0), a.log.name))
        reasons[run] = f"{_attempt_reason(attempt, run_dir)} (job {attempt.job})"
    return reasons


def _run_key(name: str):
    return tuple(float(p) if p.lstrip("-").replace(".", "", 1).isdigit() else 0 for p in name.split("_"))


def summarize(reasons: dict[str, str], max_listed: int = MAX_LISTED) -> list[str]:
    """One line per reason, most runs first: 'reason: n run(s): 1444, 1445, ...'."""
    by_reason: dict[str, list[str]] = {}
    for run, reason in reasons.items():
        by_reason.setdefault(reason, []).append(run)
    lines = []
    for reason, runs in sorted(by_reason.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        runs = sorted(runs, key=_run_key)
        shown = ", ".join(runs[:max_listed]) + (", ..." if len(runs) > max_listed else "")
        lines.append(f"{reason}: {len(runs)} run{'s' if len(runs) != 1 else ''}: {shown}")
    return lines
