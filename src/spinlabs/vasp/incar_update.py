"""Set INCAR tags consistently across every run of a defect (``spinlabs update``).

Every INCAR under a folder is updated: the ground-state ZFS runs (ZFS_*), the trees' input/ and
input/sublevels/*/, ZFS_full and its ms_*/ sublevels, and the run folders under runs/. A tag is
replaced where it is set (trees.replace_tag, so inline comments stay) and appended where it is
missing.

Only tags that must agree between all runs of a defect may be set (ALLOWED_TAGS): changing them
in one place and not another makes the perturbed runs and the ground state they are differenced
against inconsistent. Tags that legitimately differ between runs (occupations and spin, run
control, system size) are refused with the reason (REFUSED_TAGS). Code snapshots (.spinlabs-code/)
and data/ folders are never touched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from spinlabs.vasp.settings import normalize_value

ALLOWED_TAGS = {
    # accuracy and convergence
    "EDIFF", "EDIFFG", "ENCUT", "ENAUG", "PREC", "ADDGRID", "LASPH", "LREAL", "NELM", "NELMIN", "GGA",
    "ISPIN", "SIGMA",
    # symmetry
    "ISYM",
    # parallelization
    "NCORE", "NSIM", "KPAR", "NPAR",
}

_OCCUPATIONS = ("set per run: the ground state, the SCF and each spin sublevel have their own occupations "
                "and spin (create_perturbation_tree.py writes them)")
_RUN_CONTROL = ("set per run step: the SCF, the ALGO=None sublevels and the restart files need different values "
                "(create_perturbation_tree.py writes them)")
_SYSTEM = "fixed by the system: every run of a defect already has the same value, and changing it breaks the WAVECARs"
REFUSED_TAGS = {
    **dict.fromkeys(("FERWE", "FERDO", "DOCCUP", "DOCCDO", "NUPDOWN", "ISMEAR"), _OCCUPATIONS),
    **dict.fromkeys(("ALGO", "ISTART", "ICHARG", "LWAVE", "LCHARG", "LDMATRIX", "NSW", "IBRION"), _RUN_CONTROL),
    **dict.fromkeys(("NBANDS", "NELECT"), _SYSTEM),
}

SKIPPED_DIRS = (".spinlabs-code", "data")


@dataclass(frozen=True)
class IncarChange:
    """One tag of one INCAR that changes: ``old`` is None when the tag is missing (it is appended)."""

    path: Path
    tag: str
    old: str | None
    new: str
    done: bool  # the folder holds a finished run (.done): its results do not follow the new value


def parse_settings(assignments: list[str]) -> dict[str, str]:
    """TAG=VALUE strings -> {TAG: VALUE}, checked against ALLOWED_TAGS / REFUSED_TAGS (ValueError)."""
    settings: dict[str, str] = {}
    for item in assignments:
        tag, eq, value = item.partition("=")
        tag, value = tag.strip().upper(), value.strip()
        if not eq or not tag or not value:
            raise ValueError(f"expected TAG=VALUE, got {item!r}")
        settings[tag] = value
    refused = [tag for tag in settings if tag in REFUSED_TAGS]
    if refused:
        raise ValueError(
            "these tags may not be set across all runs:\n"
            + "\n".join(f"  {tag}: {REFUSED_TAGS[tag]}" for tag in refused)
        )
    unknown = sorted(tag for tag in settings if tag not in ALLOWED_TAGS)
    if unknown:
        raise ValueError(
            f"not on the list of tags that must agree across runs: {', '.join(unknown)} "
            f"(allowed: {', '.join(sorted(ALLOWED_TAGS))})"
        )
    return settings


def find_incars(root: str | Path) -> list[Path]:
    """Every INCAR under ``root``, recursively, leaving out .spinlabs-code/ snapshots and data/ folders."""
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(
        p for p in root.rglob("INCAR")
        if p.is_file() and not any(part in SKIPPED_DIRS for part in p.relative_to(root).parts[:-1])
    )


def _value(text: str, tag: str) -> str | None:
    """The last value of ``tag`` in INCAR text (VASP semantics), comments stripped; None when unset."""
    values = re.findall(rf"^\s*{tag}\s*=\s*(.+?)\s*(?:[!#].*)?$", text, re.M | re.I)
    return values[-1] if values else None


def plan_updates(root: str | Path, settings: dict[str, str], skip_done: bool = False) -> list[IncarChange]:
    """The changes setting ``settings`` makes to the INCARs under ``root`` (values already equal are left out)."""
    changes = []
    for path in find_incars(root):
        done = (path.parent / ".done").is_file()
        if done and skip_done:
            continue
        text = path.read_text(errors="replace")
        for tag, new in settings.items():
            old = _value(text, tag)
            if old is not None and normalize_value(old) == normalize_value(new):
                continue
            changes.append(IncarChange(path, tag, old, new, done))
    return changes


def apply_updates(changes: list[IncarChange]) -> list[Path]:
    """Write the changes (replace where set, append where missing); returns the INCARs written."""
    from spinlabs.vasp.trees import replace_tag

    by_path: dict[Path, list[IncarChange]] = {}
    for change in changes:
        by_path.setdefault(change.path, []).append(change)
    for path, items in by_path.items():
        text = path.read_text(errors="replace")
        for change in items:
            if change.old is None:
                text = text.rstrip("\n") + f"\n{change.tag} = {change.new}\n"
            else:
                text = replace_tag(text, change.tag, change.new)
        path.write_text(text)
    return list(by_path)


def summary(changes: list[IncarChange], root: str | Path) -> list[str]:
    """Lines per tag: each INCAR that changes, old -> new, and a note on finished runs."""
    root = Path(root)
    lines = []
    for tag in sorted({c.tag for c in changes}):
        items = [c for c in changes if c.tag == tag]
        lines.append(f"{tag}: {len(items)} INCAR(s)")
        for c in items:
            rel = c.path.relative_to(root) if root.is_dir() else c.path
            note = "  (finished run: its results keep the old value)" if c.done else ""
            lines.append(f"  {rel}: {c.old if c.old is not None else '(unset)'} -> {c.new}{note}")
    n_done = len({c.path for c in changes if c.done})
    if n_done:
        lines.append(
            f"!! {n_done} INCAR(s) belong to finished runs (.done): changing them does not change their results; "
            "rerun them, or leave them out with --skip-done"
        )
    return lines
