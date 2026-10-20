"""Checks of single perturbation runs: the applied displacement, VASP's symmetry handling
and whether the ZFS change has the shape the mode's symmetry allows.

Used by the job runner (logged per run) and by packaging (flags over a whole dataset).

D-tensor shape: in the ground state's principal frame a traceless change splits into an
A1 part (D_zz, with D_xx + D_yy = -D_zz) and an E part (D_xx - D_yy, D_xy, D_xz, D_yz).
For C3v, a displacement built only from A1/A2 modes can change D only through its A1
part, while E modes change it mainly through the E part (checked when all modes are E;
a pair mixing A and E modes has no fixed shape). On the NV data the E
fraction is <= 0.1 for A-type modes and >= 0.86 for E modes, so 0.3 separates them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spinlabs.models import PhononSpectrum
from spinlabs.parsers import parse_outcar_zfs

# Run folders sit at <defect>/<order>/pert_<x>/<basis>/runs/<modes>; the basis's
# ground-state ZFS run is <defect>/<source>/ (as trees.BASIS_SOURCE).
BASIS_SOURCE = {"all_bands": "ZFS_hyp", "defect_band_approx": "ZFS_occup"}
E_FRACTION_LIMIT = 0.3
SHAPE_FLOOR_MHZ = 0.2  # below this |dD| the shape is numerical noise
PERTURBATION_TOL = 0.01  # |scale - 1| and residual allowed for a correct displacement

# Expected symmetry operations VASP may find (upper bounds), C3v parent:
# A1/A2 keep the parent's 6, an E mode breaks it to C_1h (2).
EXPECTED_OPS = {"A1": 6, "A2": 6, "Ex": 2, "Ey": 2}


@dataclass
class RunCheck:
    """Outcome of checking one run: ``status`` is 'ok' or 'suspicious'."""

    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "suspicious" if self.problems else "ok"

    def report(self) -> str:
        head = "SUSPICIOUS" if self.problems else "OK"
        return "\n".join([f"run check: {head}", *(f"  {ln}" for ln in self.lines), *(f"  !! {p}" for p in self.problems)])


def mode_labels(spectrum: PhononSpectrum, modes: tuple[int, ...]) -> list[str]:
    """Symmetry labels of 1-based (original) mode numbers in a possibly symmetry-filtered spectrum."""
    from spinlabs.vasp.structures import _resolve_mode

    rows = [_resolve_mode(m, spectrum.original_indices, len(spectrum.frequencies_mev)) for m in modes]
    symmetries = spectrum.symmetries
    if symmetries is None:
        from spinlabs.symmetry import classify_and_pair

        symmetries, _ = classify_and_pair(spectrum)
    return [str(symmetries[r]) for r in rows]


# --------------------------------------------------------------------------- #
# displacement
# --------------------------------------------------------------------------- #
def check_perturbation(
    poscar: str | Path,
    reference: str | Path,
    spectrum: PhononSpectrum,
    modes: tuple[int, ...],
    amplitude: float,
    *,
    mass_weighted: bool = True,
) -> tuple[bool, str]:
    """Whether ``poscar`` is ``reference`` displaced by ``amplitude`` along ``modes``, with a log line."""
    from spinlabs.vasp.structures import compare_displacement, expected_displacement

    stats = compare_displacement(reference, poscar)
    disp = stats["disp"]
    expected = expected_displacement(spectrum, modes, amplitude, mass_weighted=mass_weighted)
    scale = float(np.sum(disp * expected) / np.sum(expected * expected))
    residual = float(np.linalg.norm(disp - scale * expected) / np.linalg.norm(disp)) if np.any(disp) else 1.0
    ok = abs(scale - 1) <= PERTURBATION_TOL and residual <= PERTURBATION_TOL
    norms = np.linalg.norm(expected, axis=1)
    line = (
        f"perturbation {'OK' if ok else 'WRONG'}: modes {'+'.join(map(str, modes))}, Q={amplitude:g}: "
        f"max {stats['max']:.6f} A (expected {norms.max():.6f}), rms {stats['rms']:.6f} A "
        f"(expected {np.sqrt((norms ** 2).mean()):.6f}); scale {scale:.3f} x expected, residual {residual:.1%}"
    )
    return ok, line


# --------------------------------------------------------------------------- #
# OUTCAR symmetry handling
# --------------------------------------------------------------------------- #
def outcar_symmetry(text: str) -> dict:
    """Point group, ISYM, symmetry operations, INVGRP error and completion, from OUTCAR text."""
    point = re.search(r"has the point symmetry\s+(\S+)", text)
    isym = re.search(r"ISYM\s*=\s*(-?\d+)", text, re.I)
    ops = re.findall(r"GETGRP returns: Found\s+(\d+)", text)
    return {
        "point_symmetry": point.group(1).rstrip(".") if point else None,
        "isym": int(isym.group(1)) if isym else None,
        "n_ops": int(ops[-1]) if ops else 0,
        "invgrp_error": "internal error in subroutine INVGRP" in text,
        "symmetrized": "symmetrized" in text,
        "finished": "General timing and accounting" in text,
    }


def check_symmetry_ops(sym: dict, labels: list[str]) -> tuple[list[str], list[str]]:
    """Lines and problems for VASP's symmetry handling of a run along modes with ``labels``."""
    lines = [
        f"symmetry: point group {sym['point_symmetry'] or 'n/a'}, ISYM {sym['isym'] if sym['isym'] is not None else 'n/a'}, "
        f"{sym['n_ops']} ops found"
    ]
    problems = []
    if sym["invgrp_error"]:
        problems.append("INVGRP error: VASP's symmetry analysis failed")
    expected = min((EXPECTED_OPS.get(s, -1) for s in labels), default=-1)
    if sym["point_symmetry"] is None and sym["n_ops"] == 0:
        if sym["isym"] in (0, -1):
            lines.append(f"ISYM={sym['isym']}: symmetry off, nothing to check")
        elif sym["isym"] is None and not sym["symmetrized"]:
            lines.append("no symmetry info and no symmetrization: likely ISYM=0, check INCAR")
        else:
            problems.append("could not parse symmetry information")
    elif expected >= 0:
        if sym["n_ops"] <= expected:
            lines.append(f"modes {'/'.join(labels)}: at most {expected} ops expected, {sym['n_ops']} found")
        else:
            problems.append(
                f"modes {'/'.join(labels)}: at most {expected} ops expected but VASP found {sym['n_ops']} "
                "(SYMPREC too loose? spurious symmetry may have been applied to the spin density)"
            )
    return lines, problems


# --------------------------------------------------------------------------- #
# ZFS tensor shape
# --------------------------------------------------------------------------- #
def zfs_change_parts(tensor: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """|A1| and |E| (MHz) of tensor - reference in the reference's principal frame."""
    _, rot = np.linalg.eigh(np.asarray(reference, dtype=float))
    d = rot.T @ (np.asarray(tensor, dtype=float) - np.asarray(reference, dtype=float)) @ rot
    d = d - np.trace(d) / 3 * np.eye(3)
    a1 = abs(d[2, 2]) * np.sqrt(1.5)
    e = np.sqrt(((d[0, 0] - d[1, 1]) / 2) ** 2 + d[0, 1] ** 2 + d[0, 2] ** 2 + d[1, 2] ** 2) * np.sqrt(2)
    return float(a1), float(e)


def check_zfs_shape(tensor: np.ndarray, reference: np.ndarray, labels: list[str]) -> tuple[str, str | None]:
    """Log line and problem (or None) for the ZFS change of a run along modes with ``labels``."""
    a1, e = zfs_change_parts(tensor, reference)
    size = float(np.hypot(a1, e))
    a_only = all(s in ("A1", "A2") for s in labels)
    # A pair mixing A and E modes changes D by both responses together, so its shape
    # is not fixed by symmetry alone (the pair's alpha check covers it in the analysis).
    e_only = all(s in ("Ex", "Ey") for s in labels)
    line = f"dD: |A1| {a1:.3f} MHz, |E| {e:.3f} MHz" + (f", E fraction {e / size:.3f}" if size else "")
    if size < SHAPE_FLOOR_MHZ:
        return line + " (too small to judge its shape)", None
    frac = e / size
    if a_only and frac > E_FRACTION_LIMIT:
        return line, (
            f"modes {'/'.join(labels)} can only change D through its A1 part, but the E fraction is {frac:.2f}"
            " (wrong state or geometry, or a mislabelled mode if this repeats across runs)"
        )
    if e_only and frac < E_FRACTION_LIMIT:
        return line, (
            f"modes {'/'.join(labels)} should mainly change the E part (D_xx - D_yy, off-diagonals), "
            f"but the E fraction is only {frac:.2f}"
        )
    return line, None


# --------------------------------------------------------------------------- #
# whole run
# --------------------------------------------------------------------------- #
def reference_outcar(run_dir: Path) -> Path | None:
    """The basis's ground-state OUTCAR for a run folder in a perturbation tree, if it exists."""
    run_dir = Path(run_dir).resolve()
    basis = run_dir.parent.parent
    source = BASIS_SOURCE.get(basis.name)
    if source is None or len(basis.parents) < 3:
        return None
    outcar = basis.parents[2] / source / "OUTCAR"
    return outcar if outcar.is_file() else None


def check_run(
    run_dir: str | Path,
    spectrum: PhononSpectrum,
    modes: tuple[int, ...],
    reference: str | Path | None = None,
) -> RunCheck:
    """Check a finished run: completion, VASP symmetry handling and the ZFS change's shape."""
    run_dir = Path(run_dir)
    check = RunCheck()
    outcar = run_dir / "OUTCAR"
    if not outcar.is_file():
        check.problems.append("no OUTCAR")
        return check
    text = outcar.read_text(errors="replace")
    labels = mode_labels(spectrum, modes)
    check.lines.append(f"modes {'+'.join(map(str, modes))}: {'/'.join(labels)}")
    if "\x00" in text:
        check.problems.append(f"OUTCAR holds {text.count(chr(0))} NUL bytes (written by two processes?)")
    sym = outcar_symmetry(text)
    if not sym["finished"]:
        check.problems.append("OUTCAR incomplete: the calculation did not finish")
    lines, problems = check_symmetry_ops(sym, labels)
    check.lines += lines
    check.problems += problems

    zfs = parse_outcar_zfs(outcar)
    ref_path = Path(reference) if reference else reference_outcar(run_dir)
    ref = parse_outcar_zfs(ref_path) if ref_path else None
    if zfs is None:
        check.problems.append("no ZFS tensor in OUTCAR")
    elif ref is None:
        check.lines.append("dD shape: no ground-state OUTCAR found, not checked")
    else:
        line, problem = check_zfs_shape(zfs.to_unit("MHz").matrix, ref.to_unit("MHz").matrix, labels)
        check.lines.append(line)
        if problem:
            check.problems.append(problem)
    return check


# --------------------------------------------------------------------------- #
# whole dataset (packaging)
# --------------------------------------------------------------------------- #
def dataset_shape_flags(raw, spectrum: PhononSpectrum) -> list[tuple[str, str]]:
    """(run folder, problem) for every entry of a RawZFSData whose ZFS change has the wrong shape.

    Run folders are 1-based like the cluster's runs/<i> and runs/<i>_<j>; entries whose
    modes are outside the phonon file are skipped.
    """
    data = raw.to_unit("MHz")
    if data.ground_state_zfs is None:
        return []
    reference = data.ground_state_zfs.matrix
    symmetries = spectrum.symmetries
    if symmetries is None:
        from spinlabs.symmetry import classify_and_pair

        symmetries, _ = classify_and_pair(spectrum)

    def tensor(entry) -> np.ndarray:
        return np.asarray(entry.zfs_tensor.matrix if hasattr(entry, "zfs_tensor") else entry["tensor"], dtype=float)

    flags = []
    entries = [((i,), e) for i, e in data.first_order.items()] + [(tuple(k), e) for k, e in data.second_order.items()]
    for idx, entry in entries:
        if any(i >= len(symmetries) for i in idx):
            continue
        _, problem = check_zfs_shape(tensor(entry), reference, [str(symmetries[i]) for i in idx])
        if problem:
            flags.append(("_".join(str(i + 1) for i in idx), problem))
    return flags
