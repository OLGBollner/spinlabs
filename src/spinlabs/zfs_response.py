"""Does a ZFS change have the symmetry its phonon modes allow? One check for every stage.

The same verification runs on the cluster after each perturbation run (vasp.run_checks via
check_run.py), over a whole dataset when packaging, and in ZFSManager on the derivatives.

The change is decomposed into the irreps of the defect's point group with character
projectors (symmetry.tensor_irrep_fractions), in the ground state's principal frame with z
along its distinct principal axis. Which irreps are allowed follows from the mode labels:

* ``kind="displacement"``: the raw change of a run displaced along the modes holds their
  linear response and, at second order, every product of two of them. Along A-type modes of
  C3v only the A1 part may change (exact); any E mode allows everything, so for runs along E
  modes only the empirical E-dominance check below applies.
* ``kind="derivative"``: a second derivative d2D/dQi dQj holds only the irreps of Gi x Gj
  (A1 x E: E only, A x A: A1 only). The forward finite difference that measures it also
  carries O(q D''') terms, Gi x Gi x Gj and Gi x Gj x Gj, which for A1 x E, A1 x A2 and
  A2 x E pairs include A1. Those are allowed too, so a warning means a broken run, not
  truncation; the strict Gi x Gj share is still reported (``strict_forbidden``) for a
  summary. On NV_64 the strict rule flags 807 such pairs (median 0.42 MHz), the
  finite-difference rule one broken run.

A displacement along E modes only should still change mainly the E part (D_xx - D_yy and the
off-diagonals): on the NV data that share is >= 0.86 for E modes and <= 0.1 for A-type modes,
so E_FRACTION_LIMIT = 0.3 separates them. This catches a run in the wrong state or geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spinlabs.logging_utils import get_logger
from spinlabs.symmetry import (
    CANONICAL_OPERATIONS,
    CHARACTER_TABLES,
    SymmetryDetectionError,
    _parent_label,
    _per_op_characters,
    detect_point_group_from_spectrum,
    direct_product,
    tensor_irrep_fractions,
)

LOGGER = get_logger(__name__)

SYMMETRY_TOL = 0.3  # largest norm share of the change allowed in irreps the modes forbid
E_FRACTION_LIMIT = 0.3  # smallest share in 2D irreps for a displacement along E modes only
SHAPE_FLOOR_MHZ = 0.2  # below this |dD| the shape is numerical noise


@dataclass
class ZFSResponseCheck:
    """Outcome of verify_zfs_response; ``problem`` is None when the change is fine or not judged."""

    point_group: str | None
    size_mhz: float = 0.0
    fractions: dict[str, float] = field(default_factory=dict)
    allowed: set[str] = field(default_factory=set)
    forbidden: float = 0.0
    strict_forbidden: float = 0.0  # share outside the exact symmetry rule, finite-difference terms not allowed
    line: str = ""
    problem: str | None = None


def response_point_group(spectrum) -> str | None:
    """Point group for checking ZFS responses: detected from the structure, else implied by the labels.

    Only groups with canonical operations (symmetry.CANONICAL_OPERATIONS) can decompose a
    tensor. When the structure gives no usable group (e.g. a synthetic spectrum), the mode
    labels decide if exactly one such group has them all; otherwise None (not checked).
    """
    try:
        symbol = detect_point_group_from_spectrum(spectrum).symbol
        if symbol in CANONICAL_OPERATIONS:
            return symbol
        LOGGER.info(f"ZFS response check: point group {symbol} has no canonical operations; using the mode labels")
    except SymmetryDetectionError:
        pass
    labels = {_parent_label(str(s)) for s in (spectrum.symmetries or [])}
    if not labels:
        return None
    fits = [pg for pg in CANONICAL_OPERATIONS if labels <= set(CHARACTER_TABLES[pg])]
    return fits[0] if len(fits) == 1 else None


def _times(point_group: str, irreps: set[str], label: str) -> set[str]:
    return {g for ir in irreps for g in direct_product(point_group, ir, label)}


def allowed_irreps(point_group: str, labels: list[str], kind: str, strict: bool = False) -> set[str]:
    """Irreps a ZFS change along modes with ``labels`` may contain (see the module docstring).

    For ``kind="derivative"`` the finite-difference truncation terms are included unless
    ``strict``, which gives the exact rule (G for a first, Gi x Gj for a second derivative).
    """
    parents = [_parent_label(str(s)) for s in labels]
    if kind == "derivative":
        exact = {parents[0]} if len(parents) == 1 else set(direct_product(point_group, *parents))
        if strict:
            return exact
        if len(parents) == 1:  # dD/dQ from a forward difference: + G x G at O(q)
            return exact | _times(point_group, {parents[0]}, parents[0])
        a, b = parents  # + Ga x Ga x Gb and Ga x Gb x Gb at O(q)
        return exact | _times(point_group, set(direct_product(point_group, a, a)), b) | _times(
            point_group, set(direct_product(point_group, b, b)), a
        )
    if kind != "displacement":
        raise ValueError(f"kind must be 'displacement' or 'derivative', not {kind!r}")
    allowed = set(parents)
    for a in parents:
        for b in parents:
            allowed |= set(direct_product(point_group, a, b))
    return allowed


def _dimension(table, irrep: str) -> int:
    return int(round(_per_op_characters(table, irrep)[0]))  # chi(E)


def _principal_frame(reference: np.ndarray) -> np.ndarray:
    """Rotation whose columns are the reference's principal axes, its distinct axis last (z)."""
    values, rot = np.linalg.eigh(np.asarray(reference, dtype=float))
    spread = [abs(values[k] - np.mean(np.delete(values, k))) for k in range(3)]
    z = int(np.argmax(spread))
    return rot[:, [k for k in range(3) if k != z] + [z]]


def verify_zfs_response(
    change_mhz: np.ndarray,
    labels: list[str],
    *,
    kind: str,
    point_group: str | None,
    reference: np.ndarray,
) -> ZFSResponseCheck:
    """Whether a ZFS change (MHz, at the displacement) has the symmetry its modes allow.

    ``reference`` is the ground-state tensor (any unit) whose principal frame defines the
    defect axis; ``change_mhz`` is in the same Cartesian frame. Changes below
    SHAPE_FLOOR_MHZ are reported but not judged.
    """
    if point_group is None or point_group not in CANONICAL_OPERATIONS:
        return ZFSResponseCheck(point_group, line=f"dD shape: point group {point_group or 'unknown'}, not checked")
    rot = _principal_frame(reference)
    d = rot.T @ np.asarray(change_mhz, dtype=float) @ rot
    d = d - np.trace(d) / 3 * np.eye(3)
    size = float(np.linalg.norm(d))
    fractions = tensor_irrep_fractions(d, point_group)
    allowed = allowed_irreps(point_group, labels, kind)

    def outside(irreps: set[str]) -> float:
        return float(np.sqrt(sum(f**2 for ir, f in fractions.items() if ir not in irreps)))

    forbidden = outside(allowed)
    strict_forbidden = outside(allowed_irreps(point_group, labels, kind, strict=True))
    table = CHARACTER_TABLES[point_group]
    degenerate = float(np.sqrt(sum(f**2 for ir, f in fractions.items() if _dimension(table, ir) > 1)))
    parts = ", ".join(f"|{ir}| {f * size:.3f} MHz" for ir, f in fractions.items() if f * size >= 5e-4)
    line = f"dD ({point_group}): {parts or '0'}" + (f", E fraction {degenerate:.3f}" if size else "")
    check = ZFSResponseCheck(point_group, size, fractions, allowed, forbidden, strict_forbidden, line)
    if size < SHAPE_FLOOR_MHZ:
        check.line += " (too small to judge its shape)"
        return check
    modes = "/".join(map(str, labels))
    if forbidden > SYMMETRY_TOL:
        shown = ", ".join(f"{ir} {f:.2f}" for ir, f in fractions.items() if ir not in allowed and f > 0.01)
        check.problem = (
            f"modes {modes} allow only {'/'.join(sorted(allowed))} in this {kind}, but {forbidden:.2f} of the change "
            f"is in {shown} (wrong state or geometry, or a mislabelled mode if this repeats across runs)"
        )
    elif kind == "displacement" and all(_dimension(table, _parent_label(str(s))) > 1 for s in labels):
        if degenerate < E_FRACTION_LIMIT:
            check.problem = (
                f"modes {modes} should mainly change the E part (D_xx - D_yy, off-diagonals), "
                f"but the E fraction is only {degenerate:.2f}"
            )
    return check
