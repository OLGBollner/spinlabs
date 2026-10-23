"""Generalized phonon-mode symmetry classification.

Pipeline:
1. Detect the point group of the defect structure via pymatgen's
   SpacegroupAnalyzer (spglib backend).
2. For each phonon mode, compute its character under each group
   operation (atom permutation + displacement rotation).
3. Project the character vector onto each irrep of the group's
   character table; degenerate pairs are matched with summed
   characters (2D irreps).

The analytic selection of coupling coefficients from the ZFS matrix
remains a per-group mapping table; only irrep labels cross
this seam.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np


from spinlabs.models import PhononSpectrum, check_original_indices
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Character tables (Hermann-Mauguin spglib point-group symbols as keys).
# χ vectors are ordered to match _operation_keys().
# ---------------------------------------------------------------------------

CHARACTER_TABLES: dict[str, dict[str, Any]] = {
    # C3v: operations (E, C3, C3^2, σv1, σv2, σv3)
    # 2D irreps carry sublabels: the character pattern of each component,
    # used for per-mode matching and for pairing (Ex/Ey etc.).
    "3m": {
        "A1": [1, 1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1, -1],
        "E": {
            # The stored sublabel patterns are NOT physical characters (a 2D
            # irrep's components are basis conventions). They only define the
            # parent class characters via their sum, and are kept for pairing.
            "sublabels": {
                "Ex": [1, -1, -1, 1, 1, 1],
                "Ey": [1, -1, -1, -1, -1, -1],
            },
            # True class characters of E (components mix under C3, so the
            # sublabel proxy patterns double-count there). Used for the
            # degenerate-group verification: Ex+Ey must sum to this.
            "class_chars": [2, -1, -1, 0, 0, 0],
            # Legacy convention: chi(first reflection) > 0 -> Ex, else Ey.
            "sublabel_rule": (3, "Ex", "Ey"),
        },
    },
    # C2v: operations (E, C2, σv, σv')
    "mm2": {
        "A1": [1, 1, 1, 1],
        "A2": [1, 1, -1, -1],
        "B1": [1, -1, 1, -1],
        "B2": [1, -1, -1, 1],
    },
    # D3h: operations (E, 2C3, 3C2', σh, 2S3, 3σv)
    "-6m2": {
        "A1'": [1, 1, 1, 1, 1, 1],
        "A2'": [1, 1, -1, 1, 1, -1],
        "E'": [1, -1, 0, 1, -1, 0],
        "A1''": [1, 1, 1, -1, -1, -1],
        "A2''": [1, 1, -1, -1, -1, 1],
        "E''": [1, -1, 0, -1, 1, 0],
    },
}


class SymmetryDetectionError(RuntimeError):
    """Raised when the point group of a structure cannot be determined.

    Typically synthetic fixtures with all atoms at the origin, which have
    no usable symmetry. Callers that have stored labels may catch this and
    fall back; anything else is a genuine bug and should propagate.
    """


@dataclass(frozen=True)
class PointGroup:
    """Detected point group of a structure."""

    symbol: str  # Hermann-Mauguin point-group symbol (spglib)
    order: int  # number of symmetry operations
    operations: tuple  # rotation matrices (3x3)
    space_group: str  # international space group symbol


def detect_point_group(structure: Structure, symprec: float = 1e-3) -> PointGroup:
    """Detect the point group of a (defect) supercell structure."""
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        symm_ops = sga.get_symmetry_operations()
    except Exception as exc:
        # spglib raises its own error for pathological (e.g. all-atoms-at-
        # origin) structures; translate it so callers can catch our type.
        raise SymmetryDetectionError(f"Unable to determine symmetry (symprec={symprec}): {exc}") from exc
    if not symm_ops:
        raise SymmetryDetectionError(f"No symmetry operations found (symprec={symprec}); cannot classify modes.")
    return PointGroup(
        symbol=sga.get_point_group_symbol(),
        order=len(symm_ops),
        operations=tuple(op.rotation_matrix for op in symm_ops),
        space_group=sga.get_space_group_symbol(),
    )


def detect_point_group_from_spectrum(spectrum, symprec: float = 1e-3) -> PointGroup:
    """Build a pymatgen Structure from a PhononSpectrum and detect its group."""
    structure = Structure(
        lattice=spectrum.lattice,
        species=spectrum.atom_symbols,
        coords=spectrum.atom_frac_coords,
        coords_are_cartesian=False,
    )
    return detect_point_group(structure, symprec=symprec)


def classify_modes(spectrum, tol_mev: float = 0.01, symprec: float = 1e-3) -> tuple[list[str], list[list[int]]]:
    """Classify each phonon mode into irreps of the detected point group.

    Returns (labels, deg_groups): a list of irrep labels, one per mode, and
    the degeneracy groups — lists of mode indices sharing a frequency and
    irrep. Degenerate partners of multi-dimensional irreps end up in the
    same group; accidental degeneracies between different irreps do not.
    """
    pg = detect_point_group_from_spectrum(spectrum, symprec=symprec)
    table = CHARACTER_TABLES.get(pg.symbol)
    if table is None:
        raise NotImplementedError(f"No character table for point group '{pg.symbol}'. Add one to CHARACTER_TABLES.")

    ops = defect_frame_operations(spectrum)
    chars = _mode_characters(spectrum, ops)
    # Sublabels of a 2D irrep (Ex/Ey) are a basis convention, not physics:
    # the stored convention (set by the legacy analyzer) is the sign of the
    # character on the first reflection operation. Compute that index here.
    first_reflection = next((k for k, R in enumerate(ops) if np.linalg.det(R) < 0), None)
    labels, deg_groups = _match_irreps(spectrum, chars, table, tol_mev, first_reflection=first_reflection)
    spectrum.symmetries = labels
    spectrum.sym_check = verify_deg_groups(chars, labels, deg_groups, table)
    return labels, deg_groups


def verify_deg_groups(
    chars,
    labels,
    deg_groups,
    table,
    atol: float = 0.05,
) -> dict:
    """Check that each degenerate group sums to its parent irrep's characters.

    Physics check (independent of how labels were derived): the character
    vector of a multidimensional irrep is the trace of the full
    representation, so the summed characters of a complete degenerate set
    must reproduce the parent irrep's class characters. For C3v E this is
    [2, -1, -1, 0, 0, 0] — verified on ClV: 127/127 Ex/Ey pairs.

    Returns a report dict:
      ok         True when every multi-member group passes
      failures   [(group_idx, indices, expected, got), ...] for mismatches
      unpaired   [(group_idx, indices), ...] lone members of a
                 multidimensional irrep (partner beyond tol) — unverifiable,
                 not a failure
    Singleton 1D irreps (A1/A2) are skipped: they are their own trace.
    """
    # class characters per irrep: explicit when the table carries them,
    # else the sum of its sublabel patterns (identity-safe fallback)
    class_chars: dict[str, np.ndarray] = {}
    dim: dict[str, int] = {}
    for name, entry in table.items():
        if isinstance(entry, dict):
            subs = list(entry["sublabels"].items())
            if "class_chars" in entry:
                chi = np.asarray(entry["class_chars"], dtype=float)
            else:
                chi = np.zeros(len(subs[0][1]), dtype=float)
                for _, sub_chi in subs:
                    chi += np.asarray(sub_chi, dtype=float)
            class_chars[name] = chi
            dim[name] = len(subs)
        else:
            class_chars[name] = np.asarray(entry, dtype=float)
            dim[name] = 1

    failures, unpaired = [], []
    for gi, group in enumerate(deg_groups):
        lab = labels[group[0]]
        if lab is None:
            continue
        parent = _parent_label(lab)
        if parent not in class_chars:
            continue
        if len(group) == 1:
            # lone member of a multidimensional irrep: partner missing
            if dim[parent] > 1:
                unpaired.append((gi, list(group)))
            continue
        got = np.sum([chars[i] for i in group], axis=0)
        expected = class_chars[parent]
        if not np.allclose(got, expected, atol=atol):
            failures.append((gi, list(group), expected.tolist(), got.tolist()))
    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "unpaired": unpaired,
    }


def sym_check_summary(check: dict | None) -> str:
    """Human-readable one-liner for a verify_deg_groups() report."""
    if check is None:
        return ""
    if check.get("ok"):
        n_unpaired = len(check.get("unpaired", ()))
        if n_unpaired:
            return f"ok ({n_unpaired} unpaired)"
        return "ok"
    n = len(check["failures"])
    detail = "; ".join(
        f"group {gi} ({' '.join(map(str, idx))}): expected {np.round(np.asarray(exp), 3).tolist()}, "
        f"got {np.round(np.asarray(got), 3).tolist()}"
        for gi, idx, exp, got in check["failures"][:3]
    )
    more = f" (+{n - 3} more)" if n > 3 else ""
    return f"FAILED ({n} group(s) mismatch: {detail}{more})"


def classify_and_pair(
    spectrum,
    tol_mev: float = 0.01,
    symprec: float = 1e-3,
    force: bool = False,
) -> tuple[list[str], list[list[int]]]:
    """Classify modes and build degeneracy groups, trusting stored labels by default.

    Returns (labels, deg_groups). If the spectrum already carries symmetry
    labels, they are trusted and only deg_groups are (re)built; pass
    force=True to re-derive labels via the general projection method.

    Both paths run verify_deg_groups() afterwards: the character-sum check
    (each degenerate set must sum to its parent irrep's class characters).
    The report is stored on spectrum.sym_check. If the point group cannot
    be detected (e.g. synthetic test structures), verification is skipped
    (sym_check stays None) rather than raising — labels/groups still work.
    """
    if spectrum.symmetries is not None and not force:
        labels = list(spectrum.symmetries)
        deg_groups = _build_groups_from_labels(spectrum, labels, tol_mev)
    else:
        labels, deg_groups = classify_modes(spectrum, tol_mev=tol_mev, symprec=symprec)
        spectrum.symmetries = labels
        return labels, deg_groups  # verify already ran inside classify_modes
    try:
        pg = detect_point_group_from_spectrum(spectrum, symprec=symprec)
        table = CHARACTER_TABLES.get(pg.symbol, {})
    except Exception:
        table = {}
    if table:
        ops = defect_frame_operations(spectrum)
        chars = _mode_characters(spectrum, ops)
        spectrum.sym_check = verify_deg_groups(chars, labels, deg_groups, table)
    return labels, deg_groups


def _build_groups_from_labels(spectrum, labels, tol_mev):
    """Build degeneracy groups from stored labels without re-deriving."""
    parent_of = {
        sub: parent
        for parent, entry in CHARACTER_TABLES["3m"].items()
        if isinstance(entry, dict)
        for sub in entry["sublabels"]
    }
    freqs = spectrum.frequencies_mev
    by_parent: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        if lab is not None:
            by_parent.setdefault(parent_of.get(lab, lab), []).append(i)
    groups = []
    for idxs in by_parent.values():
        # seed-based grouping (same as _match_irreps): direct |df| < tol
        # comparison, NOT frequency bucketing — rounding puts partners that
        # straddle a bucket boundary into different buckets (observed on
        # NV_512: 146.60496 vs 146.60502 meV split into two lone groups).
        remaining = list(idxs)
        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            seen = {labels[seed]}
            for j in remaining:
                if labels[j] in seen:
                    continue
                if abs(freqs[j] - freqs[seed]) >= tol_mev:
                    continue
                seen.add(labels[j])
                group.append(j)
            for j in group[1:]:
                remaining.remove(j)
            groups.append(group)
    return groups


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _defect_frame(spectrum) -> tuple[np.ndarray, np.ndarray]:
    """Returns (axis, reflection_normal) of the defect in Cartesian coords.

    Same convention as the legacy analyze_c3v_symmetry: NV (C+N) has its
    principal axis along [1, 1, 1] with a sigma_v mirror normal [1, -1, 0];
    everything else (e.g. ClV) is taken as [0, 0, 1] / [1, 0, 0].
    """
    symbols = list(spectrum.atom_symbols)
    if "C" in symbols and "N" in symbols:
        return np.array([1.0, 1.0, 1.0]), np.array([1.0, -1.0, 0.0])
    return np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])


def _defect_center(spectrum) -> np.ndarray:
    """Fractional coords of the defect center, wrapped into [0, 1).

    Exact N site for single-N (NV) and Cl-pair midpoint for ClV; otherwise
    the generic self-contained localizer (vacancy, divacancy, substitution,
    interstitial). Falls back to the cell origin if nothing is detected.
    Matches PhononSpectrum.translate_defect_to_origin. Callers are
    responsible for periodic wrapping of the result.
    """
    frac = np.mod(np.asarray(spectrum.atom_frac_coords, dtype=float), 1.0)
    symbols_list = list(spectrum.atom_symbols)
    if symbols_list.count("N") == 1:
        return frac[symbols_list.index("N")]
    if "Cl" in symbols_list:
        return frac[np.asarray(symbols_list) == "Cl"].mean(axis=0)
    from spinlabs.vasp.structures import detect_defect_position

    detected = detect_defect_position(spectrum)
    if detected is not None:
        return detected
    return np.zeros(3)


def defect_frame_operations(spectrum) -> list[np.ndarray]:
    """Builds the point-group operations in the defect-aligned frame.

    The defect is centered at the origin and the principal axis is taken as
    the group's z axis, so the operations are the standard rotations/reflections
    about that frame. Operations act about the *defect center*, not the cell
    origin: applying them to an off-center structure corrupts the atom mapping
    and hence the characters.
    """
    from spinlabs.utils import MathUtils

    axis, normal = _defect_frame(spectrum)
    axis = axis / np.linalg.norm(axis)
    ops = [np.eye(3)]
    ops.append(MathUtils.rotation_around_symmetry_axis(axis, 3))
    ops.append(MathUtils.rotation_around_symmetry_axis(axis, 3).T)  # C3^2
    for C in (None, ops[1], ops[2]):  # three sigma_v mirrors of C3v
        n = normal / np.linalg.norm(normal)
        M = np.eye(3) - 2.0 * np.outer(n, n)
        if C is not None:  # rotate the mirror plane by +-120 deg about the axis
            M = C @ M
        ops.append(M)
    return ops


def _mode_characters(spectrum, operations: list[np.ndarray]) -> np.ndarray:
    """chi_m(R) = <psi_m | R psi_m> for each mode m and operation R.

    R acts on the eigenvector by permuting atoms and rotating their
    displacement vectors. The structure is first centered on the defect
    (PBC-aware, matching the legacy code) and the operations are built in
    the defect-aligned frame, so an off-center defect does not corrupt the
    characters.
    """
    eigs = np.asarray(spectrum.eigenvectors)
    n_modes = eigs.shape[0]
    if eigs.ndim == 3:
        vecs = eigs  # (n_modes, n_atoms, 3)
    else:
        n_atoms = eigs.shape[-1] // 3  # flat (n_modes, 3N)
        vecs = eigs.reshape(n_modes, n_atoms, 3)

    lattice = np.asarray(spectrum.lattice, dtype=float)
    inv_lat = np.linalg.inv(lattice)
    frac = np.mod(np.asarray(spectrum.atom_frac_coords, dtype=float), 1.0)
    center = _defect_center(spectrum)
    frac = np.mod(frac - center, 1.0)  # PBC-aware centering, defect at origin
    cart_atoms = frac @ lattice
    symbols = np.asarray(spectrum.atom_symbols)

    chars = np.zeros((n_modes, len(operations)))
    for k, R in enumerate(operations):
        R = np.asarray(R, dtype=float)
        mapping = _atom_mapping(R, cart_atoms, frac, inv_lat, symbols, lattice)
        # Legacy convention: chi = trace(eig[mapping] @ (R @ eig.T))
        # expands to sum_i v_{sigma(i)} . R v_i: rotate the original atom's
        # displacement, compare with the mapped atom's.
        rotated = np.einsum("ij,naj->nai", R, vecs)
        chars[:, k] = np.sum(vecs[:, mapping, :] * rotated, axis=(1, 2))
    return chars


def _atom_mapping(R, cart_atoms, frac, inv_lat, symbols, lattice):
    """sigma: for each atom i, index of the atom R sends i onto."""
    n_atoms = cart_atoms.shape[0]
    mapping = np.zeros(n_atoms, dtype=int)
    rotated_cart = cart_atoms @ R.T
    rot_frac = np.mod(rotated_cart @ inv_lat, 1.0)
    for i in range(n_atoms):
        diffs = np.mod(frac - rot_frac[i] + 0.5, 1.0) - 0.5
        dists = np.linalg.norm(diffs @ np.asarray(lattice, dtype=float), axis=1)
        valid = np.where(symbols == symbols[i])[0]
        mapping[i] = valid[np.argmin(dists[valid])] if len(valid) else i
    return mapping


def _parent_label(label: str) -> str:
    """Parent irrep of a (possibly sub-)label: "Ex" -> "E", "E" -> "E"."""
    return label.rstrip("xy") if label not in ("A1", "A2") else label


def _match_irreps(spectrum, chars, table, tol_mev, first_reflection=None):
    """Assign irrep labels per mode, mirroring the legacy semantics.

    Each mode is matched independently against 1D irreps and the
    single-component character patterns of degenerate (multidimensional)
    irreps. Frequency degeneracy is used only afterwards, to build
    deg_groups: components of the same multidimensional irrep sit at the
    same frequency. Accidental degeneracies between different irreps are
    therefore harmless.
    """
    n = spectrum.n_modes
    freqs = spectrum.frequencies_mev

    # Per-irrep patterns: 1D irreps appear as-is. A multi-dimensional irrep
    # (e.g. C3v E) is irreducible, so its components have no characters of
    # their own — the sublabels (Ex/Ey) are a basis convention. Each mode is
    # therefore scored once against the parent's class characters (the sum of
    # the stored sublabel patterns), and the sublabel is then assigned by a
    # sign rule (see first_reflection in classify_modes) that mirrors the
    # legacy analyzer's convention: chi > 0 -> first sublabel, else second.
    patterns: list[tuple[str, np.ndarray]] = []
    parent_of: dict[str, str] = {}
    sublabel_rule: dict[str, tuple[int, str, str]] = {}
    for name, entry in table.items():
        if isinstance(entry, dict):
            subs = list(entry["sublabels"].items())
            parent_chi = np.zeros(len(next(iter(subs))[1]), dtype=float)
            for sub, chi in subs:
                parent_chi += np.asarray(chi, dtype=float)
                parent_of[sub] = name
            patterns.append((name, parent_chi))
            parent_of[name] = name
            if "sublabel_rule" in entry:
                sublabel_rule[name] = tuple(entry["sublabel_rule"])
        else:
            patterns.append((name, np.asarray(entry, dtype=float)))
            parent_of[name] = name

    labels: list[Optional[str]] = []
    parents: list[Optional[str]] = []  # parent irrep of each sublabel
    for i in range(n):
        v = chars[i]
        nv = np.linalg.norm(v) + 1e-12
        best, best_score, best_parent = None, -1.0, None
        for name, chi in patterns:
            score = abs(np.dot(v, chi)) / (nv * (np.linalg.norm(chi) + 1e-12))
            if score > best_score:
                best, best_score, best_parent = name, score, parent_of[name]
        if best is not None and best in sublabel_rule and first_reflection is not None:
            op_idx, pos, neg = sublabel_rule[best]
            chi_ref = v[op_idx]
            best = pos if chi_ref > 0 else neg
        labels.append(best)
        parents.append(best_parent)

    # Degeneracy groups: complete sets of one irrep's sublabels at the same
    # frequency. Modes of the same parent irrep whose sublabels cover each
    # component exactly once form a group; accidental coincidences between
    # different parents never merge. Unpaired members (e.g. a lone Ex whose
    # Ey partner sits beyond tol) are reported as their own group.
    deg_groups: list[list[int]] = []
    by_parent: dict[str, list[int]] = {}
    for i, par in enumerate(parents):
        by_parent.setdefault(par or "?", []).append(i)

    for par, idxs in by_parent.items():
        remaining = list(idxs)
        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            # partners: modes sharing the seed's frequency (within tol)
            partners = [j for j in remaining if abs(freqs[j] - freqs[seed]) < tol_mev]
            # keep at most one partner per distinct sublabel beyond the seed
            taken: set[str] = set()
            for j in partners:
                if labels[j] == labels[seed]:
                    continue  # same sublabel: accidental, not a partner
                if labels[j] in taken:
                    continue
                taken.add(labels[j])
                group.append(j)
            for j in group[1:]:
                remaining.remove(j)
            deg_groups.append(group)
    return labels, deg_groups


def group_of(deg_groups: list[list[int]] | None, i: int) -> list[int] | None:
    """Return the degeneracy group containing mode i, or None."""
    if deg_groups is None:
        return None
    for g in deg_groups:
        if i in g:
            return g
    return None


def filter_degenerate_partners(
    spectrum,
    tol_mev: float = 0.01,
    symprec: float = 1e-3,
):
    """Return a reduced spectrum keeping one mode per complete degenerate set.

    General replacement for ``PhononSpectrum.filter_sym_pairs``: classifies
    via :func:`classify_and_pair` (any point group), then drops all but one
    member of every multi-member degeneracy group. Within a group the
    surviving member is the one with the lexicographically smallest irrep
    label (for C3v this reproduces the legacy keep-all-Ex rule, since
    'Ex' < 'Ey').

    The reduced spectrum carries 0-based ``original_indices`` into the
    FULL (pre-reduction) spectrum. If the input spectrum was itself already
    reduced, the indices are mapped through so they keep pointing at the
    original DFT run positions.
    """
    labels, deg_groups = classify_and_pair(spectrum, tol_mev=tol_mev, symprec=symprec)
    drop: set[int] = set()
    for g in deg_groups:
        if len(g) < 2:
            continue
        keep = min(g, key=lambda i: labels[i] or "")
        drop.update(i for i in g if i != keep)

    n = spectrum.n_modes
    mask = np.array([i not in drop for i in range(n)])
    original = spectrum.original_indices
    kept = np.where(mask)[0]
    if original is not None:
        kept = np.asarray(original, dtype=int)[kept]
    reduced = PhononSpectrum(
        frequencies_mev=spectrum.frequencies_mev[mask],
        eigenvectors=spectrum.eigenvectors[mask],
        atom_frac_coords=spectrum.atom_frac_coords,
        atom_symbols=spectrum.atom_symbols,
        atomic_masses=spectrum.atomic_masses,
        lattice=spectrum.lattice,
        symmetries=[s for k, s in enumerate(labels) if mask[k]],
        iprs=spectrum.iprs[mask] if spectrum.iprs is not None else None,
        original_indices=kept,
        n_full=((spectrum.n_full if spectrum.n_full is not None else len(original)) if original is not None else n),
    )
    # indices point into the FULL pre-reduction spectrum, so the parent
    # size is the recorded n_full (or the input's own size when unreduced)
    n_full = spectrum.n_full if spectrum.n_full is not None else (len(original) if original is not None else n)
    check_original_indices(reduced.original_indices, reduced.n_modes, n_full=n_full)
    return reduced


def twin_of(deg_groups, i: int) -> int | None:
    """Return the degenerate partner of mode i (Ex<->Ey etc.), or None."""
    g = group_of(deg_groups, i)
    if g is None or len(g) < 2:
        return None
    return next(j for j in g if j != i)


def _op_mappings(spectrum, operations) -> list[np.ndarray]:
    """Per-operation atom mapping sigma (i -> atom R sends i onto).

    The defect frame is centered so R about the origin equals R about the
    defect. Mapping of the *centered* coordinates equals that of the raw
    ones (translation invariance of the nearest-partner search).
    """
    lattice = np.asarray(spectrum.lattice, dtype=float)
    inv_lat = np.linalg.inv(lattice)
    frac = np.mod(np.asarray(spectrum.atom_frac_coords, dtype=float), 1.0)
    center = _defect_center(spectrum)
    frac = np.mod(frac - center, 1.0)
    cart = frac @ lattice
    symbols = np.asarray(spectrum.atom_symbols)
    return [_atom_mapping(np.asarray(R, dtype=float), cart, frac, inv_lat, symbols, lattice) for R in operations]


def _group_operator_apply(mappings, operations, vecs) -> np.ndarray:
    """U_k v for each group operation: (U v)_j = R_k v_{sigma_k^-1(j)}.

    sigma maps atom i onto the atom R sends it to (per _atom_mapping), so
    the inverse mapping feeds each output atom from its preimage. This is
    the operator whose expectation value _mode_characters reports:
    chi_m(k) = <v_m | U_k | v_m>.
    """
    out = np.empty((len(operations),) + np.shape(vecs)[1:], dtype=float)
    for k, (R, m) in enumerate(zip(operations, mappings)):
        R = np.asarray(R, dtype=float)
        inv = np.argsort(m)
        out[k] = np.einsum("ij,naj->nai", R, vecs[:, inv])
    return out


def _per_op_characters(table, name: str) -> np.ndarray:
    """Character of each operation (ops ordered E, C3, C3^2, sv1..3)."""
    entry = table[name]
    if isinstance(entry, dict):
        return np.asarray(entry["class_chars"], dtype=float)
    return np.asarray(entry, dtype=float)


def _irrep_dimension(table, name: str) -> int:
    entry = table[name]
    return len(entry["sublabels"]) if isinstance(entry, dict) else 1


def _canonical_c3v() -> list[np.ndarray]:
    c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    c3 = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    mirror = np.diag([1.0, -1.0, 1.0])  # sigma_v containing x and z
    return [np.eye(3), c3, c3 @ c3, mirror, c3 @ mirror, c3 @ c3 @ mirror]


# Point-group operations about z, in the order of the group's per-operation characters in
# CHARACTER_TABLES. A group listed here can be used to decompose rank-2 tensors.
CANONICAL_OPERATIONS: dict[str, list[np.ndarray]] = {"3m": _canonical_c3v()}


def _irrep_names(table) -> list[str]:
    return [name for name in table if not name.startswith("_")]


def tensor_irrep_fractions(tensor: np.ndarray, pg_symbol: str = "3m") -> dict[str, float]:
    """Norm fraction of a 3x3 tensor in each irrep of the point group.

    Character projectors P_G T = (d_G / |G|) sum_g chi_G(g) R(g) T R(g)^T, with the group's
    canonical operations about z, so the tensor must be expressed in that frame (z = the
    defect's principal axis). The projectors are orthogonal and complete: the squared
    fractions add up to 1.
    """
    if pg_symbol not in CANONICAL_OPERATIONS:
        raise ValueError(f"No canonical operations for point group '{pg_symbol}'.")
    table = CHARACTER_TABLES[pg_symbol]
    ops = CANONICAL_OPERATIONS[pg_symbol]
    tensor = np.asarray(tensor, dtype=float)
    norm = np.linalg.norm(tensor)
    out = {}
    for name in _irrep_names(table):
        chars = _per_op_characters(table, name)
        projected = chars[0] / len(ops) * sum(c * R @ tensor @ R.T for c, R in zip(chars, ops))
        out[name] = float(np.linalg.norm(projected) / norm) if norm > 0 else 0.0
    return out


def direct_product(pg_symbol: str, sym_a: str, sym_b: str) -> list[str]:
    """Irreps in the direct product of two irreps (sublabels such as Ex/Ey count as their parent).

    By character orthogonality, n_G = (1/|G|) sum_g chi_a(g) chi_b(g) chi_G(g), over the
    per-operation characters of CHARACTER_TABLES (real for the groups listed there).
    """
    table = CHARACTER_TABLES.get(pg_symbol)
    if table is None:
        raise ValueError(f"No character table for point group '{pg_symbol}'.")
    a = _per_op_characters(table, _parent_label(sym_a))
    b = _per_op_characters(table, _parent_label(sym_b))
    out = []
    for name in _irrep_names(table):
        n = float(np.sum(a * b * _per_op_characters(table, name))) / len(a)
        reps = int(round(n))
        if abs(n - reps) > 1e-9 or reps < 0:
            raise ValueError(f"{sym_a} x {sym_b} does not decompose into irreps of '{pg_symbol}'.")
        out.extend([name] * reps)
    return out


# Sector-presence threshold: the projector applied to a vector in its own
# irrep returns norm 1 up to numerical error (~1e-5); a genuine admixture of
# another irrep shows up as a projection norm of order 0.1 or larger. 1e-2
# separates the noise floor from real projections.
_SECTOR_PRESENCE_NORM = 1e-2
# Gram-Schmidt acceptance: a residual below this is numerical noise, not a
# new independent direction.
_GS_KEEP_NORM = 1e-3


def _build_clusters(freqs: np.ndarray, tol_mev: float) -> list[list[int]]:
    """Chains of modes within tol of their neighbour, in frequency order.

    Unlike deg_groups (which only merge complete sets of one parent irrep),
    clusters may span several parents — exactly the accidental degeneracies
    that produce mixed eigenvectors.
    """
    clusters: list[list[int]] = []
    cur: list[int] = []
    for idx in np.argsort(freqs):
        if cur and freqs[idx] - freqs[cur[-1]] > tol_mev:
            clusters.append(cur)
            cur = []
        cur.append(int(idx))
    if cur:
        clusters.append(cur)
    return clusters


def _project_sector(vecs, cluster, mappings, ops, table, parent):
    """Isotypic projection of a cluster onto one irrep's sector.

    Applies P_G = (d_G / |G|) sum_k chi_G(R_k)^* U_k to every cluster
    member, keeps the nonzero results and Gram-Schmidt-orthonormalizes
    them. Returns the sector basis (may be fewer vectors than members).
    """
    chi_ops = _per_op_characters(table, parent)
    d = _irrep_dimension(table, parent)
    sector: list[np.ndarray] = []
    for i in cluster:
        U = _group_operator_apply(mappings, ops, vecs[i : i + 1])
        u = sum(float(np.conj(chi_ops[k])) * U[k] for k in range(len(ops)))
        u *= d / len(ops)
        norm = np.linalg.norm(u)
        if norm > _SECTOR_PRESENCE_NORM:
            sector.append(u / norm)
    basis: list[np.ndarray] = []
    for u in sector:
        for b in basis:
            u = u - float(np.sum(u * b)) * b
        n = np.linalg.norm(u)
        if n > _GS_KEEP_NORM:
            basis.append(u / n)
    return basis


def symmetrize_degenerate_groups(spectrum, tol_mev: float = 0.01, verbose: bool = False):
    """Re-diagonalize accidentally mixed near-degenerate modes into pure irreps.

    Physics: [H, R] = 0 for every point-group operation R, so exact
    eigenvectors transform as pure irreps. At an *accidental* degeneracy
    between different irreps (e.g. an A1 landing on top of an Ex/Ey pair),
    the eigensolver may return a rotated basis of the degenerate subspace,
    giving modes with ~7% admixture of the wrong irrep. The degenerate
    subspace is degenerate to within tol, so any orthonormal basis of it is
    an eigenbasis to that accuracy — we may as well take the symmetric one.

    Method (Wigner): for each near-degenerate cluster of modes whose
    members span more than one parent irrep, apply the isotypic projectors

        P_G = (d_G / |G|) * sum_k chi_G(R_k)^* U_k

    with U_k the full space-group operator (permute atoms + rotate
    displacements). Each member's projection onto each irrep sector is
    collected and Gram-Schmidt orthonormalized within the sector; each new
    pure vector inherits the frequency of the old mode it overlaps most.
    Clusters already spanning a single irrep are left untouched (the
    projector is a fixpoint on them).

    Returns a new PhononSpectrum (input untouched) with symmetrized
    eigenvectors and freshly derived labels/groups (``classify_and_pair``
    re-run on the cleaned vectors, so sublabels follow the new characters);
    the ``symmetrization_report`` attribute lists symmetrized and skipped
    clusters.
    """
    labels, _ = classify_and_pair(spectrum, tol_mev)
    pg = detect_point_group_from_spectrum(spectrum)
    table = CHARACTER_TABLES.get(pg.symbol)
    if table is None:
        # No character table (e.g. synthetic structures) — nothing to
        # symmetrize against; return the spectrum unchanged.
        return spectrum
    ops = defect_frame_operations(spectrum)
    mappings = _op_mappings(spectrum, ops)
    vecs = np.array(spectrum.eigenvectors, dtype=float, copy=True)
    freqs = np.asarray(spectrum.frequencies_mev, dtype=float)
    report: dict[str, list] = {"symmetrized": [], "skipped": []}

    # Frequency clusters: chains of modes within tol of their neighbour.
    # Unlike deg_groups (which only merge complete sets of one parent
    # irrep), clusters may span several parents — exactly the accidental
    # degeneracies that produce mixed eigenvectors.
    clusters = _build_clusters(freqs, tol_mev)

    for g in clusters:
        parents: list[str] = []
        for i in g:
            p = _parent_label(labels[i])
            if p is not None and p not in parents:
                parents.append(p)
        if len(parents) < 2:
            continue  # single-irrep cluster: already pure (fixpoint)

        new_vecs: list[np.ndarray] = []
        new_freqs: list[float] = []
        ok = True
        for parent in parents:
            basis = _project_sector(vecs, g, mappings, ops, table, parent)
            n_expected = _irrep_dimension(table, parent) * sum(1 for i in g if _parent_label(labels[i]) == parent)
            if len(basis) > n_expected:
                report["skipped"].append((g, parents))
                ok = False
                break
            for u in basis:
                overlaps = [abs(float(np.sum(u * vecs[i]))) for i in g]
                best = g[int(np.argmax(overlaps))]
                new_vecs.append(u)
                new_freqs.append(float(freqs[best]))
        if not ok:
            continue
        if len(new_vecs) != len(g):
            report["skipped"].append((g, parents))
            if verbose:
                LOGGER.info(f"skipped cluster {g} {parents}: {len(new_vecs)} basis vectors for {len(g)} modes")
            continue
        # keep the cluster's slots frequency-sorted: pair up new vectors
        # with their inherited frequencies, sort, then fill the slots in order
        pairs = sorted(zip(new_vecs, new_freqs), key=lambda t: t[1])
        for slot, (u, f) in zip(g, pairs):
            vecs[slot] = u
            freqs[slot] = f
        report["symmetrized"].append((g, parents))
        if verbose:
            LOGGER.info(f"symmetrized {g}: {parents}")

    # Labels and groups are re-derived from the cleaned vectors rather than
    # inherited: a symmetrization can move a mode between irreps (that is
    # its whole point), and classify_and_pair applies the sublabel sign rule
    # the same way everywhere else in the module.
    new_spec = replace(
        spectrum,
        frequencies_mev=freqs,
        eigenvectors=vecs,
        symmetries=None,
        e_pair_complete=None,
        pair_ids=None,
        deg_groups=None,
        original_indices=spectrum.original_indices,
    )
    classify_and_pair(new_spec, tol_mev)
    new_spec.symmetrization_report = report
    return new_spec
