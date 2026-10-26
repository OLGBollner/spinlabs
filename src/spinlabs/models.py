from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
from scipy import constants as Cn

from spinlabs.constants import CONSTANTS
from spinlabs.serialization import FieldSpec, SchemaError, load_npz, save_npz, _fields_from_npz, warn_unknown_keys

from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)


class EnergyUnit(str, Enum):
    MHZ = "MHz"
    JOULE = "J"
    MEV = "meV"
    GHZ = "GHz"
    THZ = "THz"


def convert_energy(values: Union[float, np.ndarray], from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """
    Converts energy / frequency values between MHz, J, meV, GHz, and THz.
    """
    if from_unit == to_unit:
        return values.copy() if isinstance(values, np.ndarray) else values

    # Convert source unit to Joules (SI)
    if from_unit == "J":
        val_j = values
    elif from_unit == "MHz":
        val_j = values * CONSTANTS["MHz2J"]
    elif from_unit == "meV":
        val_j = values * CONSTANTS["meV2J"]
    elif from_unit == "GHz":
        val_j = values * (CONSTANTS["MHz2J"] * 1e3)
    elif from_unit == "THz":
        val_j = values * (CONSTANTS["THz2meV"] * CONSTANTS["meV2J"])
    else:
        raise ValueError(f"Unsupported source unit: '{from_unit}'")

    # Convert Joules (SI) to target unit
    if to_unit == "J":
        return val_j
    elif to_unit == "MHz":
        return val_j / CONSTANTS["MHz2J"]
    elif to_unit == "meV":
        return val_j / CONSTANTS["meV2J"]
    elif to_unit == "GHz":
        return val_j / (CONSTANTS["MHz2J"] * 1e3)
    elif to_unit == "THz":
        return val_j / (CONSTANTS["THz2meV"] * CONSTANTS["meV2J"])
    else:
        raise ValueError(f"Unsupported target unit: '{to_unit}'")


# How a perturbation run displaces the atoms along mode k (e_k: unit mass-weighted eigenvector,
# m: atomic masses, step: the tree's PERT):
#   normalized:    step * u_k / |u_k| with u_k = e_k / sqrt(m) -- every mode moves the atoms by the
#                  same Cartesian length, step (Angstrom); its normal coordinate is step * n_k
#                  with n_k = 1 / |u_k| (sqrt(amu)), see PhononSpectrum.mode_norm_factors
#   mass_weighted: step * u_k -- normal coordinate Q = step (Angstrom*sqrt(amu)), the earlier default
#   unweighted:    step * e_k -- the first NV_64 trees; Q = step * effective sqrt-mass
DISPLACEMENTS = ("normalized", "mass_weighted", "unweighted")


def displacement_convention(displacement: Optional[str], mass_weighted: bool = True) -> str:
    """The displacement convention named by ``displacement``, else the legacy ``mass_weighted`` flag."""
    if displacement is None:
        return "mass_weighted" if mass_weighted else "unweighted"
    if displacement not in DISPLACEMENTS:
        raise ValueError(f"Unknown displacement convention {displacement!r}; expected one of {DISPLACEMENTS}")
    return displacement


@dataclass
class ZFSTensor:
    """
    Represents a Zero-Field Splitting (ZFS) tensor in 3x3 Cartesian matrix form.
    Default unit is MHz.
    """

    matrix: np.ndarray  # Shape (3, 3)
    unit: str = "MHz"
    reference: Optional[np.ndarray] = None  # reference tensor (3x3) pinning the x/y axis assignment

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix.shape != (3, 3):
            raise ValueError(f"ZFS tensor matrix must have shape (3, 3), got {self.matrix.shape}")

    def traceless(self) -> np.ndarray:
        """Returns the traceless part of the ZFS matrix."""
        tr = np.trace(self.matrix) / 3.0
        return self.matrix - tr * np.eye(3)

    def with_reference(self, reference: np.ndarray) -> "ZFSTensor":
        """Return a copy whose axis assignment is pinned to ``reference``.

        ``reference`` is a 3x3 tensor (or principal-frame matrix) in the same
        orientation as ``self.matrix``: the returned tensor's principal axes are
        matched to those of the reference by eigenvector overlap, so that
        near-degenerate transverse axes (|D_xx| ~ |D_yy|) never swap labels
        between runs. Pass the relaxed/Q=0 tensor.
        """
        return ZFSTensor(matrix=self.matrix.copy(), unit=self.unit, reference=np.asarray(reference, dtype=float).copy())

    def principal_components(self, reference: Optional[np.ndarray] = None) -> tuple[float, float, float, np.ndarray]:
        """
        Calculates principal values (D_xx, D_yy, D_zz) and eigenvectors.

        Default convention (no reference): standard EPR convention,
        |D_zz| >= |D_yy| >= |D_xx| (with trace=0).

        If a reference tensor is given (explicitly or via ``self.reference``,
        see :meth:`with_reference`), the axes are instead matched to the
        reference's principal axes by largest eigenvector overlap, with signs
        aligned so that each axis has positive overlap with its reference
        partner. This keeps the x/y assignment continuous across runs whose
        transverse splitting |D_xx| - |D_yy| is near zero, where a magnitude
        sort can silently swap x and y (and flip the sign of E).

        Returns:
            (D_xx, D_yy, D_zz, eigenvectors)
        """
        D_tl = self.traceless()
        evals, evecs = np.linalg.eigh(D_tl)

        ref = reference if reference is not None else self.reference
        if ref is not None:
            ref_mat = np.asarray(ref, dtype=float)
            if ref_mat.shape != (3, 3):
                raise ValueError(f"reference must be a 3x3 tensor, got shape {ref_mat.shape}")
            # Reference frame: use its own principal axes (sorted convention is fine
            # here; overlap matching removes any ambiguity).
            _, _, _, R_ref = ZFSTensor(matrix=ref_mat).principal_components()
            # Greedy match each reference axis to the closest eigenvector.
            # Eigenvectors are orthonormal on both sides, so greedy is exact.
            G = np.abs(R_ref.T @ evecs)  # [ref_axis, eigvec]
            order = []
            for j in range(3):
                k = int(np.argmax(G[j]))
                order.append(k)
                G[:, k] = -1.0  # claim it
            ix, iy, iz = order
        else:
            # Sort by absolute magnitude so that |D_zz| is largest
            abs_order = np.argsort(np.abs(evals))  # [smallest, middle, largest]
            ix, iy, iz = abs_order[0], abs_order[1], abs_order[2]

        D_xx_val = evals[ix]
        D_yy_val = evals[iy]
        D_zz_val = evals[iz]

        R = np.column_stack([evecs[:, ix], evecs[:, iy], evecs[:, iz]])
        if ref is not None:
            # Sign-align each axis with its reference partner (dot > 0).
            dots = np.diag(R_ref.T @ R)
            R = R @ np.diag(np.where(dots < 0, -1.0, 1.0))
        # Ensure right-handed coordinate system for eigenvectors
        if np.linalg.det(R) < 0:
            R[:, 0] = -R[:, 0]

        return float(D_xx_val), float(D_yy_val), float(D_zz_val), R

    @property
    def D(self) -> float:
        """Axial ZFS parameter D = 3/2 * D_zz in the principal frame."""
        _, _, D_zz_val, _ = self.principal_components()
        return 1.5 * D_zz_val

    @property
    def E(self) -> float:
        """Rhombic ZFS parameter E = (D_xx - D_yy) / 2 in the principal frame."""
        D_xx_val, D_yy_val, _, _ = self.principal_components()
        return 0.5 * (D_xx_val - D_yy_val)

    @property
    def xx(self) -> float:
        return float(self.matrix[0, 0])

    @property
    def yy(self) -> float:
        return float(self.matrix[1, 1])

    @property
    def zz(self) -> float:
        return float(self.matrix[2, 2])

    @property
    def xy(self) -> float:
        return float(self.matrix[0, 1])

    @property
    def xz(self) -> float:
        return float(self.matrix[0, 2])

    @property
    def yz(self) -> float:
        return float(self.matrix[1, 2])

    def to_unit(self, target_unit: str) -> ZFSTensor:
        """Convert tensor matrix into target energy unit using convert_energy."""
        if self.unit == target_unit:
            return ZFSTensor(matrix=self.matrix.copy(), unit=self.unit, reference=self.reference)
        out_mat = convert_energy(self.matrix, self.unit, target_unit)
        return ZFSTensor(matrix=out_mat, unit=target_unit, reference=self.reference)

    def rotate(self, rotation_matrix: np.ndarray) -> ZFSTensor:
        """Rotate tensor: D' = R @ D @ R.T"""
        R = np.asarray(rotation_matrix, dtype=float)
        rotated_mat = R @ self.matrix @ R.T
        # Rotate the pinned reference frame along with the tensor, so the
        # axis assignment stays consistent with the tensor's new orientation.
        ref = None if self.reference is None else R @ np.asarray(self.reference, dtype=float) @ R.T
        return ZFSTensor(matrix=rotated_mat, unit=self.unit, reference=ref)


def check_original_indices(
    original_indices: np.ndarray,
    n_modes: int,
    n_full: Optional[int] = None,
) -> None:
    """Raise if ``original_indices`` violates the 0-based idx convention.

    It must be a 1-D integer array, one entry per kept mode, strictly
    increasing (full-spectrum order preserved), non-negative, and — when
    ``n_full`` is given — within ``[0, n_full)``.
    """
    oi = np.asarray(original_indices)
    if oi.ndim != 1 or not np.issubdtype(oi.dtype, np.integer):
        raise TypeError("original_indices must be a 1-D integer array")
    if len(oi) != n_modes:
        raise ValueError(f"original_indices has length {len(oi)} but spectrum has {n_modes} modes — inconsistent")
    if len(oi) > 1 and np.any(np.diff(oi) <= 0):
        raise ValueError("original_indices must be strictly increasing (0-based, full-spectrum order preserved)")
    if (oi < 0).any():
        raise ValueError("original_indices must be 0-based (no negative indices)")
    if n_full is not None and (oi >= n_full).any():
        raise ValueError(f"original_indices must be < n_full ({n_full}); the 0-based idx convention is broken")


@dataclass
class PhononMode:
    """Represents a single vibrational mode."""

    index: int
    frequency_mev: float
    eigenvector: np.ndarray  # Shape (N_atoms, 3)
    symmetry: Optional[str] = None
    ipr: Optional[float] = None
    pair_id: int = -1  # index of degenerate partner mode; -1 = unknown (no spectrum)
    original_index: int = -1  # mode index in the full DFT run; -1 = unknown

    @property
    def frequency_thz(self) -> float:
        return self.frequency_mev / CONSTANTS["THz2meV"]

    @property
    def frequency_rads(self) -> float:
        return self.frequency_mev * CONSTANTS["meV2rads"]


@dataclass
class PhononSpectrum:
    """Represents a collection of phonon modes in a supercell."""

    frequencies_mev: np.ndarray  # Shape (N_modes,)
    eigenvectors: np.ndarray  # Shape (N_modes, N_atoms, 3)
    atom_frac_coords: np.ndarray  # Shape (N_atoms, 3)
    atom_symbols: list[str]  # Length N_atoms
    atomic_masses: np.ndarray  # Shape (N_atoms,)
    lattice: np.ndarray  # Shape (3, 3)
    symmetries: Optional[list[str]] = None  # Length N_modes
    iprs: Optional[np.ndarray] = None  # Shape (N_modes,)
    e_pair_complete: Optional[list[bool]] = None  # Length N_modes
    pair_ids: Optional[np.ndarray] = (
        None  # Shape (N_modes,): degenerate partner index, n_modes if unpaired (out-of-bounds sentinel)
    )
    deg_groups: Optional[list] = None  # Lists of mode indices forming complete irrep component sets
    original_indices: Optional[np.ndarray] = None  # Shape (N_modes,): mode index in the full DFT run
    n_full: Optional[int] = None  # size of the full spectrum original_indices point into
    frequency_unit: str = "meV"

    def __post_init__(self):
        # Symmetry labels + degeneracy groups come from the general symmetry module.
        # Stored labels are trusted; classification only runs when labels are absent.
        if self.symmetries is None and self.eigenvectors is not None and len(self.eigenvectors) > 0:
            try:
                from .symmetry import classify_and_pair

                self.symmetries, self.deg_groups = classify_and_pair(self)
            except Exception:
                # General detection unavailable (no table, or structure too
                # degenerate for spglib, e.g. synthetic test data); fall back
                # to the legacy pattern-based C3v analyzer.
                self.analyze_c3v_symmetry()
        if self.e_pair_complete is None and self.symmetries is not None and len(self.frequencies_mev) > 0:
            self.check_e_pair_completeness()
        if self.original_indices is None:
            self.original_indices = np.arange(self.n_modes, dtype=int)
        if self.n_full is None:
            self.n_full = self.n_modes

    @property
    def n_modes(self) -> int:
        return len(self.frequencies_mev)

    @property
    def n_atoms(self) -> int:
        return len(self.atom_symbols)

    def check_e_pair_completeness(self, tol_mev: float = 0.05) -> list[bool]:
        """
        Determines whether each vibrational mode has both Ex and Ey pairs in the dataset.
        For A1 and A2 modes, the value is False.
        For E modes (Ex or Ey), it is True if a matching partner of opposite symmetry
        exists within frequency tolerance tol_mev, otherwise False.
        """
        if self.symmetries is None or len(self.frequencies_mev) == 0:
            self.e_pair_complete = None
            return []

        n = len(self.frequencies_mev)
        completeness = [False] * n
        matched = set()

        for i in range(n):
            sym_i = self.symmetries[i]
            if sym_i not in ("Ex", "Ey") or i in matched:  # a paired mode must not pair again
                continue

            target_sym = "Ey" if sym_i == "Ex" else "Ex"
            freq_i = self.frequencies_mev[i]

            best_j = None
            min_diff = float("inf")
            for j in range(n):
                if j == i or j in matched:
                    continue
                if self.symmetries[j] == target_sym:
                    diff = abs(self.frequencies_mev[j] - freq_i)
                    if diff < tol_mev and diff < min_diff:
                        min_diff = diff
                        best_j = j

            if best_j is not None:
                completeness[i] = True
                completeness[best_j] = True
                matched.add(i)
                matched.add(best_j)

        self.e_pair_complete = completeness
        return completeness

    def build_pair_ids(self, tol_mev: float = 0.05) -> np.ndarray:
        """Deprecated: use spinlabs.symmetry.classify_and_pair (deg_groups)."""
        """
        Builds the symmetric pair_ids mapping between degenerate E-mode partners.

        pair_ids[i] is the index of mode i's degenerate partner (Ex <-> Ey within
        tol_mev), or n_modes if unpaired — the sentinel is deliberately out of bounds
        so that unguarded indexing (e.g. freqs[pair_ids[i]]) raises IndexError instead
        of silently wrapping around to the last element (as -1 would).
        The relation must be a strict involution:
        pair_ids[i] = j implies pair_ids[j] = i. Degenerate chains (a mode whose
        partner is already paired with a different mode) raise ValueError loudly.
        """
        if self.symmetries is None:
            self.pair_ids = np.full(self.n_modes, self.n_modes, dtype=int)
            return self.pair_ids

        n = self.n_modes
        pair_ids = np.full(n, n, dtype=int)
        # General path: use degeneracy groups from the symmetry module when present.
        if self.deg_groups is not None:
            for group in self.deg_groups:
                if len(group) == 2:
                    i, j = int(group[0]), int(group[1])
                    pair_ids[i] = j
                    pair_ids[j] = i
                # groups of 1 or >2 stay unpaired (n sentinel); chains impossible by construction
            self.pair_ids = pair_ids
            return self.pair_ids

        for i in range(n):
            sym_i = self.symmetries[i]
            if sym_i not in ("Ex", "Ey") or pair_ids[i] != n:
                continue

            target_sym = "Ey" if sym_i == "Ex" else "Ex"
            freq_i = self.frequencies_mev[i]

            best_j = None
            min_diff = float("inf")
            for j in range(n):
                if j == i or pair_ids[j] != n:
                    continue
                if self.symmetries[j] == target_sym:
                    diff = abs(self.frequencies_mev[j] - freq_i)
                    if diff < tol_mev and diff < min_diff:
                        min_diff = diff
                        best_j = j

            if best_j is not None:
                # Strict involution: best_j must be unpaired here because of the
                # pair_ids[j] != n guard above, so no chains can form.
                pair_ids[i] = best_j
                pair_ids[best_j] = i

        self.pair_ids = pair_ids
        return self.pair_ids

    def validate_pair_ids(self) -> None:
        """
        Raises ValueError if pair_ids is not a strict symmetric involution
        (pair_ids[i] = j implies pair_ids[j] = i). Chains fail loudly.
        """
        if self.pair_ids is None:
            return
        n = len(self.pair_ids)
        for i, j in enumerate(self.pair_ids):
            if j >= n:
                continue
            if self.pair_ids[j] != i:
                raise ValueError(
                    f"Invalid pair_ids: mode {i} pairs with {j}, but mode {j} "
                    f"pairs with {self.pair_ids[j]}. "
                    "pair_ids must be a strict symmetric involution (no degenerate chains)."
                )

    def index_of_original(self, original_index: int) -> Optional[int]:
        """
        Maps a mode index from the full DFT run (original indexing) to its
        position in this spectrum. Returns None if the mode was dropped.
        """
        if self.original_indices is None:
            return original_index if 0 <= original_index < self.n_modes else None
        hits = np.where(self.original_indices == original_index)[0]
        return int(hits[0]) if len(hits) else None

    def infer_original_indices(self, raw_perturbations: dict[int, float], tol_rel: float = 0.05) -> bool:
        """
        Legacy fallback: reconstructs original_indices by matching raw perturbation
        amplitudes to phonon frequencies. Since q = q0 * sqrt(2*omega/hbar), the
        perturbation amplitude squared is proportional to the mode frequency, so a
        rank matching (sorted pert^2 vs sorted frequency) recovers the mapping.

        Only runs when the raw index set clearly exceeds this spectrum's modes
        (i.e. the phonon file dropped modes without recording which). Returns True
        if a consistent mapping was found, False otherwise.

        Assumes degenerate E pairs share a frequency; ties are matched in index order.
        """
        n_raw = max(raw_perturbations) + 1 if raw_perturbations else 0
        if n_raw <= self.n_modes:
            return False

        # Fit the proportionality constant q0^2 from rank matching
        raw_sorted = sorted(raw_perturbations.items(), key=lambda kv: kv[1])
        freq_pos = np.argsort(self.frequencies_mev)
        if len(raw_sorted) != self.n_modes:
            LOGGER.warning(
                f"raw ZFS data covers {len(raw_sorted)} modes but spectrum has "
                f"{self.n_modes}; cannot infer original_indices by rank matching. "
                "Modes will be dropped loudly instead of silently mismatched."
            )
            return False

        # Check proportionality quality first
        p2 = np.array([p for _, p in raw_sorted]) ** 2
        f = self.frequencies_mev[freq_pos]
        valid = f > 0
        if valid.sum() < 2:
            return False
        c = np.corrcoef(p2[valid], f[valid])[0, 1]
        if c < 1 - tol_rel:
            LOGGER.warning(
                f"perturbation-frequency correlation is only {c:.4f}; "
                "refusing to infer original_indices from a dubious match."
            )
            return False

        q0_2 = float(np.median(p2[valid] / f[valid]))
        residuals = np.abs(p2 - q0_2 * f) / (q0_2 * f + 1e-30)
        bad = int(np.sum(residuals[valid] > tol_rel))
        if bad > 0:
            LOGGER.warning(
                f"{bad} modes deviate > {tol_rel:.0%} from the fitted "
                f"q0^2 = {q0_2:.3e}; inferred original_indices may be unreliable."
            )

        original_indices = np.full(self.n_modes, -1, dtype=int)
        # Raw indices are 1-based folders minus 1; they map in sorted-perturbation order
        # to sorted-frequency positions. Degenerate twins (same frequency) are matched
        # in raw-index order, mirroring how the perturbation folders were generated.
        for pos, (raw_idx, _) in zip(freq_pos, raw_sorted):
            original_indices[pos] = raw_idx

        self.original_indices = original_indices
        LOGGER.info(
            f"Inferred original_indices for legacy phonon file by amplitude-frequency "
            f"rank matching (corr={c:.5f}, q0^2={q0_2:.3e}). "
            "Regenerate the phonon npz with explicit original_indices for a principled mapping."
        )
        return True

    def get_mode(self, idx: int) -> PhononMode:
        sym = self.symmetries[idx] if self.symmetries is not None else None
        ipr_val = float(self.iprs[idx]) if self.iprs is not None else None
        pair = int(self.pair_ids[idx]) if self.pair_ids is not None else -1  # -1 = unknown (no spectrum attached)
        orig = int(self.original_indices[idx]) if self.original_indices is not None else idx
        return PhononMode(
            index=idx,
            frequency_mev=float(self.frequencies_mev[idx]),
            eigenvector=self.eigenvectors[idx],
            symmetry=sym,
            ipr=ipr_val,
            pair_id=pair,
            original_index=orig,
        )

    def filter_by_energy(self, min_mev: float = -np.inf, max_mev: float = np.inf) -> np.ndarray:
        """Returns mode indices within energy range [min_mev, max_mev]."""
        mask = (self.frequencies_mev >= min_mev) & (self.frequencies_mev <= max_mev)
        return np.where(mask)[0]

    def filter_by_symmetry(self, symmetry_label: str) -> np.ndarray:
        """Returns mode indices matching the given symmetry label."""
        if self.symmetries is None:
            return np.array([], dtype=int)
        return np.array([i for i, sym in enumerate(self.symmetries) if sym == symmetry_label], dtype=int)

    def frequencies_to_unit(self, target_unit: str) -> np.ndarray:
        """Convert mode frequencies from meV to target energy/frequency unit."""
        return convert_energy(self.frequencies_mev, "meV", target_unit)

    def translate_defect_to_origin(
        self, defect_pos: Optional[np.ndarray] = None, wrap: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        frac_atoms = self.atom_frac_coords
        lattice = self.lattice
        inv_lat = np.linalg.inv(lattice)
        symbols = self.atom_symbols

        if defect_pos is None:
            if "N" in symbols and "C" in symbols:
                n_indices = [i for i, s in enumerate(symbols) if s == "N"]
                if len(n_indices) != 1:
                    raise ValueError(f"Expected exactly one N, found {len(n_indices)}")
                defect_frac = frac_atoms[n_indices[0]]
            elif "Cl" in symbols and "Si" in symbols:
                cl_indices = [i for i, s in enumerate(symbols) if s == "Cl"]
                defect_frac = frac_atoms[cl_indices].mean(axis=0)
            else:
                # Generic fallback: self-contained point-defect localizer
                # (local import avoids the structures -> models cycle).
                from spinlabs.vasp.structures import detect_defect_position

                detected = detect_defect_position(self)
                if detected is not None:
                    defect_frac = detected
                else:
                    LOGGER.warning("Could not identify defect centre, no shift applied.")
                    return frac_atoms, np.zeros(3)
        else:
            defect_pos = np.asarray(defect_pos, dtype=float)
            defect_frac = defect_pos @ inv_lat

        shifted_frac = frac_atoms - defect_frac
        if wrap:
            shifted_frac = np.mod(shifted_frac, 1.0)

        return shifted_frac, defect_frac

    def analyze_c3v_symmetry(self) -> list[str]:
        """
        Analyzes C3v point group symmetry representations (A1, A2, Ex, Ey) for each phonon mode.

        Deprecated: use spinlabs.symmetry.classify_modes, which detects the
        point group and classifies via general projection. This
        legacy method is kept as a validation reference only.
        """
        import warnings

        warnings.warn(
            "analyze_c3v_symmetry is deprecated; use spinlabs.symmetry.classify_modes "
            "(general projection). Kept as a validation reference.",
            DeprecationWarning,
            stacklevel=2,
        )
        from spinlabs.utils import MathUtils

        try:
            frac_atoms, _ = self.translate_defect_to_origin()
        except Exception:
            self.symmetries = ["A1"] * self.n_modes
            return self.symmetries

        symbols = np.array(self.atom_symbols)
        freqs = self.frequencies_mev
        eigs = self.eigenvectors
        lattice = self.lattice

        if "Si" in symbols and "Cl" in symbols:
            principal_axis = [0, 0, 1]
            reflection_normal = [1, 0, 0]
        elif "C" in symbols and "N" in symbols:
            principal_axis = [1, 1, 1]
            reflection_normal = [1, -1, 0]
        else:
            principal_axis = [0, 0, 1]
            reflection_normal = [1, 0, 0]

        R_C3 = MathUtils.rotation_around_symmetry_axis(principal_axis, 3)
        R_sv = MathUtils.reflection_matrix(reflection_normal)

        inv_lat = np.linalg.inv(lattice)
        num_atoms = frac_atoms.shape[0]
        num_modes = eigs.shape[0]
        cart_atoms = frac_atoms @ lattice

        def get_mapping(R):
            mapping = np.zeros(num_atoms, dtype=int)
            rotated_cart = cart_atoms @ R.T
            rot_frac = np.mod(rotated_cart @ inv_lat, 1.0)
            orig_frac = np.mod(frac_atoms, 1.0)

            for i in range(num_atoms):
                diffs = np.mod(orig_frac - rot_frac[i] + 0.5, 1.0) - 0.5
                dists = np.linalg.norm(diffs @ lattice, axis=1)
                valid_indices = np.where(symbols == symbols[i])[0]
                if len(valid_indices) == 0:
                    mapping[i] = i
                else:
                    mapping[i] = valid_indices[np.argmin(dists[valid_indices])]
            return mapping

        map_C3 = get_mapping(R_C3)
        map_sv = get_mapping(R_sv)

        results_dict = {"idx": [], "freqs": [], "sym": [], "char_C3": [], "char_sv": []}

        for m in range(num_modes):
            eig = eigs[m]
            char_C3 = np.trace(np.dot(eig[map_C3], R_C3 @ eig.T))
            char_sv = np.trace(np.dot(eig[map_sv], R_sv @ eig.T))

            if char_C3 > 0.8:
                sym = "A1" if char_sv > 0.0 else "A2"
            else:
                sym = "Ex" if char_sv > 0.0 else "Ey"

            results_dict["idx"].append(m)
            results_dict["freqs"].append(freqs[m])
            results_dict["sym"].append(sym)
            results_dict["char_C3"].append(char_C3)
            results_dict["char_sv"].append(char_sv)

        sort_indices = np.argsort(results_dict["idx"])
        output_dict = {key: np.array(value)[sort_indices] for key, value in results_dict.items()}

        self.symmetries = [str(s) for s in output_dict["sym"]]
        return self.symmetries

    def get_phonon_pert(
        self,
        perturbation_scale_si: float = 1.0,
        perturbation_scale: Optional[float] = None,
        *,
        mass_weighted: bool = True,
        displacement: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Computes mass-weighted perturbation displacements (SI), frequencies (J), symmetries, and IPRs.

        The perturbation amplitude for each mode follows the mass-weighted
        phonon coordinate  q = Q * sqrt(2*omega/hbar), with Q the mode's normal
        coordinate for a step q0 (the scale). Modes with frequency <= 0 get None.

        ``displacement`` (else ``mass_weighted``) names how the runs were displaced
        (see DISPLACEMENTS): mass_weighted runs have Q = q0; unweighted runs (q0 * e)
        have Q = q0 times the mode's effective sqrt-mass (:meth:`effective_sqrt_mass`);
        normalized runs (a Cartesian step q0 along every mode) have Q = q0 * n_k
        (:meth:`mode_norm_factors`).
        """
        scale = perturbation_scale if perturbation_scale is not None else perturbation_scale_si
        omega_rads = self.frequencies_mev * CONSTANTS["meV2rads"]
        convention = displacement_convention(displacement, mass_weighted)
        if convention == "normalized":
            weights = self.mode_norm_factors()
        elif convention == "unweighted":
            weights = self.effective_sqrt_mass()
        else:
            weights = np.ones(self.n_modes)
        displacements = [
            scale * w * np.sqrt(2.0 * omega / Cn.hbar) if freq > 0 else None
            for freq, omega, w in zip(self.frequencies_mev, omega_rads, weights)
        ]
        freqs_j = self.frequencies_to_unit("J")
        iprs = self.get_ipr()
        syms = self.symmetries if self.symmetries is not None else ["A1"] * self.n_modes

        return {
            "disp": displacements,
            "freqs": freqs_j,
            "sym": syms,
            "ipr": iprs,
            "eigs": self.eigenvectors,
        }

    def effective_sqrt_mass(self) -> np.ndarray:
        """Per-mode sum_j sqrt(m_j) |e_kj|^2 / sum_j |e_kj|^2, in sqrt(amu).

        Displacing the atoms by Q * e_k (no 1/sqrt(m)) moves the mass-weighted
        normal coordinate of mode k by Q times this factor: sqrt(12.011) for
        an all-carbon mode, more where the mode moves heavier atoms.
        """
        if self.atomic_masses is None:
            raise ValueError("Phonon spectrum has no atomic masses; cannot correct non-mass-weighted perturbations")
        weight = np.sum(np.abs(np.asarray(self.eigenvectors)) ** 2, axis=2)  # (n_modes, n_atoms)
        sqrt_m = np.sqrt(np.asarray(self.atomic_masses, dtype=float))
        return (weight @ sqrt_m) / weight.sum(axis=1)

    def mode_norm_factors(self) -> np.ndarray:
        """Per-mode n_k = 1 / |e_k / sqrt(m)|, in sqrt(amu).

        A normalized run displaces the atoms by step * u_k / |u_k| with u_k = e_k / sqrt(m),
        a Cartesian length `step` for every mode; that moves the normal coordinate of mode k by
        step * n_k. sqrt(12.011) for an all-carbon mode, up to sqrt(m) of the heaviest atom.
        """
        if self.atomic_masses is None:
            raise ValueError("Phonon spectrum has no atomic masses; cannot normalise the displacements")
        eigs = np.abs(np.asarray(self.eigenvectors))
        eigs = eigs.reshape(eigs.shape[0], -1, 3)
        sqrt_m = np.sqrt(np.asarray(self.atomic_masses, dtype=float))
        return 1.0 / np.linalg.norm((eigs / sqrt_m[None, :, None]).reshape(eigs.shape[0], -1), axis=1)

    def calc_ipr(self) -> np.ndarray:
        """Calculates and caches the Inverse Participation Ratio (IPR) for all phonon modes."""
        from spinlabs.utils import MathUtils

        if self.iprs is None:
            self.iprs = MathUtils.calc_ipr(self.eigenvectors)
        return self.iprs

    def get_ipr(self) -> np.ndarray:
        """Returns the IPR array, computing it if not already present."""
        if self.iprs is None:
            return self.calc_ipr()
        return self.iprs

    def filter_sym_pairs(self, tol_mev: float = 0.01) -> "PhononSpectrum":
        """
        Removes redundant degenerate partner modes from Ex/Ey doublets.

        For every Ey mode, if an Ex mode lies within ``tol_mev`` in
        frequency, the Ey twin is dropped (the doublet is redundant — E is
        doubly degenerate, Ex and Ey span the same 2-D representation). All
        Ex modes are kept. Group-theoretic bookkeeping: for a spectrum with
        N_atoms modes, A1 + A2 + 2*E = 3*N_atoms, so after filtering
        A1 + A2 + E = 2*N_atoms + 1 modes remain. The reduced spectrum
        carries 0-based
        ``original_indices`` pointing into the *full* spectrum, so downstream
        code can map each reduced mode back to its source position.

        Requires symmetry labels: computes them if not already present.
        Validates the returned ``original_indices`` against the 0-based
        convention (see ``check_original_indices``); raises on violation.
        """
        syms = self.symmetries if self.symmetries is not None else self.analyze_c3v_symmetry()
        freqs = self.frequencies_mev
        n = self.n_modes

        ex_idx = np.array([i for i in range(n) if syms[i] == "Ex"], dtype=int)
        ex_freqs = freqs[ex_idx]
        skip_indices = set()
        for i in range(n):
            if syms[i] == "Ey" and np.any(np.abs(ex_freqs - freqs[i]) < tol_mev):
                skip_indices.add(i)

        mask = np.array([i not in skip_indices for i in range(n)])
        # 0-based indices into the ORIGINAL (full) spectrum: when this
        # spectrum is itself already reduced, map through its own indices
        original = self.original_indices
        kept = np.where(mask)[0]
        if original is not None:
            kept = np.asarray(original, dtype=int)[kept]
        reduced = PhononSpectrum(
            frequencies_mev=freqs[mask],
            eigenvectors=self.eigenvectors[mask],
            atom_frac_coords=self.atom_frac_coords,
            atom_symbols=self.atom_symbols,
            atomic_masses=self.atomic_masses,
            lattice=self.lattice,
            symmetries=[s for k, s in enumerate(syms) if mask[k]],
            iprs=self.iprs[mask] if self.iprs is not None else None,
            # 0-based indices into the ORIGINAL (full) spectrum, so downstream
            # code can map each reduced mode back to its source position.
            original_indices=kept,
            n_full=((self.n_full if self.n_full is not None else len(original)) if original is not None else n),
        )
        n_full = self.n_full if self.n_full is not None else (len(original) if original is not None else n)
        check_original_indices(reduced.original_indices, reduced.n_modes, n_full=n_full)
        return reduced

    SCHEMA = [
        FieldSpec(
            "frequencies_mev",
            "array",
            npz_key="frequencies",
            shape=("n_modes",),
            save_aliases=("frequencies_mev", "freqs"),
        ),
        FieldSpec("frequency_unit", "str", optional=True),
        FieldSpec("eigenvectors", "array", shape=("n_modes", "n_atoms", 3), save_aliases=("eigs",)),
        FieldSpec("atom_frac_coords", "array", shape=("n_atoms", 3)),
        FieldSpec("atom_symbols", "list_str"),
        FieldSpec("atomic_masses", "array", shape=("n_atoms",), save_aliases=("masses",)),
        FieldSpec("lattice", "array", shape=(3, 3)),
        FieldSpec("symmetries", "list_str", optional=True),
        FieldSpec("iprs", "array", shape=("n_modes",), optional=True),
        FieldSpec("e_pair_complete", "list_bool", optional=True),
        FieldSpec("pair_ids", "array", shape=("n_modes",), optional=True),
        FieldSpec("deg_groups", "array", optional=True),
        FieldSpec("original_indices", "array", shape=("n_modes",), optional=True, save_aliases=("idx",)),
        FieldSpec("n_full", "int", optional=True),
    ]

    def save(self, out_path: Union[str, Path]) -> str:
        """Saves spectrum to .npz file with explicit frequency unit tag."""
        return save_npz(self, out_path, self.SCHEMA)

    @classmethod
    def load(cls, in_path: Union[str, Path]) -> PhononSpectrum:
        """Loads spectrum from .npz file, converting frequencies using explicit unit tags."""
        data = np.load(str(in_path), allow_pickle=True)

        def _pre(d: Any, kw: Dict[str, Any]) -> Dict[str, Any]:
            unit = str(d["frequency_unit"]) if "frequency_unit" in d else "meV"
            freqs_raw = d["frequencies"] if "frequencies" in d else d["frequencies_mev"]
            kw["frequencies_mev"] = convert_energy(freqs_raw, unit, "meV")
            kw["frequency_unit"] = "meV"
            if kw.get("symmetries") is not None:
                kw["symmetries"] = list(kw["symmetries"])
            return kw

        warn_unknown_keys(data.files, cls.SCHEMA)
        return load_npz(cls, in_path, cls.SCHEMA, pre_decode=_pre)


@dataclass
class PerturbationEntry:
    """Represents a single 1D or 2D perturbed calculation."""

    order: int  # 1 for 1D (dD/dq), 2 for 2D (d2D/dq_i dq_j)
    mode_indices: tuple[int, ...]
    amplitude: Union[float, tuple[float, float]]
    zfs_tensor: ZFSTensor
    energy: Optional[float] = None


def _decode_2d_array(value: Any, data: Any) -> Optional[np.ndarray]:
    """2-d coupling arrays: empty or malformed arrays come back as None."""
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0:
        return None
    return arr if arr.ndim >= 2 and arr.size > 0 else None


def _decode_calc_method(value: str, data: Any) -> Optional[str]:
    """Legacy files may store the string "None" or "" for an absent method."""
    return None if value in ("None", "") else value


def _encode_first_order(d: dict) -> dict:
    """PerturbationEntry -> plain dict; amplitude Å*sqrt(amu) -> SI when unconverted."""
    if not d:
        return None
    saved = {}
    for idx, entry in d.items():
        if isinstance(entry, PerturbationEntry):
            amp = entry.amplitude
            if isinstance(amp, (int, float)) and amp > 1e-4:
                amp = amp * CONSTANTS["ang_amu2SI"]
            saved[idx] = {"tensor": entry.zfs_tensor.matrix, "unit": entry.zfs_tensor.unit, "pert": amp}
        elif isinstance(entry, dict):
            saved[idx] = entry
        else:
            raise SchemaError(f"first_order[{idx}]: unsupported entry type {type(entry).__name__}")
    return saved


def _encode_second_order(d: dict) -> dict:
    """2D entries: tuple keys -> "i_j" strings; amplitude conversion as in 1D."""
    if not d:
        return None
    saved = {}
    for idx, entry in d.items():
        key = f"{idx[0]}_{idx[1]}" if isinstance(idx, tuple) else str(idx)
        if isinstance(entry, PerturbationEntry):
            amp = entry.amplitude
            if isinstance(amp, (tuple, list)):
                amp = tuple(
                    a * CONSTANTS["ang_amu2SI"] if (isinstance(a, (int, float)) and a > 1e-4) else a for a in amp
                )
            elif isinstance(amp, (int, float)) and amp > 1e-4:
                amp = (amp * CONSTANTS["ang_amu2SI"], amp * CONSTANTS["ang_amu2SI"])
            saved[key] = {"tensor": entry.zfs_tensor.matrix, "unit": entry.zfs_tensor.unit, "pert": amp}
        elif isinstance(entry, dict):
            saved[key] = entry
        else:
            raise SchemaError(f"second_order[{key}]: unsupported entry type {type(entry).__name__}")
    return saved


def _decode_order_dict(value: Any, data: Any) -> dict:
    """Parse "i_j" string keys back into integer tuples."""
    if isinstance(value, np.ndarray):
        value = value[()]
    if not isinstance(value, dict):
        raise SchemaError(f"expected dict of perturbation entries, got {type(value).__name__}")
    out = {}
    for k, v in value.items():
        parsed = tuple(int(x) for x in k.split("_")) if isinstance(k, str) and "_" in k else k
        out[parsed] = v
    return out


# Factor 3/2 scaling applied to defect_band_approx (ZFS_occup) tensors: the
# defect-band approximation overestimates the ZFS splitting. The parser applies
# this factor at intake and stores spin_correction_applied=True; legacy npz
# files (unscaled, flag=False) are scaled by RawZFSData.load once on load.
SPIN_CORRECTION_SCALE = 1.5


def apply_spin_correction(tensor: ZFSTensor) -> ZFSTensor:
    """A defect-band-approx ZFS tensor scaled by :data:`SPIN_CORRECTION_SCALE` (3/2)."""
    return ZFSTensor(matrix=tensor.matrix * SPIN_CORRECTION_SCALE, unit=tensor.unit)


def _scale_correction_fields(fields: dict) -> dict:
    """Returns a copy of RawZFSData constructor kwargs with approx tensors scaled by 3/2.

    Shared by RawZFSData.load (per-file, before merging) and _apply_spin_correction,
    so both paths scale identically. NOTE: PerturbationEntry objects inside the
    kwargs are mutated in place — pass copies if the caller must keep originals.
    """
    out = dict(fields)
    gs = out.get("ground_state_zfs")
    if gs is not None:
        out["ground_state_zfs"] = apply_spin_correction(gs)
    for name in ("first_order", "second_order"):
        entries = dict(out.get(name) or {})
        for key, entry in entries.items():
            if isinstance(entry, PerturbationEntry):
                entry.zfs_tensor = apply_spin_correction(entry.zfs_tensor)
            elif isinstance(entry, dict) and entry.get("tensor") is not None:
                entry["tensor"] = np.asarray(entry["tensor"], dtype=float) * SPIN_CORRECTION_SCALE
        out[name] = entries
    return out


def normalize_calc_method(m: Optional[str]) -> Optional[str]:
    """Normalizes calc_method aliases ('approx' -> 'defect_band_approx', etc.)."""
    if not m or m in ("None", ""):
        return None
    if m in ("all", "all_bands"):
        return "all_bands"
    if m in ("approx", "defect_band_approx"):
        return "defect_band_approx"
    return str(m)


def _merge_vasp_settings(a: Optional[dict], b: Optional[dict]) -> Optional[dict]:
    """Settings of combined data: the trees of both, one ground state, all problems."""
    if not a or not b:
        return a or b
    return {
        "ground_state": a.get("ground_state") or b.get("ground_state"),
        "trees": {**a.get("trees", {}), **b.get("trees", {})},
        "problems": list(a.get("problems", [])) + [p for p in b.get("problems", []) if p not in a.get("problems", [])],
    }


@dataclass
class RawZFSData:
    """Container for raw unperturbed and perturbed ZFS simulation data."""

    defect: str
    cell_size: int
    pert_scale: float
    calc_method: Optional[str] = None
    spin_correction_applied: bool = False
    # False for runs displaced by Q * e instead of Q * e / sqrt(m) (before mass
    # weighting was added); the couplings then rescale q, see PhononSpectrum.get_phonon_pert.
    mass_weighted: bool = True
    # One of DISPLACEMENTS; None for data from before the key, read through mass_weighted.
    displacement: Optional[str] = None
    order: Optional[int] = None
    ground_state_zfs: Optional[ZFSTensor] = None
    eigen_rotation: Optional[np.ndarray] = None
    first_order: dict[int, Union[PerturbationEntry, dict[str, Any]]] = field(default_factory=dict)
    second_order: dict[tuple[int, int], Union[PerturbationEntry, dict[str, Any]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # VASP settings of the ground state and of each perturbation tree, recorded by
    # ``spinlabs package`` (spinlabs.vasp.settings): {"ground_state", "trees", "problems"}.
    vasp_settings: Optional[dict[str, Any]] = None

    @property
    def displacement_convention(self) -> str:
        """How the runs were displaced (one of DISPLACEMENTS); legacy data falls back on mass_weighted."""
        return displacement_convention(self.displacement, self.mass_weighted)

    @property
    def has_second_order(self) -> bool:
        """Returns True if second-order perturbation data is present and non-empty."""
        return len(self.second_order) > 0

    @property
    def has_first_order(self) -> bool:
        """Returns True if first-order perturbation data is present and non-empty."""
        return len(self.first_order) > 0

    @property
    def combined_order(self) -> int:
        """Effective order: 2 if second-order data is present, else order or 1."""
        return 2 if self.has_second_order else (self.order or 1)

    @property
    def order_suffix(self) -> str:
        """Filename suffix showing which orders are present: '1d', '2d' or '12d' (both)."""
        if self.has_first_order and self.has_second_order:
            return "12d"
        return f"{self.combined_order}d"

    def combine(self, other: RawZFSData) -> RawZFSData:
        """
        Combines two RawZFSData containers (e.g. 1D and 2D datasets).

        Verifies that metadata (defect, cell_size, pert_scale, calc_method) matches.
        Raises ValueError if there is any metadata mismatch.
        """
        if not isinstance(other, RawZFSData):
            raise TypeError(f"Cannot combine RawZFSData with object of type {type(other)}")

        # 1. Verify defect
        d1 = self.defect if self.defect not in (None, "None", "") else None
        d2 = other.defect if other.defect not in (None, "None", "") else None
        if d1 is not None and d2 is not None and d1 != d2:
            raise ValueError(f"Cannot combine RawZFSData: defect mismatch ('{d1}' != '{d2}').")

        # 2. Verify cell_size
        c1 = self.cell_size if self.cell_size not in (None, 0) else None
        c2 = other.cell_size if other.cell_size not in (None, 0) else None
        if c1 is not None and c2 is not None and c1 != c2:
            raise ValueError(f"Cannot combine RawZFSData: cell_size mismatch ({c1} != {c2}).")

        # 3. Verify pert_scale
        p1 = self.pert_scale if self.pert_scale not in (None, 0.0) else None
        p2 = other.pert_scale if other.pert_scale not in (None, 0.0) else None
        if p1 is not None and p2 is not None and not np.isclose(p1, p2, rtol=1e-3, atol=1e-5):
            raise ValueError(f"Cannot combine RawZFSData: pert_scale mismatch ({p1} != {p2}).")

        # 4. Verify calc_method
        m1 = normalize_calc_method(self.calc_method)
        m2 = normalize_calc_method(other.calc_method)
        if m1 is not None and m2 is not None and m1 != m2:
            raise ValueError(f"Cannot combine RawZFSData: calc_method mismatch ('{m1}' != '{m2}').")

        # 5. Verify common metadata dictionary keys
        if self.metadata and other.metadata:
            for k in set(self.metadata.keys()) & set(other.metadata.keys()):
                if k in ("sim_path", "source_file", "source_files", "order", "calc_method"):
                    continue
                v1, v2 = self.metadata[k], other.metadata[k]
                if v1 is not None and v2 is not None and v1 != v2:
                    raise ValueError(f"Cannot combine RawZFSData: metadata mismatch for '{k}' ({v1} != {v2}).")

        # 6. Verify ground state ZFS if both have it
        gs = self.ground_state_zfs
        if gs is None and other.ground_state_zfs is not None:
            gs = other.ground_state_zfs
        elif gs is not None and other.ground_state_zfs is not None:
            gs_self_mhz = gs.to_unit("MHz")
            gs_other_mhz = other.ground_state_zfs.to_unit("MHz")
            if not np.allclose(gs_self_mhz.matrix, gs_other_mhz.matrix, rtol=1e-2, atol=1e-1):
                raise ValueError("Cannot combine RawZFSData: ground_state_zfs tensor mismatch.")

        # 7. Merge perturbation entries
        merged_first = dict(self.first_order)
        merged_first.update(other.first_order)

        merged_second = dict(self.second_order)
        merged_second.update(other.second_order)

        eff_order = 2 if len(merged_second) > 0 else (self.order or other.order or 1)
        merged_defect = d1 or d2 or "defect"
        merged_cell = c1 or c2 or 0
        merged_scale = p1 or p2 or 0.0
        merged_method = m1 or m2 or self.calc_method or other.calc_method
        merged_rotation = self.eigen_rotation if self.eigen_rotation is not None else other.eigen_rotation
        # Never re-scale a merged dataset: if either part already carries the
        # 3/2 correction, mark the combination as corrected (see _apply_spin_correction).
        merged_correction = self.spin_correction_applied or other.spin_correction_applied
        if self.spin_correction_applied != other.spin_correction_applied:
            raise ValueError(
                "Cannot combine datasets with inconsistent spin-correction flags: "
                "one part is corrected (x3/2) and the other is not. Scaling the "
                "merged tensors would silently mix corrected and uncorrected values. "
                "Correct the raw dataset first (RawZFSData.load with apply_corrections=True)."
            )

        if self.displacement_convention != other.displacement_convention:
            raise ValueError(
                f"Cannot combine {self.displacement_convention} and {other.displacement_convention} perturbation "
                "datasets: their perturbation amplitudes differ per mode."
            )

        merged_metadata = dict(self.metadata)
        for k, v in other.metadata.items():
            if k == "sim_path":
                p_existing = merged_metadata.get("sim_path")
                if p_existing:
                    if isinstance(p_existing, list):
                        merged_metadata["sim_path"] = p_existing + [v]
                    else:
                        merged_metadata["sim_path"] = [p_existing, v]
                else:
                    merged_metadata["sim_path"] = v
            elif k not in merged_metadata or merged_metadata[k] is None:
                merged_metadata[k] = v

        return RawZFSData(
            defect=merged_defect,
            cell_size=merged_cell,
            pert_scale=merged_scale,
            calc_method=merged_method,
            spin_correction_applied=merged_correction,
            mass_weighted=self.mass_weighted,
            displacement=self.displacement,
            order=eff_order,
            ground_state_zfs=gs,
            eigen_rotation=merged_rotation,
            first_order=merged_first,
            second_order=merged_second,
            metadata=merged_metadata,
            vasp_settings=_merge_vasp_settings(self.vasp_settings, other.vasp_settings),
        )

    def __add__(self, other: RawZFSData) -> RawZFSData:
        return self.combine(other)

    def to_unit(self, target_unit: str) -> RawZFSData:
        """Converts all tensors in the container to the specified unit."""
        new_gs = self.ground_state_zfs.to_unit(target_unit) if self.ground_state_zfs else None

        new_first = {}
        for k, v in self.first_order.items():
            if isinstance(v, PerturbationEntry):
                new_first[k] = PerturbationEntry(
                    order=v.order,
                    mode_indices=v.mode_indices,
                    amplitude=v.amplitude,
                    zfs_tensor=v.zfs_tensor.to_unit(target_unit),
                    energy=v.energy,
                )
            elif isinstance(v, dict) and "tensor" in v:
                curr_unit = v.get("unit", "J")
                new_v = dict(v)
                new_v["tensor"] = convert_energy(v["tensor"], curr_unit, target_unit)
                new_v["unit"] = target_unit
                new_first[k] = new_v
            else:
                new_first[k] = v

        new_second = {}
        for k, v in self.second_order.items():
            if isinstance(v, PerturbationEntry):
                new_second[k] = PerturbationEntry(
                    order=v.order,
                    mode_indices=v.mode_indices,
                    amplitude=v.amplitude,
                    zfs_tensor=v.zfs_tensor.to_unit(target_unit),
                    energy=v.energy,
                )
            elif isinstance(v, dict) and "tensor" in v:
                curr_unit = v.get("unit", "J")
                new_v = dict(v)
                new_v["tensor"] = convert_energy(v["tensor"], curr_unit, target_unit)
                new_v["unit"] = target_unit
                new_second[k] = new_v
            else:
                new_second[k] = v

        return RawZFSData(
            defect=self.defect,
            cell_size=self.cell_size,
            pert_scale=self.pert_scale,
            calc_method=self.calc_method,
            spin_correction_applied=self.spin_correction_applied,
            mass_weighted=self.mass_weighted,
            displacement=self.displacement,
            order=self.order,
            ground_state_zfs=new_gs,
            eigen_rotation=self.eigen_rotation.copy() if self.eigen_rotation is not None else None,
            first_order=new_first,
            second_order=new_second,
            metadata=dict(self.metadata),
            vasp_settings=self.vasp_settings,
        )

    def enrich_with_spectrum(self, spectrum: PhononSpectrum) -> None:
        """
        Enriches first_order and second_order entries with mode-dependent SI displacements,
        C3v symmetries, and IPRs from the given phonon spectrum.
        """
        pert_SI = (self.pert_scale or 1.0) * CONSTANTS["ang_amu2SI"]
        phonon_pert = spectrum.get_phonon_pert(pert_SI, displacement=self.displacement_convention)
        disps = phonon_pert.get("disp", [])
        syms = phonon_pert.get("sym", [])
        iprs = phonon_pert.get("ipr", [])

        for idx, entry in list(self.first_order.items()):
            disp = disps[idx] if idx < len(disps) else None
            sym = syms[idx] if idx < len(syms) else None
            ipr = iprs[idx] if idx < len(iprs) else None
            if isinstance(entry, PerturbationEntry):
                if disp is not None:
                    entry.amplitude = disp
            elif isinstance(entry, dict):
                if disp is not None:
                    entry["pert"] = disp
                if sym is not None:
                    entry["symmetry"] = sym
                if ipr is not None:
                    entry["ipr"] = ipr

        for (i, j), entry in list(self.second_order.items()):
            disp_i = disps[i] if i < len(disps) else None
            disp_j = disps[j] if j < len(disps) else None
            sym_i = syms[i] if i < len(syms) else None
            sym_j = syms[j] if j < len(syms) else None
            ipr_i = iprs[i] if i < len(iprs) else None
            ipr_j = iprs[j] if j < len(iprs) else None
            pair_disp = (disp_i, disp_j) if (disp_i is not None and disp_j is not None) else None
            if isinstance(entry, PerturbationEntry):
                if pair_disp is not None:
                    entry.amplitude = pair_disp
            elif isinstance(entry, dict):
                if pair_disp is not None:
                    entry["pert"] = pair_disp
                if sym_i is not None and sym_j is not None:
                    entry["symmetry"] = (sym_i, sym_j)
                if ipr_i is not None and ipr_j is not None:
                    entry["ipr"] = (ipr_i, ipr_j)

    def _default_name(self):
        return f"{self.defect}_{self.cell_size}_raw_zfs_data_{self.calc_method}_{self.order_suffix}.npz"

    SCHEMA = [
        FieldSpec("defect", "str"),
        FieldSpec("cell_size", "int"),
        FieldSpec("pert_scale", "float"),
        FieldSpec("calc_method", "str", optional=True, decode=_decode_calc_method),
        FieldSpec("spin_correction_applied", "bool", optional=True),
        FieldSpec("mass_weighted", "bool", optional=True),
        FieldSpec("displacement", "str", optional=True),
        FieldSpec("order", "int", optional=True),
        FieldSpec("ground_state_zfs", "zfstensor", optional=True, legacy_keys=("zfs_relaxed",)),
        FieldSpec("eigen_rotation", "array", optional=True),
        FieldSpec(
            "first_order",
            "dict",
            optional=True,
            legacy_keys=("zfs_tensors",),
            encode=_encode_first_order,
            decode=_decode_order_dict,
        ),
        FieldSpec(
            "second_order",
            "dict",
            optional=True,
            legacy_keys=("zfs_tensors_2d",),
            encode=_encode_second_order,
            decode=_decode_order_dict,
        ),
        FieldSpec(
            "vasp_settings",
            "str",
            optional=True,
            encode=lambda v: json.dumps(v, sort_keys=True),
            decode=lambda v, _: json.loads(str(v)),
        ),
    ]

    def save(
        self,
        out_path: Optional[Union[str, Path]] = None,
        spectrum: Optional[PhononSpectrum] = None,
    ) -> str:
        """Saves RawZFSData to a .npz file with latest naming conventions and explicit unit metadata."""
        if spectrum is not None:
            self.enrich_with_spectrum(spectrum)

        # Effective order derived from which perturbation dicts are present
        eff_order = self.order
        if self.has_first_order and self.has_second_order:
            eff_order = 12  # combined 1d+2d data: unified order marker everywhere
        elif eff_order is None or eff_order == 1:
            if self.second_order:
                eff_order = 2
            elif self.first_order:
                eff_order = 1

        return save_npz(
            self, self._default_name() if out_path is None else out_path, self.SCHEMA, overrides={"order": eff_order}
        )

    @classmethod
    def load(
        cls,
        in_path: Union[str, Path, Sequence[Union[str, Path]]],
        apply_corrections: bool = True,
    ) -> RawZFSData:
        """
        Loads RawZFSData from one or more .npz files.
        Maintains backward compatibility with legacy keys (zfs_tensors, zfs_tensors_2d, zfs_relaxed).
        If apply_corrections is True (default), defect_band_approx tensors are scaled by 3/2
        exactly once on load; files already carrying the correction are skipped.
        """
        if isinstance(in_path, (list, tuple)):
            paths = [Path(p) for p in in_path]
        else:
            paths = [Path(in_path)]

        raw_data = [np.load(str(p), allow_pickle=True) for p in paths]
        warn_unknown_keys(raw_data[0].files, cls.SCHEMA)

        # Scale EACH file before merging, based on that file's own method and
        # flag. Scaling the merged object once would mis-handle mixed inputs:
        # a corrected file could be scaled again, an uncorrected one missed.
        def _per_file(npz: np.lib.npyio.NpzFile, strict: bool) -> tuple[dict, bool]:
            fields = _fields_from_npz(npz, cls, cls.SCHEMA, strict=strict)
            if fields.get("spin_correction_applied"):
                fields["spin_correction_applied"] = True
                return fields, False
            fields["spin_correction_applied"] = False
            if not apply_corrections or normalize_calc_method(fields.get("calc_method")) != "defect_band_approx":
                return fields, False
            scaled = _scale_correction_fields(
                {
                    "ground_state_zfs": fields.get("ground_state_zfs"),
                    "first_order": fields.get("first_order"),
                    "second_order": fields.get("second_order"),
                }
            )
            fields.update(scaled)
            fields["spin_correction_applied"] = True
            return fields, True

        scaled_pairs = [_per_file(d, i == 0) for i, d in enumerate(raw_data)]
        scaled_fields = [f for f, _ in scaled_pairs]

        base = scaled_fields[0]
        first_order = dict(base.get("first_order") or {})
        second_order = dict(base.get("second_order") or {})

        for fields in scaled_fields[1:]:
            for k, v in (fields.get("first_order") or {}).items():
                first_order.setdefault(k, v)
            for k, v in (fields.get("second_order") or {}).items():
                second_order.setdefault(k, v)

        # Files without the keys predate them and were mass-weighted.
        conventions = {
            displacement_convention(f.get("displacement"), f.get("mass_weighted") is not False) for f in scaled_fields
        }
        if len(conventions) > 1:
            raise ValueError(f"Cannot load perturbation files with different displacements together: {conventions}")
        convention = conventions.pop()

        order = base.get("order")
        if second_order:
            order = 2
        elif order is None:
            order = 1 if first_order else None

        obj = cls(
            defect=base["defect"],
            cell_size=base["cell_size"],
            pert_scale=base["pert_scale"],
            calc_method=base.get("calc_method"),
            # all(): if some files are uncorrected (only possible with
            # apply_corrections=False), never report the merge as corrected.
            spin_correction_applied=all(f.get("spin_correction_applied") for f in scaled_fields),
            mass_weighted=convention != "unweighted",
            displacement=convention,
            order=order,
            ground_state_zfs=base.get("ground_state_zfs"),
            eigen_rotation=base.get("eigen_rotation"),
            first_order=first_order,
            second_order=second_order,
        )
        for fields in scaled_fields:
            obj.vasp_settings = _merge_vasp_settings(obj.vasp_settings, fields.get("vasp_settings"))
        for p, (_, scaled_now) in zip(paths, scaled_pairs):
            if apply_corrections and scaled_now:
                LOGGER.info(f"Applied spin correction scaling: {SPIN_CORRECTION_SCALE:g} (x3/2) [{p.name}]")
        return obj.to_unit("J")

    def _apply_spin_correction(self) -> None:
        """Scales defect_band_approx tensors by 3/2 exactly once.

        npz files written before the correction moved into the loader/parser
        store unscaled tensors and carry spin_correction_applied=False; they
        are scaled here. Files written after carry the flag=True and are
        skipped (the parser applies the correction at intake). all_bands
        data is never scaled.
        """
        if self.spin_correction_applied:
            return
        if normalize_calc_method(self.calc_method) != "defect_band_approx":
            return
        scaled = _scale_correction_fields(
            {
                "ground_state_zfs": self.ground_state_zfs,
                "first_order": self.first_order,
                "second_order": self.second_order,
            }
        )
        self.ground_state_zfs = scaled["ground_state_zfs"]
        self.first_order = scaled["first_order"]
        self.second_order = scaled["second_order"]
        self.spin_correction_applied = True
        LOGGER.info(f"Applied spin correction scaling: {SPIN_CORRECTION_SCALE:g} (x3/2)")


@dataclass
class SpinPhononCouplingData:
    """
    Container for calculated spin-phonon coupling coefficients (V-tensors),
    finite-difference derivatives, and frequencies with explicit unit tracking.
    """

    order: int
    defect: str
    cell_size: int
    pert_scale: float
    calc_method: Optional[str] = None
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    frequency_unit: str = "J"
    V_0_0: np.ndarray = field(default_factory=lambda: np.array([]))
    V_p_m: np.ndarray = field(default_factory=lambda: np.array([]))
    V_0_pm: np.ndarray = field(default_factory=lambda: np.array([]))
    coupling_unit: str = "J"
    ground_state_zfs: Optional[ZFSTensor] = None
    zfs_derivs: Optional[np.ndarray] = None
    derivs_unit: str = "J"
    symmetries: Optional[list[str]] = None
    iprs: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Second-order (2D) coupling coefficients, optional — used when the run
    # combines first- and second-order data (1d + 2d perturbation sets).
    V2_0_0: Optional[np.ndarray] = None
    V2_p_m: Optional[np.ndarray] = None
    V2_0_pm: Optional[np.ndarray] = None
    zfs_2nd_derivs: Optional[np.ndarray] = None

    @property
    def has_second_order(self) -> bool:
        """Returns True if valid 2nd-order coupling data is present (non-None and non-empty)."""
        if self.V2_0_0 is None:
            return False
        arr = np.asarray(self.V2_0_0)
        return arr.ndim >= 2 and arr.size > 0

    @property
    def combined_order(self) -> int:
        """Effective order for combined runs: 2 when 2d data present, else order."""
        return 2 if self.has_second_order else self.order

    def to_unit(self, target_unit: str) -> SpinPhononCouplingData:
        """
        Returns a new instance with coupling coefficients (V_0_0, V_p_m, V_0_pm, zfs_derivs)
        and ground_state_zfs converted to target_unit.
        """
        if self.coupling_unit == target_unit:
            new_v00 = self.V_0_0.copy()
            new_vpm = self.V_p_m.copy()
            new_v0pm = self.V_0_pm.copy()
        else:
            new_v00 = convert_energy(self.V_0_0, self.coupling_unit, target_unit)
            new_vpm = convert_energy(self.V_p_m, self.coupling_unit, target_unit)
            new_v0pm = convert_energy(self.V_0_pm, self.coupling_unit, target_unit)

        new_derivs = None
        if self.zfs_derivs is not None:
            # Check for numpy 0-d object array holding None
            if not (getattr(self.zfs_derivs, "shape", None) == () and self.zfs_derivs.item() is None):
                new_derivs = convert_energy(self.zfs_derivs, self.derivs_unit, target_unit)

        new_gs = self.ground_state_zfs.to_unit(target_unit) if self.ground_state_zfs else None

        return SpinPhononCouplingData(
            order=self.order,
            defect=self.defect,
            cell_size=self.cell_size,
            pert_scale=self.pert_scale,
            calc_method=self.calc_method,
            frequencies=self.frequencies.copy(),
            frequency_unit=self.frequency_unit,
            V_0_0=new_v00,
            V_p_m=new_vpm,
            V_0_pm=new_v0pm,
            coupling_unit=target_unit,
            ground_state_zfs=new_gs,
            zfs_derivs=new_derivs,
            derivs_unit=target_unit,
            symmetries=list(self.symmetries) if self.symmetries is not None else None,
            iprs=self.iprs.copy() if self.iprs is not None else None,
            metadata=dict(self.metadata),
            V2_0_0=convert_energy(self.V2_0_0, self.coupling_unit, target_unit) if self.V2_0_0 is not None else None,
            V2_p_m=convert_energy(self.V2_p_m, self.coupling_unit, target_unit) if self.V2_p_m is not None else None,
            V2_0_pm=convert_energy(self.V2_0_pm, self.coupling_unit, target_unit) if self.V2_0_pm is not None else None,
            zfs_2nd_derivs=convert_energy(self.zfs_2nd_derivs, self.derivs_unit, target_unit)
            if self.zfs_2nd_derivs is not None
            else None,
        )

    def frequencies_to_unit(self, target_unit: str) -> SpinPhononCouplingData:
        """Returns a new instance with mode frequencies converted to target_unit."""
        if self.frequency_unit == target_unit:
            new_freqs = self.frequencies.copy()
        else:
            new_freqs = convert_energy(self.frequencies, self.frequency_unit, target_unit)

        return SpinPhononCouplingData(
            order=self.order,
            defect=self.defect,
            cell_size=self.cell_size,
            pert_scale=self.pert_scale,
            calc_method=self.calc_method,
            frequencies=new_freqs,
            frequency_unit=target_unit,
            V_0_0=self.V_0_0.copy(),
            V_p_m=self.V_p_m.copy(),
            V_0_pm=self.V_0_pm.copy(),
            coupling_unit=self.coupling_unit,
            ground_state_zfs=self.ground_state_zfs,
            zfs_derivs=self.zfs_derivs.copy() if self.zfs_derivs is not None else None,
            derivs_unit=self.derivs_unit,
            symmetries=list(self.symmetries) if self.symmetries is not None else None,
            iprs=self.iprs.copy() if self.iprs is not None else None,
            metadata=dict(self.metadata),
            V2_0_0=self.V2_0_0.copy() if self.V2_0_0 is not None else None,
            V2_p_m=self.V2_p_m.copy() if self.V2_p_m is not None else None,
            V2_0_pm=self.V2_0_pm.copy() if self.V2_0_pm is not None else None,
            zfs_2nd_derivs=self.zfs_2nd_derivs.copy() if self.zfs_2nd_derivs is not None else None,
        )

    SCHEMA = [
        FieldSpec("order", "int", optional=True),
        FieldSpec("defect", "str", optional=True),
        FieldSpec("cell_size", "int", optional=True),
        FieldSpec("pert_scale", "float", optional=True),
        FieldSpec("calc_method", "str", optional=True, decode=_decode_calc_method),
        FieldSpec("frequencies", "array", legacy_keys=("freqs",)),
        FieldSpec("frequency_unit", "str", optional=True),
        FieldSpec("V_0_0", "array"),
        FieldSpec("V_p_m", "array"),
        FieldSpec("V_0_pm", "array"),
        FieldSpec("coupling_unit", "str", optional=True),
        FieldSpec("ground_state_zfs", "zfstensor", optional=True),
        FieldSpec("zfs_derivs", "array", optional=True),
        FieldSpec("derivs_unit", "str", optional=True),
        FieldSpec("symmetries", "list_str", optional=True, legacy_keys=("sym",), save_aliases=("sym",)),
        FieldSpec("iprs", "array", optional=True, legacy_keys=("ipr",), save_aliases=("ipr",)),
        FieldSpec("V2_0_0", "array", optional=True, decode=_decode_2d_array),
        FieldSpec("V2_p_m", "array", optional=True, decode=_decode_2d_array),
        FieldSpec("V2_0_pm", "array", optional=True, decode=_decode_2d_array),
        FieldSpec("zfs_2nd_derivs", "array", optional=True, decode=_decode_2d_array),
    ]

    def save(self, out_path: Union[str, Path]) -> str:
        """Saves coupling data to .npz file with explicit unit metadata and legacy keys."""
        gs_d_joule = self.ground_state_zfs.to_unit("J").D if self.ground_state_zfs else 0.0
        return save_npz(
            self,
            out_path,
            self.SCHEMA,
            extras={"zfs": gs_d_joule},  # legacy scalar D in Joules
        )

    @classmethod
    def load(cls, in_path: Union[str, Path]) -> SpinPhononCouplingData:
        """Loads SpinPhononCouplingData from a .npz file, parsing explicit units."""
        data = np.load(str(in_path), allow_pickle=True)
        warn_unknown_keys(data.files, cls.SCHEMA, extra_ignore=("zfs",))

        def _pre(d, kwargs):
            # Legacy frequency unit inference when the tag is absent
            if kwargs.get("frequency_unit") is None:
                freqs = kwargs.get("frequencies")
                freqs_pos = freqs[freqs > 0] if freqs is not None and getattr(freqs, "ndim", 0) else []
                kwargs["frequency_unit"] = "J" if len(freqs_pos) and float(np.mean(freqs_pos)) < 1e-15 else "meV"
            kwargs["coupling_unit"] = kwargs.get("coupling_unit") or "J"
            kwargs["derivs_unit"] = kwargs.get("derivs_unit") or kwargs["coupling_unit"]

            # Legacy 2d files stored their coefficients under the first-order
            # keys with order=2 (no V2_* present). Detect and remap.
            if kwargs.get("V2_0_0") is None and "V2_0_0" not in d and int(d.get("order", 1)) == 2:
                kwargs["V2_0_0"] = kwargs.get("V_0_0")
                kwargs["V2_p_m"] = kwargs.get("V_p_m")
                kwargs["V2_0_pm"] = kwargs.get("V_0_pm")
                kwargs["zfs_2nd_derivs"] = kwargs.get("zfs_2nd_derivs") or kwargs.get("zfs_derivs")

            # Legacy scalar D in Joules -> diagonal ZFSTensor
            if kwargs.get("ground_state_zfs") is None and "zfs" in d:
                d_val_j = float(d["zfs"])
                kwargs["ground_state_zfs"] = ZFSTensor(
                    matrix=np.diag([-d_val_j / 3.0, -d_val_j / 3.0, 2.0 * d_val_j / 3.0]), unit="J"
                )

            # Legacy defaults for base fields
            if kwargs.get("defect") is None:
                kwargs["defect"] = "unknown"
            if kwargs.get("cell_size") is None:
                kwargs["cell_size"] = 0
            if kwargs.get("pert_scale") is None:
                kwargs["pert_scale"] = 0.0
            if kwargs.get("order") is None:
                kwargs["order"] = 1
            return kwargs

        return load_npz(cls, in_path, cls.SCHEMA, pre_decode=_pre)
