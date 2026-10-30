from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional
import numpy as np

from spinlabs.constants import CONSTANTS
from spinlabs.symmetry import twin_of, _build_groups_from_labels
from spinlabs.models import ZFSTensor, PhononSpectrum, PerturbationEntry, RawZFSData, SpinPhononCouplingData, normalize_calc_method
from spinlabs.utils import MathUtils
from spinlabs.zfs_response import (
    SHAPE_FLOOR_MHZ,
    SYMMETRY_TOL,
    ZFSResponseCheck,
    response_point_group,
    verify_zfs_response,
)

from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)

# Healthy second-order entries have alpha ~ 0 (|alpha| < 0.05 for NV_64): the
# curvature is far below the linear response at the perturbation amplitude.
# Over-displaced runs give alpha ~ 1 or more; runs whose change cancels the
# linear response (wrong state or geometry) give alpha ~ -1. Both are excluded.
INCONSISTENT_ALPHA = 0.5
# alpha is the remainder relative to the expected change; below this size (Frobenius
# norm, MHz) the expected change is too small for that ratio to mean anything, so
# the entry is not judged.
MIN_EXPECTED_MHZ = 0.05


class ZFSManager:
    """
    Manages Zero-Field Splitting (ZFS) tensor calculations, coordinate transformations,
    finite-difference derivatives (1st and 2nd order), and spin-phonon coupling coefficients (V-tensors).
    Operates directly on the PhononSpectrum dataclass.
    """

    def __init__(
        self,
        spectrum: Optional[PhononSpectrum] = None,
        raw_data: Optional[RawZFSData] = None,
        exclude_inconsistent_pairs: bool = False,
        degenerate_pairs_from_symmetry: bool = False,
        offdiag_from_diagonal: bool = False,
    ):
        # spectrum is the primary dataclass
        if spectrum is not None:
            # Always reclassify fresh via the general projection method and
            # cross-check against labels stored in the phonon npz (they are
            # legacy/baked-in and may predate the current classification).
            # When the structure cannot be symmetry-detected (e.g. synthetic
            # test fixtures with all atoms at the origin), fall back to the
            # stored labels rather than failing the whole pipeline.
            from .symmetry import SymmetryDetectionError, classify_and_pair

            stored_syms = list(spectrum.symmetries) if spectrum.symmetries is not None else None
            try:
                fresh_labels, deg_groups = classify_and_pair(spectrum, force=True)
            except (SymmetryDetectionError, NotImplementedError) as exc:
                # NotImplementedError: point group detected but no character
                # table; SymmetryDetectionError: no symmetry found at all.
                # Both mean "cannot classify this structure" — a genuine bug
                # in the classifier must NOT be caught here.
                if stored_syms is None:
                    raise
                LOGGER.warning(f"symmetry classification of the phonon modes failed ({exc}); using the stored labels")
                fresh_labels = stored_syms
                # Stored labels are trusted as-is here, so sym_check cannot be
                # verified (no detectable point group to build characters
                # from) and stays None for this spectrum.
                deg_groups = _build_groups_from_labels(spectrum, stored_syms, 0.01)
            if stored_syms is not None and fresh_labels is not stored_syms:
                mismatched = [i for i, (a, b) in enumerate(zip(stored_syms, fresh_labels)) if a != b]
                if mismatched:
                    LOGGER.warning(
                        f"{len(mismatched)} stored symmetry label(s) differ from a fresh classification "
                        f"(modes {', '.join(str(i + 1) for i in mismatched[:10])}"
                        f"{', ...' if len(mismatched) > 10 else ''}); the fresh labels are used"
                    )
            spectrum.symmetries = fresh_labels
            spectrum.deg_groups = deg_groups
            # An incomplete spectrum (missing degenerate E twins) must be rejected:
            # the ZFS pipeline needs all 3*N_atoms modes. Regenerate the phonon npz
            # from the full calculation instead.
            if spectrum.e_pair_complete is None:
                spectrum.check_e_pair_completeness()
            incomplete = [
                i for i, ok in enumerate(spectrum.e_pair_complete or []) if not ok and fresh_labels[i] in ("Ex", "Ey")
            ]
            if incomplete:
                raise ValueError(
                    f"Phonon spectrum is missing the degenerate partners of "
                    f"{len(incomplete)} E modes (positions {incomplete[:10]}"
                    f"{'...' if len(incomplete) > 10 else ''}). The ZFS pipeline "
                    f"requires a complete phonon file with all 3*N_atoms modes; "
                    f"regenerate it from the full calculation (e.g. phonon_data.npz) "
                    f"rather than the symmetry-adapted subset."
                )
            self.spectrum = spectrum
        else:
            self.spectrum = None
        # Point group for the ZFS symmetry checks (None without a spectrum: not checked).
        self.point_group = response_point_group(spectrum) if spectrum is not None else None

        self.raw_data = raw_data

        # Defect metadata
        self.defect: Optional[str] = raw_data.defect if raw_data else None
        self.cell_size: Optional[int] = raw_data.cell_size if raw_data else None
        self.pert_scale: Optional[float] = raw_data.pert_scale if raw_data else None
        self.calc_method: Optional[str] = (
            raw_data.calc_method or raw_data.metadata.get("calc_method") if raw_data else None
        )
        # How the runs were displaced (spinlabs.models.DISPLACEMENTS); sets each mode's amplitude q.
        self.displacement: str = raw_data.displacement_convention if raw_data else "mass_weighted"
        self.mass_weighted: bool = self.displacement != "unweighted"
        self.exclude_inconsistent_pairs = exclude_inconsistent_pairs
        # Second-order entries failing the linear-response check, as
        # (i, j, alpha, |remainder| in MHz) with 0-based modes; see calculate_second_order_derivatives.
        self.inconsistent_pairs: list[tuple[int, int, float, float]] = []
        # Replace each degenerate Ex-Ey second-order coupling by its symmetry value
        # from the Ex-Ex curvature; see _degenerate_pairs_from_symmetry.
        self.degenerate_pairs_from_symmetry = degenerate_pairs_from_symmetry
        # (Ex, Ey) pairs, 0-based, whose couplings were replaced that way.
        self.symmetry_derived_pairs: list[tuple[int, int]] = []
        # Fill every second-order pair (i, j), i != j, that no run or symmetry rule gave with the
        # geometric mean of the two diagonal couplings; see _offdiag_from_diagonal.
        self.offdiag_from_diagonal = offdiag_from_diagonal
        self.n_offdiag_filled = 0  # pairs (i < j) filled that way
        self.n_offdiag_filled_a_x_e = 0  # of which A x E pairs, filled by the mixed-type root
        if self.displacement != "mass_weighted":
            if spectrum is None:
                raise ValueError(
                    f"{self.displacement} perturbation data needs the phonon spectrum (eigenvectors and masses) "
                    "to get each mode's perturbation amplitude"
                )
            if self.displacement == "unweighted":
                LOGGER.info("displacements: not mass-weighted; q scaled by each mode's effective sqrt(mass)")
            else:
                LOGGER.info("displacements: normalized (same Cartesian step per mode); q scaled by each mode's n_k")

        # Processed ZFS data in defect principal frame
        self.zfs_relaxed: Optional[np.ndarray] = None  # Shape (3, 3) in J
        self.eigen_rotation: Optional[np.ndarray] = None  # Shape (3, 3)
        self.first_order: dict[int, dict[str, Any]] = {}  # 1D perturbations
        # 1D perturbations at -Q: with them first order and the diagonal second order are central differences
        self.first_order_minus: dict[int, dict[str, Any]] = {}
        self.second_order: dict[tuple[int, int], dict[str, Any]] = {}  # 2D perturbations
        self.treated_modes: set[int] = set()

        if raw_data is not None:
            self._ingest_raw_data(raw_data)

    @property
    def zfs_tensors(self) -> dict[int, dict[str, Any]]:
        """Backward-compatible alias for first_order perturbations."""
        return self.first_order

    @zfs_tensors.setter
    def zfs_tensors(self, val: dict[int, dict[str, Any]]):
        self.first_order = val

    @property
    def zfs_tensors_2d(self) -> dict[tuple[int, int], dict[str, Any]]:
        """Backward-compatible alias for second_order perturbations."""
        return self.second_order

    @zfs_tensors_2d.setter
    def zfs_tensors_2d(self, val: dict[tuple[int, int], dict[str, Any]]):
        self.second_order = val

    @property
    def nmodes(self) -> int:
        if self.spectrum is not None:
            return self.spectrum.n_modes
        return 0

    def get_phonon_frequencies(self) -> np.ndarray:
        if self.spectrum is not None:
            return self.spectrum.frequencies_mev
        return np.array([])

    def get_phonon_pert(self, pert_scale_si: float) -> Optional[dict[str, Any]]:
        if self.spectrum is not None:
            return self.spectrum.get_phonon_pert(pert_scale_si, displacement=self.displacement)
        return None

    def _ingest_raw_data(self, raw: RawZFSData):
        """Populates internal arrays and applies principal frame rotation from RawZFSData."""
        # Derivatives, couplings and thresholds here are in J. RawZFSData.load already
        # converts, but data parsed from simulation folders arrives in MHz.
        raw = raw.to_unit("J")
        self.defect = raw.defect
        self.cell_size = raw.cell_size
        self.pert_scale = raw.pert_scale
        self.calc_method = raw.metadata.get("calc_method", "calc")

        if raw.ground_state_zfs is not None:
            D_xx, D_yy, D_zz, R = raw.ground_state_zfs.principal_components()
            self.eigen_rotation = R
            self.zfs_relaxed = np.diag([D_xx, D_yy, D_zz])

        # eigen_rotation columns are the ground-state principal-frame eigenvectors,
        # so rotating into that frame is R.T @ tensor @ R (NOT R @ tensor @ R.T).
        eigen_rot = self.eigen_rotation if self.eigen_rotation is not None else np.eye(3)
        eigen_rot_t = eigen_rot.T

        pert_SI = self.pert_scale * CONSTANTS["ang_amu2SI"]
        phonon_pert = self.get_phonon_pert(pert_SI)
        # Remap raw perturbation indices (full DFT-run indexing) to spectrum positions.
        # The spectrum owns mode identity: original_indices records which DFT modes were kept.
        spectrum = self.spectrum
        # Legacy fallback: if raw indices clearly exceed the spectrum and the spectrum
        # has no recorded original_indices, infer the mapping from perturbation amplitudes
        # vs frequencies. This replaces the old silent index truncation (raw idx >= n_modes
        # were dropped or misaligned) with an explicit, validated remap.
        raw_idx_set = set(raw.first_order.keys()) | {i for i, j in raw.second_order.keys()}
        identity = (
            spectrum is not None
            and spectrum.original_indices is not None
            and np.array_equal(spectrum.original_indices, np.arange(spectrum.n_modes))
        )
        if (
            spectrum is not None
            and raw_idx_set
            and max(raw_idx_set) >= spectrum.n_modes
            and (spectrum.original_indices is None or identity)
        ):
            pert_amps = {}
            for idx, entry in list(raw.first_order.items()) + list((k[0], v) for k, v in raw.second_order.items()):
                pert = entry.get("pert") if isinstance(entry, dict) else None
                if pert is None:
                    continue
                amp = pert[0] if isinstance(pert, (list, tuple, np.ndarray)) else pert
                if amp is not None:
                    pert_amps[idx] = float(amp)
            if pert_amps and spectrum.infer_original_indices(pert_amps):
                pass
        remap_needed = (
            spectrum is not None
            and spectrum.original_indices is not None
            and not np.array_equal(spectrum.original_indices, np.arange(spectrum.n_modes))
        )
        n_phon = spectrum.n_modes if spectrum is not None else None
        dropped = []

        def remap(idx: int) -> Optional[int]:
            """Raw DFT index -> spectrum position; logs and returns None if dropped."""
            if not remap_needed:
                return idx if n_phon is None or idx < n_phon else None
            mapped = spectrum.index_of_original(idx)
            if mapped is None:
                dropped.append(idx)
            return mapped

        def ingest_first(entries: dict, sign: int) -> dict[int, dict[str, Any]]:
            """First-order runs at sign * pert_scale, rotated into the defect frame, keyed by spectrum position."""
            out = {}
            for idx, entry in entries.items():
                sp = remap(idx)
                if sp is None:
                    continue
                if isinstance(entry, PerturbationEntry):
                    tensor_mhz = entry.zfs_tensor.matrix
                    rotated = eigen_rot_t @ tensor_mhz @ eigen_rot if self.eigen_rotation is not None else tensor_mhz
                    tensor_j = rotated
                    out[sp] = {
                        "tensor": tensor_j,
                        "symmetry": phonon_pert["sym"][sp] if phonon_pert else None,
                        "pert": sign * phonon_pert["disp"][sp] if phonon_pert else None,
                        "ipr": phonon_pert["ipr"][sp] if phonon_pert else None,
                        "original_index": idx,
                    }
                elif isinstance(entry, dict):
                    # Loaded from saved raw_zfs_data .npz: rotate into the defect
                    # eigenframe with R.T @ tensor @ R, exactly as the
                    # PerturbationEntry branch does above, otherwise the derivative
                    # subtraction mixes frames and A1/A1 modes become non-diagonal.
                    enriched = dict(entry)
                    tensor = np.asarray(enriched.get("tensor"), dtype=float)
                    rotated = eigen_rot_t @ tensor @ eigen_rot if self.eigen_rotation is not None else tensor
                    enriched["tensor"] = rotated
                    # When phonon data is available, always update symmetry/pert/ipr
                    # so that mode-dependent SI displacements q_i = q0 * sqrt(2*omega/hbar)
                    # are used for derivatives instead of a raw pert_scale.
                    if phonon_pert is not None:
                        if phonon_pert.get("sym") is not None and sp < len(phonon_pert["sym"]):
                            enriched["symmetry"] = phonon_pert["sym"][sp]
                        if phonon_pert.get("disp") is not None and sp < len(phonon_pert["disp"]):
                            enriched["pert"] = sign * phonon_pert["disp"][sp]
                        if phonon_pert.get("ipr") is not None and sp < len(phonon_pert["ipr"]):
                            enriched["ipr"] = phonon_pert["ipr"][sp]
                    enriched["original_index"] = idx
                    out[sp] = enriched
            return out

        raw.check_central()
        if raw.first_order:
            self.first_order = ingest_first(raw.first_order, 1)
        if raw.first_order_minus:
            self.first_order_minus = ingest_first(raw.first_order_minus, -1)
            self.second_order = self._central_diagonal_entries()

        if raw.second_order:
            self.second_order = {}
            for (i, j), entry in raw.second_order.items():
                si, sj = remap(i), remap(j)
                if si is None or sj is None:
                    continue
                if isinstance(entry, PerturbationEntry):
                    tensor_mhz = entry.zfs_tensor.matrix
                    rotated = eigen_rot_t @ tensor_mhz @ eigen_rot if self.eigen_rotation is not None else tensor_mhz
                    tensor_j = rotated
                    self.second_order[(si, sj)] = {
                        "tensor": tensor_j,
                        "symmetry": (phonon_pert["sym"][si], phonon_pert["sym"][sj]) if phonon_pert else None,
                        "pert": (phonon_pert["disp"][si], phonon_pert["disp"][sj]) if phonon_pert else None,
                        "ipr": (phonon_pert["ipr"][si], phonon_pert["ipr"][sj]) if phonon_pert else None,
                        "original_index": (i, j),
                    }
                elif isinstance(entry, dict):
                    # Loaded from saved raw_zfs_data .npz: rotate into the defect
                    # eigenframe with R.T @ tensor @ R, exactly as the
                    # PerturbationEntry branch does above.
                    enriched = dict(entry)
                    tensor = np.asarray(enriched.get("tensor"), dtype=float)
                    rotated = eigen_rot_t @ tensor @ eigen_rot if self.eigen_rotation is not None else tensor
                    enriched["tensor"] = rotated
                    if phonon_pert is not None:
                        if (
                            phonon_pert.get("sym") is not None
                            and si < len(phonon_pert["sym"])
                            and sj < len(phonon_pert["sym"])
                        ):
                            enriched["symmetry"] = (phonon_pert["sym"][si], phonon_pert["sym"][sj])
                        if (
                            phonon_pert.get("disp") is not None
                            and si < len(phonon_pert["disp"])
                            and sj < len(phonon_pert["disp"])
                        ):
                            enriched["pert"] = (phonon_pert["disp"][si], phonon_pert["disp"][sj])
                        if (
                            phonon_pert.get("ipr") is not None
                            and si < len(phonon_pert["ipr"])
                            and sj < len(phonon_pert["ipr"])
                        ):
                            enriched["ipr"] = (phonon_pert["ipr"][si], phonon_pert["ipr"][sj])
                    enriched["original_index"] = (i, j)
                    self.second_order[(si, sj)] = enriched

        if dropped:
            LOGGER.warning(
                f"{len(dropped)} raw mode indices have no partner in the phonon spectrum ({n_phon} modes kept) "
                f"and are skipped: {sorted(set(dropped))[:10]}{'...' if len(set(dropped)) > 10 else ''}; "
                "their couplings are inherited through degeneracy groups where available"
            )

        self.treated_modes = self._get_symmetry_factor()

    def _get_symmetry_factor(self) -> set[int]:
        total_modes = self.nmodes
        n_1d = len(self.zfs_tensors)
        n_2d = len(self.zfs_tensors_2d)
        if self.first_order_minus:  # central: the diagonal entries are built from the first-order runs
            treated = self.zfs_tensors.keys()
        else:
            treated = self.zfs_tensors.keys() if (0 < n_1d <= n_2d or n_2d == 0) else self.zfs_tensors_2d.keys()
        mode_set = {x for item in treated for x in (item if isinstance(item, tuple) else (item,))}
        LOGGER.info(
            f"ZFS tensors: {n_1d} first order, {n_2d} second order; {len(mode_set)} of {total_modes} modes treated"
        )
        return mode_set

    def _central_diagonal_entries(self) -> dict[tuple[int, int], dict[str, Any]]:
        """The (i, i) second-order entries of the central differences, one per mode run at +Q and -Q.

        Each holds the +Q tensor and, as ``tensor_minus``, the -Q one; calculate_second_order_derivatives
        takes (D(Q) + D(-Q) - 2 D0)/Q^2 from them. A mode without its -Q run gets no diagonal entry.
        """
        entries = {}
        for i, plus in self.first_order.items():
            minus = self.first_order_minus.get(i)
            if minus is None:
                continue
            entries[(i, i)] = {
                "tensor": plus["tensor"],
                "tensor_minus": minus["tensor"],
                "central": True,
                "pert": (plus.get("pert"), plus.get("pert")),
                "symmetry": (plus.get("symmetry"), plus.get("symmetry")),
                "ipr": (plus.get("ipr"), plus.get("ipr")),
                "original_index": (plus.get("original_index"), plus.get("original_index")),
            }
        missing = sorted(set(self.first_order) - set(self.first_order_minus))
        if missing:
            LOGGER.warning(
                f"central differences: {len(missing)} modes have no -Q run, so no diagonal second-order coupling "
                "and a forward first-order derivative: "
                + ", ".join(str(self.first_order[i].get("original_index", i) + 1) for i in missing[:20])
                + (", ..." if len(missing) > 20 else "")
            )
        extra = sorted(set(self.first_order_minus) - set(self.first_order))
        if extra:
            LOGGER.warning(f"central differences: {len(extra)} -Q runs without their +Q run are not used")
        return entries

    def calculate_first_order_derivatives(self, ipr_thresh: Optional[float] = None):
        """
        Calculates 1st order finite difference derivatives dD/dq and spin-phonon coupling coefficients.
        """
        n_modes = self.nmodes

        zfs_deriv = np.zeros((n_modes, 3, 3))
        V_0_0 = np.zeros(n_modes)
        V_0_pm = np.zeros(n_modes)
        V_p_m = np.zeros(n_modes)

        # The V formulas are rotationally invariant in the xy-plane, so the
        # Ex/Ey ordering convention of the eigenvectors does not affect them;
        # no per-defect/cell-size switch is needed.

        inherit_coefficients = False
        spectrum = self.spectrum
        if len(self.treated_modes) < n_modes and spectrum is not None:
            inherit_coefficients = True
            LOGGER.debug("Degenerate modes inherit coefficients")

        for i, item in sorted(self.zfs_tensors.items()):
            if i not in self.treated_modes:
                continue

            if ipr_thresh is not None and item.get("ipr", 0) < ipr_thresh:
                continue

            D_i = item["tensor"]
            q = item["pert"]
            if q is None:
                continue
            sym = item.get("symmetry")
            if sym is None:
                continue

            minus = self.first_order_minus.get(i)
            if minus is not None:
                # central: (D(Q) - D(-Q)) / 2Q, free of D0 and of the Q^2 term
                dD = (D_i - minus["tensor"]) / 2
            else:
                dD = D_i - self.zfs_relaxed
            self._check_symmetry(dD, [sym], i, kind="displacement")

            dD_dq = dD / q
            zfs_deriv[i] = dD_dq

            trace_in_plane = dD_dq[0, 0] + dD_dq[1, 1]
            diff_in_plane = dD_dq[0, 0] - dD_dq[1, 1]
            off_diag_in_plane = dD_dq[0, 1]

            if sym == "A1":
                # Appendix A.1 Eq. A.5:
                # Traceless projection: dD_zz - 0.5 * (dD_xx + dD_yy) = 1.5 * d\tilde{D}_zz
                # V_00 = 0.5 * d\tilde{D}_zz = 1/3 * (dD_zz - 0.5 * (dD_xx + dD_yy))
                V_0_0[i] = np.abs(dD_dq[2, 2] - 0.5 * trace_in_plane) / 3.0
            elif sym in ("Ex", "Ey"):
                # Appendix A.1 Eq. A.6 & A.7 (rotationally invariant in xy-plane):
                # V_+- = 0.5 * sqrt((dD_xx - dD_yy)^2 + 4 * dD_xy^2)
                # V_0+- = sqrt(dD_xz^2 + dD_yz^2) / sqrt(2)
                V_p_m[i] = 0.5 * np.sqrt(diff_in_plane**2 + 4.0 * off_diag_in_plane**2)
                V_0_pm[i] = np.sqrt(dD_dq[0, 2] ** 2 + dD_dq[1, 2] ** 2) / np.sqrt(2)

            # Inherit V-coefficients for the degenerate twin via pair_id.
            # The spectrum owns the pairing (strict involution); no frequency guessing.
            # Fill-only: never overwrite a twin's own non-zero coefficients (Ex
            # yields V_pm, Ey yields V_0pm; both are real data on a complete file).
            if inherit_coefficients:
                twin = twin_of(spectrum.deg_groups, i)
                if twin is not None and twin != i:
                    V_0_pm[twin] = np.where(V_0_pm[twin] == 0, V_0_pm[i], V_0_pm[twin])
                    V_p_m[twin] = np.where(V_p_m[twin] == 0, V_p_m[i], V_p_m[twin])

            if LOGGER.isEnabledFor(logging.DEBUG):
                self._debug_derivs(
                    dD_dq / CONSTANTS["MHz2J"],
                    q,
                    sym,
                    i + 1,
                    V_0_0[i] / CONSTANTS["MHz2J"],
                    V_0_pm[i] / CONSTANTS["MHz2J"],
                    V_p_m[i] / CONSTANTS["MHz2J"],
                )

        LOGGER.info(
            f"first-order couplings (nonzero): V_00 {np.sum(V_0_0 > 0)}, V_pm {np.sum(V_p_m > 0)}, "
            f"V_0pm {np.sum(V_0_pm > 0)}"
        )

        return zfs_deriv, V_0_0, V_p_m, V_0_pm

    def calculate_second_order_derivatives(self, zfs_1d_derivs: np.ndarray, ipr_thresh: Optional[float] = None):
        """
        Calculates 2nd order finite difference derivatives d2D/dqi dqj and 2-phonon coupling coefficients.
        """
        n_modes = self.nmodes
        phonon_energies = self.get_phonon_frequencies()

        zfs_2nd_derivs = np.zeros((n_modes, n_modes, 3, 3))
        V_0_0_2nd = np.zeros((n_modes, n_modes))
        V_p_m_2nd = np.zeros((n_modes, n_modes))
        V_0_pm_2nd = np.zeros((n_modes, n_modes))

        no_first_order = []
        diagonal_done: set[int] = set()  # modes whose own (i, i) coupling was computed
        computed_pairs: set[tuple[int, int]] = set()  # (i, j), i <= j, whose coupling came from a run
        # Entries with content outside the exact Gi x Gj rule (MHz of that content), for the summary.
        outside_strict: list[float] = []
        n_judged = 0
        for (i, j), item in sorted(self.zfs_tensors_2d.items()):
            # The curvature subtracts both modes' first-order response; a mode without one
            # (derivative 0) would leave its linear change in d2D as a spurious coupling.
            if i not in self.treated_modes or j not in self.treated_modes:
                if i in self.treated_modes or j in self.treated_modes:
                    no_first_order.append((i, j))
                continue

            if ipr_thresh is not None:
                ipr_i, ipr_j = item["ipr"]
                if ipr_i < ipr_thresh and ipr_j < ipr_thresh:
                    continue

            (q_i, q_j) = item["pert"]
            if q_i is None or q_j is None or q_i == 0 or q_j == 0:
                continue

            dD_qi = zfs_1d_derivs[i]
            dD_qj = zfs_1d_derivs[j]
            sym_i, sym_j = item["symmetry"]
            sym = MathUtils.calc_symmetry(sym_i, sym_j)

            D_qi_qj = item["tensor"]
            if item.get("central"):
                # (D(Q) + D(-Q) - 2 D0) / Q^2 from the +-Q first-order runs: no pair run, nothing to check it against
                d2D_dqidqj = (D_qi_qj + item["tensor_minus"] - 2 * self.zfs_relaxed) / (q_i * q_j)
            else:
                d2D_dqidqj = (D_qi_qj - self.zfs_relaxed) / (q_i * q_j) - dD_qi / q_j - dD_qj / q_i

            # The pair displaces Q_i + Q_j, so its ZFS change is expected to be the
            # single-mode changes dD_i q_i + dD_j q_j plus a small remainder (the
            # cross curvature). An A2 mode cannot change D linearly (C3v), so its
            # single-mode change is pure curvature, which grows as Q^2: along 2Q
            # (the diagonal pair) it is 4x the change at Q, not 2x. A remainder that
            # is a sizeable multiple of the expectation (|alpha| > INCONSISTENT_ALPHA)
            # means the response is strongly nonlinear at this Q, or the run was
            # displaced too far or is stale; the geometry check tells these apart.
            if i in self.treated_modes and j in self.treated_modes and not item.get("central"):
                if i == j and sym_i == "A2":
                    expected = 4.0 * dD_qi * q_i
                else:
                    expected = dD_qi * q_i + dD_qj * q_j
                remainder = (D_qi_qj - self.zfs_relaxed) - expected
                norm = np.sum(expected * expected)
                if norm > (MIN_EXPECTED_MHZ * CONSTANTS["MHz2J"]) ** 2:
                    alpha = float(np.sum(remainder * expected) / norm)
                    if abs(alpha) > INCONSISTENT_ALPHA:
                        self.inconsistent_pairs.append(
                            (i, j, alpha, float(np.linalg.norm(remainder) / CONSTANTS["MHz2J"]))
                        )
                        if self.exclude_inconsistent_pairs:
                            continue

            check = self._check_symmetry(d2D_dqidqj * q_i * q_j, [sym_i, sym_j], (i, j), kind="derivative")
            if check is not None and check.size_mhz >= SHAPE_FLOOR_MHZ:
                n_judged += 1
                if check.strict_forbidden > SYMMETRY_TOL:
                    outside_strict.append(check.strict_forbidden * check.size_mhz)

            zfs_2nd_derivs[i, j] = d2D_dqidqj
            zfs_2nd_derivs[j, i] = d2D_dqidqj

            trace_in_plane = d2D_dqidqj[0, 0] + d2D_dqidqj[1, 1]
            diff_in_plane = d2D_dqidqj[0, 0] - d2D_dqidqj[1, 1]
            off_diag_in_plane = d2D_dqidqj[0, 1]

            if sym == ["A1"]:
                V_0_0_2nd[i, j] = np.abs(d2D_dqidqj[2, 2] - 0.5 * trace_in_plane) / 3.0
            elif {sym_i, sym_j} == {"Ex"}:
                V_0_0_2nd[i, j] = np.abs(d2D_dqidqj[2, 2] - 0.5 * trace_in_plane) / 3.0
                V_0_pm_2nd[i, j] = np.sqrt(d2D_dqidqj[0, 2] ** 2 + d2D_dqidqj[1, 2] ** 2) / (2.0 * np.sqrt(2))
                V_p_m_2nd[i, j] = 0.25 * np.sqrt(diff_in_plane**2 + 4.0 * off_diag_in_plane**2)
            elif {sym_i, sym_j} == {"Ey"}:
                V_0_0_2nd[i, j] = np.abs(d2D_dqidqj[2, 2] - 0.5 * trace_in_plane) / 3.0
                V_0_pm_2nd[i, j] = np.sqrt(d2D_dqidqj[0, 2] ** 2 + d2D_dqidqj[1, 2] ** 2) / (2.0 * np.sqrt(2))
                V_p_m_2nd[i, j] = 0.25 * np.sqrt(diff_in_plane**2 + 4.0 * off_diag_in_plane**2)
            elif {sym_i, sym_j} == {"Ex", "Ey"}:
                # The D_xy part of Ex x Ey is the E partner of the D_xx - D_yy part of Ex x Ex, of equal
                # size, so the same factors apply.
                V_0_pm_2nd[i, j] = np.sqrt(d2D_dqidqj[0, 2] ** 2 + d2D_dqidqj[1, 2] ** 2) / (2.0 * np.sqrt(2))
                V_p_m_2nd[i, j] = 0.25 * np.sqrt(diff_in_plane**2 + 4.0 * off_diag_in_plane**2)
            elif {sym_i, sym_j} & {"A1", "A2"} and {sym_i, sym_j} & {"Ex", "Ey"}:
                # A x E = E: a pure E curvature, shaped like a first-order E derivative, so no V_00.
                V_0_pm_2nd[i, j] = np.sqrt(d2D_dqidqj[0, 2] ** 2 + d2D_dqidqj[1, 2] ** 2) / (2.0 * np.sqrt(2))
                V_p_m_2nd[i, j] = 0.25 * np.sqrt(diff_in_plane**2 + 4.0 * off_diag_in_plane**2)

            if i == j:
                diagonal_done.add(i)
            computed_pairs.add((min(i, j), max(i, j)))

            V_0_0_2nd[j, i] = V_0_0_2nd[i, j]
            V_p_m_2nd[j, i] = V_p_m_2nd[i, j]
            V_0_pm_2nd[j, i] = V_0_pm_2nd[i, j]

            if LOGGER.isEnabledFor(logging.DEBUG):
                self._debug_derivs(
                    d2D_dqidqj / CONSTANTS["MHz2J"],
                    item["pert"],
                    item["symmetry"],
                    (i + 1, j + 1),
                    V_0_0_2nd[i, j] / CONSTANTS["MHz2J"],
                    V_0_pm_2nd[i, j] / CONSTANTS["MHz2J"],
                    V_p_m_2nd[i, j] / CONSTANTS["MHz2J"],
                )

        # A degenerate twin without its own diagonal run inherits its partner's diagonal
        # coupling (the spectrum owns the pairing). Fill-only, as for first order: a twin
        # with its own computed (i, i) keeps it, whatever order the entries came in.
        if len(self.treated_modes) < n_modes and len(phonon_energies) == n_modes and self.spectrum is not None:
            for i in sorted(diagonal_done):
                twin = twin_of(self.spectrum.deg_groups, i)
                if twin is not None and twin not in diagonal_done:
                    for V in (V_0_0_2nd, V_0_pm_2nd, V_p_m_2nd):
                        V[twin, twin] = V[i, i]

        if self.degenerate_pairs_from_symmetry:
            self._degenerate_pairs_from_symmetry(zfs_2nd_derivs, V_0_0_2nd, V_p_m_2nd, V_0_pm_2nd)
            computed_pairs.update((min(x, y), max(x, y)) for x, y in self.symmetry_derived_pairs)

        if self.offdiag_from_diagonal:
            self._offdiag_from_diagonal(computed_pairs, V_0_0_2nd, V_p_m_2nd, V_0_pm_2nd)

        if outside_strict:
            LOGGER.debug(
                f"second-order symmetry: {len(outside_strict)} of {n_judged} entries hold content outside "
                f"Gi x Gj (median {np.median(outside_strict):.2f} MHz, max {np.max(outside_strict):.2f} MHz); "
                "expected O(q D''') finite-difference terms, warned about only beyond those"
            )

        if no_first_order:
            LOGGER.warning(
                f"{len(no_first_order)} second-order entries skipped: a mode has no first-order derivative, so "
                "its linear response cannot be removed; run folders "
                + ", ".join(f"{i + 1}_{j + 1}" for i, j in no_first_order[:20])
                + (", ..." if len(no_first_order) > 20 else "")
            )

        if self.inconsistent_pairs:
            if self.exclude_inconsistent_pairs:
                action, hint = "excluded (--exclude-inconsistent-pairs)", ""
            else:
                action = "kept"
                hint = "; check the geometry with compare_displacement.py, --exclude-inconsistent-pairs drops them"
            LOGGER.warning(
                f"{len(self.inconsistent_pairs)} of {len(self.zfs_tensors_2d)} second-order entries {action}: "
                f"their ZFS change departs from the one expected from the first-order runs by a multiple of it "
                f"(|alpha| > {INCONSISTENT_ALPHA}: strongly nonlinear at this Q, or an over-displaced or stale "
                f"run{hint}); run folders "
                + ", ".join(f"{i + 1}_{j + 1} (alpha={a:.2f})" for i, j, a, _ in self.inconsistent_pairs)
            )

        LOGGER.info(
            f"second-order couplings (nonzero): V_00 {np.sum(V_0_0_2nd > 0)}, V_pm {np.sum(V_p_m_2nd > 0)}, "
            f"V_0pm {np.sum(V_0_pm_2nd > 0)}"
        )

        return zfs_2nd_derivs, V_0_0_2nd, V_p_m_2nd, V_0_pm_2nd

    def _degenerate_pairs_from_symmetry(
        self, zfs_2nd_derivs: np.ndarray, V_0_0_2nd: np.ndarray, V_p_m_2nd: np.ndarray, V_0_pm_2nd: np.ndarray
    ) -> None:
        """Replace the coupling of each degenerate Ex-Ey pair by its C3v value from the Ex-Ex curvature.

        Within one E pair the quadratic terms Qx^2, Qy^2, QxQy span A1 + E, so the
        second-order response has one constant per irrep of D (A1, and the two E
        parts), all of them present in d2D/dQx^2. For any orientation of the pair
        in its degenerate subspace this gives |w_xy| = |w_xx| and |p_xy| = |p_xx|,
        with w = (D_xx - D_yy) + 2i D_xy and p = D_xz + i D_yz, while the A1 part of
        d2D/dQxdQy vanishes. The Ex-Ey V-coefficients therefore follow from the
        Ex-Ex derivative alone, so the Ex-Ey runs are not needed. Only the
        V-coefficients are replaced; zfs_2nd_derivs keeps the computed tensors.
        Pairs without an Ex-Ex derivative (not run, or excluded) keep their values.
        """
        spectrum = self.spectrum
        if spectrum is None or spectrum.deg_groups is None or spectrum.symmetries is None:
            LOGGER.warning("degenerate Ex-Ey pairs from symmetry need the phonon spectrum; option ignored")
            return

        syms = list(spectrum.symmetries)
        n_pairs = 0
        self.symmetry_derived_pairs = []
        for group in spectrum.deg_groups:
            if len(group) != 2 or {syms[k] for k in group} != {"Ex", "Ey"}:
                continue
            n_pairs += 1
            x, y = group if syms[group[0]] == "Ex" else group[::-1]
            d2D_xx = zfs_2nd_derivs[x, x]
            if not np.any(d2D_xx):
                continue

            diff_in_plane = d2D_xx[0, 0] - d2D_xx[1, 1]
            off_diag_in_plane = d2D_xx[0, 1]
            # The Ex-Ey formulas of calculate_second_order_derivatives, with |w| and |p| of the Ex-Ex derivative.
            V_0_0_2nd[x, y] = V_0_0_2nd[y, x] = 0.0
            V_p_m_2nd[x, y] = V_p_m_2nd[y, x] = 0.25 * np.sqrt(diff_in_plane**2 + 4.0 * off_diag_in_plane**2)
            V_0_pm_2nd[x, y] = V_0_pm_2nd[y, x] = np.sqrt(d2D_xx[0, 2] ** 2 + d2D_xx[1, 2] ** 2) / (
                2.0 * np.sqrt(2)
            )
            self.symmetry_derived_pairs.append((int(x), int(y)))

        n_kept = n_pairs - len(self.symmetry_derived_pairs)
        LOGGER.info(
            f"degenerate Ex-Ey couplings from symmetry: {len(self.symmetry_derived_pairs)} of {n_pairs} pairs"
            + (f" ({n_kept} without an Ex-Ex derivative keep their computed values)" if n_kept else "")
        )

    def _offdiag_from_diagonal(
        self, computed: set[tuple[int, int]], V_0_0_2nd: np.ndarray, V_p_m_2nd: np.ndarray, V_0_pm_2nd: np.ndarray
    ) -> None:
        """Approximate each second-order pair (i, j), i != j, that was not computed, from the diagonal.

        From the diagonal couplings (the (i, i) runs, inherited by degenerate twins), and only for
        the types the pair's symmetry allows, by the same rules as the computed pairs in
        calculate_second_order_derivatives. Same-type pairs take the root of the same coupling type,
        V_T(i, j) = sqrt(V_T(i, i) V_T(j, j)): A1 x A1 and A2 x A2 give V_00; Ex x Ex and Ey x Ey give
        V_00, V_0pm and V_pm; Ex x Ey gives V_0pm and V_pm.

        A x E pairs (A1 x E = A2 x E = E) are allowed V_0pm and V_pm, but an A mode has no diagonal
        V_0pm or V_pm, so that root would be zero. They take the mixed-type root instead,
        V_T(a, e) = sqrt(V_00(a, a) V_T(e, e)), and no V_00: how strongly the A mode modulates D,
        with how strongly the E mode flips the spin. Any other pair (A1 x A2) gets nothing.

        Both roots are heuristics: neither follows from symmetry, and pair by pair they scatter by a
        factor of about ten. In NV_64, where every pair is computed, the fill without the A x E pairs
        gives a T1 2.5-4.4x the all-pairs one (100-500 K, sigma 1-7.5 meV); with them 0.74-1.46x.
        Pairs in ``computed`` (run, or derived by symmetry) keep their values.
        """
        n = V_0_0_2nd.shape[0]
        labels = list(self.spectrum.symmetries) if self.spectrum is not None and self.spectrum.symmetries else []
        if len(labels) != n:
            LOGGER.warning("off-diagonal couplings from the diagonal need the modes' symmetry labels; option ignored")
            return
        irrep = np.array([str(s).upper() for s in labels])
        is_ex, is_ey = irrep == "EX", irrep == "EY"
        same = irrep[:, None] == irrep[None, :]
        a_pair = same & np.isin(irrep, ["A1", "A2"])[:, None]  # A1 x A1, A2 x A2: product A1
        e_same = same & (is_ex | is_ey)[:, None]  # Ex x Ex, Ey x Ey
        e_mixed = (is_ex[:, None] & is_ey[None, :]) | (is_ey[:, None] & is_ex[None, :])  # Ex x Ey
        is_a, is_e = np.isin(irrep, ["A1", "A2"]), is_ex | is_ey
        a_x_e = (is_a[:, None] & is_e[None, :]) | (is_e[:, None] & is_a[None, :])  # A1 x E, A2 x E
        allowed = {"V_00": a_pair | e_same, "V_pm": e_same | e_mixed, "V_0pm": e_same | e_mixed}

        keep = np.eye(n, dtype=bool)
        for i, j in computed:
            keep[i, j] = keep[j, i] = True
        v00_diag = np.abs(np.diag(V_0_0_2nd))  # before the loop: it fills only off-diagonal entries
        for name, V in (("V_00", V_0_0_2nd), ("V_pm", V_p_m_2nd), ("V_0pm", V_0_pm_2nd)):
            diag = np.abs(np.diag(V))
            fill = ~keep & allowed[name]
            V[fill] = np.sqrt(np.outer(diag, diag))[fill]
            if name != "V_00":  # A x E: the A mode's V_00 with the E mode's spin-flip coupling
                a_side, e_side = np.where(is_a, v00_diag, 0.0), np.where(is_e, diag, 0.0)
                fill = ~keep & a_x_e
                V[fill] = np.sqrt(np.outer(a_side, e_side) + np.outer(e_side, a_side))[fill]
        any_allowed = allowed["V_00"] | allowed["V_pm"] | a_x_e
        self.n_offdiag_filled = int(np.count_nonzero(np.triu(~keep & any_allowed, 1)))
        self.n_offdiag_filled_a_x_e = int(np.count_nonzero(np.triu(~keep & a_x_e, 1)))
        LOGGER.info(
            f"off-diagonal couplings from the diagonal, symmetry-allowed types only: {self.n_offdiag_filled} "
            f"pairs filled ({self.n_offdiag_filled_a_x_e} A x E by sqrt(V00_aa V_ee), the rest by sqrt(V_ii V_jj)), "
            f"{int(np.count_nonzero(np.triu(keep, 1)))} computed off-diagonal pairs kept"
        )

    def process_first_order_perturbations(self, output_filename: Optional[str] = None) -> list[str]:
        if not self.zfs_tensors:
            raise ValueError("First order ZFS data not loaded.")

        pert_SI = self.pert_scale * CONSTANTS["ang_amu2SI"]
        phonon_pert = self.get_phonon_pert(pert_SI)
        zfs_derivs, V_0_0, V_p_m, V_0_pm = self.calculate_first_order_derivatives()

        save_name = (
            f"{output_filename}.npz"
            if output_filename
            else f"derivatives/{self.defect}_{self.cell_size}_zfs_coefficients_"
            f"{self.calc_method}_{self.pert_scale}_.npz"
        )

        save_path = self.save_data(
            save_name,
            zfs=1.5 * self.zfs_relaxed[2, 2],
            zfs_derivs=zfs_derivs,
            V_0_0=V_0_0,
            V_p_m=V_p_m,
            V_0_pm=V_0_pm,
            freqs=phonon_pert["freqs"],
            sym=phonon_pert["sym"],
            ipr=phonon_pert["ipr"],
        )
        return [save_path]

    def process_second_order_perturbations(
        self, first_order: Optional[SpinPhononCouplingData] = None, output_filename: Optional[str] = None
    ) -> list[str]:
        if not self.zfs_tensors_2d:
            raise ValueError("Second order ZFS data not loaded.")

        save_name = (
            f"{output_filename}.npz"
            if output_filename
            else f"derivatives/{self.defect}_{self.cell_size}_zfs2d_coefficients_"
            f"{self.calc_method}_{self.pert_scale}_.npz"
        )

        # Second-order couplings are built on the first-order derivatives, so with
        # 1d tensors loaded this is the combined analysis: one file, V_* + V2_*.
        if self.zfs_tensors:
            return self.process_both_orders(output_filename=save_name[: -len(".npz")])
        if first_order is None or first_order.zfs_derivs is None:
            raise ValueError("First order ZFS data not loaded (need 1d perturbation set or first_order coupling data).")

        if first_order.coupling_unit != "J" or first_order.derivs_unit != "J":
            raise ValueError("first_order coupling data must be in J (coupling_unit and derivs_unit).")
        zfs_1d_derivs = first_order.zfs_derivs
        zfs_2nd_derivs, V2_0_0, V2_p_m, V2_0_pm = self.calculate_second_order_derivatives(zfs_1d_derivs)

        pert_SI = self.pert_scale * CONSTANTS["ang_amu2SI"]
        phonon_pert = self.get_phonon_pert(pert_SI)

        # Keep the first-order couplings from first_order under V_*; the
        # second-order ones go under V2_* so the two never share keys.
        save_path = self.save_data(
            save_name,
            second_order=True,
            zfs=1.5 * self.zfs_relaxed[2, 2],
            zfs_derivs=zfs_1d_derivs,
            V_0_0=first_order.V_0_0,
            V_p_m=first_order.V_p_m,
            V_0_pm=first_order.V_0_pm,
            V2_0_0=V2_0_0,
            V2_p_m=V2_p_m,
            V2_0_pm=V2_0_pm,
            zfs_2nd_derivs=zfs_2nd_derivs,
            freqs=phonon_pert["freqs"],
            sym=phonon_pert["sym"],
            ipr=phonon_pert["ipr"],
        )
        return [save_path]

    def process_both_orders(
        self,
        output_filename: Optional[str] = None,
        ipr_thresh_first: Optional[float] = None,
        ipr_thresh_second: Optional[float] = None,
    ) -> list[str]:
        """
        Calculates first- AND second-order derivatives/couplings in one go and
        saves them together in a single SpinPhononCouplingData .npz with both
        V_* (first order) and V2_* (second order) coefficient keys.

        The later transition-rate and T_1 stages need both orders, so a combined
        run_dir output is the natural container: one file, all couplings.
        """
        if not self.zfs_tensors:
            raise ValueError("First order ZFS data not loaded (need 1d perturbation set).")
        if not self.zfs_tensors_2d:
            raise ValueError("Second order ZFS data not loaded (need 2d perturbation set).")

        pert_SI = self.pert_scale * CONSTANTS["ang_amu2SI"]
        phonon_pert = self.get_phonon_pert(pert_SI)
        if phonon_pert is None:
            raise ValueError(
                "Phonon data (frequencies/symmetry/ipr) is required for combined 1d+2d analysis. "
                "Provide --phonon_file (phonon_data.npz or phonopy.yaml)."
            )

        # 1) First order
        zfs_derivs, V_0_0, V_p_m, V_0_pm = self.calculate_first_order_derivatives(ipr_thresh=ipr_thresh_first)

        # 2) Second order (needs 1d derivs)
        zfs_2nd_derivs, V2_0_0, V2_p_m, V2_0_pm = self.calculate_second_order_derivatives(
            zfs_derivs, ipr_thresh=ipr_thresh_second
        )

        save_name = (
            f"{output_filename}.npz"
            if output_filename
            else f"derivatives/{self.defect}_{self.cell_size}_zfs_coupling_"
            f"{self.calc_method}_{self.pert_scale}_both.npz"
        )
        save_path = self.save_data(
            save_name,
            second_order=True,
            zfs=1.5 * self.zfs_relaxed[2, 2],
            zfs_derivs=zfs_derivs,
            V_0_0=V_0_0,
            V_p_m=V_p_m,
            V_0_pm=V_0_pm,
            V2_0_0=V2_0_0,
            V2_p_m=V2_p_m,
            V2_0_pm=V2_0_pm,
            zfs_2nd_derivs=zfs_2nd_derivs,
            freqs=phonon_pert["freqs"],
            sym=phonon_pert["sym"],
            ipr=phonon_pert["ipr"],
        )
        return [save_path]

    def save_data(self, save_name: str, **kwargs) -> str:
        """
        Saves simulation or derivative datasets using structured dataclasses
        with explicit unit metadata.
        """
        if not save_name.endswith(".npz"):
            save_name += ".npz"
        Path(save_name).parent.mkdir(parents=True, exist_ok=True)

        # 1. If saving calculated coupling coefficients (SpinPhononCouplingData)
        if "V_0_0" in kwargs:
            gs_tensor = ZFSTensor(matrix=self.zfs_relaxed, unit="J") if self.zfs_relaxed is not None else None
            coupling_obj = SpinPhononCouplingData(
                order=kwargs.get("order", 2 if kwargs.get("second_order", False) or "2nd" in save_name else 1),
                defect=self.defect,
                cell_size=self.cell_size,
                pert_scale=self.pert_scale,
                calc_method=self.calc_method,
                frequencies=kwargs.get("freqs", np.array([])),
                frequency_unit=kwargs.get("frequency_unit", "J"),
                V_0_0=kwargs["V_0_0"],
                V_p_m=kwargs.get("V_p_m", np.array([])),
                V_0_pm=kwargs.get("V_0_pm", np.array([])),
                coupling_unit=kwargs.get("coupling_unit", "J"),
                ground_state_zfs=gs_tensor,
                zfs_derivs=kwargs.get("zfs_derivs"),
                derivs_unit=kwargs.get("derivs_unit", "J"),
                symmetries=kwargs.get("sym"),
                iprs=kwargs.get("ipr"),
                V2_0_0=kwargs.get("V2_0_0"),
                V2_p_m=kwargs.get("V2_p_m"),
                V2_0_pm=kwargs.get("V2_0_pm"),
                zfs_2nd_derivs=kwargs.get("zfs_2nd_derivs"),
            )
            out = coupling_obj.save(save_name)
            LOGGER.info(f"saved coupling data: {out}")
            return out

        # 2. If saving raw dataset (RawZFSData)
        first = kwargs.get("first_order", kwargs.get("zfs_tensors"))
        second = kwargs.get("second_order", kwargs.get("zfs_tensors_2d"))
        if first is not None or second is not None:
            gs_tensor = ZFSTensor(matrix=self.zfs_relaxed, unit="J") if self.zfs_relaxed is not None else None
            raw_obj = RawZFSData(
                defect=self.defect,
                cell_size=self.cell_size,
                pert_scale=self.pert_scale,
                calc_method=self.calc_method,
                # Tensors stored on the manager were ingested via parse/load,
                # which apply the 3/2 correction for approx data at intake.
                spin_correction_applied=normalize_calc_method(self.calc_method) == "defect_band_approx",
                mass_weighted=self.mass_weighted,
                displacement=self.displacement,
                order=kwargs.get("order", 1 if first is not None else 2),
                ground_state_zfs=gs_tensor,
                eigen_rotation=self.eigen_rotation,
                first_order=first or {},
                second_order=second or {},
            )
            out = raw_obj.save(save_name)
            LOGGER.info(f"saved raw ZFS data: {out}")
            return out

        # 3. Fallback for generic arrays
        metadata = {
            "defect": self.defect,
            "cell_size": self.cell_size,
            "calc_method": self.calc_method,
            "pert_scale": self.pert_scale,
        }
        np.savez(save_name, **metadata, **kwargs)
        LOGGER.info(f"saved ZFS arrays: {save_name}")
        return save_name

    def _check_symmetry(self, change: np.ndarray, labels: list[str], idx: Any, kind: str) -> Optional[ZFSResponseCheck]:
        """Warn when a ZFS change (J, at the displacement) has symmetry its modes forbid.

        The check shared with the cluster runs and packaging (zfs_response.verify_zfs_response):
        first-order entries are raw displacement changes, second-order ones curvature terms
        d2D/dQi dQj * qi * qj (Gi x Gj plus the finite difference's O(q D''') terms).
        """
        if self.zfs_relaxed is None:  # no ground state: no defect frame to check in
            return None
        check = verify_zfs_response(
            change / CONSTANTS["MHz2J"], labels, kind=kind, point_group=self.point_group, reference=self.zfs_relaxed
        )
        if check.problem:
            folder = "_".join(str(k + 1) for k in idx) if isinstance(idx, tuple) else str(idx + 1)
            LOGGER.warning(f"symmetry mismatch in run folder {folder}: {check.problem}; {check.line}")
        return check

    def _debug_derivs(self, dD, q, symmetry, idx, V_0_0, V_0_pm, V_p_m):
        max_val = np.max(np.abs(dD))
        col_width = 12
        total_width = (col_width * 3) + 10
        double_line = "=" * total_width

        freqs = self.get_phonon_frequencies()
        if isinstance(idx, tuple):
            energy = (freqs[idx[0] - 1], freqs[idx[1] - 1]) if len(freqs) >= max(idx) else (0, 0)
        else:
            energy = freqs[idx - 1] if len(freqs) >= idx else 0

        LOGGER.debug(f"{double_line}\n DEBUG DERIVATIVES\n{'-' * total_width}")
        LOGGER.debug(
            f" Mode: {idx}\n"
            f" Symmetry: {symmetry}\n"
            f" Energy: {energy}\n"
            f" Perturbation: {MathUtils.fmt(q)}\n"
            f"{double_line}\n"
            f" Max Value: {max_val:<10.6f}\n"
            f" V_0_0: {V_0_0}\n"
            f" V_0_pm: {V_0_pm}\n"
            f" V_p_m: {V_p_m}"
        )
        LOGGER.debug(" Tensor Structure (3x3):\n")
        for row in dD:
            formatted_row = "  ".join(f"{val:>{col_width}.6f}" for val in row)
            LOGGER.debug(f"  [ {formatted_row} ]")
        LOGGER.debug(f"{double_line}\n")
