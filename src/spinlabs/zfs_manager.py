from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np

from spinlabs.constants import CONSTANTS
from spinlabs.symmetry import twin_of, _build_groups_from_labels
from spinlabs.models import ZFSTensor, PhononSpectrum, PerturbationEntry, RawZFSData, SpinPhononCouplingData, normalize_calc_method
from spinlabs.utils import MathUtils

from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)

# Healthy second-order entries have alpha ~ 0 (|alpha| < 0.05 for NV_64): the
# curvature is far below the linear response at the perturbation amplitude.
# Over-displaced runs give alpha ~ 1 or more; runs whose change cancels the
# linear response (wrong state or geometry) give alpha ~ -1. Both are excluded.
INCONSISTENT_ALPHA = 0.5


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
        exclude_inconsistent_pairs: bool = True,
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
                warnings.warn(
                    f"Fresh symmetry classification failed ({exc}); falling back to stored labels.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                fresh_labels = stored_syms
                # Stored labels are trusted as-is here, so sym_check cannot be
                # verified (no detectable point group to build characters
                # from) and stays None for this spectrum.
                deg_groups = _build_groups_from_labels(spectrum, stored_syms, 0.01)
            if stored_syms is not None and fresh_labels is not stored_syms:
                mismatched = [i for i, (a, b) in enumerate(zip(stored_syms, fresh_labels)) if a != b]
                if mismatched:
                    LOGGER.warning(
                        f"{len(mismatched)} stored symmetry label(s) differ "
                        f"from fresh classification: {mismatched[:10]}"
                        f"{'...' if len(mismatched) > 10 else ''}. "
                        f"Using fresh classification."
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

        self.raw_data = raw_data

        # Defect metadata
        self.defect: Optional[str] = raw_data.defect if raw_data else None
        self.cell_size: Optional[int] = raw_data.cell_size if raw_data else None
        self.pert_scale: Optional[float] = raw_data.pert_scale if raw_data else None
        self.calc_method: Optional[str] = (
            raw_data.calc_method or raw_data.metadata.get("calc_method") if raw_data else None
        )
        self.mass_weighted: bool = raw_data.mass_weighted if raw_data else True
        self.exclude_inconsistent_pairs = exclude_inconsistent_pairs
        # Second-order entries failing the linear-response check, as
        # (i, j, alpha, |remainder| in MHz) with 0-based modes; see calculate_second_order_derivatives.
        self.inconsistent_pairs: list[tuple[int, int, float, float]] = []
        if not self.mass_weighted:
            if spectrum is None:
                raise ValueError(
                    "Non-mass-weighted perturbation data needs the phonon spectrum (eigenvectors and masses) "
                    "to correct the perturbation amplitudes"
                )
            LOGGER.info("Perturbations are not mass-weighted: scaling q by each mode's effective sqrt(mass)")

        # Processed ZFS data in defect principal frame
        self.zfs_relaxed: Optional[np.ndarray] = None  # Shape (3, 3) in J
        self.eigen_rotation: Optional[np.ndarray] = None  # Shape (3, 3)
        self.first_order: dict[int, dict[str, Any]] = {}  # 1D perturbations
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
            return self.spectrum.get_phonon_pert(pert_scale_si, mass_weighted=self.mass_weighted)
        return None

    def _ingest_raw_data(self, raw: RawZFSData):
        """Populates internal arrays and applies principal frame rotation from RawZFSData."""
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

        if raw.first_order:
            self.first_order = {}
            for idx, entry in raw.first_order.items():
                sp = remap(idx)
                if sp is None:
                    continue
                if isinstance(entry, PerturbationEntry):
                    tensor_mhz = entry.zfs_tensor.matrix
                    rotated = eigen_rot_t @ tensor_mhz @ eigen_rot if self.eigen_rotation is not None else tensor_mhz
                    tensor_j = rotated
                    self.first_order[sp] = {
                        "tensor": tensor_j,
                        "symmetry": phonon_pert["sym"][sp] if phonon_pert else None,
                        "pert": phonon_pert["disp"][sp] if phonon_pert else None,
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
                            enriched["pert"] = phonon_pert["disp"][sp]
                        if phonon_pert.get("ipr") is not None and sp < len(phonon_pert["ipr"]):
                            enriched["ipr"] = phonon_pert["ipr"][sp]
                    enriched["original_index"] = idx
                    self.first_order[sp] = enriched

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
                f"Note: {len(dropped)} raw mode indices have no partner in the "
                f"phonon spectrum ({n_phon} modes kept) and were skipped: "
                f"{sorted(set(dropped))[:10]}{'...' if len(set(dropped)) > 10 else ''}. "
                "Their V-coefficients will be inherited via degeneracy groups where available."
            )

        self.treated_modes = self._get_symmetry_factor()

    def _get_symmetry_factor(self) -> set[int]:
        total_modes = self.nmodes
        n_1d = len(self.zfs_tensors)
        n_2d = len(self.zfs_tensors_2d)
        LOGGER.info(f"Tensor data size: {n_1d} 1D, {n_2d} 2D out of {total_modes} total modes")

        treated = self.zfs_tensors.keys() if (0 < n_1d <= n_2d or n_2d == 0) else self.zfs_tensors_2d.keys()
        mode_set = {x for item in treated for x in (item if isinstance(item, tuple) else (item,))}
        LOGGER.info(f"Treated unique modes: {len(mode_set)}")
        return mode_set

    def calculate_first_order_derivatives(self, ipr_thresh: Optional[float] = None):
        """
        Calculates 1st order finite difference derivatives dD/dq and spin-phonon coupling coefficients.
        """
        LOGGER.info("Calculating first-order derivatives...")
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

            dD = D_i - self.zfs_relaxed
            self._check_symmetry(dD, sym, i)

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

        LOGGER.info("Symmetry adjusted coefficients (1st order):")
        LOGGER.info(f"V_00:  {np.sum(V_0_0 > 0)}")
        LOGGER.info(f"V_pm:  {np.sum(V_p_m > 0)}")
        LOGGER.info(f"V_0pm: {np.sum(V_0_pm > 0)}")

        return zfs_deriv, V_0_0, V_p_m, V_0_pm

    def calculate_second_order_derivatives(self, zfs_1d_derivs: np.ndarray, ipr_thresh: Optional[float] = None):
        """
        Calculates 2nd order finite difference derivatives d2D/dqi dqj and 2-phonon coupling coefficients.
        """
        LOGGER.info("Calculating second-order derivatives...")
        n_modes = self.nmodes
        phonon_energies = self.get_phonon_frequencies()

        zfs_2nd_derivs = np.zeros((n_modes, n_modes, 3, 3))
        V_0_0_2nd = np.zeros((n_modes, n_modes))
        V_p_m_2nd = np.zeros((n_modes, n_modes))
        V_0_pm_2nd = np.zeros((n_modes, n_modes))

        for (i, j), item in sorted(self.zfs_tensors_2d.items()):
            if i not in self.treated_modes and j not in self.treated_modes:
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
            d2D_dqidqj = (D_qi_qj - self.zfs_relaxed) / (q_i * q_j) - dD_qi / q_j - dD_qj / q_i

            # The pair displaces Q_i + Q_j, so its ZFS change is the first-order
            # response dD_i q_i + dD_j q_j plus a small remainder (the curvature).
            # A remainder that is a sizeable multiple of that linear response
            # (|alpha| > INCONSISTENT_ALPHA) means the run was displaced too far,
            # its result is stale, or it cancels the response; not physical curvature.
            if i in self.treated_modes and j in self.treated_modes:
                linear = dD_qi * q_i + dD_qj * q_j
                remainder = (D_qi_qj - self.zfs_relaxed) - linear
                norm = np.sum(linear * linear)
                if norm > (0.05 * CONSTANTS["MHz2J"]) ** 2:
                    alpha = float(np.sum(remainder * linear) / norm)
                    if abs(alpha) > INCONSISTENT_ALPHA:
                        self.inconsistent_pairs.append(
                            (i, j, alpha, float(np.linalg.norm(remainder) / CONSTANTS["MHz2J"]))
                        )
                        if self.exclude_inconsistent_pairs:
                            continue

            self._check_symmetry(d2D_dqidqj, (sym_i, sym_j), (i, j))

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
                V_0_pm_2nd[i, j] = np.sqrt(d2D_dqidqj[0, 2] ** 2 + d2D_dqidqj[1, 2] ** 2) / (2.0 * np.sqrt(2))
                V_p_m_2nd[i, j] = 0.5 * np.sqrt(diff_in_plane**2 + 4.0 * off_diag_in_plane**2)

            if len(self.treated_modes) < n_modes and len(phonon_energies) == n_modes and i == j:
                # Inherit V-coefficients for the degenerate twin via pair_id.
                # The spectrum owns the pairing (strict involution); no frequency guessing.
                spectrum = self.spectrum
                if spectrum is not None and (twin := twin_of(spectrum.deg_groups, i)) is not None:
                    if twin != j:  # avoid overwriting an (i,j) cross-coupling with the pair copy
                        V_0_0_2nd[twin, twin] = V_0_0_2nd[i, j]
                        V_0_pm_2nd[twin, twin] = V_0_pm_2nd[i, j]
                        V_p_m_2nd[twin, twin] = V_p_m_2nd[i, j]
                    else:
                        # Twin IS j: copy the (i,i) pair coefficients to (j,j)
                        V_0_0_2nd[j, j] = V_0_0_2nd[i, j]
                        V_0_pm_2nd[j, j] = V_0_pm_2nd[i, j]
                        V_p_m_2nd[j, j] = V_p_m_2nd[i, j]

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

        if self.inconsistent_pairs:
            action = "Excluded" if self.exclude_inconsistent_pairs else "Kept (--keep-inconsistent-pairs)"
            LOGGER.warning(
                f"{action} {len(self.inconsistent_pairs)} of {len(self.zfs_tensors_2d)} second-order entries "
                f"whose ZFS change departs from the first-order response by a multiple of it (|alpha| > "
                f"{INCONSISTENT_ALPHA}: over-displaced, stale or inconsistent runs). Run folders (1-based): "
                + ", ".join(f"{i + 1}_{j + 1} (alpha={a:.2f})" for i, j, a, _ in self.inconsistent_pairs)
            )

        LOGGER.info("Symmetry adjusted coefficients (2nd order):")
        LOGGER.info(f"V_00:  {np.sum(V_0_0_2nd > 0)}")
        LOGGER.info(f"V_pm:  {np.sum(V_p_m_2nd > 0)}")
        LOGGER.info(f"V_0pm: {np.sum(V_0_pm_2nd > 0)}")

        return zfs_2nd_derivs, V_0_0_2nd, V_p_m_2nd, V_0_pm_2nd

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
            LOGGER.info(f"Saved ZFS spin-phonon coupling data to: {out}")
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
                order=kwargs.get("order", 1 if first is not None else 2),
                ground_state_zfs=gs_tensor,
                eigen_rotation=self.eigen_rotation,
                first_order=first or {},
                second_order=second or {},
            )
            out = raw_obj.save(save_name)
            LOGGER.info(f"Saved raw ZFS dataset to: {out}")
            return out

        # 3. Fallback for generic arrays
        metadata = {
            "defect": self.defect,
            "cell_size": self.cell_size,
            "calc_method": self.calc_method,
            "pert_scale": self.pert_scale,
        }
        np.savez(save_name, **metadata, **kwargs)
        LOGGER.info(f"Saved ZFS data to: {save_name}")
        return save_name

    @staticmethod
    def _check_symmetry(d_tensor: np.ndarray, symmetry: Union[str, tuple[str, str]], idx: Any) -> None:
        sym_prod = MathUtils.calc_symmetry(*symmetry) if isinstance(symmetry, tuple) else [symmetry]
        if sym_prod == ["A1"]:
            diag = np.diag(d_tensor)
            d_xx, d_yy, d_zz = diag
            tensor_scale = np.max(np.abs(diag))
            dyn_tol = 1.0 * tensor_scale + 1e-4

            is_axial = np.abs(d_xx - d_yy) <= dyn_tol
            is_traceless_axial = np.abs(2 * d_xx + d_zz) <= dyn_tol
            off_diag_mask = ~np.eye(3, dtype=bool)
            off_diags_zero = np.all(np.abs(d_tensor[off_diag_mask]) <= dyn_tol)

            if not (is_axial and is_traceless_axial and off_diags_zero):
                LOGGER.warning(
                    f"Symmetry mismatch at index {idx} with symmetry {sym_prod} "
                    f"(threshold: {dyn_tol:.2e} | scale: {tensor_scale:.2f})"
                )

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
