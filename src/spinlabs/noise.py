"""Noise of the second-order couplings and couplings with it removed (for error bands on rates and T1).

The diagonal second differences of a dataset carry the SCF noise of its runs. Production takes the
magnitudes of the coupling components (|p| for V2_0_pm, |w| for V2_p_m), so the noise biases every
diagonal coupling upward, most for the many weakly coupled modes, and the rates with it.

The noise is measured from the dataset's own runs, where symmetry says the second derivative is zero:
  - A modes (A1, A2): their second derivative is pure A1, so the E-type parts are noise: the out-of-plane
    components (xz, yz) and the in-plane ones ((xx - yy)/sqrt 2, xy).
  - E modes aligned with a mirror, central data only: the mirror-antisymmetric part (one out-of-plane and
    one in-plane component). Forward differences put their odd-in-Q error there as well, so they are not used.
The noise is anisotropic (in-plane several times out-of-plane), so it is measured per orientation:
out-of-plane for Delta1 (V2_0_pm), in-plane for Delta2 (V2_p_m) and, as a stand-in, for Delta0 (V2_0_0),
whose noise no forbidden part shows. Each is sigma = median|x| / 0.674 (robust to the few runs with
larger errors).

denoised_couplings removes the noise expected in each diagonal coupling in quadrature,
V -> sqrt(max(V^2 - sigma_V^2, 0)): sigma_V is the noise of the components the coupling is made of, over
the mode's amplitude q^2; for modes the mirror projection treated (ZFSManager mirror_projection), only the
allowed component. The filled pairs follow the diagonals (E x E: sqrt(V_ii V_jj); A x E: sqrt(V00_aa V_ee)).
It is an estimate (the quadrature subtraction assumes noise of this size in every mode), for an error band
next to the production values, not a replacement for them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spinlabs.constants import CONSTANTS

# median |x| of a zero-mean Gaussian in units of its sigma
_MEDIAN_ABS = 0.6745
# central diagonals of modes aligned with their mirror at least this well count as E samples
_ALIGNED = 0.99


@dataclass(frozen=True)
class SecondOrderNoise:
    """Noise sigma of one component of the diagonal second difference (MHz, times the mode's q^2 in coupling units)."""

    out_of_plane: float  # xz, yz: Delta1
    in_plane: float  # (xx - yy)/sqrt 2, xy: Delta2 (and the Delta0 stand-in)
    n_out: int
    n_in: int
    # share of the runs VASP symmetrized (ISYM 2 or 3, from the package's recorded settings), None if unknown
    symmetrized_runs: float | None = None
    from_e_modes: bool = False  # whether aligned central E modes contributed samples

    def per_component(self, channel: str) -> tuple[float, float]:
        """(allowed, forbidden) noise in coupling units x q^2 (MHz) of one channel's diagonal coupling.

        Delta0: |zz - (xx + yy)/2| / 6, scalar (no forbidden part). Delta1: |p| / (2 sqrt 2), p = (xz, yz).
        Delta2: |w| / 4, w = (xx - yy, 2 xy): allowed xx - yy (noise sqrt 2 in-plane sigma), forbidden 2 xy.
        """
        if channel == "V2_0_0":
            return self.in_plane / 6.0, 0.0
        if channel == "V2_0_pm":
            return self.out_of_plane / (2 * np.sqrt(2)), self.out_of_plane / (2 * np.sqrt(2))
        if channel == "V2_p_m":
            return np.sqrt(2) * self.in_plane / 4.0, 2 * self.in_plane / 4.0
        raise ValueError(f"unknown channel {channel!r}")


def _second_difference(manager, i: int, item: dict) -> np.ndarray | None:
    """The diagonal second difference of mode i (J, D0 principal frame), not yet divided by q^2."""
    d0 = manager.zfs_relaxed
    if item.get("central"):
        return np.asarray(item["tensor"]) + np.asarray(item["tensor_minus"]) - 2 * d0
    first = manager.first_order.get(i)
    if first is None:
        return None
    return np.asarray(item["tensor"]) - 2 * np.asarray(first["tensor"]) + d0


def second_difference_noise(manager) -> SecondOrderNoise:
    """The noise of a dataset's diagonal second differences, from the parts symmetry sets to zero."""
    labels = [str(s) for s in manager.spectrum.symmetries]
    rot = manager.eigen_rotation if manager.eigen_rotation is not None else np.eye(3)
    parity = manager._mirror_parities() if any(lab in ("Ex", "Ey") for lab in labels) else {}
    out_samples: list[float] = []
    in_samples: list[float] = []
    from_e = False  # whether aligned central E modes contributed samples
    for (i, j), item in manager.zfs_tensors_2d.items():
        if i != j:
            continue
        d = _second_difference(manager, i, item)
        if d is None:
            continue
        if labels[i] in ("A1", "A2"):
            out_samples += [d[0, 2], d[1, 2]]
            in_samples += [(d[0, 0] - d[1, 1]) / np.sqrt(2), d[0, 1]]
        elif labels[i] in ("Ex", "Ey") and item.get("central") and i in parity and abs(parity[i][1]) >= _ALIGNED:
            M = rot.T @ parity[i][0] @ rot
            anti = 0.5 * (d - M @ d @ M.T)  # one out-of-plane and one in-plane component, in a frame-free form
            out_samples.append(np.hypot(anti[0, 2], anti[1, 2]))
            in_samples.append(0.5 * np.hypot(anti[0, 0] - anti[1, 1], 2 * anti[0, 1]))  # |w_anti| = 2 |xy|
            from_e = True
    if not out_samples or not in_samples:
        raise ValueError("no symmetry-forbidden parts to measure the noise from (no A modes or aligned central E modes)")
    mhz = CONSTANTS["MHz2J"]
    noise = SecondOrderNoise(
        out_of_plane=float(np.median(np.abs(out_samples)) / _MEDIAN_ABS / mhz),
        in_plane=float(np.median(np.abs(in_samples)) / _MEDIAN_ABS / mhz),
        n_out=len(out_samples),
        n_in=len(in_samples),
        symmetrized_runs=_symmetrized_share(manager),
        from_e_modes=from_e,
    )
    if not from_e and (noise.symmetrized_runs or 0) > 0.5:
        from spinlabs.logging_utils import get_logger

        get_logger(__name__).warning(
            f"second-order noise measured on A modes only, and {noise.symmetrized_runs:.0%} of the runs were "
            "symmetrized by VASP (ISYM 2/3), which can keep an A-mode displacement's forbidden parts near zero: "
            "the noise may be understated"
        )
    return noise


def _symmetrized_share(manager) -> float | None:
    """Share of a dataset's perturbation runs with ISYM 2 or 3, from the settings spinlabs package recorded."""
    settings = getattr(manager.raw_data, "vasp_settings", None) if manager.raw_data is not None else None
    if isinstance(settings, str):
        import json

        try:
            settings = json.loads(settings)
        except ValueError:
            return None
    if not isinstance(settings, dict):
        return None
    counts: dict[str, int] = {}
    for tree in (settings.get("trees") or {}).values():
        for value, count in ((tree.get("runs") or {}).get("ISYM") or {}).items():
            counts[str(value)] = counts.get(str(value), 0) + int(count)
    total = sum(counts.values())
    return sum(c for v, c in counts.items() if v in ("2", "3")) / total if total else None


def mode_amplitudes(manager) -> np.ndarray:
    """Each mode's amplitude q of its diagonal second-order run (SI); twins without a run take their partner's."""
    n = manager.spectrum.n_modes
    q = np.zeros(n)
    for (i, j), item in manager.zfs_tensors_2d.items():
        if i == j and item.get("pert") and item["pert"][0]:
            q[i] = float(item["pert"][0])
    for group in manager.spectrum.deg_groups or []:
        known = [q[k] for k in group if q[k]]
        for k in group:
            q[k] = q[k] or (known[0] if known else 0.0)
    return q


def denoised_couplings(manager, coupling: dict, noise: SecondOrderNoise | None = None) -> dict:
    """``coupling`` (a coupling npz as a dict, the manager's output) with the noise removed from V2_*.

    Every diagonal coupling loses its expected noise in quadrature; the filled pairs (sqrt(V_ii V_jj)) follow
    by sqrt(s_i s_j). Off-diagonal pairs the dataset computed keep their values: their own noise is not
    estimated, and the diagonals' factors do not apply to them. Modes the manager's mirror projection treated
    carry only the allowed component's noise.
    """
    noise = noise or second_difference_noise(manager)
    labels = [str(s) for s in manager.spectrum.symmetries]
    n = len(labels)
    q = mode_amplitudes(manager)
    projected = set(getattr(manager, "mirror_projected", [])) | set(getattr(manager, "mirror_projected_central", []))
    projected |= {i for i, _ in getattr(manager, "mirror_inherited", [])}
    for group in manager.spectrum.deg_groups or []:  # a twin without a run inherits its projected partner's value
        if any(k in projected for k in group):
            projected |= set(group)
    # q is in SI and the manager works in J; a coupling file may store the couplings in MHz instead
    stored = str(np.asarray(coupling.get("coupling_unit", "J"))) if "coupling_unit" in coupling else "J"
    if stored not in ("J", "MHz"):
        raise ValueError(f"couplings in {stored}: denoised_couplings handles J or MHz")
    to_stored = 1.0 if stored == "J" else 1.0 / CONSTANTS["MHz2J"]
    scale: dict[str, np.ndarray] = {}
    for channel in ("V2_0_0", "V2_0_pm", "V2_p_m"):
        allowed, forbidden = noise.per_component(channel)
        power = np.array([allowed**2 + (0.0 if k in projected else forbidden**2) for k in range(n)])
        # noise of the coupling (J): sigma in MHz -> J, over q^2; then into the file's unit
        sigma_v = np.where(q > 0, np.sqrt(power) * CONSTANTS["MHz2J"] / np.where(q > 0, q, 1.0) ** 2, 0.0) * to_stored
        diag = np.abs(np.diag(np.asarray(coupling[channel], dtype=float)))
        clean = np.sqrt(np.maximum(diag**2 - sigma_v**2, 0.0))
        scale[channel] = np.where(diag > 0, clean / np.where(diag > 0, diag, 1.0), 1.0)
    out = dict(coupling)
    for channel in ("V2_0_pm", "V2_p_m"):  # A modes enter Delta1/Delta2 through their Delta0 (A x E fill)
        vec = np.where([lab.startswith("A") for lab in labels], scale["V2_0_0"], scale[channel])
        out[channel] = np.asarray(coupling[channel], dtype=float) * np.sqrt(np.outer(vec, vec))
    out["V2_0_0"] = np.asarray(coupling["V2_0_0"], dtype=float) * np.sqrt(np.outer(scale["V2_0_0"], scale["V2_0_0"]))
    computed = [(i, j) for (i, j) in manager.zfs_tensors_2d if i != j]
    for channel in ("V2_0_0", "V2_0_pm", "V2_p_m"):
        original = np.asarray(coupling[channel], dtype=float)
        for i, j in computed:
            out[channel][i, j], out[channel][j, i] = original[i, j], original[j, i]
    return out
