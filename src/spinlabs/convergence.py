"""D vs displacement (ZFS) with fit residuals from a packed perturbation npz.

Loads the .npz written by pack_perturbation_runs.py, rotates each ZFS
tensor into the ground-state principal frame (fixed rotation built once
from the Q=0 tensor, as in spinlabs's ZFSManager), and plots the diagonal
components against the perturbation scale together with the residuals
from a linear fit.

Usage (anywhere, needs numpy + matplotlib):
    python plot_pert_convergence.py pert_runs.npz [-m MODE] [-o out.png]

With --quad, also fit y = a + b x + c x^2 per principal component and
report 2c (the second-order ZFS derivative d2D/dQ2) next to the linear
slope, so the quadratic-regression estimate can be cross-checked against
a forward-difference value computed from the two smallest-|Q| points.
When Q=0 is present (either as a pert in the npz or via the 'd0' key
written by pack_perturbation_runs.py --ground-state) it is included as
an anchored data point and a central second difference
(f(+Q) + f(-Q) - 2 f(0)) / Q^2 is reported per |Q| as well.
"""
from __future__ import annotations




import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from spinlabs.models import ZFSTensor

from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)


def principal_frame(D0: np.ndarray) -> np.ndarray:
    """Ground-state principal-axis rotation R, via spinlabs.ZFSTensor.

    ZFSTensor(D0).principal_components() returns (D_xx, D_yy, D_zz, R)
    in EPR convention (|D_zz| >= |D_yy| >= |D_xx|, traceless), with R's
    columns the ground-state eigenvectors. Built ONCE; every perturbed
    tensor is then rotated into this fixed frame (R.T @ D @ R), exactly
    as ZFSManager._ingest_raw_data does.
    """
    _, _, _, R = ZFSTensor(matrix=D0).principal_components()
    return R


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """OLS fit y = a + b x. Returns (a, b, y_fit, residuals)."""
    b, a = np.polyfit(x, y, 1)
    y_fit = a + b * x
    return a, b, y_fit, y - y_fit


def quad_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """OLS fit y = a + b x + c x^2. Returns (a, b, c, y_fit, residuals)."""
    c, b, a = np.polyfit(x, y, 2)
    y_fit = a + b * x + c * x**2
    return a, b, c, y_fit, y - y_fit


# --------------------------------------------------------------------------- #
# signal-to-noise and a safe perturbation amplitude
# --------------------------------------------------------------------------- #
SNR_TARGET = 3.0  # finite difference at least this many times its error (noise + D0 offset)
TRUNC_TOL = 0.1  # relative truncation error allowed (next Taylor term / estimate)
SIGNIFICANCE = 2.0  # a fitted coefficient below this many standard errors is unresolved


def traceless_parts(tensor: np.ndarray, R: np.ndarray) -> np.ndarray:
    """The 5 traceless components of ``tensor`` in the frame R, as (A1, E1..E4).

    A1 = (2 D_zz - D_xx - D_yy)/sqrt 6, E = ((D_xx - D_yy)/sqrt 2, D_xy, D_xz, D_yz):
    combinations with the noise of a single tensor element, so a per-element noise
    sigma applies to each of them.
    """
    d = R.T @ np.asarray(tensor, dtype=float) @ R
    return np.array([
        (2 * d[2, 2] - d[0, 0] - d[1, 1]) / np.sqrt(6),
        (d[0, 0] - d[1, 1]) / np.sqrt(2),
        d[0, 1],
        d[0, 2],
        d[1, 2],
    ])


def cubic_fit(perts: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit values(Q) = a + b Q + c Q^2 + d Q^3 to the Q != 0 points.

    The intercept is free and D0 is left out, so a is where the perturbed runs meet
    at Q=0 and a - D0 measures D0's offset from them. values is (n_Q, ...).
    Returns (coef (4, ...) = a, b, c, d; residuals at the Q != 0 points; design matrix).
    """
    perts = np.asarray(perts, dtype=float)
    keep = ~np.isclose(perts, 0.0)
    q = perts[keep]
    X = np.vander(q, 4, increasing=True)
    flat = values[keep].reshape(len(q), -1)
    coef = np.linalg.lstsq(X, flat, rcond=None)[0]
    resid = flat - X @ coef
    return coef.reshape((4,) + values.shape[1:]), resid.reshape(values[keep].shape), X


def pooled_noise(perts: list[np.ndarray], parts_by_mode: list[np.ndarray]) -> tuple[float, int]:
    """Noise sigma (MHz per tensor element) from the cubic-fit scatter of every mode's D(Q).

    Each mode's 5 traceless parts at Q != 0 are fitted with a cubic (cubic_fit); the
    residuals of all modes are pooled. Returns (sigma, degrees of freedom); (nan, 0)
    without 5+ amplitudes besides Q=0. Any response beyond Q^3 also lands in the
    residual, so this is an upper bound.
    """
    sq, dof = 0.0, 0
    for p, parts in zip(perts, parts_by_mode):
        n = int(np.sum(~np.isclose(p, 0.0)))
        if n < 5:
            continue
        sq += float(np.sum(cubic_fit(p, parts)[1] ** 2))
        dof += (n - 4) * parts.shape[1]
    return (float(np.sqrt(sq / dof)), dof) if dof else (float("nan"), 0)


def d0_offset(perts: list[np.ndarray], parts_by_mode: list[np.ndarray], d0_parts: np.ndarray) -> np.ndarray:
    """Offset (5 traceless parts, MHz) of D0 from where the perturbed runs meet at Q=0.

    The median over modes of the cubic fits' intercept minus D0. D0 is usually a
    separate ground-state run; any difference in its settings or convergence shows
    up here, the same for every mode, and enters every finite difference as
    offset/Q (first order) and offset/Q^2 (second order).
    """
    icpt = [cubic_fit(p, parts)[0][0] for p, parts in zip(perts, parts_by_mode)
            if np.sum(~np.isclose(p, 0.0)) >= 5]
    return np.median(np.array(icpt), axis=0) - d0_parts if icpt else np.zeros(5)


@dataclass
class SNRReport:
    """Noise, D0 offset, Taylor coefficients and the safe amplitude window of one mode.

    Coefficients are norms over the 5 traceless parts of D(Q) = a + b Q + c Q^2 + d Q^3,
    fitted to the perturbed runs. The production schemes measure against D0:
    first order (D(Q) - D0)/Q, with error hypot(sqrt 2 sigma, offset)/Q and
    truncation c Q, and second order (D(2Q) - 2 D(Q) + D0)/Q^2 (the i = i pair), with
    error hypot(sqrt 6 sigma, offset)/Q^2 and truncation 6 d Q on 2c. An unresolved
    coefficient is replaced by its upper bound, so the window's upper end is then a
    lower limit.
    """

    sigma: float
    sigma_source: str
    offset: float
    b: float
    c: float
    d: float
    c_resolved: bool
    d_resolved: bool
    q_lo1: float
    q_hi1: float
    q_lo2: float
    q_hi2: float
    target: float = SNR_TARGET
    tol: float = TRUNC_TOL
    b_resolved: bool = True

    @property
    def err1(self) -> float:
        """Error of D(Q) - D0 (MHz): noise of two runs and D0's offset."""
        return float(np.hypot(np.sqrt(2) * self.sigma, self.offset))

    @property
    def err2(self) -> float:
        """Error of D(2Q) - 2 D(Q) + D0 (MHz): noise of three runs and D0's offset."""
        return float(np.hypot(np.sqrt(6) * self.sigma, self.offset))

    def first_order(self, q) -> tuple[np.ndarray, np.ndarray]:
        """(noise + offset, truncation) relative errors of the first-order coefficient at q."""
        q = np.asarray(q, dtype=float)
        return self.err1 / (self.b * q), self._c_up * q / self.b

    def second_order(self, q) -> tuple[np.ndarray, np.ndarray]:
        """(noise + offset, truncation) relative errors of the diagonal second-order coefficient at q."""
        q = np.asarray(q, dtype=float)
        return self.err2 / (2 * self.c * q**2), 3 * self._d_up * q / self.c

    @property
    def _c_up(self) -> float:
        return self.tol * self.b / self.q_hi1

    @property
    def _d_up(self) -> float:
        return self.tol * self.c / (3 * self.q_hi2)


def snr_report(
    perts: np.ndarray,
    parts: np.ndarray,
    sigma: float | None = None,
    sigma_source: str = "",
    offset: np.ndarray | None = None,
    target: float = SNR_TARGET,
    tol: float = TRUNC_TOL,
) -> SNRReport | None:
    """Fit D(Q) per traceless part with a cubic and derive SNR and the safe Q window.

    perts must contain Q=0 (D0); parts is (n_Q, 5) from traceless_parts. The cubic is
    fitted to the other points (cubic_fit). Without ``sigma`` the noise is its residual,
    without ``offset`` (5 parts) D0's offset is this mode's intercept minus D0.
    Returns None without D0 or with fewer than 5 other amplitudes.
    """
    perts = np.asarray(perts, dtype=float)
    zero = np.isclose(perts, 0.0)
    if not zero.any() or np.sum(~zero) < 5:
        return None
    coef, resid, X = cubic_fit(perts, parts)
    if sigma is None:
        dof = (int(np.sum(~zero)) - 4) * parts.shape[1]
        sigma = float(np.sqrt(np.sum(resid**2) / dof))
        sigma_source = f"this mode's cubic-fit residuals, {dof} dof"
    if offset is None:
        offset = coef[0] - parts[zero][0]
    delta = float(np.linalg.norm(offset))
    se = sigma * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
    # the norm over 5 parts of pure-noise coefficients is ~ sqrt 5 standard errors
    floor = SIGNIFICANCE * se * np.sqrt(parts.shape[1])
    b, c, d = np.linalg.norm(coef[1:], axis=1)
    c_up, d_up = max(c, floor[2]), max(d, floor[3])
    err1, err2 = np.hypot(np.sqrt(2) * sigma, delta), np.hypot(np.sqrt(6) * sigma, delta)
    # a response that stays within noise + D0 offset even at the largest |Q| is ~ 0
    qmax = float(np.max(np.abs(perts)))
    return SNRReport(
        sigma=sigma,
        sigma_source=sigma_source,
        offset=delta,
        b=float(b),
        c=float(c),
        d=float(d),
        b_resolved=bool(b > floor[1] and b * qmax > SIGNIFICANCE * err1),
        c_resolved=bool(c > floor[2] and 2 * c * qmax**2 > SIGNIFICANCE * err2),
        d_resolved=bool(d > floor[3]),
        q_lo1=float(target * err1 / b),
        q_hi1=float(tol * b / c_up),
        q_lo2=float(np.sqrt(target * err2 / (2 * c))),
        q_hi2=float(tol * c / (3 * d_up)),
        target=target,
        tol=tol,
    )


def format_snr(report: SNRReport, qs: list[float]) -> str:
    """Text block: noise, coefficients, per-|Q| SNR and truncation, and the safe windows."""
    r = report
    lines = [
        f"noise sigma {r.sigma:.3g} MHz per tensor element ({r.sigma_source}); D0 offset {r.offset:.3g} MHz",
        f"|b| {r.b:.4g} MHz/Q, |c| {r.c:.4g} MHz/Q^2{'' if r.c_resolved else ' (unresolved)'}, "
        f"|d| {r.d:.4g} MHz/Q^3{'' if r.d_resolved else ' (unresolved)'}",
        "  SNR = coefficient / (noise + D0 offset)",
        f"  {'|Q|':>8}  {'SNR 1st':>9}  {'trunc 1st':>10}  {'SNR 2nd':>9}  {'trunc 2nd':>10}",
    ]
    for q in qs:
        n1, t1 = r.first_order(q)
        n2, t2 = r.second_order(q)
        lines.append(
            f"  {q:>8.4g}  {1 / n1:>9.3g}  {'<' if not r.c_resolved else ' '}{t1:>9.2%}"
            f"  {1 / n2:>9.3g}  {'<' if not r.d_resolved else ' '}{t2:>9.2%}"
        )

    def window(lo: float, hi: float, bound: bool) -> str:
        if lo <= hi:
            return f"{lo:.3g} .. {'>= ' if bound else ''}{hi:.3g}"
        if bound:  # the truncation bound is only what the noise allows; the window may still exist
            return f">= {lo:.3g}; upper end unknown (next Taylor term not resolved by these amplitudes)"
        return f"none (SNR {r.target:g} needs |Q| >= {lo:.3g}, truncation exceeds {r.tol:.0%} above {hi:.3g})"

    lines.append(f"safe |Q| (SNR >= {r.target:g}, truncation <= {r.tol:.0%}; '<' marks an upper bound):")
    lines.append(f"  first order : {window(r.q_lo1, r.q_hi1, not r.c_resolved)}")
    lines.append(f"  second order: {window(r.q_lo2, r.q_hi2, not r.d_resolved)}")
    return "\n".join(lines)


def one_sided_d2(p: np.ndarray, v: np.ndarray) -> float | None:
    """Second derivative from the three smallest-|Q| points (Lagrange).

    Evaluated at the smallest-|Q| point (closest to Q=0):
      f''(x0) = 2 [ f0/(h1(h1+h2)) - f1/(h1 h2) + f2/(h2(h1+h2)) ]
    with h1 = p1-p0, h2 = p2-p1. Returns None if fewer than 3 points or
    the points are collinear/not strictly ordered in |Q| spacing.
    """
    if len(p) < 3:
        return None
    srt = np.argsort(np.abs(p))
    p_s, v_s = p[srt], v[srt]
    if np.isclose(p_s[0], 0.0) and ((p_s > 0).sum() >= 2 or (p_s < 0).sum() >= 2):
        # with a Q=0 anchor: triple (0, Q1, Q2) on whichever side has points,
        # evaluated at Q=0 itself
        pos_idx = np.where(p_s > 0)[0][:2]
        neg_idx = np.where(p_s < 0)[0][:2]
        idx = np.concatenate(([0], pos_idx if len(pos_idx) == 2 else neg_idx))
    else:
        # no anchor: three consecutive same-side points, evaluated at the
        # smallest-|Q| one
        side = p_s > 0 if (p_s > 0).sum() >= 3 else p_s < 0
        idx = np.where(side)[0][:3]
    if len(idx) < 3:
        return None
    p3, v3 = p_s[idx], v_s[idx]
    h1, h2 = p3[1] - p3[0], p3[2] - p3[1]
    if h1 <= 0 or h2 <= 0:
        return None
    return 2 * (v3[0] / (h1 * (h1 + h2)) - v3[1] / (h1 * h2) + v3[2] / (h2 * (h1 + h2)))


PART_LABELS = ("A1", "(xx−yy)/√2", "xy", "xz", "yz")


def _plot_finite_differences(axes, perts: np.ndarray, parts: np.ndarray, report: SNRReport) -> None:
    """Bottom row: the measured finite differences of each traceless part vs |Q|.

    The 5 traceless parts (traceless_parts) hold the E response of E modes too, which
    sits in the off-diagonal elements the principal-value panels above do not show.
    Parts with < 2% of the largest response are left out. All differences are taken
    against D0, as in the coupling calculation. Error bars are the noise (sqrt 2
    sigma/Q first order, sqrt 6 sigma/Q^2 second order); dashed lines are the Q -> 0
    values b and 2c of the cubic fitted to the perturbed runs. A point drifting off
    its line at large |Q| is truncation; at small |Q| it is D0's offset (growing as
    1/Q, 1/Q^2) or noise (wide bars).
    """
    f0 = parts[np.isclose(perts, 0.0)][0]
    coef = cubic_fit(perts, parts)[0]  # (4, 5): a, b, c, d per part
    qmax = float(np.max(np.abs(perts)))
    sigma = report.sigma
    shown1 = np.abs(coef[1]) >= 0.02 * np.max(np.abs(coef[1]))
    shown2 = np.abs(coef[2]) >= 0.02 * np.max(np.abs(coef[2]))
    for k in range(parts.shape[1]):
        color = f"C{k}"
        label = PART_LABELS[k]
        # first order (D(Q) - D0)/Q on both sides
        for side, face in ((1, color), (-1, "white")) if shown1[k] else ():
            q = np.sort(np.abs(perts[np.sign(perts) == side]))
            if not len(q):
                continue
            vals = [(parts[np.isclose(perts, side * x)][0, k] - f0[k]) / (side * x) for x in q]
            axes[0].errorbar(q, vals, yerr=np.sqrt(2) * sigma / q, fmt="o", color=color, mfc=face, ms=5, capsize=3,
                             label=f"{label} {'+Q' if side > 0 else '−Q'}")
        if shown1[k]:
            axes[0].axhline(coef[1, k], color=color, ls="--", lw=0.8)
        if not shown2[k]:
            continue
        # second order: central at each |Q|, and the i = i pair scheme (step Q, uses 2Q)
        cq, cv = [], []
        for x in sorted({abs(float(v)) for v in perts if v > 0}):
            pos, neg = np.isclose(perts, x), np.isclose(perts, -x)
            if pos.any() and neg.any():
                cq.append(x)
                cv.append((parts[pos][0, k] + parts[neg][0, k] - 2 * f0[k]) / x**2)
        if cq:
            cq = np.array(cq)
            axes[1].errorbar(cq, cv, yerr=np.sqrt(6) * sigma / cq**2, fmt="o", color=color, ms=5, capsize=3,
                             label=f"{label} central")
        for side in (1, -1):
            fq, fv = [], []
            for x in sorted({abs(float(v)) for v in perts if np.sign(v) == side}):
                one, two = np.isclose(perts, side * x), np.isclose(perts, 2 * side * x)
                if one.any() and two.any():
                    fq.append(x)
                    fv.append((parts[two][0, k] - 2 * parts[one][0, k] + f0[k]) / x**2)
            if fq:
                fq = np.array(fq)
                axes[1].errorbar(fq * (1.04 if side > 0 else 0.96), fv, yerr=np.sqrt(6) * sigma / fq**2, fmt="s",
                                 color=color, mfc=color if side > 0 else "white", ms=5, capsize=3,
                                 label=f"{label} pair {'+' if side > 0 else '−'}Q, 2Q")
        axes[1].axhline(2 * coef[2, k], color=color, ls="--", lw=0.8)
    for ax, lo, hi, resolved, title in (
        (axes[0], report.q_lo1, report.q_hi1, report.c_resolved, "first order  (D(Q) − D0)/Q"),
        (axes[1], report.q_lo2, report.q_hi2, report.d_resolved, "second order  d²D/dQ² estimates"),
    ):
        right = 1.15 * qmax
        if lo < min(hi, right) or (not resolved and lo < right):
            ax.axvspan(max(lo, 0), right if not resolved else min(hi, right), color="#7fb77e", alpha=0.2, lw=0)
        window = f"{lo:.3g}–{'≥' if not resolved else ''}{hi:.3g}" if lo <= hi or not resolved else "none"
        ax.set_title(f"{title}\nsafe |Q|: {window}  (SNR ≥ {report.target:g}, trunc ≤ {report.tol:.0%})", fontsize=9)
        ax.set_xlim(0, right)
        ax.set_xlabel("|Q|  (dashed: Q→0 value of the cubic fit; bars: noise ±1σ)", fontsize=8)
        ax.axhline(0, color="#bbbbbb", lw=0.6)
        ax.legend(fontsize=6, ncol=2)
    axes[0].set_ylabel("traceless part (MHz/Q, MHz/Q²)")


def common_window(reports: dict[int, SNRReport]) -> str:
    """One |Q| range for all modes: the largest lower end and the smallest upper end.

    A mode whose coefficient is not resolved from zero (e.g. no first-order coupling
    by symmetry) sets no lower end for that order, and an unresolved next Taylor term
    sets no upper end.
    """
    lines = [
        "safe |Q| per mode and for all modes together:",
        f"  {'mode':>6}  {'first order':>22}  {'second order':>22}",
    ]

    def cell(lo: float, hi: float, signal: bool, next_term: bool) -> str:
        if not signal:
            return "coefficient ~ 0"
        return f"{lo:.3g} .. {hi:.3g}" if next_term else f"{lo:.3g} .. (>= {hi:.3g})"

    for m, r in reports.items():
        lines.append(
            f"  {m:>6}  {cell(r.q_lo1, r.q_hi1, r.b_resolved, r.c_resolved):>22}"
            f"  {cell(r.q_lo2, r.q_hi2, r.c_resolved, r.d_resolved):>22}"
        )
    for order, lo_key, hi_key, signal, next_term in (
        ("first order ", "q_lo1", "q_hi1", "b_resolved", "c_resolved"),
        ("second order", "q_lo2", "q_hi2", "c_resolved", "d_resolved"),
    ):
        los = {m: getattr(r, lo_key) for m, r in reports.items() if getattr(r, signal)}
        his = {m: getattr(r, hi_key) for m, r in reports.items() if getattr(r, signal) and getattr(r, next_term)}
        if not los:
            lines.append(f"  all modes, {order}: no mode has a resolved coefficient")
            continue
        lo_mode = max(los, key=los.get)
        lo_txt = f"{los[lo_mode]:.3g} (mode {lo_mode})"
        if his:
            hi_mode = min(his, key=his.get)
            hi_txt = f"{his[hi_mode]:.3g} (mode {hi_mode})"
            verdict = "OK" if los[lo_mode] <= his[hi_mode] else "no common |Q|"
        else:
            hi_txt, verdict = "not limited by these amplitudes", "OK"
        lines.append(f"  all modes, {order}: {lo_txt} .. {hi_txt}: {verdict}")
    return "\n".join(lines)


def read_conv_metadata(npz_path: Path) -> dict:
    """Return the JSON 'metadata' key of a packed convergence npz ({} if absent)."""
    data = np.load(npz_path)
    if "metadata" not in data:
        return {}
    try:
        return json.loads(str(data["metadata"]))
    except (ValueError, TypeError):
        return {}


PC_NAMES = ("xx", "yy", "zz")


def plot_pert_convergence(
    npz_path: Path,
    output_path: Path | None = None,
    mode: list[int] | None = None,
    quad: bool = False,
    deviation: bool = False,
    csv_path: Path | None = None,
    show: bool = False,
    drop_anchor: bool = False,
    noise: float | None = None,
    snr_target: float = SNR_TARGET,
    trunc_tol: float = TRUNC_TOL,
) -> list[Path]:
    """Plot ZFS principal components vs perturbation scale from a packed npz.

    With the Q=0 anchor and 5+ amplitudes a third row shows the finite differences
    the coupling calculation uses, evaluated at each measured |Q| with their noise
    error bars: first order (D(Q) - D0)/Q and second order, central
    (D(Q) + D(-Q) - 2 D0)/Q^2 and the i = i pair's (D(2Q) - 2 D(Q) + D0)/Q^2. The
    noise sigma is the cubic-fit scatter pooled over all modes of the npz (or
    ``noise``); the shaded |Q| range has SNR >= snr_target and truncation <= trunc_tol,
    and the log ends with the range common to all modes.

    Returns the list of written output files.
    """
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("Error: matplotlib not available")

    data = np.load(npz_path)
    modes = data["modes"]
    perts = data["perts"]
    tensors = data["tensors"]

    # Anchor the Q=0 point: take it from the perts array if present, else
    # from the 'd0' key written by pack_perturbation_runs.py --ground-state.
    if 0.0 not in set(perts.tolist()):
        if "d0" in data:
            perts = np.concatenate(([0.0], perts))
            tensors = np.concatenate((np.broadcast_to(data["d0"], (tensors.shape[0], 1, 3, 3)), tensors), axis=1)
        else:
            LOGGER.warning(
                "no Q=0 point (no 0 in perts, no 'd0' key in npz); fits use the unanchored points only",
            )

    # Q=0 is the relaxed structure, so the ZFS tensor there is identical for
    # every mode: if a mode lacks its own Q=0 run (absent column or NaN),
    # borrow the Q=0 tensor from whichever mode has one.
    izero_g = int(np.argmin(np.abs(perts)))
    q0_shared = None
    if np.isclose(perts[izero_g], 0.0):
        for i in range(tensors.shape[0]):
            t0 = tensors[i][izero_g]
            if not np.isnan(t0).any():
                q0_shared = t0
                break
        if q0_shared is not None:
            patched = 0
            for i in range(tensors.shape[0]):
                if np.isnan(tensors[i][izero_g]).any():
                    tensors[i][izero_g] = q0_shared
                    patched += 1
            if patched:
                LOGGER.info(
                    f"borrowed the Q=0 tensor for {patched} mode(s) that lack "
                    "their own pert=0 run (identical relaxed structure)",
                )

    wanted = mode if mode else [int(m) for m in modes]
    missing = [m for m in wanted if m not in set(modes.tolist())]
    if missing:
        LOGGER.warning(f"modes {missing} not in npz")
    if output_path is None:
        output_path = npz_path.with_name(npz_path.stem + "_convergence.png")

    csv_rows: list[tuple] = []
    out_files: list[Path] = []
    reports: dict[int, SNRReport] = {}

    # noise and D0's offset from the convergence runs themselves, pooled over all modes
    offset = None
    noise_source = "--noise"
    if q0_shared is not None:
        frame = principal_frame(q0_shared)
        finished = [~np.isnan(tensors[i]).any(axis=(1, 2)) for i in range(tensors.shape[0])]
        p_modes = [perts[ok] for ok in finished]
        parts_modes = [np.stack([traceless_parts(t, frame) for t in tensors[i][ok]]) for i, ok in enumerate(finished)]
        if noise is None:
            noise, n_dof = pooled_noise(p_modes, parts_modes)
            noise_source = f"cubic-fit scatter of all {tensors.shape[0]} modes, {n_dof} dof"
        offset = d0_offset(p_modes, parts_modes, traceless_parts(q0_shared, frame))
        LOGGER.info(
            f"noise sigma {noise:.3g} MHz per tensor element ({noise_source}); D0 sits "
            f"{np.linalg.norm(offset):.3g} MHz from where the perturbed runs meet at Q=0 "
            f"(A1, E parts: {np.array2string(offset, precision=4)})"
        )

    for mode in wanted:
        if mode not in set(modes.tolist()):
            continue
        i = int(np.where(modes == mode)[0][0])
        # drop unfinished runs (NaN tensors) BEFORE diagonalizing: eigh on a
        # NaN slot raises LinAlgError for the whole mode
        ok0 = ~np.isnan(tensors[i]).any(axis=(1, 2))
        dropped = perts[~ok0]
        if len(dropped):
            LOGGER.warning(
                f"mode {mode}: dropping {len(dropped)} run(s) with no ZFS tensor "
                f"(unfinished or failed): {np.array2string(dropped)}",
            )
        if ok0.sum() < 2:
            LOGGER.warning(f"mode {mode}: fewer than 2 finished runs, skipping")
            continue
        p_all, t_all = perts[ok0], tensors[i][ok0]
        # fixed principal frame from the Q=0 (relaxed) tensor; plot the
        # diagonal components of every tensor rotated into that frame
        izero = int(np.argmin(np.abs(p_all)))
        if not np.isclose(p_all[izero], 0.0):
            LOGGER.warning(
                f"mode {mode}: warning: no Q=0 tensor; principal frame taken "
                "from the smallest-|Q| perturbation instead",
            )
        R = principal_frame(t_all[izero])
        # reference for --deviation: the Q=0 values BEFORE any drop, so the
        # subtraction stays correct even when --drop-anchor removes the anchor
        anchor_vals = np.diag(R.T @ t_all[izero] @ R) if deviation else None
        p_use = p_all
        t_use = t_all
        if drop_anchor:
            p_use = p_all[np.abs(p_all) > 0]
            t_use = t_all[np.abs(p_all) > 0]
            if len(p_use) == 0:
                LOGGER.warning(f"mode {mode}: --drop-anchor leaves no points, skipping")
                continue
        vals = np.stack([np.diag(R.T @ t @ R) for t in t_use])
        trace = vals.sum(axis=1)
        if deviation:
            # plot D(Q) - D(Q=0) so the response is visible on its own scale
            vals = vals - anchor_vals
        # plot order: ascending signed value, so lines connect monotonically
        # instead of zigzagging between +Q and -Q
        sgn = np.argsort(p_use)

        report = None
        if noise is not None and np.isfinite(noise) and np.isclose(p_all, 0.0).any():
            parts = np.stack([traceless_parts(t, R) for t in t_all])
            report = snr_report(p_all, parts, noise, noise_source, offset, snr_target, trunc_tol)
        ncols = 2
        nrows = 3 if report is not None else 2
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 9 if report else 6), gridspec_kw={"height_ratios": [3, 1, 2.4][:nrows]}
        )
        for j in range(ncols):
            axes[0, j].sharex(axes[1, j])
            axes[0, j].tick_params(labelbottom=False)
            if report is not None:
                axes[1, j].axhspan(-report.sigma, report.sigma, color="#d9d9d9", alpha=0.6, lw=0)
        trace_labeled = False
        for k in range(3):
            a, b, y_lin, res = linear_fit(p_use, vals[:, k])
            qa, qb, qc, y_q, qres = quad_fit(p_use, vals[:, k])
            axes[0, 0].plot(p_use[sgn], vals[sgn, k], "o", label=rf"$D_{{\mathrm{{{PC_NAMES[k]}}}}}")
            axes[0, 0].plot(p_use[sgn], y_lin[sgn], "-", alpha=0.4, color=axes[0, 0].lines[-1].get_color())
            axes[1, 0].plot(p_use[sgn], res[sgn], "o-")
            axes[1, 0].axhline(np.max(np.abs(res)), ls="--", lw=0.8, color="grey")
            axes[1, 0].axhline(-np.max(np.abs(res)), ls="--", lw=0.8, color="grey")
            trace_label = "Tr D" if not trace_labeled else None
            trace_labeled = True
            axes[0, 0].plot(p_use[sgn], trace[sgn], "-", lw=0.8, color="gray", label=trace_label)
            axes[0, 1].plot(p_use[sgn], trace[sgn], "-", lw=0.8, color="gray", label=trace_label)
            pc_label = rf"$D_{{\mathrm{{{PC_NAMES[k]}}}}}$ (b={b:.4g}, 2c={2 * qc:.4g})"
            axes[0, 1].plot(p_use[sgn], vals[sgn, k], "o", label=pc_label)
            axes[0, 1].plot(p_use[sgn], y_q[sgn], "-", alpha=0.4, color=axes[0, 1].lines[-1].get_color())
            axes[1, 1].plot(p_use[sgn], qres[sgn], "o-")
            axes[1, 1].axhline(np.max(np.abs(qres)), ls="--", lw=0.8, color="grey")
            axes[1, 1].axhline(-np.max(np.abs(qres)), ls="--", lw=0.8, color="grey")
            row: list = [mode, k + 1, f"{b:.6g}", f"{np.max(np.abs(res)):.4g}"]
            msg = (
                f"mode {mode} PC {k + 1}: slope {b:.4g} MHz/unit-pert, max |res| {np.max(np.abs(res)):.4g} MHz (signed)"
            )
            if quad:
                qa, qb, qc, _, qres = quad_fit(p_use, vals[:, k])
                d2 = 2 * qc
                fwd = one_sided_d2(p_use, vals[:, k])
                fwd_txt = f"{fwd:.4g}" if fwd is not None else "n/a"
                span = float(np.ptp(vals[:, k])) or 1e-30
                rel = (
                    f" ({abs(d2 - fwd) / max(abs(fwd), 1e-30) * 100:.2f}% from fwd)"
                    if fwd is not None and abs(fwd) > 0.05 * span
                    else ""
                )
                msg += f"\n    quad: D(Q) = {qa:.4g} + {qb:.4g} Q + {qc:.4g} Q^2 (dof {len(p_use) - 3})"
                msg += f"\n    d2D/dQ2: regression {d2:.4g}, one-sided fwd {fwd_txt} MHz/pert^2{rel}"
                row += [f"{d2:.6g}", fwd_txt]
                # central second differences (f(+Q) + f(-Q) - 2 f(0)) / Q^2
                if 0.0 in set(p_use.tolist()):
                    f0 = vals[p_use == 0.0, k][0]
                    cents = []
                    for q in sorted({abs(x) for x in p_use if x > 0}):
                        neg, pos = np.isclose(p_use, -q), np.isclose(p_use, q)
                        if neg.any() and pos.any():
                            cent = (vals[pos, k][0] + vals[neg, k][0] - 2 * f0) / q**2
                            cents.append(f"{q:g}:{cent:.4g}")
                            row[-1] = f"{cent:.6g}"
                    if cents:
                        msg += f"\n    central d2 (|Q|:value): {', '.join(cents)} MHz/pert^2"
            LOGGER.info(20 * "-")
            LOGGER.info(msg)
            LOGGER.info(20 * "-")
            csv_rows.append(tuple(row))
        for j, title in enumerate(["ax + b", "ax² + bx + c"]):
            axes[0, j].set_title(title)
            axes[0, j].set_ylabel("principal value (MHz)")
            axes[1, j].set_xlabel("perturbation scale")
            axes[1, j].xaxis.set_major_locator(plt.MaxNLocator(5))
            axes[1, j].set_ylabel("residual (MHz)" + ("\n(grey: ±σ noise)" if report else ""))
        if report is not None:
            _plot_finite_differences(axes[2], p_all, parts, report)
            reports[mode] = report
            qs = sorted({abs(float(x)) for x in p_all if x != 0})
            LOGGER.info(f"mode {mode}: signal-to-noise\n" + format_snr(report, qs))
        fig.suptitle(f"mode {mode}: $D_\\lambda$" if not deviation else f"mode {mode}: $D_\\lambda - D_0$")
        axes[0, 1].legend(fontsize=7)
        fig.tight_layout()
        stem, ext = output_path.stem, output_path.suffix or ".png"
        out = output_path.with_name(f"{stem}_mode{mode}{ext}")
        fig.savefig(out, dpi=150)
        out_files.append(out)
        LOGGER.info(f"  -> {out}")
        if show:
            plt.show()

    if csv_path:
        header = ["mode", "pc", "slope", "max_abs_res"]
        if quad:
            header += ["d2_reg", "d2_3pt"]  # 3-pt = central diff if Q=0 present, else one-sided
        with csv_path.open("w") as fh:
            fh.write(",".join(header) + "\n")
            for row in csv_rows:
                fh.write(",".join(str(v) for v in row) + "\n")
        LOGGER.info(f"  -> {csv_path}")
    if reports:
        LOGGER.info(common_window(reports))
    return out_files
