"""Spin-phonon coupling channels from a convergence pack.

The three channels are the ms-difference channels of the production coupling
calculation:

    V_{Delta 0}   = V_0_0    (ms conserved)
    V_{Delta +-1} = V_0_pm   (one spin flip)
    V_{Delta +-2} = V_p_m    (two spin flips)

They are evaluated from the same forward (or central) finite differences the
convergence summary uses, on the convergence perturbation scale (MHz per unit
Q).  The production ZFSManager values are per SI displacement; converting them
needs the phonon spectrum and is deliberately not done here.

The safe windows reuse the conservative norm-based error of
convergence.difference_errors for every channel.  A channel's lower end is where
its fitted Q -> 0 coefficient reaches the SNR target; its upper end is where the
next Taylor term reaches the truncation tolerance.  Only symmetry-allowed
channels enter a common window; forbidden channels are still returned so the
GUI can draw them as a zero check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spinlabs.convergence import (
    SIGNIFICANCE,
    SNR_TARGET,
    TRUNC_TOL,
    cubic_fit,
    d0_offset,
    difference_errors,
    fit_points,
    helper_step,
    load_convergence,
    poly_fit,
    pooled_noise,
    principal_frame,
    traceless_parts,
)
from spinlabs.convergence_summary import WEAK_FRACTION
from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)

CHANNELS = ("d0", "d1", "d2")
CHANNEL_LABELS = {
    "d0": r"$V_{\Delta 0}$",
    "d1": r"$V_{\Delta \pm 1}$",
    "d2": r"$V_{\Delta \pm 2}$",
}
CHANNEL_NAMES = {"d0": "V_0_0", "d1": "V_0_pm", "d2": "V_p_m"}
CHANNEL_MARKERS = {"d0": "o", "d1": "^", "d2": "s"}
# Euclidean weight of each channel functional on the five traceless parts:
# V_0_0 = |A1|/sqrt 6, V_0_pm = ||(E3,E4)||/sqrt 2, V_p_m = sqrt(E1^2/2 + E2^2).
CHANNEL_WEIGHTS = {"d0": 1.0 / np.sqrt(6.0), "d1": 1.0 / np.sqrt(2.0), "d2": 1.0}
ORDER_NAMES = {1: "first order", 2: "second order"}
SCHEMES = ("forward", "central")


def channel_value(parts, channel: str):
    """One coupling channel from five traceless parts (A1, E1, E2, E3, E4).

    ``parts`` may be a (5,) vector or an array whose last axis has length 5.
    The formulas are the production ones in the ground-state principal frame.
    """
    p = np.asarray(parts, dtype=float)
    a, e1, e2, e3, e4 = p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4]
    if channel == "d0":
        return np.abs(a) / np.sqrt(6.0)
    if channel == "d1":
        return np.sqrt(e3**2 + e4**2) / np.sqrt(2.0)
    if channel == "d2":
        return np.sqrt(e1**2 / 2.0 + e2**2)
    raise ValueError(f"unknown coupling channel {channel!r} (known: {', '.join(CHANNELS)})")


def channel_values(parts) -> dict[str, float]:
    """All three channels of one five-part vector."""
    return {key: float(channel_value(parts, key)) for key in CHANNELS}


def allowed_channels(symmetry: str | None, order: int) -> frozenset[str]:
    """Channels the production symmetry rules allow for one mode and order.

    First order: A1 -> Delta 0, E -> Delta +-1 and +-2, A2 -> none.
    Second order: A1/A2 -> Delta 0, E -> all three (E x E contains A1).
    Unknown symmetry returns all three; callers should warn.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    sym = None if symmetry is None else str(symmetry)
    if sym is None or sym not in ("A1", "A2", "Ex", "Ey", "E"):
        return frozenset(CHANNELS)
    if order == 1:
        if sym == "A1":
            return frozenset({"d0"})
        if sym == "A2":
            return frozenset()
        return frozenset({"d1", "d2"})
    if sym in ("A1", "A2"):
        return frozenset({"d0"})
    return frozenset(CHANNELS)


@dataclass
class ChannelWindow:
    """One channel's safe |Q| window and flags."""

    q_lo: float = float("nan")
    q_hi: float = float("nan")
    upper_known: bool = False
    resolved: bool = False
    allowed: bool = True
    weak: bool = False
    reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.allowed and self.resolved and not self.weak and np.isfinite(self.q_lo) and np.isfinite(self.q_hi)


@dataclass
class ChannelSeries:
    """Measured channel values of one mode, order and scheme."""

    mode: int
    order: int
    channel: str
    scheme: str
    q: np.ndarray
    values: np.ndarray
    q0: float | None
    window: ChannelWindow


@dataclass
class ChannelMode:
    mode: int
    symmetry: str | None = None
    frequency: float | None = None
    series: list[ChannelSeries] = field(default_factory=list)
    allowed: dict[int, frozenset[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def get(self, order: int, channel: str, scheme: str) -> ChannelSeries | None:
        return next(
            (s for s in self.series if s.order == order and s.channel == channel and s.scheme == scheme), None
        )


@dataclass
class ChannelAnalysis:
    path: Path
    sigma: float
    offset: float
    modes: list[ChannelMode]
    errors: dict[str, tuple[float, float]]
    common: dict[tuple[int, str], tuple[float, float, bool] | None]
    warnings: list[str] = field(default_factory=list)
    sigma_source: str = ""

    def mode(self, number: int) -> ChannelMode | None:
        return next((m for m in self.modes if m.mode == number), None)


def _at(perts: np.ndarray, parts: np.ndarray, q: float):
    i = np.flatnonzero(np.isclose(perts, q))
    return parts[i[0]] if len(i) else None


def _standard_errors(sigma: float, X: np.ndarray) -> np.ndarray:
    """Standard errors of the polynomial coefficients (same for every one of the 5 parts).

    A rank-deficient design (duplicate steps) gives NaN floors instead of raising:
    the channels then show as unresolved rather than taking the whole tab down.
    """
    if X.size == 0:
        return np.full(X.shape[1], np.nan)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return np.full(X.shape[1], np.nan)
    return sigma * np.sqrt(np.diag(np.linalg.pinv(X.T @ X)))


def _window(
    *,
    channel: str,
    order: int,
    scheme: str,
    coef: np.ndarray,
    next_coef: np.ndarray,
    floor_se: float,
    next_se: float,
    sigma: float,
    offset: float,
    target: float,
    tol: float,
    qmax: float,
    allowed: bool,
) -> tuple[float | None, ChannelWindow]:
    """The Q -> 0 channel value and its safe window.

    ``coef`` is the fitted coefficient of the order's leading term (b for first,
    c for second); ``next_coef`` the next Taylor coefficient (c/d for first,
    d/e for second).  Error scalars are the norm-based difference errors.
    """
    weight = CHANNEL_WEIGHTS[channel]
    value = float(channel_value(coef, channel))
    window = ChannelWindow(allowed=allowed)
    if not allowed:
        window.reason = "symmetry-forbidden (display only)"
    err1, err2 = difference_errors(sigma, offset, central=(scheme == "central"))
    if not np.isfinite(value) or value <= 0 or qmax <= 0:
        window.reason = window.reason or "no resolved finite-difference coefficient"
        return (None if not np.isfinite(value) else value), window
    # The standard error of this channel's coefficient from the local Euclidean
    # weight.  It is the channel analogue of the norm's sqrt(5) * se floor.
    floor = SIGNIFICANCE * float(floor_se) * weight
    next_value = float(channel_value(next_coef, channel))
    next_floor = SIGNIFICANCE * float(next_se) * weight
    next_up = max(next_value, next_floor)
    upper_known = next_value >= next_floor
    if order == 1:
        window.resolved = bool(value > floor and value * qmax > SIGNIFICANCE * err1)
        window.q_lo = float(target * err1 / value)
        window.q_hi = float(tol * value / next_up) if scheme == "forward" else float(np.sqrt(tol * value / next_up))
    else:
        window.resolved = bool(value > floor and 2.0 * value * qmax**2 > SIGNIFICANCE * err2)
        window.q_lo = float(np.sqrt(target * err2 / (2.0 * value)))
        window.q_hi = (
            float(tol * value / (3.0 * next_up))
            if scheme == "forward"
            else float(np.sqrt(tol * value / next_up))
        )
    window.upper_known = bool(upper_known)
    if not window.resolved:
        window.reason = window.reason or "coefficient unresolved from noise"
    if not upper_known:
        window.reason = (window.reason + "; " if window.reason else "") + "next Taylor term unresolved (upper bound)"
    return value, window


def analyse_channels(
    path: Path,
    *,
    modes: set[int] | None = None,
    noise: float | None = None,
    target: float = SNR_TARGET,
    tol: float = TRUNC_TOL,
    weak_fraction: float = WEAK_FRACTION,
    schemes: tuple[str, ...] = SCHEMES,
    mode_info: dict[int, tuple[str | None, float | None]] | None = None,
) -> ChannelAnalysis:
    """Per-mode coupling-channel curves and safe windows of one convergence pack.

    ``mode_info`` maps a mode number to (symmetry, frequency in meV); missing
    entries are treated as unknown symmetry.  ``schemes`` selects forward and/or
    central differences.  Raises ValueError when the pack has no Q=0 anchor.
    """
    path = Path(path)
    conv = load_convergence(path)
    if conv.q0 is None:
        raise ValueError(f"{path} has no Q=0 point (no 0 in perts and no d0 key); cannot measure differences")
    frame = principal_frame(conv.q0)
    p0 = traceless_parts(conv.q0, frame)
    finished = [~np.isnan(t).any(axis=(1, 2)) for t in conv.tensors]
    perts = [conv.perts[ok] for ok in finished]
    parts = [np.stack([traceless_parts(t, frame) for t in tensors[ok]]) for tensors, ok in zip(conv.tensors, finished)]

    helper = helper_step(conv.perts)
    fit = [fit_points(p, helper) for p in perts]
    fit_perts = [p[f] for p, f in zip(perts, fit)]
    fit_parts = [x[f] for x, f in zip(parts, fit)]
    sigma, dof = pooled_noise(fit_perts, fit_parts)
    sigma_source = f"pooled, {dof} dof"
    if noise is not None:
        sigma, sigma_source = float(noise), "--noise"
    offset_vec = d0_offset(fit_perts, fit_parts, p0)
    offset = float(np.linalg.norm(offset_vec))
    errors = {scheme: difference_errors(sigma, offset, central=(scheme == "central")) for scheme in schemes}

    info = mode_info or {}
    selected = None if modes is None else {int(m) for m in modes}
    warnings: list[str] = []
    mode_results: list[ChannelMode] = []
    for i, mode in enumerate(conv.modes):
        mode_number = int(mode)
        if selected is not None and mode_number not in selected:
            continue
        p = perts[i]
        x = parts[i]
        sym, freq = info.get(mode_number, (None, None))
        result = ChannelMode(mode=mode_number, symmetry=sym, frequency=freq)
        result.allowed = {1: allowed_channels(sym, 1), 2: allowed_channels(sym, 2)}
        if sym is None:
            result.warnings.append("symmetry unknown; all three channels included conservatively")
        n_nonzero = int(np.sum(~np.isclose(p, 0.0)))
        if n_nonzero < 5:
            result.warnings.append("fewer than 5 perturbed runs; no channel fit or safe window")
        if "central" in schemes and n_nonzero < 6:
            result.warnings.append("fewer than 6 perturbed runs; no central channel fit")

        # (coef, se, leading index, next index, qmax) per (order, scheme).
        fits: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, int, int, float]] = {}
        if "forward" in schemes and n_nonzero >= 5:
            coef, _resid, X = cubic_fit(p[fit[i]], x[fit[i]])
            fits[(1, "forward")] = (coef, _standard_errors(sigma, X), 1, 2,
                                    float(np.max(np.abs(p[fit[i]]))))
            coef, _resid, X = cubic_fit(p, x)
            fits[(2, "forward")] = (coef, _standard_errors(sigma, X), 2, 3,
                                    float(np.max(np.abs(p))))
        if "central" in schemes and n_nonzero >= 6:
            coef, _resid, X = poly_fit(p, x, 4)
            se = _standard_errors(sigma, X)
            qmax = float(np.max(np.abs(p)))
            fits[(1, "central")] = (coef, se, 1, 3, qmax)
            fits[(2, "central")] = (coef, se, 2, 4, qmax)

        # Measured (|Q|, channel values) per order and scheme.
        measured: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
        no_fit_reason = {
            (1, "forward"): "not enough perturbed runs",
            (2, "forward"): "not enough 2Q runs",
            (1, "central"): "not enough +-Q pairs",
            (2, "central"): "not enough +-Q pairs",
        }
        for scheme in schemes:
            first_steps = sorted({abs(float(q)) for q in p[fit[i]] if not np.isclose(q, 0.0)})
            if scheme == "forward":
                qs, values = [], []
                for q in first_steps:
                    one = _at(p, x, q)
                    if one is None:
                        one = _at(p, x, -q)
                    if one is None:
                        continue
                    qs.append(q)
                    values.append([float(channel_value(one - p0, key) / q) for key in CHANNELS])
                measured[(1, scheme)] = (np.array(qs), np.array(values, dtype=float).reshape(-1, len(CHANNELS)))
                qs, values = [], []
                for q in sorted({abs(float(q)) for q in p if not np.isclose(q, 0.0)}):
                    plus, two_plus = _at(p, x, q), _at(p, x, 2 * q)
                    minus, two_minus = _at(p, x, -q), _at(p, x, -2 * q)
                    if plus is not None and two_plus is not None:
                        d2 = two_plus - 2 * plus + p0
                    elif minus is not None and two_minus is not None:
                        d2 = two_minus - 2 * minus + p0
                    else:
                        continue
                    qs.append(q)
                    values.append([float(channel_value(d2, key) / (2.0 * q**2)) for key in CHANNELS])
                measured[(2, scheme)] = (np.array(qs), np.array(values, dtype=float).reshape(-1, len(CHANNELS)))
            else:
                qs, first_values, second_values = [], [], []
                for q in sorted({abs(float(q)) for q in p if not np.isclose(q, 0.0)}):
                    plus, minus = _at(p, x, q), _at(p, x, -q)
                    if plus is None or minus is None:
                        continue
                    qs.append(q)
                    first_values.append(
                        [float(channel_value((plus - minus) / 2.0, key) / q) for key in CHANNELS]
                    )
                    second_values.append(
                        [float(channel_value(plus + minus - 2 * p0, key) / (2.0 * q**2)) for key in CHANNELS]
                    )
                measured[(1, scheme)] = (
                    np.array(qs), np.array(first_values, dtype=float).reshape(-1, len(CHANNELS))
                )
                measured[(2, scheme)] = (
                    np.array(qs), np.array(second_values, dtype=float).reshape(-1, len(CHANNELS))
                )

        for (order, scheme), (qs, values) in measured.items():
            if not len(qs):
                continue
            for k, channel in enumerate(CHANNELS):
                fit_info = fits.get((order, scheme))
                if fit_info is not None:
                    coef, se, lead, nxt, qmax = fit_info
                    q0, window = _window(
                        channel=channel, order=order, scheme=scheme, coef=coef[lead], next_coef=coef[nxt],
                        floor_se=se[lead], next_se=se[nxt], sigma=sigma, offset=offset, target=target, tol=tol,
                        qmax=qmax, allowed=channel in result.allowed[order],
                    )
                else:
                    q0, window = None, ChannelWindow(
                        allowed=channel in result.allowed[order], reason=no_fit_reason[(order, scheme)]
                    )
                result.series.append(
                    ChannelSeries(mode_number, order, channel, scheme, qs, values[:, k], q0, window)
                )
        result.series.sort(key=lambda s: (s.order, s.channel, s.scheme))
        mode_results.append(result)
        warnings.extend(f"mode {mode_number}: {w}" for w in result.warnings)

    # Weak-channel flags: compare each allowed channel with the strongest allowed
    # channel of its order.  Forbidden channels are zero in the production results.
    for order in (1, 2):
        strongest = 0.0
        for result in mode_results:
            for series in result.series:
                if series.order != order or not series.window.allowed or series.q0 is None:
                    continue
                strongest = max(strongest, abs(series.q0))
        if strongest <= 0:
            continue
        for result in mode_results:
            for series in result.series:
                if series.order != order or not series.window.allowed or series.q0 is None:
                    continue
                if abs(series.q0) < weak_fraction * strongest:
                    series.window.weak = True
                    series.window.reason = (
                        f"coefficient {abs(series.q0):.3g} is {abs(series.q0) / strongest:.1%} "
                        f"of the strongest allowed channel ({strongest:.3g})"
                    )

    # Common safe window per order and scheme over allowed, resolved, non-weak channels.
    common: dict[tuple[int, str], tuple[float, float, bool] | None] = {}
    for order in (1, 2):
        for scheme in schemes:
            windows = [
                s.window
                for result in mode_results
                for s in result.series
                if s.order == order and s.scheme == scheme and s.window.allowed
                and s.window.resolved and not s.window.weak and np.isfinite(s.window.q_lo)
            ]
            if not windows:
                common[(order, scheme)] = None
                continue
            lo = max(w.q_lo for w in windows)
            known = [w.q_hi for w in windows if w.upper_known and np.isfinite(w.q_hi)]
            hi = min(known) if known else float("nan")
            common[(order, scheme)] = (lo, hi, bool(known))

    return ChannelAnalysis(
        path=path, sigma=sigma, offset=offset, modes=mode_results, errors=errors, common=common,
        warnings=warnings, sigma_source=sigma_source,
    )
