"""One figure per defect from its convergence packs: finite differences and their safe step sizes.

It shows how the finite differences spinlabs takes from the ZFS tensor behave as the step
size changes. Each pack (one convergence folder, ``spinlabs package --conv``) is one column
of two rows:

- first order, |D(Q) - D0| / Q per mode against |Q| (filled +Q, open -Q), with noise error
  bars sqrt 2 sigma / Q and the cubic fit's Q -> 0 value b as a line;
- second order, the forward difference (D(2Q) - 2 D(Q) + D0) / Q^2 that
  ZFSManager.calculate_second_order_derivatives takes from the i = i pair run (D(2Q)) and the
  first-order run (D(Q)), for every Q whose 2Q was run on the same side (filled +Q, open -Q),
  with noise error bars sqrt 6 sigma / Q^2 and the fit's 2c as a line.

In both rows a band along each mode's line marks its safe |Q|: from where the finite
difference's SNR (b Q / err1, 2 c Q^2 / err2) reaches the target to where its truncation
error (c Q / b, 3 d Q / c) reaches the tolerance. The shaded column is the |Q| safe for every
mode of the pack (convergence.common_window), leaving out a mode whose finite difference stays
below the SNR target at every measured step: its coupling is too weak to matter (its band is faint).

Every norm is over the 5 traceless parts (convergence.traceless_parts), as the fit and the
SNR use them. The grey dashed curve in a quotient row is the error of that quotient with
D0's offset included, err1 / Q and err2 / Q^2: a point near or below it is noise, not a
measurement. A line drawn faint has its coefficient unresolved (b for first order, c for
second order): the fitted value is within noise of zero, so that order is not measurable at
these amplitudes and the line is no estimate of its size or SNR.

Noise sigma and D0's offset are pooled over all modes of a pack and the coefficients come
from convergence.snr_report, as the per-mode convergence plots report them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator

from spinlabs.convergence import (
    SNR_TARGET,
    TRUNC_TOL,
    d0_offset,
    load_convergence,
    pooled_noise,
    principal_frame,
    snr_report,
    traceless_parts,
)
from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)

# One categorical slot per mode, in pack order; more modes than slots are not plotted
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
METHODS = {"all": "all bands", "approx": "defect band approx"}  # pack metadata 'method' -> column title
STANDARD_KEYS = {"test_all", "test_occup"}  # folder suffixes the column title need not repeat
FAINT = 0.4  # alpha of an unresolved coefficient's lines and points
BAND_WIDTH, BAND_ALPHA = 10, 0.25  # a mode's safe |Q| band along its fit line
COMMON_COLOR = "#e5f4e4"  # the |Q| column safe for every mode
_ORDER_NAMES = {1: "first order", 2: "second order"}


def quotients(p, parts, p0):
    """The measured finite differences of one mode, as (signed Q, norm) arrays.

    first: |D(Q) - D0| / |Q| per run; second: the forward difference |D(2Q) - 2 D(Q) + D0| / Q^2
    per Q whose 2Q was run on the same side.
    """
    def at(q):
        i = np.flatnonzero(np.isclose(p, q))
        return parts[i[0]] if len(i) else None

    nonzero = p[~np.isclose(p, 0.0)]
    first = [(q, np.linalg.norm(at(q) - p0) / abs(q)) for q in nonzero]
    second = [(q, np.linalg.norm(at(2 * q) - 2 * at(q) + p0) / q**2) for q in nonzero if at(2 * q) is not None]
    return tuple(np.array(rows, dtype=float).reshape(-1, 2) for rows in (first, second))


def analyse(path, modes=None, noise=None, target=SNR_TARGET, tol=TRUNC_TOL):
    """Per mode of one pack: (mode, perts, first, second quotients, SNR report); plus noise and D0 offset.

    ``modes`` keeps only those modes (1-based, as packed); ``noise`` replaces the pooled sigma.
    """
    path = Path(path)
    conv = load_convergence(path)
    if conv.q0 is None:
        raise ValueError(f"{path} has no Q=0 point (no 0 in perts, no d0 key); cannot measure differences")
    frame = principal_frame(conv.q0)
    p0 = traceless_parts(conv.q0, frame)
    finished = [~np.isnan(t).any(axis=(1, 2)) for t in conv.tensors]
    perts = [conv.perts[ok] for ok in finished]
    parts = [np.stack([traceless_parts(t, frame) for t in tensors[ok]]) for tensors, ok in zip(conv.tensors, finished)]
    sigma, dof = pooled_noise(perts, parts)
    source = f"pooled, {dof} dof"
    if noise is not None:
        sigma, source = noise, "--noise"
    offset = d0_offset(perts, parts, p0)
    missing = sorted(set(modes or []) - {int(m) for m in conv.modes})
    if missing:
        LOGGER.warning(f"modes {missing} not in {path.name}; not plotted")
    rows = []
    for mode, p, mode_parts in zip(conv.modes, perts, parts):
        if modes and int(mode) not in modes:
            continue
        report = snr_report(p, mode_parts, sigma, source, offset, target, tol)
        if report is None:
            LOGGER.warning(f"{path.name} mode {mode}: too few finished amplitudes for an SNR estimate; not plotted")
            continue
        rows.append((int(mode), p, *quotients(p, mode_parts, p0), report))
    return rows, sigma, float(np.linalg.norm(offset))


def column_title(path) -> str:
    """'NV_512 all bands', plus the folder suffix when it is not the standard test_all/test_occup."""
    meta = load_convergence(path).metadata
    name = "_".join(str(meta[k]) for k in ("defect", "cell") if meta.get(k) is not None) or Path(path).stem
    key = meta.get("method_key")
    title = f"{name} {METHODS.get(meta.get('method'), key or '')}".strip()
    return f"{title} [{key}]" if key and key not in STANDARD_KEYS else title


def _registered_symmetries(defect: str, cell: int, n_needed: int) -> tuple[tuple[str, ...], Path] | None:
    """Symmetry labels of the first registered full phonon spectrum of defect_cell with n_needed modes or more."""
    from spinlabs.parsers import parse_phonon_data
    from spinlabs.registry import ensure_db, find_phonon
    from spinlabs.symmetry import classify_and_pair

    for path in find_phonon(ensure_db(), defect, cell):
        spectrum = parse_phonon_data(path)
        if spectrum.n_modes < n_needed:
            continue
        if spectrum.symmetries is None:
            classify_and_pair(spectrum)
        return tuple(str(s) for s in spectrum.symmetries), path
    return None


def mode_symmetries(path, modes, cache: dict | None = None) -> dict[int, str]:
    """Symmetry label per mode (1-based numbers in the full spectrum) from the defect's registered phonon data.

    Empty, with a warning, when the pack names no defect/cell or no registered spectrum covers the modes.
    ``cache`` keeps parsed spectra between the packs of one figure (a defect's columns share one).
    """
    cache = {} if cache is None else cache
    meta = load_convergence(path).metadata
    name = f"{meta.get('defect')}_{meta.get('cell')}"
    if not modes:
        return {}
    if meta.get("defect") is None or meta.get("cell") is None:
        LOGGER.warning(f"{Path(path).name} names no defect/cell; legend without mode symmetries")
        return {}
    key = (str(meta["defect"]), int(meta["cell"]), max(modes))
    if key not in cache:
        cache[key] = _registered_symmetries(*key)
    found = cache[key]
    if found is None:
        LOGGER.warning(
            f"no registered phonon data for {name} covers modes up to {max(modes)}; legend without mode symmetries "
            "(package the defect folder to register its phonon_data.npz)"
        )
        return {}
    labels, source = found
    LOGGER.debug(f"mode symmetries of {name} from {source}")
    return {m: labels[m - 1] for m in modes}


def safe_window(r, order: int) -> tuple[float, float, bool] | None:
    """(lower, upper, upper_known) of one mode's safe |Q| for this order, or None when its coefficient is unresolved.

    The lower end is where the finite difference reaches the SNR target, the upper end where its
    truncation error reaches the tolerance. With the next Taylor term unresolved the upper end is
    only a lower limit (``upper_known`` False): these amplitudes do not limit it.
    """
    if order == 1:
        return (r.q_lo1, r.q_hi1, r.c_resolved) if r.b_resolved else None
    return (r.q_lo2, r.q_hi2, r.d_resolved) if r.c_resolved else None


def below_target_everywhere(r, measured, order: int) -> bool:
    """Whether a mode's finite difference stays under the SNR target at every step that was measured.

    Such a mode is within noise at all these amplitudes: its coupling is too weak to matter next to
    the measured ones, so it does not get to narrow the |Q| safe for all modes.
    """
    q = np.unique(np.abs(np.asarray(measured, dtype=float)))
    if not len(q):
        return False
    relative_error = r.first_order(q)[0] if order == 1 else r.second_order(q)[0]
    return bool(np.all(1 / relative_error < r.target))


def shade_safe_windows(ax, rows, order: int, xlim: tuple[float, float], name: str) -> list[int]:
    """Each mode's safe |Q| as a band along its fit line, and the |Q| safe for every mode as a column.

    As convergence.common_window: the column runs from the largest lower end to the smallest
    known upper end; a mode whose upper end is not limited by these amplitudes reaches the axis.
    A mode below the SNR target at every measured step (below_target_everywhere) is left out of
    the column and its band drawn faint. Returns those neglected modes.
    """
    los, his, neglected = [], [], []
    for color, (mode, _p, first, second, r) in zip(COLORS, rows):
        window = safe_window(r, order)
        if window is None:
            continue
        lo, hi, known = window
        weak = below_target_everywhere(r, (first if order == 1 else second)[:, 0], order)
        if weak:
            neglected.append(mode)
            LOGGER.info(f"{name}: mode {mode} is below SNR {r.target:g} at every measured step; left out of the "
                        "|Q| safe for all modes")
        else:
            los.append(lo)
            if known:
                his.append(hi)
        start, end = max(lo, xlim[0]), min(hi, xlim[1]) if known else xlim[1]
        if start < end:
            level = r.b if order == 1 else 2 * r.c
            ax.plot([start, end], [level, level], lw=BAND_WIDTH, alpha=BAND_ALPHA * (FAINT if weak else 1),
                    color=color, solid_capstyle="butt", zorder=1)
        else:
            LOGGER.info(f"{name}: mode {mode} has no safe |Q| ({lo:.3g} .. {hi:.3g})")
    if not los:
        return neglected
    lo, hi = max(los), min(his, default=xlim[1])
    if lo < hi:
        ax.axvspan(max(lo, xlim[0]), min(hi, xlim[1]), color=COMMON_COLOR, zorder=0, lw=0)
        LOGGER.info(f"{name}: safe |Q| for all modes {lo:.3g} .. {hi:.3g}" + ("" if his else " (upper end not limited)"))
    else:
        LOGGER.info(f"{name}: no |Q| is safe for all modes ({lo:.3g} .. {hi:.3g})")
    return neglected


def plot_signed(ax, rows, marker, color, yerr, alpha, big=9, small=5):
    """Points at |Q|; nearly equal at +-Q, so the -Q ring first and a smaller +Q dot inside it."""
    for side, face, ms in ((-1, "white", big), (1, color, small)):
        sel = np.sign(rows[:, 0]) == side
        q = np.abs(rows[sel, 0])
        ax.errorbar(q, rows[sel, 1], yerr=yerr(q), fmt=marker, ms=ms, color=color, mfc=face, mew=1.5,
                    capsize=3, elinewidth=1, alpha=alpha)


def plot_convergence_summary(
    packs, output, *, modes=None, noise=None, target=SNR_TARGET, tol=TRUNC_TOL, dpi=200, show=False
) -> Path:
    """One figure, one column per pack (e.g. a defect's all-bands and defect-band packs); returns its path."""
    packs = [Path(p) for p in packs]
    # each column ticks its own pack's step sizes, so only the rows of one column share x
    fig, axes = plt.subplots(2, len(packs), figsize=(7.5 * len(packs), 9.5), squeeze=False, sharex="col")
    spectra: dict = {}  # registered phonon symmetries, parsed once per figure
    for col, path in enumerate(packs):
        rows, sigma, offset = analyse(path, modes, noise, target, tol)
        title = column_title(path)
        if len(rows) > len(COLORS):
            LOGGER.warning(
                f"{title}: {len(rows)} modes but {len(COLORS)} colours; only modes "
                f"{', '.join(str(r[0]) for r in rows[:len(COLORS)])} are plotted (pick others with --mode)"
            )
            rows = rows[:len(COLORS)]
        LOGGER.info(f"{title}: {len(rows)} modes, noise sigma {sigma:.3g} MHz, D0 offset {offset:.3g} MHz ({path})")
        symmetries = mode_symmetries(path, [r[0] for r in rows], spectra)
        first_ax, second_ax = axes[:, col]
        steps = np.concatenate([np.abs(p[~np.isclose(p, 0.0)]) for _, p, *_ in rows]) if rows else np.array([0.01, 0.3])
        q = np.geomspace(steps.min() / 2, steps.max() * 1.5, 200)
        for color, (mode, p, first, second, r) in zip(COLORS, rows):
            a1 = 1.0 if r.b_resolved else FAINT
            a2 = 1.0 if r.c_resolved else FAINT
            plot_signed(first_ax, first, "o", color, lambda x: np.sqrt(2) * sigma / x, a1)
            first_ax.axhline(r.b, color=color, lw=1, alpha=0.7 * a1)
            plot_signed(second_ax, second, "s", color, lambda x: np.sqrt(6) * sigma / x**2, a2)
            second_ax.axhline(2 * r.c, color=color, lw=1, alpha=0.7 * a2)
        # the quotient's error with D0's offset: points near or below it are not measured
        err1 = float(np.hypot(np.sqrt(2) * sigma, offset))
        err2 = float(np.hypot(np.sqrt(6) * sigma, offset))
        first_ax.plot(q, err1 / q, "--", color="0.35", lw=1)
        second_ax.plot(q, err2 / q**2, "--", color="0.35", lw=1)
        # log |Q| axis ticked at the step sizes that were run, not at decades
        run_steps = np.unique(np.round(steps, 12))
        for ax in (first_ax, second_ax):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(run_steps, [f"{s:g}" for s in run_steps])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(True, which="major", color="0.9", linewidth=0.6)
            ax.set_xlim(q[0], q[-1])
        neglected = {
            order: shade_safe_windows(ax, rows, order, (q[0], q[-1]), f"{title} {_ORDER_NAMES[order]}")
            for ax, order in ((first_ax, 1), (second_ax, 2))
        }
        first_ax.set_title(f"{title}: σ = {sigma:.2g} MHz, D0 offset {offset:.2g} MHz", fontsize=12)
        first_ax.set_ylabel(r"first order $\|D(Q) - D_0\| / Q$ (MHz/Q)")
        second_ax.set_ylabel(r"second order $\|D(2Q) - 2D(Q) + D_0\| / Q^2$ (MHz/Q²)")
        second_ax.set_xlabel("|Q| (step size)")
        # "<mode>: <symmetry>", or the bare mode number when the symmetry is unknown
        mode_handles = [
            Line2D([], [], marker="o", ls="", color=c, ms=7, label=f"{m}: {symmetries[m]}" if m in symmetries else f"{m}")
            for c, (m, *_) in zip(COLORS, rows)
        ]
        error_handle = Line2D([], [], ls="--", color="0.35", label="error incl. D0 offset")
        def window_handles(order):
            handles = [
                Line2D([], [], lw=BAND_WIDTH, alpha=BAND_ALPHA, color="0.35", solid_capstyle="butt",
                       label=f"mode's safe |Q| (SNR ≥ {target:g}, truncation ≤ {tol:.0%})"),
                Patch(color=COMMON_COLOR, label="safe |Q| for all modes"),
            ]
            if neglected[order]:
                handles.insert(1, Line2D([], [], lw=BAND_WIDTH, alpha=BAND_ALPHA * FAINT, color="0.35",
                                         solid_capstyle="butt", label=f"left out: SNR < {target:g} at every step"))
            return handles
        modes_legend = first_ax.legend(handles=mode_handles, loc="lower right", fontsize=10)
        first_ax.add_artist(modes_legend)
        first_ax.legend(handles=[
            Line2D([], [], marker="o", ls="", color="0.35", mfc="0.35", label="+Q"),
            Line2D([], [], marker="o", ls="", color="0.35", mfc="white", label="−Q"),
            Line2D([], [], ls="-", color="0.35", label="b (fit, Q → 0)"),
            error_handle,
            *window_handles(1),
        ], loc="lower left", fontsize=10)
        second_ax.legend(handles=[
            Line2D([], [], marker="s", ls="", color="0.35", mfc="0.35", label="+Q"),
            Line2D([], [], marker="s", ls="", color="0.35", mfc="white", label="−Q"),
            Line2D([], [], ls="-", color="0.35", label="2c (fit, Q → 0)"),
            error_handle,
            *window_handles(2),
        ], loc="lower left", fontsize=10)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    LOGGER.info(f"saved figure: {output}")
    if show:
        plt.show()
    plt.close(fig)
    return output
