"""Figure builders shared by the GUI and its headless tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

from spinlabs.convergence import (
    PART_LABELS,
    PC_NAMES,
    SNRReport,
    SNR_TARGET,
    TRUNC_TOL,
    _plot_finite_differences,
    cubic_fit,
    d0_offset,
    difference_errors,
    fit_points,
    helper_step,
    linear_fit,
    one_sided_d2,
    poly_fit,
    pooled_noise,
    principal_frame,
    quad_fit,
    snr_report,
    traceless_parts,
)
from spinlabs.convergence_channels import (
    CHANNELS,
    CHANNEL_LABELS,
    CHANNEL_MARKERS,
    CHANNEL_NAMES,
    ChannelAnalysis,
)
from spinlabs.convergence_summary import (
    COMMON_COLOR,
    FAINT,
    WEAK_FRACTION,
    plot_signed,
    shade_safe_windows,
)
from spinlabs.gui.analysis import ModeEntry, NormResult, PackSession, mode_color
from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)

_CHANNEL_LS = {"d0": "-", "d1": "--", "d2": ":"}
_CENTRAL_MARKER = {"o": "D", "s": "v"}
_COMMON_ALPHA = 0.55


@dataclass
class NormColumnPlot:
    label: str
    modes: dict[int, ModeEntry]
    forward: NormResult | None
    central: NormResult | None
    note: str = ""


@dataclass
class ChannelColumnPlot:
    label: str
    modes: dict[int, ModeEntry]
    analysis: ChannelAnalysis
    note: str = ""


@dataclass
class ModeFigureResult:
    figure: Figure
    report: SNRReport | None
    noise: float | None
    offset: float | None
    label: str
    warnings: list[str]
    extra_lines: list[str] = field(default_factory=list)
    reports: dict[str, SNRReport] = field(default_factory=dict)


def _rows_by_mode(rows: list) -> dict[int, tuple]:
    return {int(row[0]): row for row in rows}


def _steps_of(rows: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, SNRReport]]) -> np.ndarray:
    steps = [np.abs(row[1][~np.isclose(row[1], 0.0)]) for row in rows]
    if not steps:
        return np.array([0.01, 0.1])
    all_steps = np.concatenate(steps)
    if not len(all_steps):
        return np.array([0.01, 0.1])
    return np.unique(np.round(all_steps, 12))


def _log_axes(ax, q: np.ndarray, run_steps: np.ndarray) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(run_steps, [f"{s:g}" for s in run_steps])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", color="0.9", linewidth=0.6)
    ax.set_xlim(q[0], q[-1])


def _mode_handles(mode_ids: list[int], modes: dict[int, ModeEntry], colors: list[str]) -> list[Line2D]:
    handles = []
    for mode, color in zip(mode_ids, colors):
        entry = modes.get(mode, ModeEntry(mode))
        label = f"{mode}: {entry.symmetry}" if entry.symmetry else str(mode)
        handles.append(Line2D([], [], marker="o", ls="", color=color, ms=7, label=label))
    return handles


def _empty(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=10, color="0.35")
    ax.set_axis_off()


def message_figure(text: str, *, figsize: tuple[float, float] = (8.0, 5.0), dpi: int = 100) -> Figure:
    """A blank figure carrying a message (empty selection, missing data, ...)."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _empty(ax, text)
    return fig


def build_norm_summary_figure(
    columns: list[NormColumnPlot],
    *,
    target: float = SNR_TARGET,
    tol: float = TRUNC_TOL,
    weak_fraction: float = WEAK_FRACTION,
    colors: list[str] | None = None,
    dpi: int = 100,
) -> Figure:
    """The CLI summary figure for GUI columns, with optional forward/central overlay."""
    if not columns:
        fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=dpi)
        _empty(ax, "Open a convergence pack to see the summary")
        return fig
    fig, axes = plt.subplots(
        2, len(columns), figsize=(max(7.5, 8.0 * len(columns)), 9.5), squeeze=False, sharex="col", sharey="row",
        dpi=dpi,
    )
    all_modes: set[int] = set()
    for column in columns:
        for result in (column.forward, column.central):
            if result is not None:
                all_modes.update(int(row[0]) for row in result.rows)
    global_color = {mode: mode_color(i) for i, mode in enumerate(sorted(all_modes))}
    for col, column in enumerate(columns):
        first_ax, second_ax = axes[:, col]
        forward_rows = _rows_by_mode(column.forward.rows) if column.forward is not None else {}
        central_rows = _rows_by_mode(column.central.rows) if column.central is not None else {}
        mode_ids = sorted(set(forward_rows) | set(central_rows))
        color_of = {mode: global_color.get(mode, mode_color(0)) for mode in mode_ids}
        col_colors = [color_of[mode] for mode in mode_ids]
        steps = _steps_of([*forward_rows.values(), *central_rows.values()])
        q = np.geomspace(steps.min() / 2, steps.max() * 1.5, 200)
        if not mode_ids:
            _empty(first_ax, column.label + "\n" + (column.note or "no analysable modes"))
            second_ax.set_axis_off()
            continue

        for mode in mode_ids:
            color = color_of[mode]
            for scheme, rows, result in (
                ("forward", forward_rows, column.forward),
                ("central", central_rows, column.central),
            ):
                row = rows.get(mode)
                if row is None or result is None:
                    continue
                _mode, _p, first, second, report = row
                a1 = 1.0 if report.b_resolved else FAINT
                a2 = 1.0 if report.c_resolved else FAINT
                if scheme == "forward":
                    plot_signed(first_ax, first, "o", color, a1)
                    plot_signed(second_ax, second, "s", color, a2)
                    first_ax.axhline(report.b, color=color, lw=1, alpha=0.7 * a1, ls="-")
                    second_ax.axhline(2 * report.c, color=color, lw=1, alpha=0.7 * a2, ls="-")
                else:
                    first_ax.plot(first[:, 0], first[:, 1], marker=_CENTRAL_MARKER["o"], ls="", color=color,
                                  mfc="white", mec=color, mew=1.2, ms=6, alpha=a1)
                    second_ax.plot(second[:, 0], second[:, 1], marker=_CENTRAL_MARKER["s"], ls="", color=color,
                                   mfc="white", mec=color, mew=1.2, ms=6, alpha=a2)
                    first_ax.axhline(report.b, color=color, lw=1, alpha=0.5 * a1, ls="--")
                    second_ax.axhline(2 * report.c, color=color, lw=1, alpha=0.5 * a2, ls="--")

        # The grey error curve: forward solid, central dotted.
        for scheme, result, style in (
            ("forward", column.forward, "--"),
            ("central", column.central, ":"),
        ):
            if result is None or not np.isfinite(result.sigma):
                continue
            err1, err2 = difference_errors(result.sigma, result.offset, central=(scheme == "central"))
            first_ax.plot(q, err1 / q, style, color="0.35", lw=1)
            second_ax.plot(q, err2 / q**2, style, color="0.35", lw=1)

        # Safe bands come from Forward when it is available, otherwise Central.
        shading = column.forward if column.forward is not None and column.forward.rows else column.central
        shading_rows = _rows_by_mode(shading.rows) if shading is not None else {}
        if shading_rows:
            order = sorted(shading_rows)
            shade_safe_windows(
                first_ax, [shading_rows[m] for m in order], 1, (q[0], q[-1]), f"{column.label} first",
                weak_fraction, [color_of[m] for m in order],
            )
            shade_safe_windows(
                second_ax, [shading_rows[m] for m in order], 2, (q[0], q[-1]), f"{column.label} second",
                weak_fraction, [color_of[m] for m in order],
            )

        _log_axes(first_ax, q, steps)
        _log_axes(second_ax, q, steps)
        first_ax.set_title(f"{column.label}: σ = {_sigma_text(column)}, D0 offset {_offset_text(column)}",
                           fontsize=11)
        first_ax.set_ylabel(r"first order $\|D(Q) - D_0\| / Q$ (MHz/Q)")
        second_ax.set_ylabel(r"second order $\|D(2Q) - 2D(Q) + D_0\| / Q^2$ (MHz/Q$^2$)")
        second_ax.set_xlabel("|Q| (step size)")
        _norm_legends(first_ax, second_ax, mode_ids, column.modes, col_colors, column)

    fig.tight_layout()
    return fig


def _sigma_text(column: NormColumnPlot) -> str:
    result = column.forward or column.central
    return f"{result.sigma:.2g} MHz" if result is not None and np.isfinite(result.sigma) else "n/a"


def _offset_text(column: NormColumnPlot) -> str:
    result = column.forward or column.central
    return f"{result.offset:.2g} MHz" if result is not None and np.isfinite(result.offset) else "n/a"


def _norm_legends(ax, second_ax, mode_ids, modes, colors, column: NormColumnPlot) -> None:
    mode_legend = ax.legend(handles=_mode_handles(mode_ids, modes, colors), loc="lower right", fontsize=9)
    ax.add_artist(mode_legend)
    handles = [
        Line2D([], [], marker="o", ls="", color="0.35", mfc="0.35", label="+Q"),
        Line2D([], [], marker="o", ls="", color="0.35", mfc="white", label="−Q"),
        Line2D([], [], ls="-", color="0.35", label="fit, Q → 0"),
    ]
    if column.forward is not None:
        handles.append(Line2D([], [], ls="--", color="0.35", label="forward error"))
    if column.central is not None:
        handles += [
            Line2D([], [], marker=_CENTRAL_MARKER["o"], ls="", color="0.35", mfc="white",
                   label="central (diamonds)"),
            Line2D([], [], ls=":", color="0.35", label="central error"),
        ]
    ax.legend(handles=handles, loc="lower left", fontsize=8)


def build_channel_summary_figure(
    columns: list[ChannelColumnPlot],
    *,
    schemes: tuple[str, ...] = ("forward", "central"),
    target: float = SNR_TARGET,
    tol: float = TRUNC_TOL,
    dpi: int = 100,
) -> Figure:
    """The V_{Delta n} channel summary for GUI columns."""
    if not columns:
        fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=dpi)
        _empty(ax, "Open a convergence pack to see the coupling channels")
        return fig
    fig, axes = plt.subplots(
        2, len(columns), figsize=(max(7.5, 8.0 * len(columns)), 9.5), squeeze=False, sharex="col", sharey="row",
        dpi=dpi,
    )
    all_modes = sorted({mode.mode for column in columns for mode in column.analysis.modes})
    global_color = {mode: mode_color(i) for i, mode in enumerate(all_modes)}
    for col, column in enumerate(columns):
        first_ax, second_ax = axes[:, col]
        analysis = column.analysis
        mode_ids = [m.mode for m in sorted(analysis.modes, key=lambda m: m.mode)]
        col_colors = [global_color.get(mode, mode_color(0)) for mode in mode_ids]
        color_of = dict(zip(mode_ids, col_colors))
        qs = [s.q for mode in analysis.modes for s in mode.series if len(s.q)]
        if not qs:
            _empty(first_ax, column.label + "\n" + (column.note or "no channel data"))
            second_ax.set_axis_off()
            continue
        steps = np.unique(np.round(np.abs(np.concatenate(qs)), 12))
        steps = steps[steps > 0]
        if not len(steps):
            steps = np.array([0.01, 0.1])
        q = np.geomspace(steps.min() / 2, steps.max() * 1.5, 200)

        zero_marks: dict[int, list[tuple[str, str]]] = {1: [], 2: []}
        for mode in analysis.modes:
            color = color_of.get(mode.mode, mode_color(0))
            for order, ax in ((1, first_ax), (2, second_ax)):
                for channel in CHANNELS:
                    for scheme in schemes:
                        series = mode.get(order, channel, scheme)
                        if series is None or not len(series.q):
                            continue
                        window = series.window
                        alpha = 1.0 if window.allowed else 0.5
                        if window.allowed and (not window.resolved or window.weak):
                            alpha = 0.55
                        ax.plot(
                            series.q, series.values, marker=CHANNEL_MARKERS[channel], ls="", color=color,
                            mfc=color if scheme == "forward" else "white", mec=color, mew=1.2, ms=6, alpha=alpha,
                        )
                        if series.q0 is not None and np.isfinite(series.q0) and series.q0 > 0:
                            ax.axhline(
                                series.q0, color=color, ls=_CHANNEL_LS[channel], lw=1, alpha=0.75 * alpha,
                            )
                            if window.usable:
                                start = max(window.q_lo, q[0])
                                end = min(window.q_hi, q[-1]) if window.upper_known else q[-1]
                                if start < end:
                                    ax.plot(
                                        [start, end], [series.q0, series.q0], color=color, lw=5,
                                        alpha=0.22, solid_capstyle="butt", zorder=1,
                                    )
                        elif series.q0 is not None and np.isfinite(series.q0):
                            # An exactly zero coefficient is the visual zero check; log axes
                            # cannot show 0, so mark it at the floor after autoscaling.
                            zero_marks[order].append((color, channel))
        # Conservative norm error curves.
        for scheme, style in (("forward", "--"), ("central", ":")):
            if scheme not in schemes or scheme not in analysis.errors:
                continue
            err1, err2 = analysis.errors[scheme]
            first_ax.plot(q, err1 / q, style, color="0.35", lw=1)
            second_ax.plot(q, err2 / q**2, style, color="0.35", lw=1)
        # One common window: Forward when it is enabled, else Central.
        shade_scheme = "forward" if "forward" in schemes else "central"
        for order, ax in ((1, first_ax), (2, second_ax)):
            common = analysis.common.get((order, shade_scheme))
            if common is None:
                continue
            lo, hi, known = common
            if not np.isfinite(lo):
                continue
            end = min(hi, q[-1]) if known and np.isfinite(hi) else q[-1]
            if lo < end:
                ax.axvspan(max(lo, q[0]), end, color=COMMON_COLOR, alpha=_COMMON_ALPHA, zorder=0, lw=0)

        _log_axes(first_ax, q, steps)
        _log_axes(second_ax, q, steps)
        for order, ax in ((1, first_ax), (2, second_ax)):
            if not zero_marks[order]:
                continue
            floor = ax.get_ylim()[0] * 1.02
            for color, channel in zero_marks[order]:
                ax.axhline(floor, color=color, ls=_CHANNEL_LS[channel], lw=0.7, alpha=0.35)
        first_ax.set_title(f"{column.label}: σ = {analysis.sigma:.2g} MHz, D0 offset {analysis.offset:.2g} MHz",
                           fontsize=11)
        first_ax.set_ylabel(r"first order $V_{\Delta n}$ (MHz/Q)")
        second_ax.set_ylabel(r"second order $V_{\Delta n}$ (MHz/Q$^2$)")
        second_ax.set_xlabel("|Q| (step size)")
        _channel_legends(first_ax, second_ax, mode_ids, column.modes, col_colors, schemes)
    fig.tight_layout()
    return fig


def _channel_legends(ax, second_ax, mode_ids, modes, colors, schemes) -> None:
    mode_legend = ax.legend(handles=_mode_handles(mode_ids, modes, colors), loc="lower right", fontsize=9)
    ax.add_artist(mode_legend)
    handles = [
        Line2D([], [], marker=CHANNEL_MARKERS[c], ls=_CHANNEL_LS[c], color="0.35", label=CHANNEL_LABELS[c])
        for c in CHANNELS
    ]
    if "forward" in schemes:
        handles.append(Line2D([], [], marker="o", ls="", color="0.35", mfc="0.35", label="forward"))
    if "central" in schemes:
        handles.append(Line2D([], [], marker="o", ls="", color="0.35", mfc="white", label="central"))
    handles.append(Line2D([], [], ls="--", color="0.35", label="conservative error"))
    ax.legend(handles=handles, loc="lower left", fontsize=8)


def _q0_label(value: float | None) -> str:
    return f"{value:.3g}" if value is not None and np.isfinite(value) else "n/a"


def channel_report_lines(analysis: ChannelAnalysis) -> list[str]:
    """Plain-text channel table for the report pane's Details tab."""
    lines = [
        f"noise sigma {analysis.sigma:.4g} MHz per tensor element ({analysis.sigma_source}); "
        f"D0 offset {analysis.offset:.4g} MHz",
        "channels: V_d0 = V_0_0, V_d±1 = V_0_pm, V_d±2 = V_p_m; "
        "forbidden channels are display-only",
    ]
    for mode in analysis.modes:
        sym = mode.symmetry or "?"
        head = f"mode {mode.mode} ({sym}"
        if mode.frequency is not None:
            head += f", {mode.frequency:.2f} meV"
        lines.append(head + ")")
        for order in (1, 2):
            for channel in CHANNELS:
                series = mode.get(order, channel, "forward") or mode.get(order, channel, "central")
                if series is None:
                    continue
                window = series.window
                flags = []
                if not window.allowed:
                    flags.append("forbidden")
                if not window.resolved:
                    flags.append("unresolved")
                if window.weak:
                    flags.append("weak")
                lines.append(
                    f"  {CHANNEL_NAMES[channel]:>6} {('first' if order == 1 else 'second'):>6}  "
                    f"Q→0 {_q0_label(series.q0):>10}  window {window.q_lo:.3g} .. "
                    f"{window.q_hi:.3g}{'' if window.upper_known else ' (upper bound)'}"
                    + (f"  [{', '.join(flags)}]" if flags else "")
                )
    return lines


def _scheme_finite_differences(axes, perts, parts, report, helper, scheme: str, sigma: float | None) -> None:
    """The per-mode finite-difference row limited to one scheme.

    The CLI row shows forward first order and both forward-pair and central second
    order.  Here a single scheme is drawn; both schemes keep the CLI row.
    """
    if scheme not in ("forward", "central"):
        raise ValueError(f"unknown scheme {scheme!r}")
    f0 = parts[np.isclose(perts, 0.0)][0]
    qmax = float(np.max(np.abs(perts)))
    if scheme == "forward":
        fit = fit_points(perts, helper)
        p_fit, x_fit = perts[fit], parts[fit]
        coef = cubic_fit(p_fit, x_fit)[0]
        pair_coef = cubic_fit(perts, parts)[0]
        b, c = coef[1], 2 * pair_coef[2]
        err1 = np.sqrt(2.0) * sigma if sigma is not None and np.isfinite(sigma) else None
        err2 = np.sqrt(6.0) * sigma if sigma is not None and np.isfinite(sigma) else None
        title1 = "first order  (D(Q) − D0)/Q"
        title2 = "second order  (D(2Q) − 2D(Q) + D0)/Q²"
    else:
        coef = poly_fit(perts, parts, 4)[0]
        b, c = coef[1], 2 * coef[2]
        err1 = sigma / np.sqrt(2.0) if sigma is not None and np.isfinite(sigma) else None
        err2 = np.sqrt(6.0) * sigma if sigma is not None and np.isfinite(sigma) else None
        title1 = "first order  (D(Q) − D(−Q))/2Q"
        title2 = "second order  (D(Q) + D(−Q) − 2D0)/Q²"

    shown1 = np.abs(b) >= 0.02 * np.max(np.abs(b)) if len(b) else np.zeros_like(b, bool)
    shown2 = np.abs(c) >= 0.02 * np.max(np.abs(c)) if len(c) else np.zeros_like(c, bool)
    for k in range(parts.shape[1]):
        color = f"C{k}"
        label = PART_LABELS[k]
        if shown1[k]:
            if scheme == "forward":
                for side, face in ((1, color), (-1, "white")):
                    q = np.sort(np.abs(p_fit[np.sign(p_fit) == side]))
                    if not len(q):
                        continue
                    vals = [(x_fit[np.isclose(p_fit, side * v)][0, k] - f0[k]) / (side * v) for v in q]
                    axes[0].errorbar(q, vals, yerr=None if err1 is None else err1 / q, fmt="o", color=color,
                                     mfc=face, ms=5, capsize=3, label=f"{label} {'+Q' if side > 0 else '−Q'}")
            else:
                q = np.array([abs(float(v)) for v in perts if v > 0 and _has(perts, -v)])
                vals = []
                for v in q:
                    plus = parts[np.isclose(perts, v)][0, k]
                    minus = parts[np.isclose(perts, -v)][0, k]
                    vals.append((plus - minus) / (2 * v))
                if len(q):
                    axes[0].errorbar(q, vals, yerr=None if err1 is None else err1 / q, fmt="D", color=color,
                                     mfc="white", mec=color, mew=1.2, ms=6, capsize=3, label=f"{label} central")
            axes[0].axhline(b[k], color=color, ls="--", lw=0.8)
        if shown2[k]:
            qs, vals = [], []
            for v in sorted({abs(float(x)) for x in perts if not np.isclose(x, 0.0)}):
                if scheme == "forward":
                    one = _at_pert(perts, v)
                    two = _at_pert(perts, 2 * v)
                    if one is not None and two is not None:
                        qs.append(v)
                        vals.append((parts[two][k] - 2 * parts[one][k] + f0[k]) / v**2)
                else:
                    plus, minus = _at_pert(perts, v), _at_pert(perts, -v)
                    if plus is not None and minus is not None:
                        qs.append(v)
                        vals.append(
                            (parts[plus][k] + parts[minus][k] - 2 * f0[k]) / v**2
                        )
            if qs:
                qs = np.array(qs)
                axes[1].errorbar(qs, vals, yerr=None if err2 is None else err2 / qs**2,
                                 fmt="s" if scheme == "forward" else "v", color=color,
                                 mfc=color if scheme == "forward" else "white", mec=color, ms=6, capsize=3,
                                 label=f"{label} {scheme}")
            axes[1].axhline(c[k], color=color, ls="--", lw=0.8)
    for ax, lo, hi, resolved, title in (
        (axes[0], report.q_lo1, report.q_hi1, report.c_resolved, title1) if report else
        (axes[0], float("nan"), float("nan"), False, title1),
        (axes[1], report.q_lo2, report.q_hi2, report.d_resolved, title2) if report else
        (axes[1], float("nan"), float("nan"), False, title2),
    ):
        right = 1.15 * qmax
        if report is not None and (lo < min(hi, right) or (not resolved and lo < right)):
            ax.axvspan(max(lo, 0), right if not resolved else min(hi, right), color="#7fb77e", alpha=0.2, lw=0)
        window = f"{lo:.3g}–{'≥' if not resolved else ''}{hi:.3g}" if np.isfinite(lo) and np.isfinite(hi) else "no SNR estimate"
        ax.set_title(f"{title}\nsafe |Q|: {window}", fontsize=9)
        ax.set_xlim(0, right)
        ax.set_xlabel("|Q|", fontsize=8)
        ax.axhline(0, color="#bbbbbb", lw=0.6)
        ax.legend(fontsize=6, ncol=2)
    axes[0].set_ylabel("traceless part (MHz/Q, MHz/Q²)")


def _has(array, value):
    return np.flatnonzero(np.isclose(array, value))


def _at_pert(perts, value):
    found = _has(perts, value)
    return int(found[0]) if len(found) else None


def build_mode_figure(
    session: PackSession,
    mode: int,
    *,
    quad: bool = False,
    deviation: bool = False,
    drop_anchor: bool = False,
    noise: float | None = None,
    snr_target: float = SNR_TARGET,
    trunc_tol: float = TRUNC_TOL,
    schemes: tuple[str, ...] = ("forward", "central"),
    dpi: int = 100,
) -> ModeFigureResult:
    """The CLI's per-mode 3x2 figure, returned instead of saved."""
    conv = session.conv
    modes, perts, tensors = conv.modes, conv.perts, conv.tensors
    label = session.label
    warnings: list[str] = []
    wanted = {int(m) for m in modes}
    if mode not in wanted:
        raise ValueError(f"mode {mode} is not in {session.path.name} (modes: {', '.join(map(str, modes))})")

    helper = helper_step(perts)
    offset = None
    offset_vec = None
    noise_source = "--noise"
    q0_shared = conv.q0
    if q0_shared is not None:
        frame0 = principal_frame(q0_shared)
        fit = fit_points(perts, helper)
        finished = [~np.isnan(tensors[i]).any(axis=(1, 2)) & fit for i in range(tensors.shape[0])]
        p_modes = [perts[ok] for ok in finished]
        parts_modes = [np.stack([traceless_parts(t, frame0) for t in tensors[i][ok]]) for i, ok in enumerate(finished)]
        if noise is None:
            noise, n_dof = pooled_noise(p_modes, parts_modes)
            noise_source = f"cubic-fit scatter of all {tensors.shape[0]} modes, {n_dof} dof"
        offset_vec = d0_offset(p_modes, parts_modes, traceless_parts(q0_shared, frame0))
        offset = float(np.linalg.norm(offset_vec))

    i = int(np.where(modes == mode)[0][0])
    ok0 = ~np.isnan(tensors[i]).any(axis=(1, 2))
    dropped = perts[~ok0]
    if len(dropped):
        warnings.append(f"mode {mode}: {len(dropped)} run(s) without a ZFS tensor left out: "
                        f"steps {np.array2string(dropped)}")
    if ok0.sum() < 2:
        raise ValueError(f"mode {mode}: fewer than 2 finished runs; not plotted")
    p_full, t_full = perts[ok0], tensors[i][ok0]
    keep = fit_points(p_full, helper)
    p_all, t_all = p_full[keep], t_full[keep]
    izero = int(np.argmin(np.abs(p_all)))
    if not np.isclose(p_all[izero], 0.0):
        warnings.append(f"mode {mode}: no Q=0 tensor; principal frame from the smallest |Q| perturbation")
    R = principal_frame(t_all[izero])
    anchor_vals = np.diag(R.T @ t_all[izero] @ R) if deviation else None
    p_use, t_use = p_all, t_all
    if drop_anchor:
        p_use = p_all[np.abs(p_all) > 0]
        t_use = t_all[np.abs(p_all) > 0]
        if len(p_use) == 0:
            raise ValueError(f"mode {mode}: dropping the Q=0 anchor leaves no points")
    vals = np.stack([np.diag(R.T @ t @ R) for t in t_use])
    trace = vals.sum(axis=1)
    if deviation:
        vals = vals - anchor_vals
    sgn = np.argsort(p_use)

    reports: dict[str, SNRReport] = {}
    parts = parts_full = None
    if noise is not None and np.isfinite(noise) and np.isclose(p_all, 0.0).any():
        parts = np.stack([traceless_parts(t, R) for t in t_all])
        parts_full = np.stack([traceless_parts(t, R) for t in t_full])
        if "forward" in schemes:
            reports["forward"] = snr_report(
                p_all, parts, noise, noise_source, offset_vec, snr_target, trunc_tol,
                pair=(p_full, parts_full),
            )
        if "central" in schemes:
            reports["central"] = snr_report(
                p_full, parts_full, noise, noise_source, offset_vec, snr_target, trunc_tol, central=True
            )
    report = reports.get("forward") or reports.get("central")
    band_report = report

    ncols = 2
    nrows = 3 if schemes and parts_full is not None else 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 9 if nrows == 3 else 6),
        gridspec_kw={"height_ratios": [3, 1, 2.4][:nrows]}, dpi=dpi,
    )
    for j in range(ncols):
        axes[0, j].sharex(axes[1, j])
        axes[0, j].tick_params(labelbottom=False)
        if band_report is not None:
            axes[1, j].axhspan(-band_report.sigma, band_report.sigma, color="#d9d9d9", alpha=0.6, lw=0)
    trace_labeled = False
    extra_lines: list[str] = []
    for k in range(3):
        _a, b, y_lin, res = linear_fit(p_use, vals[:, k])
        _qa, _qb, qc, y_q, qres = quad_fit(p_use, vals[:, k])
        if quad:
            d2 = 2 * qc
            forward_d2 = one_sided_d2(p_use, vals[:, k])
            extra_lines.append(
                f"PC {PC_NAMES[k]}: d2D/dQ2 regression {d2:.4g}"
                + (f", one-sided {forward_d2:.4g}" if forward_d2 is not None else "")
                + " MHz/pert^2"
            )
            if np.isclose(p_use, 0.0).any():
                f0 = vals[p_use == 0.0, k][0]
                cents = []
                for q in sorted({abs(float(x)) for x in p_use if x > 0}):
                    neg, pos = np.isclose(p_use, -q), np.isclose(p_use, q)
                    if neg.any() and pos.any():
                        cents.append(f"{q:g}:{(vals[pos, k][0] + vals[neg, k][0] - 2 * f0) / q**2:.4g}")
                if cents:
                    extra_lines.append(f"PC {PC_NAMES[k]}: central d2 (|Q|:value) {', '.join(cents)}")
        axes[0, 0].plot(p_use[sgn], vals[sgn, k], "o", label=rf"$D_{{\mathrm{{{PC_NAMES[k]}}}}}$")
        axes[0, 0].plot(p_use[sgn], y_lin[sgn], "-", alpha=0.4, color=axes[0, 0].lines[-1].get_color())
        axes[1, 0].plot(p_use[sgn], res[sgn], "o-")
        axes[1, 0].axhline(np.max(np.abs(res)), ls="--", lw=0.8, color="grey")
        axes[1, 0].axhline(-np.max(np.abs(res)), ls="--", lw=0.8, color="grey")
        trace_label = "Tr D" if not trace_labeled else None
        trace_labeled = True
        axes[0, 0].plot(p_use[sgn], trace[sgn], "-", lw=0.8, color="gray", label=trace_label)
        axes[0, 1].plot(p_use[sgn], trace[sgn], "-", lw=0.8, color="gray", label=trace_label)
        axes[0, 1].plot(
            p_use[sgn], vals[sgn, k], "o",
            label=rf"$D_{{\mathrm{{{PC_NAMES[k]}}}}}$ (b={b:.4g}, 2c={2 * qc:.4g})",
        )
        axes[0, 1].plot(p_use[sgn], y_q[sgn], "-", alpha=0.4, color=axes[0, 1].lines[-1].get_color())
        axes[1, 1].plot(p_use[sgn], qres[sgn], "o-")
        axes[1, 1].axhline(np.max(np.abs(qres)), ls="--", lw=0.8, color="grey")
        axes[1, 1].axhline(-np.max(np.abs(qres)), ls="--", lw=0.8, color="grey")
    for j, title in enumerate(["ax + b", "ax² + bx + c"]):
        axes[0, j].set_title(title)
        axes[0, j].set_ylabel("principal value (MHz)")
        axes[1, j].set_xlabel("perturbation scale")
        axes[1, j].xaxis.set_major_locator(plt.MaxNLocator(5))
        axes[1, j].set_ylabel("residual (MHz)" + ("\n(grey: ±σ noise)" if band_report else ""))
    if nrows == 3 and parts_full is not None:
        if "forward" in schemes and "central" in schemes and reports.get("forward") and reports.get("central"):
            _plot_finite_differences(axes[2], p_full, parts_full, reports["forward"], helper)
        elif "forward" in schemes:
            _scheme_finite_differences(
                axes[2], p_full, parts_full, reports.get("forward"), helper, "forward",
                band_report.sigma if band_report is not None else None,
            )
        elif "central" in schemes:
            _scheme_finite_differences(
                axes[2], p_full, parts_full, reports.get("central"), helper, "central",
                band_report.sigma if band_report is not None else None,
            )
    fig.suptitle(f"mode {mode}: $D_\\lambda$" if not deviation else f"mode {mode}: $D_\\lambda - D_0$")
    axes[0, 1].legend(fontsize=7)
    fig.tight_layout()
    return ModeFigureResult(
        figure=fig, report=report, noise=noise, offset=offset, label=label, warnings=warnings,
        extra_lines=extra_lines, reports=reports,
    )
