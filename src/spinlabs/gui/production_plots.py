"""Figure builders of the comparison views: couplings, spectral functions, 2D maps, rates and T1.

Every builder takes the entries to draw as ``(label, colour, data)`` and returns a Matplotlib Figure; one
colour per entry, so several defects, cells or methods compare in one graph. No Tk here (headless tests).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from spinlabs.gui.plots import message_figure
from spinlabs.gui.production import CHANNEL_LABELS, SPIN_FLIP, rate_direction

ENTRY_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2",
                "#7f7f7f", "#bcbd22")
# Experimental points: dark grey, one marker per series (the entries keep the colours)
EXPERIMENT_COLOR = "#333333"
EXPERIMENT_MARKERS = ("o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "p")


def experiment_marker(index: int) -> str:
    return EXPERIMENT_MARKERS[index % len(EXPERIMENT_MARKERS)]


def _draw_experiment(ax, points) -> None:
    """``points``: (legend, marker, colour, x, y) drawn as open markers (colour None: dark grey)."""
    for legend, marker, color, x, y in points:
        ax.plot(x, y, ls="none", marker=marker, ms=4.5, mfc="none", mec=color or EXPERIMENT_COLOR, mew=0.9,
                label=f"exp.: {legend}", zorder=3)


PROCESS_STYLES = {"total": "-", "direct": "--", "two-phonon": ":", "Raman (1st order)": "-."}
MAP_DYNAMIC_RANGE = 1e-6


def entry_color(index: int) -> str:
    return ENTRY_COLORS[index % len(ENTRY_COLORS)]


def _layout(n: int, figsize, *, rows: int = 1, dpi: int = 100):
    fig, axes = plt.subplots(rows, max(1, n), figsize=figsize or (4.2 * max(1, n), 3.6 * rows), dpi=dpi,
                             squeeze=False, layout="constrained")
    return fig, axes


def _legend(fig: Figure, axes) -> None:
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        axes.flat[0].legend(handles, labels, fontsize=7, loc="best", frameon=False)


COUPLING_LOG_RANGE = 1e-6  # on the log scale, the lines start at most this far below a panel's largest |V|


def _coupling_lines(ax, series, log: bool) -> None:
    """Each mode as a vertical line from the baseline to |V| at its phonon energy, as the older coupling
    plots (plotter.plot_vlines_sorted_by_magnitude). ``series``: (label, colour, frequencies, values).

    The entry with the larger mean coupling goes behind, so a smaller set drawn over it stays visible;
    the legend keeps the entry order. On the log scale the lines start just below the smallest coupling
    shown (at most COUPLING_LOG_RANGE below the largest), as there is no zero to start from."""
    shown = [(label, color, x[v > 0] if log else x, v[v > 0] if log else v) for label, color, x, v in series]
    values = np.concatenate([v for *_, v in shown]) if shown else np.empty(0)
    if log and values.size:
        top = float(values.max())
        base = max(float(values.min()), top * COUPLING_LOG_RANGE) / 1.5
        ax.set_ylim(base, top * 1.5)
    else:
        base = 0.0
        ax.set_ylim(0, 1.05 * float(values.max()) if values.size and values.max() > 0 else 1.0)
    behind = np.argsort([-float(np.mean(v)) if v.size else 0.0 for *_, v in shown])  # largest first
    zorder = {int(i): 2 + 0.01 * rank for rank, i in enumerate(behind)}
    for k, (label, color, x, v) in enumerate(shown):
        ax.vlines(x, base, v, color=color, lw=1.0, alpha=0.75, label=label, zorder=zorder[k])


def build_coupling_figure(entries, channels, *, log: bool = True, figsize=None, dpi: int = 100) -> Figure:
    """The couplings against the phonon energy, laid out as the spectral functions: one column per channel
    (transition), first order |V| per mode on top and the diagonal second order |V^(ll)| below. Each mode
    is a vertical line at its energy (:func:`_coupling_lines`). ``entries``: (label, colour, Couplings)."""
    channels = [c for c in channels if c in CHANNEL_LABELS]
    if not entries or not channels:
        return message_figure("Add entries and select a channel", dpi=dpi)
    fig, axes = _layout(len(channels), figsize, rows=2, dpi=dpi)
    for col, channel in enumerate(channels):
        for row, ylabel in enumerate((r"$|V|$ (MHz)", r"$|V^{(ll)}|$ (MHz)")):
            ax = axes[row, col]
            series = [(label, color, np.asarray(c.frequencies, float),
                       np.abs(np.asarray(c.first[channel] if row == 0 else np.diag(c.second[channel]), float)))
                      for label, color, c in entries]
            if log:
                ax.set_yscale("log")
            _coupling_lines(ax, series, log)
            ax.set_xlim(left=0)
            ax.grid(True, ls=":", lw=0.5, alpha=0.6)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(f"{CHANNEL_LABELS[channel]}, first order", fontsize=9)
            else:
                ax.set_title(r"second order, diagonal $V^{(ll)}$", fontsize=9)
                ax.set_xlabel(r"$\hbar\omega$ (meV)")
    _legend(fig, axes)
    return fig


def build_spectral_figure(entries, channels, sigma: float, *, log: bool = True, figsize=None, dpi: int = 100) -> Figure:
    """One-phonon J1(w) (top) and two-phonon J2(w, w) (bottom), one column per channel.
    ``entries``: (label, colour, compute_spectral result)."""
    channels = [c for c in channels if c in CHANNEL_LABELS]
    if not entries or not channels:
        return message_figure("Add entries and select a channel", dpi=dpi)
    fig, axes = _layout(len(channels), figsize, rows=2, dpi=dpi)
    for col, channel in enumerate(channels):
        for row, (kind, ylabel) in enumerate((("one", r"$J^{(1)}$ (MHz$^2$/meV)"),
                                              ("two", r"$J^{(2)}(\omega,\omega)$ (MHz$^2$/meV$^2$)"))):
            ax = axes[row, col]
            top = 0.0
            for label, color, spec in entries:
                omega, density = spec[kind][channel]
                ax.plot(omega, density, color=color, lw=1.1, label=label)
                top = max(top, float(np.max(density)) if len(density) else 0.0)
            if log and top > 0:
                ax.set_yscale("log")
                ax.set_ylim(top * 1e-5, top * 3)
            ax.set_xlim(left=0)
            ax.grid(True, ls=":", lw=0.5, alpha=0.6)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(f"{CHANNEL_LABELS[channel]}, $\\sigma$ = {sigma:g} meV", fontsize=9)
            else:
                ax.set_xlabel(r"$\hbar\omega$ (meV)")
    _legend(fig, axes)
    return fig


def build_map_figure(label: str, maps, channels, sigma: float, *, figsize=None, dpi: int = 100) -> Figure:
    """The full two-phonon density J2(w_x, w_y) of one entry, one panel per channel (log colour scale over
    MAP_DYNAMIC_RANGE). ``maps``: compute_spectral_map result."""
    channels = [c for c in channels if c in CHANNEL_LABELS]
    if maps is None or not channels:
        return message_figure("Select an entry and a channel", dpi=dpi)
    fig, axes = _layout(len(channels), figsize, dpi=dpi)
    for ax, channel in zip(axes[0], channels):
        omega, omega_y, density = maps[channel]
        top = float(np.max(density))
        if top <= 0:
            ax.text(0.5, 0.5, "zero for this channel", ha="center", va="center", transform=ax.transAxes)
        else:
            norm = LogNorm(vmin=top * MAP_DYNAMIC_RANGE, vmax=top)
            mesh = ax.pcolormesh(omega, omega_y, np.clip(density, top * MAP_DYNAMIC_RANGE, None), norm=norm,
                                 cmap="viridis", shading="auto", rasterized=True)
            fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03).set_label(r"MHz$^2$/meV$^2$", fontsize=8)
            hi = float(max(omega.max(), omega_y.max()))
            ax.plot([0, hi], [0, hi], color="white", lw=0.6, ls="--")
            ax.set_xlim(0, hi)
            ax.set_ylim(0, hi)
            ax.set_aspect("equal")
        ax.set_xlabel(r"$\hbar\omega_x$ (meV)")
        ax.set_ylabel(r"$\hbar\omega_y$ (meV)")
        ax.set_title(f"{label}: {CHANNEL_LABELS[channel]}, $\\sigma$ = {sigma:g} meV", fontsize=9)
    return fig


def _direction_text(key: str) -> str:
    return key.replace("_to_", "→").replace("0.5", "½").replace("1.5", "3⁄2")


def build_rates_figure(entries, channels, processes, *, mirror: bool = False, experiment=None, figsize=None,
                       dpi: int = 100) -> Figure:
    """Rates against temperature, one panel per spin-flip channel (the transition of rate_direction), line style
    per process. ``entries``: (label, colour, Rates); ``processes``: {name: tuple of rate processes}. With
    ``mirror`` (a magnetic field: +m and -m differ) a second row shows the transitions from m = -S.
    ``experiment``: {channel: [(legend, marker, colour, temperatures, rate)]}, measured points drawn in the +S row (the
    measurements are at zero or small field)."""
    channels = [c for c in channels if c in SPIN_FLIP]
    if not entries or not channels or not processes:
        return message_figure("Add entries, and select a spin-flip channel (Δ1, Δ2) and a process", dpi=dpi)
    signs = (1, -1) if mirror else (1,)
    fig, axes = _layout(len(channels), figsize, rows=len(signs), dpi=dpi)
    for row, sign in enumerate(signs):
        for ax, channel in zip(axes[row], channels):
            directions = set()
            for label, color, r in entries:
                key = rate_direction(r.spin, channel, sign)
                total_by_direction = {name: r.total(members) for name, members in processes.items()}
                if any(key not in by_direction for by_direction in total_by_direction.values()):
                    continue
                directions.add(_direction_text(key))
                for name, by_direction in total_by_direction.items():
                    y = by_direction[key]
                    ok = np.isfinite(y) & (y > 0)
                    ax.plot(r.temperatures[ok], y[ok], color=color, ls=PROCESS_STYLES.get(name, "-"), lw=1.2,
                            label=f"{label}, {name}" if len(processes) > 1 else label)
            ax.set_yscale("log")
            ax.set_xlabel("Temperature (K)")
            field = ""
            if mirror:
                fields = sorted({r.b_field_mt for _l, _c, r in entries})
                field = f", B = {', '.join(f'{b:g}' for b in fields)} mT"
            if row == 0 and experiment:
                _draw_experiment(ax, experiment.get(channel, []))
            ax.set_title(f"{CHANNEL_LABELS[channel]} rate (m$_s$ {', '.join(sorted(directions))}{field})", fontsize=9)
            ax.grid(True, ls=":", lw=0.5, alpha=0.6)
        axes[row, 0].set_ylabel(r"rate (s$^{-1}$)")
    _legend(fig, axes)
    for ax in axes[0, 1:]:  # another channel's measured points are not in the first panel's legend
        pairs = [(h, lab) for h, lab in zip(*ax.get_legend_handles_labels()) if lab.startswith("exp.:")]
        if pairs:
            ax.legend(*zip(*pairs), fontsize=7, loc="best", frameon=False)
    return fig


def build_t1_figure(entries, *, ratio: bool = False, experiment=None, figsize=None, dpi: int = 100) -> Figure:
    """T1 against temperature, one line per entry; with ``ratio`` a second panel of T1 / T1(first entry).
    ``entries``: (label, colour, (temperatures, T1, initial state)); ``experiment``: [(legend, marker,
    colour, temperatures, T1)] drawn as points in the T1 panel."""
    if not entries:
        return message_figure("Add entries", dpi=dpi)
    rows = 2 if ratio and len(entries) > 1 else 1
    fig, axes = plt.subplots(rows, 1, figsize=figsize or (6.4, 3.4 * rows), dpi=dpi, squeeze=False,
                             sharex=True, layout="constrained", height_ratios=[2.2, 1][:rows])
    ax = axes[0, 0]
    for label, color, (temps, t1, state) in entries:
        ok = np.isfinite(t1) & (t1 > 0)
        ax.plot(temps[ok], t1[ok], color=color, lw=1.4, label=f"{label} ({state})")
    if experiment:
        _draw_experiment(ax, experiment)
    ax.set_yscale("log")
    ax.set_ylabel(r"$T_1$ (s)")
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=7, frameon=False)
    if rows == 2:
        bx = axes[1, 0]
        ref_label, _c, (ref_t, ref_t1, _s) = entries[0]
        for label, color, (temps, t1, _state) in entries[1:]:
            common, i, j = np.intersect1d(temps, ref_t, return_indices=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_values = t1[i] / ref_t1[j]
            ok = np.isfinite(ratio_values) & (ratio_values > 0)
            bx.plot(common[ok], ratio_values[ok], color=color, lw=1.2)
        bx.axhline(1.0, color="0.4", lw=0.6, ls=":")
        bx.set_ylabel(f"T$_1$ / T$_1$({ref_label})", fontsize=8)
        bx.grid(True, ls=":", lw=0.5, alpha=0.6)
    axes[-1, 0].set_xlabel("Temperature (K)")
    return fig
