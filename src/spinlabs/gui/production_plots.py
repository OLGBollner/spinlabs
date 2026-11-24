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
PROCESS_STYLES = {"total": "-", "direct": "--", "two-phonon": ":", "Raman (1st order)": "-."}
_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "*")
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


def build_coupling_figure(entries, channels, *, log: bool = True, figsize=None, dpi: int = 100) -> Figure:
    """The couplings against the phonon energy, laid out as the spectral functions: one column per channel
    (transition), first order |V| per mode on top and the diagonal second order |V^(ll)| below.
    ``entries``: (label, colour, Couplings)."""
    channels = [c for c in channels if c in CHANNEL_LABELS]
    if not entries or not channels:
        return message_figure("Add entries and select a channel", dpi=dpi)
    fig, axes = _layout(len(channels), figsize, rows=2, dpi=dpi)
    for col, channel in enumerate(channels):
        for row, ylabel in enumerate((r"$|V|$ (MHz)", r"$|V^{(ll)}|$ (MHz)")):
            ax = axes[row, col]
            for k, (label, color, c) in enumerate(entries):
                values = c.first[channel] if row == 0 else np.abs(np.diag(c.second[channel]))
                ok = values > 0 if log else np.ones(len(values), bool)
                ax.plot(c.frequencies[ok], values[ok], ls="none", marker=_MARKERS[k % len(_MARKERS)], ms=3.5,
                        mfc=color, mec="white", mew=0.3, alpha=0.85, label=label)
            if log:
                ax.set_yscale("log")
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


def build_rates_figure(entries, channels, processes, *, figsize=None, dpi: int = 100) -> Figure:
    """Rates against temperature, one panel per spin-flip channel (the transition of rate_direction), line style
    per process. ``entries``: (label, colour, Rates); ``processes``: {name: tuple of rate processes}."""
    channels = [c for c in channels if c in SPIN_FLIP]
    if not entries or not channels or not processes:
        return message_figure("Add entries, and select a spin-flip channel (Δ1, Δ2) and a process", dpi=dpi)
    fig, axes = _layout(len(channels), figsize, dpi=dpi)
    for ax, channel in zip(axes[0], channels):
        directions = set()
        for label, color, r in entries:
            key = rate_direction(r.spin, channel)
            directions.add(key.replace("_to_", "→").replace("0.5", "½").replace("1.5", "3⁄2"))
            for name, members in processes.items():
                y = r.total(members)[key]
                ok = np.isfinite(y) & (y > 0)
                ax.plot(r.temperatures[ok], y[ok], color=color, ls=PROCESS_STYLES.get(name, "-"), lw=1.2,
                        label=f"{label}, {name}" if len(processes) > 1 else label)
        ax.set_yscale("log")
        ax.set_xlabel("Temperature (K)")
        ax.set_title(f"{CHANNEL_LABELS[channel]} rate (m$_s$ {', '.join(sorted(directions))})", fontsize=9)
        ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    axes[0, 0].set_ylabel(r"rate (s$^{-1}$)")
    _legend(fig, axes)
    return fig


def build_t1_figure(entries, *, ratio: bool = False, figsize=None, dpi: int = 100) -> Figure:
    """T1 against temperature, one line per entry; with ``ratio`` a second panel of T1 / T1(first entry).
    ``entries``: (label, colour, (temperatures, T1, initial state))."""
    if not entries:
        return message_figure("Add entries", dpi=dpi)
    rows = 2 if ratio and len(entries) > 1 else 1
    fig, axes = plt.subplots(rows, 1, figsize=figsize or (6.4, 3.4 * rows), dpi=dpi, squeeze=False,
                             sharex=True, layout="constrained", height_ratios=[2.2, 1][:rows])
    ax = axes[0, 0]
    for label, color, (temps, t1, state) in entries:
        ok = np.isfinite(t1) & (t1 > 0)
        ax.plot(temps[ok], t1[ok], color=color, lw=1.4, label=f"{label} ({state})")
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
