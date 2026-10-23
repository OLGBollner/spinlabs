"""One figure per defect from its perturbation-convergence packs: finite differences and SNR against step size.

It shows how the finite differences spinlabs takes from the ZFS tensor behave as the step
size changes. For each defect the newest pack per method
(``data/<defect>_<date>_conv_test_<all|occup>.npz``, written by pack_convergence) gives one
column of three rows:

- first order, |D(Q) - D0| / Q per mode against |Q| (filled +Q, open -Q), with noise error
  bars sqrt 2 sigma / Q and the cubic fit's Q -> 0 value b as a line;
- second order, the forward difference (D(2Q) - 2 D(Q) + D0) / Q^2 that
  ZFSManager.calculate_second_order_derivatives takes from the i = i pair run (D(2Q)) and the
  first-order run (D(Q)), for every Q whose 2Q was run on the same side (filled +Q, open -Q),
  with noise error bars sqrt 6 sigma / Q^2 and the fit's 2c as a line;
- the signal-to-noise of those finite differences: first order b Q / err1 and the forward
  second order 2 c Q^2 / err2, against the SNR target (SNR above it shaded), with markers
  at the amplitudes that were run.

Every norm is over the 5 traceless parts (spinlabs.convergence.traceless_parts), as the fit
and the SNR use them. The grey dashed curve in a quotient row is the error of that quotient
with D0's offset included, err1 / Q and err2 / Q^2: a point near or below it is noise, not
a measurement. A line drawn faint has its coefficient unresolved (b for first order, c for
second order): the fitted value is within noise of zero, so that order is not measurable at
these amplitudes and the line is no estimate of its size or SNR.

Noise sigma and D0's offset are pooled over all modes of a pack and the coefficients come
from spinlabs.convergence.snr_report, as ``spinlabs plot convergence`` reports them.
"""

import logging
import re
from argparse import ArgumentParser as Parser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from spinlabs.convergence import (
    SNR_TARGET,
    d0_offset,
    load_convergence,
    pooled_noise,
    principal_frame,
    snr_report,
    traceless_parts,
)

# One categorical slot per mode, in pack order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
METHODS = {"all": "all bands", "occup": "defect band approx"}
PACK = re.compile(r"(?P<defect>.+)_(?P<date>\d{8})_conv_test_(?P<method>all|occup)\.npz")
FAINT = 0.4  # alpha of an unresolved coefficient's lines and points


def find_packs(data_dir):
    """Newest pack per (defect, method) in data_dir, as {defect: {method: path}}."""
    packs = {}
    for path in sorted(Path(data_dir).glob("*_conv_test_*.npz")):
        m = PACK.fullmatch(path.name)
        if m:  # sorted by name, so a later date replaces an earlier one
            packs.setdefault(m["defect"], {})[m["method"]] = path
    return packs


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


def analyse(path):
    """Per mode of one pack: (mode, perts, first, second quotients, SNR report); plus noise and D0 offset."""
    conv = load_convergence(path)
    if conv.q0 is None:
        raise SystemExit(f"{path} has no Q=0 point (no 0 in perts, no d0 key); cannot measure differences")
    frame = principal_frame(conv.q0)
    p0 = traceless_parts(conv.q0, frame)
    finished = [~np.isnan(t).any(axis=(1, 2)) for t in conv.tensors]
    perts = [conv.perts[ok] for ok in finished]
    parts = [np.stack([traceless_parts(t, frame) for t in tensors[ok]]) for tensors, ok in zip(conv.tensors, finished)]
    sigma, dof = pooled_noise(perts, parts)
    offset = d0_offset(perts, parts, p0)
    modes = []
    for mode, p, mode_parts in zip(conv.modes, perts, parts):
        report = snr_report(p, mode_parts, sigma, f"pooled, {dof} dof", offset)
        if report is None:
            logging.warning(f"{path.name} mode {mode}: too few finished amplitudes for an SNR estimate, skipped")
            continue
        modes.append((int(mode), p, *quotients(p, mode_parts, p0), report))
    return modes, sigma, float(np.linalg.norm(offset))


def plot_signed(ax, rows, marker, color, yerr, alpha, big=9, small=5):
    """Points at |Q|; nearly equal at +-Q, so the -Q ring first and a smaller +Q dot inside it."""
    for side, face, ms in ((-1, "white", big), (1, color, small)):
        sel = np.sign(rows[:, 0]) == side
        q = np.abs(rows[sel, 0])
        ax.errorbar(q, rows[sel, 1], yerr=yerr(q), fmt=marker, ms=ms, color=color, mfc=face, mew=1.5,
                    capsize=3, elinewidth=1, alpha=alpha)


def plot_defect(defect, packs, output):
    methods = [m for m in METHODS if m in packs]
    fig, axes = plt.subplots(3, len(methods), figsize=(7.5 * len(methods), 13), squeeze=False, sharex=True)
    q = np.geomspace(1e-3, 0.3, 200)
    for col, method in enumerate(methods):
        modes, sigma, offset = analyse(packs[method])
        first_ax, second_ax, snr_ax = axes[:, col]
        for color, (mode, p, first, second, r) in zip(COLORS, modes):
            a1 = 1.0 if r.b_resolved else FAINT
            a2 = 1.0 if r.c_resolved else FAINT
            plot_signed(first_ax, first, "o", color, lambda x: np.sqrt(2) * sigma / x, a1)
            first_ax.axhline(r.b, color=color, lw=1, alpha=0.7 * a1)
            plot_signed(second_ax, second, "s", color, lambda x: np.sqrt(6) * sigma / x**2, a2)
            second_ax.axhline(2 * r.c, color=color, lw=1, alpha=0.7 * a2)
            # SNR = 1 / relative noise-and-offset error of each finite difference
            snr_ax.plot(q, 1 / r.first_order(q)[0], "-", color=color, lw=1.5, alpha=a1)
            snr_ax.plot(q, 1 / r.second_order(q)[0], "--", color=color, lw=1.5, alpha=a2)
            measured = np.unique(np.abs(p[~np.isclose(p, 0.0)]))
            snr_ax.plot(measured, 1 / r.first_order(measured)[0], "o", ms=6, color=color, alpha=a1)
            snr_ax.plot(measured, 1 / r.second_order(measured)[0], "s", ms=6, color=color, alpha=a2)
        # the quotient's error with D0's offset: points near or below it are not measured
        err1 = float(np.hypot(np.sqrt(2) * sigma, offset))
        err2 = float(np.hypot(np.sqrt(6) * sigma, offset))
        first_ax.plot(q, err1 / q, "--", color="0.35", lw=1)
        second_ax.plot(q, err2 / q**2, "--", color="0.35", lw=1)
        snr_ax.axhline(SNR_TARGET, color="0.35", ls="--", lw=1)
        snr_ax.annotate(f"SNR ≥ {SNR_TARGET:g}", (0.0, SNR_TARGET), xycoords=("axes fraction", "data"),
                        xytext=(4, 4), textcoords="offset points", fontsize=10, color="0.35")
        for ax in (first_ax, second_ax, snr_ax):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="major", color="0.9", linewidth=0.6)
        # shade SNR >= target behind the curves, keeping the limits the curves set
        ylo, yhi = snr_ax.get_ylim()
        snr_ax.axhspan(SNR_TARGET, yhi, color="#e5f4e4", zorder=0, lw=0)
        snr_ax.set_ylim(ylo, yhi)
        first_ax.set_title(f"{defect} {METHODS[method]}: σ = {sigma:.2g} MHz, D0 offset {offset:.2g} MHz", fontsize=12)
        first_ax.set_ylabel(r"first order $\|D(Q) - D_0\| / Q$ (MHz/Q)")
        second_ax.set_ylabel(r"second order $\|D(2Q) - 2D(Q) + D_0\| / Q^2$ (MHz/Q²)")
        snr_ax.set_ylabel("SNR of the finite difference")
        snr_ax.set_xlabel("|Q| (step size)")
        mode_handles = [Line2D([], [], marker="o", ls="", color=c, ms=7, label=f"mode {m}") for c, (m, *_) in
                        zip(COLORS, modes)]
        error_handle = Line2D([], [], ls="--", color="0.35", label="error incl. D0 offset")
        modes_legend = first_ax.legend(handles=mode_handles, loc="lower right", fontsize=10)
        first_ax.add_artist(modes_legend)
        first_ax.legend(handles=[
            Line2D([], [], marker="o", ls="", color="0.35", mfc="0.35", label="+Q"),
            Line2D([], [], marker="o", ls="", color="0.35", mfc="white", label="−Q"),
            Line2D([], [], ls="-", color="0.35", label="b (fit, Q → 0)"),
            error_handle,
        ], loc="lower left", fontsize=10)
        second_ax.legend(handles=[
            Line2D([], [], marker="s", ls="", color="0.35", mfc="0.35", label="+Q"),
            Line2D([], [], marker="s", ls="", color="0.35", mfc="white", label="−Q"),
            Line2D([], [], ls="-", color="0.35", label="2c (fit, Q → 0)"),
            error_handle,
        ], loc="lower left", fontsize=10)
        snr_ax.legend(handles=[
            Line2D([], [], ls="-", marker="o", color="0.35", label="first order (D(Q) − D0)/Q"),
            Line2D([], [], ls="--", marker="s", color="0.35", label="second order (D(2Q) − 2D(Q) + D0)/Q²"),
            Line2D([], [], ls="-", color="0.35", alpha=FAINT, label="first order, b unresolved"),
            Line2D([], [], ls="--", color="0.35", alpha=FAINT, label="second order, c unresolved"),
        ], loc="lower right", fontsize=10)
        print(f"{defect} {method}: {packs[method]}, {len(modes)} modes, sigma {sigma:.3g} MHz, offset {offset:.3g} MHz")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = Parser("Plots one convergence figure per defect: finite differences and SNR against step size.")
    parser.add_argument("--data", default="data", help="Folder with the <defect>_<date>_conv_test_*.npz packs")
    parser.add_argument("--defect", nargs="+", help="Defects to plot, e.g. NV_512 (default: every pack found)")
    parser.add_argument("-o", "--output-dir", default=".", help="Folder for <defect>_convergence_summary.png")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    packs = find_packs(args.data)
    if not packs:
        raise SystemExit(f"No *_conv_test_*.npz packs in {args.data}")
    for defect in args.defect or sorted(packs):
        if defect not in packs:
            raise SystemExit(f"No convergence packs for {defect} in {args.data} (found: {', '.join(sorted(packs))})")
        plot_defect(defect, packs[defect], Path(args.output_dir) / f"{defect}_convergence_summary.png")
