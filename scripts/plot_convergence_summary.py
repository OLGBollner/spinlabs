"""One figure per defect from its perturbation-convergence packs: ZFS change and SNR against step size.

For each defect the newest pack per method (``data/<defect>_<date>_conv_test_<all|occup>.npz``,
written by pack_convergence) gives one column:

- top, |D(Q) - D0| per mode against |Q|, filled +Q and open -Q, and its even part
  |D(Q) + D(-Q) - 2 D0| / 2 (squares), where the linear term cancels. Lines are the cubic
  fit's linear and quadratic terms |b| Q and |c| Q^2, and the error of a difference against
  D0 (noise of two runs and D0's offset). Every norm is over the 5 traceless parts
  (spinlabs.convergence.traceless_parts), as the fit and the SNR use them;
- bottom, the signal-to-noise of the finite differences the coupling calculation uses:
  first order b Q / err1 and the i = i pair's second order 2 c Q^2 / err2, against the
  SNR target, with markers at the amplitudes that were run. A second-order line drawn faint
  has c unresolved: the fitted curvature is within noise of zero, so second order is not
  measurable at these amplitudes and the line is no estimate of its SNR.

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


def find_packs(data_dir):
    """Newest pack per (defect, method) in data_dir, as {defect: {method: path}}."""
    packs = {}
    for path in sorted(Path(data_dir).glob("*_conv_test_*.npz")):
        m = PACK.fullmatch(path.name)
        if m:  # sorted by name, so a later date replaces an earlier one
            packs.setdefault(m["defect"], {})[m["method"]] = path
    return packs


def analyse(path):
    """Per mode of one pack: the changes against D0 and the SNR report; plus the pack's noise and D0 offset.

    Each mode is (mode, perts, |D(Q) - D0| per run, (|Q|, even part) per amplitude run at
    both signs, report), every norm over the 5 traceless parts.
    """
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
        diff = np.linalg.norm(mode_parts - p0, axis=1)
        even = []
        for q in sorted({float(x) for x in p if x > 0}):
            pos, neg = np.isclose(p, q), np.isclose(p, -q)
            if neg.any():
                even.append((q, float(np.linalg.norm(mode_parts[pos][0] + mode_parts[neg][0] - 2 * p0)) / 2))
        modes.append((int(mode), p, diff, np.array(even).reshape(-1, 2), report))
    return modes, sigma, float(np.linalg.norm(offset))


def plot_defect(defect, packs, output):
    methods = [m for m in METHODS if m in packs]
    fig, axes = plt.subplots(2, len(methods), figsize=(7.5 * len(methods), 9), squeeze=False, sharex=True)
    q = np.geomspace(1e-3, 0.3, 200)
    for col, method in enumerate(methods):
        modes, sigma, offset = analyse(packs[method])
        diff_ax, snr_ax = axes[0, col], axes[1, col]
        for color, (mode, p, diff, even, r) in zip(COLORS, modes):
            # |D(Q) - D0| is nearly the same at +-Q: the -Q ring first, a smaller +Q dot inside it
            for side, face, ms in ((-1, "white", 9), (1, color, 5)):
                sel = np.sign(p) == side
                diff_ax.plot(np.abs(p[sel]), diff[sel], "o", ms=ms, color=color, mfc=face, mew=1.5)
            diff_ax.plot(even[:, 0], even[:, 1], "s", ms=7, color=color, alpha=0.9 if r.c_resolved else 0.4)
            diff_ax.plot(q, r.b * q, "-", color=color, lw=1, alpha=0.7)
            diff_ax.plot(q, r.c * q**2, ":", color=color, lw=1.2, alpha=0.9 if r.c_resolved else 0.4)
            # SNR = 1 / relative noise-and-offset error of each finite difference
            snr_ax.plot(q, 1 / r.first_order(q)[0], "-", color=color, lw=1.5)
            snr_ax.plot(q, 1 / r.second_order(q)[0], "--", color=color, lw=1.5, alpha=1.0 if r.c_resolved else 0.4)
            measured = np.unique(np.abs(p[~np.isclose(p, 0.0)]))
            snr_ax.plot(measured, 1 / r.first_order(measured)[0], "o", ms=6, color=color)
            snr_ax.plot(measured, 1 / r.second_order(measured)[0], "s", ms=6, color=color,
                        alpha=1.0 if r.c_resolved else 0.4)
        err1 = float(np.hypot(np.sqrt(2) * sigma, offset))
        diff_ax.axhline(err1, color="0.35", ls="--", lw=1)
        # left end, below the line: the measured points sit at the right
        diff_ax.annotate(f"error of D(Q) − D0: {err1:.2g} MHz", (0.0, err1), xycoords=("axes fraction", "data"),
                         xytext=(4, -4), textcoords="offset points", va="top", fontsize=10, color="0.35")
        snr_ax.axhline(SNR_TARGET, color="0.35", ls="--", lw=1)
        snr_ax.annotate(f"SNR {SNR_TARGET:g}", (0.0, SNR_TARGET), xycoords=("axes fraction", "data"),
                        xytext=(4, 4), textcoords="offset points", fontsize=10, color="0.35")
        for ax in (diff_ax, snr_ax):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="major", color="0.9", linewidth=0.6)
        diff_ax.set_title(f"{defect} {METHODS[method]}: σ = {sigma:.2g} MHz, D0 offset {offset:.2g} MHz", fontsize=12)
        diff_ax.set_ylabel(r"$\|D(Q) - D_0\|$, traceless parts (MHz)")
        snr_ax.set_ylabel("SNR of the finite difference")
        snr_ax.set_xlabel("|Q| (step size)")
        mode_handles = [Line2D([], [], marker="o", ls="", color=c, ms=7, label=f"mode {m}") for c, (m, *_) in
                        zip(COLORS, modes)]
        style_handles = [
            Line2D([], [], marker="o", ls="", color="0.35", mfc="0.35", label="+Q"),
            Line2D([], [], marker="o", ls="", color="0.35", mfc="white", label="−Q"),
            Line2D([], [], marker="s", ls="", color="0.35", label="(D(Q) + D(−Q) − 2D0)/2"),
            Line2D([], [], ls="-", color="0.35", label="|b| Q (fit)"),
            Line2D([], [], ls=":", color="0.35", label="|c| Q² (fit)"),
        ]
        modes_legend = diff_ax.legend(handles=mode_handles, loc="upper left", fontsize=10)
        diff_ax.add_artist(modes_legend)
        diff_ax.legend(handles=style_handles, loc="lower right", fontsize=10)
        snr_ax.legend(handles=[
            Line2D([], [], ls="-", marker="o", color="0.35", label="first order (D(Q) − D0)/Q"),
            Line2D([], [], ls="--", marker="s", color="0.35", label="second order (D(2Q) − 2D(Q) + D0)/Q²"),
            Line2D([], [], ls="--", color="0.35", alpha=0.4, label="second order, c unresolved"),
        ], loc="lower right", fontsize=10)
        print(f"{defect} {method}: {packs[method]}, {len(modes)} modes, sigma {sigma:.3g} MHz, offset {offset:.3g} MHz")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = Parser("Plots one convergence figure per defect: ZFS change and SNR against step size.")
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
