"""First- against second-order ZFS changes per mode, for several perturbation scales.

For each perturbation scale this plots, against phonon energy:

- left, the Frobenius norms of the raw tensor differences D_qi - D_0 (first-order
  run along mode i) and D_qi_qj - D_0 (second-order run along modes i and j);
- right, the spin-phonon coupling coefficients V_00, V_+-, V_0+- ZFSManager
  computes from them, with their spectral functions F(w): V^2 broadened by a
  Gaussian of width --sigma, as plot_1d_spectral_functions draws them;
- bottom, |D_qi_qi - D_0| / |D_qi - D_0| for the diagonal pairs. A diagonal
  pair displaces 2 q_i, so a linear response gives 2 and a quadratic one 4.

Each package is loaded as ``spinlabs run`` loads it (latest per configuration in
the registry, with the registered phonon file unless --phonon-file is given).
"""

import logging
from argparse import ArgumentParser as Parser

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from spinlabs.constants import CONSTANTS
from spinlabs.pipeline import load_raw_inputs
from spinlabs.registry import ensure_db, find_packaged
from spinlabs.utils import MathUtils
from spinlabs.zfs_manager import ZFSManager

# One categorical slot per perturbation scale, largest first
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LABELS = {1: r"$\|D_{q_i} - D_0\|$", 2: r"$\|D_{q_i q_j} - D_0\|$"}
MARKERS = {1: "o", 2: "s"}
# Coupling coefficients in the order ZFSManager returns them: label, marker, spectral-function line style
COUPLINGS = {
    "V_00": ("$V_{00}$", "o", "-"),
    "V_pm": ("$V_{+-}$", "^", "--"),
    "V_0pm": (r"$V_{0\pm}$", "s", ":"),
}
SPECTRAL_RES = 1.0  # meV grid of the spectral functions


def find_package(defect, cell, method, pert):
    conn = ensure_db()
    paths = find_packaged(
        conn, defect=defect, cell_size=cell, calc_method=method, pert_scale=pert, pert_order=2, latest=True
    )
    conn.close()
    if not paths:
        raise SystemExit(f"No packaged second-order data for {defect}_{cell} {method} pert {pert} in the registry")
    return paths[0]


def load_scale(path, defect, cell, phonon_file=None):
    """Tensor-difference norms and coupling coefficients (MHz) of one package.

    Returns ``(diffs, couplings, spectrum, mass_weighted, phonon_path)``. ``diffs[1]`` maps
    spectrum positions i to |D_qi - D_0|, ``diffs[2]`` pairs (i, j) to |D_qi_qj - D_0|.
    ``couplings[order][name]`` is the per-mode coefficient array; for second order the
    diagonal V(i, i), as plot_run_coupling takes it.
    """
    raw, spectrum, ph_path = load_raw_inputs(raw_zfs_file=path, phonon_file=phonon_file, defect=defect, cell_size=cell)
    if spectrum is None:
        raise SystemExit(f"No phonon spectrum found for {path}; pass --phonon-file")
    manager = ZFSManager(raw_data=raw, spectrum=spectrum)
    d1, *V1 = manager.calculate_first_order_derivatives()
    _, *V2 = manager.calculate_second_order_derivatives(d1)
    # ZFSManager holds tensors and couplings in J
    to_mhz = 1.0 / CONSTANTS["MHz2J"]
    D_0 = manager.zfs_relaxed
    diffs = {
        1: {i: np.linalg.norm(e["tensor"] - D_0) * to_mhz for i, e in manager.zfs_tensors.items()},
        2: {k: np.linalg.norm(e["tensor"] - D_0) * to_mhz for k, e in manager.zfs_tensors_2d.items()},
    }
    couplings = {
        1: {name: np.asarray(V) * to_mhz for name, V in zip(COUPLINGS, V1)},
        2: {name: np.diag(V) * to_mhz for name, V in zip(COUPLINGS, V2)},
    }
    return diffs, couplings, spectrum, manager.mass_weighted, ph_path


if __name__ == "__main__":
    parser = Parser("Plots first- against second-order ZFS changes and couplings for several perturbation scales.")
    parser.add_argument("--defect", default="NV")
    parser.add_argument("--cell", type=int, default=512)
    parser.add_argument("--method", default="all_bands", help="Calculation method (all_bands, defect_band_approx)")
    parser.add_argument("--perts", type=float, nargs="+", default=[0.1, 0.025], help="Perturbation scales to compare")
    parser.add_argument("--phonon-file", help="Phonon npz (default: the registry's match for defect and cell)")
    parser.add_argument("--sigma", type=float, default=7.5, help="Gaussian broadening of the spectral functions (meV)")
    parser.add_argument("-o", "--output", help="Output filename for the plot")
    parser.add_argument("-p", "--plot", action="store_true", help="Show the plot instead of saving it")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    perts = sorted(args.perts, reverse=True)
    diffs, couplings = {}, {}
    for pert in perts:
        path = find_package(args.defect, args.cell, args.method, pert)
        diffs[pert], couplings[pert], spectrum, weighted, ph_path = load_scale(
            path, args.defect, args.cell, args.phonon_file
        )
        freqs = spectrum.frequencies_mev
        print(f"pert {pert}: {path}")
        print(f"  phonons {ph_path}, {'mass-weighted' if weighted else 'not mass-weighted'}, "
              f"{len(diffs[pert][1])} modes, {len(diffs[pert][2])} pairs")

    plt.rcParams.update({"axes.labelsize": 14, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11})
    # Tensor differences on the left, couplings and spectral functions on the right,
    # the second- over first-order ratio across the bottom
    fig = plt.figure(figsize=(17, 10))
    grid = GridSpec(3, 2, figure=fig)
    ratio_ax = fig.add_subplot(grid[2, :])
    diff_axes = [fig.add_subplot(grid[r, 0], sharex=ratio_ax) for r in range(2)]
    coupling_axes = [fig.add_subplot(grid[r, 1], sharex=ratio_ax) for r in range(2)]
    title = f"{args.defect}_{args.cell} {args.method}"
    scale_handles = [Line2D([], [], marker="o", linestyle="", color=c, label=f"pert {p}") for c, p in zip(COLORS, perts)]

    # Pairs are placed at the energy of mode i
    for ax, order in zip(diff_axes, (1, 2)):
        for color, pert in zip(COLORS, perts):
            keys = sorted(diffs[pert][order])
            x = [freqs[k if order == 1 else k[0]] for k in keys]
            ax.scatter(x, [diffs[pert][order][k] for k in keys], s=10, marker=MARKERS[order], color=color,
                       alpha=0.7, linewidths=0, label=f"pert {pert}")
        ax.set_yscale("log")
        ax.set_ylabel(f"{LABELS[order]} (MHz)")
        ax.set_title(f"{title}, {'first' if order == 1 else 'second'}-order tensor differences", fontsize=12)
        ax.grid(True, which="major", color="0.9", linewidth=0.6)
        ax.legend(loc="lower right", markerscale=2)
        ax.tick_params(labelbottom=order == 2)

    for ax, order in zip(coupling_axes, (1, 2)):
        spectral_ax = ax.twinx()
        for color, pert in zip(COLORS, perts):
            for name, (label, marker, style) in COUPLINGS.items():
                V = couplings[pert][order][name]
                nonzero = np.flatnonzero(V)
                ax.scatter(freqs[nonzero], V[nonzero], s=12, marker=marker, color=color, alpha=0.5, linewidths=0)
                w, F = MathUtils.smear_data(freqs, V**2, SPECTRAL_RES, args.sigma)
                spectral_ax.plot(w, F, color=color, linestyle=style, linewidth=1.5)
        ax.set_yscale("log")
        ax.set_ylabel("Coupling (MHz)")
        spectral_ax.set_ylabel(r"Spectral function (MHz$^2$/meV)")
        spectral_ax.set_ylim(bottom=0)
        ax.set_title(f"{title}, {'first' if order == 1 else 'second'}-order couplings, "
                     rf"$\sigma$ = {args.sigma:g} meV", fontsize=12)
        ax.grid(True, which="major", color="0.9", linewidth=0.6)
        # Colour is the scale; marker (coupling) and line style (spectral function) the coefficient
        scales = spectral_ax.legend(handles=scale_handles, loc="upper left", fontsize=9)
        spectral_ax.add_artist(scales)
        spectral_ax.legend(
            handles=[Line2D([], [], marker=m, linestyle=ls, color="0.35", label=lab) for lab, m, ls in COUPLINGS.values()],
            loc="upper right", ncol=3, fontsize=9,
        )
        ax.tick_params(labelbottom=order == 2)

    # Second- over first-order change of the diagonal pairs, per scale
    ax = ratio_ax
    for color, pert in zip(COLORS, perts):
        first = diffs[pert][1]
        keys = sorted(k for k in diffs[pert][2] if k[0] == k[1] and k[0] in first and first[k[0]] > 0)
        ratio = [diffs[pert][2][k] / first[k[0]] for k in keys]
        ax.scatter([freqs[i] for i, _ in keys], ratio, s=10, marker="o", color=color, alpha=0.7, linewidths=0,
                   label=f"pert {pert} ({len(keys)})")
    # The two lines sit close on the log axis: label linear below its line, quadratic above
    for value, style, name, dy, va in ((2.0, "--", "linear", -3, "top"), (4.0, ":", "quadratic", 3, "bottom")):
        ax.axhline(value, color="0.35", linestyle=style, linewidth=1)
        ax.annotate(f"{name} ({value:g})", (0.0, value), xycoords=("axes fraction", "data"), xytext=(4, dy),
                    textcoords="offset points", va=va, fontsize=10, color="0.35")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\|D_{q_i q_i} - D_0\| \,/\, \|D_{q_i} - D_0\|$")
    ax.set_title("Second- against first-order change (pair displaced 2$q_i$)", fontsize=12)
    ax.grid(True, which="major", color="0.9", linewidth=0.6)
    ax.legend(loc="lower right", markerscale=2)
    ax.set_xlabel("Phonon energy of mode $i$ (meV)")

    fig.tight_layout()
    if args.plot:
        plt.show()
    else:
        output = args.output or f"{args.defect}_{args.cell}_{args.method}_first_vs_second_order.png"
        fig.savefig(output, dpi=200)
        print(f"Saved {output}")
