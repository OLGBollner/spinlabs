"""First- and second-order ZFS derivatives per mode, compared between perturbation scales.

For each perturbation scale this plots, against phonon energy, the Frobenius norms of

    dD/dq_i         = (D_qi - D_0) / q_i
    d2D/dq_i dq_j   = (D_qi_qj - D_0) / (q_i q_j) - (dD/dq_i) / q_j - (dD/dq_j) / q_i

as ZFSManager computes them, and the ratio of the norms between the largest and
smallest scale. Next to each derivative panel are the spin-phonon coupling
coefficients V_00, V_+-, V_0+- computed from those derivatives. q is the dimensionless mode coordinate q0 * sqrt(2 omega / hbar),
including the effective sqrt(mass) correction for non-mass-weighted data, so a
linear response gives the same dD/dq at every scale and a quadratic one the same
d2D/dq2: the ratios sit at 1.

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
from spinlabs.symmetry import twin_of
from spinlabs.zfs_manager import ZFSManager

# One categorical slot per perturbation scale (largest first) or per derivative order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LABELS = {1: r"$\|\partial D / \partial q_i\|$", 2: r"$\|\partial^2 D / \partial q_i \partial q_j\|$"}
MARKERS = {1: "o", 2: "s"}
# Coupling coefficients in the order ZFSManager returns them, a marker each
COUPLINGS = {"V_00": ("$V_{00}$", "o"), "V_pm": ("$V_{+-}$", "^"), "V_0pm": (r"$V_{0\pm}$", "s")}


def find_package(defect, cell, method, pert):
    conn = ensure_db()
    paths = find_packaged(
        conn, defect=defect, cell_size=cell, calc_method=method, pert_scale=pert, pert_order=2, latest=True
    )
    conn.close()
    if not paths:
        raise SystemExit(f"No packaged second-order data for {defect}_{cell} {method} pert {pert} in the registry")
    return paths[0]


def derivative_norms(path, defect, cell, phonon_file=None):
    """Norms (MHz) of dD/dq per mode and of d2D/dq_i dq_j per pair, keyed by spectrum position.

    Also returns the non-zero coupling coefficients (MHz) per order and coefficient, keyed the
    same way, the phonon spectrum, whether the data is mass-weighted and the phonon file used.
    """
    raw, spectrum, ph_path = load_raw_inputs(raw_zfs_file=path, phonon_file=phonon_file, defect=defect, cell_size=cell)
    if spectrum is None:
        raise SystemExit(f"No phonon spectrum found for {path}; pass --phonon-file")
    manager = ZFSManager(raw_data=raw, spectrum=spectrum)
    d1, *V1 = manager.calculate_first_order_derivatives()
    d2, *V2 = manager.calculate_second_order_derivatives(d1)
    # Untreated modes and skipped pairs are left at zero by ZFSManager; its tensors are in J
    first = {i: np.linalg.norm(d1[i]) / CONSTANTS["MHz2J"] for i in manager.zfs_tensors if np.any(d1[i])}
    second = {
        (i, j): np.linalg.norm(d2[i, j]) / CONSTANTS["MHz2J"] for i, j in manager.zfs_tensors_2d if np.any(d2[i, j])
    }
    couplings = {
        1: {name: {i: V[i] / CONSTANTS["MHz2J"] for i in first if V[i]} for name, V in zip(COUPLINGS, V1)},
        2: {name: {k: V[k] / CONSTANTS["MHz2J"] for k in second if V[k]} for name, V in zip(COUPLINGS, V2)},
    }
    return first, second, couplings, spectrum, manager.mass_weighted, ph_path


def match_with_twins(big, small, deg_groups, order):
    """Keys of ``big`` paired with the same key in ``small``, else with its degenerate twin's.

    The scales may have run different partners of a degenerate E pair (Ex at one,
    Ey at the other). Their derivative norms are equal by symmetry: |dD/dEx| = |dD/dEy|,
    and d2D/dEx2, d2D/dEy2 share their A1 part and have opposite E parts.
    """
    def twin(i):
        t = twin_of(deg_groups, i)
        return i if t is None else t

    matched = []
    for key in sorted(big):
        partner = key
        if key not in small:
            partner = twin(key) if order == 1 else (twin(key[0]), twin(key[1]))
        if partner in small:
            matched.append((key, partner))
    return matched


if __name__ == "__main__":
    parser = Parser("Plots the first- and second-order ZFS derivatives per mode for several perturbation scales.")
    parser.add_argument("--defect", default="NV")
    parser.add_argument("--cell", type=int, default=512)
    parser.add_argument("--method", default="all_bands", help="Calculation method (all_bands, defect_band_approx)")
    parser.add_argument("--perts", type=float, nargs="+", default=[0.1, 0.025], help="Perturbation scales to compare")
    parser.add_argument("--phonon-file", help="Phonon npz (default: the registry's match for defect and cell)")
    parser.add_argument("-o", "--output", help="Output filename for the plot")
    parser.add_argument("-p", "--plot", action="store_true", help="Show the plot instead of saving it")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    perts = sorted(args.perts, reverse=True)
    norms, couplings = {}, {}
    for pert in perts:
        path = find_package(args.defect, args.cell, args.method, pert)
        first, second, coupling, spectrum, weighted, ph_path = derivative_norms(
            path, args.defect, args.cell, args.phonon_file
        )
        freqs = spectrum.frequencies_mev
        norms[pert] = {1: first, 2: second}
        couplings[pert] = coupling
        print(f"pert {pert}: {path}")
        print(f"  phonons {ph_path}, {'mass-weighted' if weighted else 'not mass-weighted'}, "
              f"{len(first)} modes, {len(second)} pairs")

    plt.rcParams.update({"axes.labelsize": 14, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11})
    # Derivatives on the left, their coupling coefficients on the right, the ratio across the bottom
    fig = plt.figure(figsize=(17, 10))
    grid = GridSpec(3, 2, figure=fig)
    ratio_ax = fig.add_subplot(grid[2, :])
    deriv_axes = [fig.add_subplot(grid[r, 0], sharex=ratio_ax) for r in range(2)]
    coupling_axes = [fig.add_subplot(grid[r, 1], sharex=ratio_ax) for r in range(2)]
    title = f"{args.defect}_{args.cell} {args.method}"

    # Pairs are placed at the energy of mode i
    for ax, order in zip(deriv_axes, (1, 2)):
        for color, pert in zip(COLORS, perts):
            keys = sorted(norms[pert][order])
            x = [freqs[k if order == 1 else k[0]] for k in keys]
            ax.scatter(x, [norms[pert][order][k] for k in keys], s=10, marker=MARKERS[order], color=color,
                       alpha=0.7, linewidths=0, label=f"pert {pert}")
        ax.set_yscale("log")
        ax.set_ylabel(f"{LABELS[order]} (MHz)")
        ax.set_title(f"{title}, {'first' if order == 1 else 'second'} order", fontsize=12)
        ax.grid(True, which="major", color="0.9", linewidth=0.6)
        ax.legend(loc="lower right", markerscale=2)
        ax.tick_params(labelbottom=order == 2)

    for ax, order in zip(coupling_axes, (1, 2)):
        for color, pert in zip(COLORS, perts):
            for name, (label, marker) in COUPLINGS.items():
                values = couplings[pert][order][name]
                keys = sorted(values)
                x = [freqs[k if order == 1 else k[0]] for k in keys]
                ax.scatter(x, [values[k] for k in keys], s=12, marker=marker, color=color, alpha=0.7,
                           linewidths=0)
        ax.set_yscale("log")
        ax.set_ylabel("Coupling (MHz)")
        ax.set_title(f"{title}, {'first' if order == 1 else 'second'}-order couplings", fontsize=12)
        ax.grid(True, which="major", color="0.9", linewidth=0.6)
        # Colour is the scale, marker the coefficient: one legend each
        scales = ax.legend(
            handles=[Line2D([], [], marker="o", linestyle="", color=c, label=f"pert {p}") for c, p in zip(COLORS, perts)],
            loc="lower left", fontsize=9,
        )
        ax.add_artist(scales)
        ax.legend(
            handles=[Line2D([], [], marker=m, linestyle="", color="0.35", label=lab) for lab, m in COUPLINGS.values()],
            loc="lower right", ncol=3, fontsize=9,
        )
        ax.tick_params(labelbottom=order == 2)

    # Ratio between the largest and smallest perturbation scale on the shared modes/pairs,
    # a degenerate partner standing in for a mode the other scale did not run
    big, small = perts[0], perts[-1]
    ax = ratio_ax
    for color, order in zip(COLORS, (1, 2)):
        shared = match_with_twins(norms[big][order], norms[small][order], spectrum.deg_groups, order)
        x = [freqs[k if order == 1 else k[0]] for k, _ in shared]
        ratio = [norms[big][order][k] / norms[small][order][p] for k, p in shared]
        n_twins = sum(k != p for k, p in shared)
        ax.scatter(x, ratio, s=10, marker=MARKERS[order], color=color, alpha=0.7, linewidths=0,
                   label=f"{LABELS[order]} ({len(shared)}, {n_twins} via twin)")
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel(f"Ratio pert {big} / {small}")
    ax.set_title("Scale independence (1 = same derivative at both scales)", fontsize=12)
    ax.grid(True, which="major", color="0.9", linewidth=0.6)
    ax.legend(loc="lower right", markerscale=2)
    ax.set_xlabel("Phonon energy of mode $i$ (meV)")

    fig.tight_layout()
    if args.plot:
        plt.show()
    else:
        output = args.output or f"{args.defect}_{args.cell}_{args.method}_derivative_scaling.png"
        fig.savefig(output, dpi=200)
        print(f"Saved {output}")
