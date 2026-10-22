"""First- and second-order ZFS derivatives per mode, compared between perturbation scales.

For each perturbation scale this plots, against phonon energy, the Frobenius norms of

    dD/dq_i         = (D_qi - D_0) / q_i
    d2D/dq_i dq_j   = (D_qi_qj - D_0) / (q_i q_j) - (dD/dq_i) / q_j - (dD/dq_j) / q_i

as ZFSManager computes them, and the ratio of the norms between the largest and
smallest scale. q is the dimensionless mode coordinate q0 * sqrt(2 omega / hbar),
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

from spinlabs.constants import CONSTANTS
from spinlabs.pipeline import load_raw_inputs
from spinlabs.registry import ensure_db, find_packaged
from spinlabs.zfs_manager import ZFSManager

# One categorical slot per perturbation scale (largest first) or per derivative order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LABELS = {1: r"$\|\partial D / \partial q_i\|$", 2: r"$\|\partial^2 D / \partial q_i \partial q_j\|$"}
MARKERS = {1: "o", 2: "s"}


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

    Also returns the phonon energies (meV), whether the data is mass-weighted and the phonon file used.
    """
    raw, spectrum, ph_path = load_raw_inputs(raw_zfs_file=path, phonon_file=phonon_file, defect=defect, cell_size=cell)
    if spectrum is None:
        raise SystemExit(f"No phonon spectrum found for {path}; pass --phonon-file")
    manager = ZFSManager(raw_data=raw, spectrum=spectrum)
    d1, *_ = manager.calculate_first_order_derivatives()
    d2, *_ = manager.calculate_second_order_derivatives(d1)
    # Untreated modes and skipped pairs are left at zero by ZFSManager; its tensors are in J
    first = {i: np.linalg.norm(d1[i]) / CONSTANTS["MHz2J"] for i in manager.zfs_tensors if np.any(d1[i])}
    second = {
        (i, j): np.linalg.norm(d2[i, j]) / CONSTANTS["MHz2J"] for i, j in manager.zfs_tensors_2d if np.any(d2[i, j])
    }
    return first, second, spectrum.frequencies_mev, manager.mass_weighted, ph_path


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
    norms = {}
    for pert in perts:
        path = find_package(args.defect, args.cell, args.method, pert)
        first, second, freqs, weighted, ph_path = derivative_norms(path, args.defect, args.cell, args.phonon_file)
        norms[pert] = {1: first, 2: second}
        print(f"pert {pert}: {path}")
        print(f"  phonons {ph_path}, {'mass-weighted' if weighted else 'not mass-weighted'}, "
              f"{len(first)} modes, {len(second)} pairs")

    plt.rcParams.update({"axes.labelsize": 14, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11})
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    title = f"{args.defect}_{args.cell} {args.method}"

    # Pairs are placed at the energy of mode i
    for ax, order in zip(axes, (1, 2)):
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

    # Ratio between the largest and smallest perturbation scale on the shared modes/pairs
    big, small = perts[0], perts[-1]
    ax = axes[-1]
    for color, order in zip(COLORS, (1, 2)):
        shared = sorted(set(norms[big][order]) & set(norms[small][order]))
        x = [freqs[k if order == 1 else k[0]] for k in shared]
        ratio = [norms[big][order][k] / norms[small][order][k] for k in shared]
        ax.scatter(x, ratio, s=10, marker=MARKERS[order], color=color, alpha=0.7, linewidths=0,
                   label=f"{LABELS[order]} ({len(shared)})")
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
