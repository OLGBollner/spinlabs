"""Norms of the first- and second-order ZFS tensor differences per mode pair.

For every second-order pair (i, j) in the packaged data this plots the Frobenius
norms of

    D_qi    - D_0   (first-order run along mode i)
    D_qj    - D_0   (first-order run along mode j)
    D_qi_qj - D_0   (second-order run along modes i and j)

for each perturbation scale, and the ratio of the norms between the largest and
smallest scale on the pairs both share. A difference linear in the displacement
scales with the ratio of the perturbation scales, a quadratic one with its square.
The packages are looked up in the spinlabs registry (latest per configuration).
"""

from argparse import ArgumentParser as Parser

import matplotlib.pyplot as plt
import numpy as np

from spinlabs.registry import ensure_db, find_packaged

COLORS = {"D_qi": "#2a78d6", "D_qj": "#eb6834", "D_qi_qj": "#1baf7a"}
MARKERS = {"D_qi": "o", "D_qj": "x", "D_qi_qj": "s"}
LABELS = {
    "D_qi": r"$\|D_{q_i} - D_0\|$",
    "D_qj": r"$\|D_{q_j} - D_0\|$",
    "D_qi_qj": r"$\|D_{q_i q_j} - D_0\|$",
}


def tensor_difference_norms(path):
    """Frobenius norms (MHz) of D_qi, D_qj and D_qi_qj minus D_0, keyed by pair (i, j).

    Pairs whose first-order run along i or j is missing are skipped.
    """
    data = np.load(path, allow_pickle=True)
    D_0 = data["ground_state_zfs_matrix"]
    first = data["first_order"].item()
    second = data["second_order"].item()
    norms = {}
    for key, entry in second.items():
        i, j = (int(k) for k in key.split("_"))
        if i not in first or j not in first:
            continue
        norms[(i, j)] = {
            "D_qi": np.linalg.norm(first[i]["tensor"] - D_0),
            "D_qj": np.linalg.norm(first[j]["tensor"] - D_0),
            "D_qi_qj": np.linalg.norm(entry["tensor"] - D_0),
        }
    return norms


def find_package(defect, cell, method, pert):
    conn = ensure_db()
    paths = find_packaged(
        conn, defect=defect, cell_size=cell, calc_method=method, pert_scale=pert, pert_order=2, latest=True
    )
    conn.close()
    if not paths:
        raise SystemExit(f"No packaged second-order data for {defect}_{cell} {method} pert {pert} in the registry")
    return paths[0]


def pair_label(pairs):
    return "mode $i$" if all(i == j for i, j in pairs) else "pair $(i, j)$, sorted by $i$"


if __name__ == "__main__":
    parser = Parser("Plots the norms of the ZFS tensor differences D_qi, D_qj and D_qi_qj.")
    parser.add_argument("--defect", default="NV")
    parser.add_argument("--cell", type=int, default=512)
    parser.add_argument("--method", default="all_bands", help="Calculation method (all_bands, defect_band_approx)")
    parser.add_argument("--perts", type=float, nargs="+", default=[0.1, 0.025], help="Perturbation scales to compare")
    parser.add_argument("-o", "--output", help="Output filename for the plot")
    parser.add_argument("-p", "--plot", action="store_true", help="Show the plot instead of saving it")
    args = parser.parse_args()

    perts = sorted(args.perts, reverse=True)
    norms = {}
    for pert in perts:
        path = find_package(args.defect, args.cell, args.method, pert)
        print(f"pert {pert}: {path}")
        norms[pert] = tensor_difference_norms(path)
        print(f"  {len(norms[pert])} pairs with both first-order runs")

    plt.rcParams.update({"axes.labelsize": 14, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11})
    fig, axes = plt.subplots(len(perts) + 1, 1, figsize=(9, 3.2 * (len(perts) + 1)), sharex=True)

    for ax, pert in zip(axes, perts):
        pairs = sorted(norms[pert])
        x = [i for i, _ in pairs]
        for name in LABELS:
            y = [norms[pert][p][name] for p in pairs]
            ax.scatter(x, y, s=10 if name != "D_qj" else 14, marker=MARKERS[name], color=COLORS[name],
                       alpha=0.7, linewidths=1, label=LABELS[name])
        ax.set_yscale("log")
        ax.set_ylabel("Norm (MHz)")
        ax.set_title(f"{args.defect}_{args.cell} {args.method}, pert {pert}", fontsize=12)
        ax.grid(True, which="major", color="0.9", linewidth=0.6)
        ax.legend(loc="lower right", markerscale=2)

    # Ratio between the largest and smallest perturbation scale on the shared pairs
    big, small = perts[0], perts[-1]
    shared = sorted(set(norms[big]) & set(norms[small]))
    ax = axes[-1]
    x = [i for i, _ in shared]
    for name in LABELS:
        ratio = [norms[big][p][name] / norms[small][p][name] for p in shared]
        ax.scatter(x, ratio, s=10 if name != "D_qj" else 14, marker=MARKERS[name], color=COLORS[name],
                   alpha=0.7, linewidths=1, label=LABELS[name])
    scale = big / small
    for power, style in ((1, "--"), (2, ":")):
        ax.axhline(scale**power, color="0.35", linestyle=style, linewidth=1)
        ax.annotate(f"{'linear' if power == 1 else 'quadratic'} ({scale**power:g})", (1.0, scale**power),
                    xycoords=("axes fraction", "data"), xytext=(-4, 3), textcoords="offset points",
                    ha="right", fontsize=10, color="0.35")
    ax.set_yscale("log")
    ax.set_ylabel(f"Ratio pert {big} / {small}")
    ax.set_title(f"{len(shared)} shared pairs", fontsize=12)
    ax.grid(True, which="major", color="0.9", linewidth=0.6)
    ax.legend(loc="lower left", markerscale=2)
    ax.set_xlabel(pair_label(shared or sorted(norms[big])))

    fig.tight_layout()
    if args.plot:
        plt.show()
    else:
        output = args.output or f"{args.defect}_{args.cell}_{args.method}_tensor_differences.png"
        fig.savefig(output, dpi=200)
        print(f"Saved {output}")
