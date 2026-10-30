"""2D two-phonon spectral functions of one raw ZFS package: all computed pairs against an approximation.

One row per spin-flip coupling (J_0pm, J_pm): the two maps on one log colour scale, their ratio
(approximation / computed), and both along the diagonal w_x = w_y, the line the difference process
(absorb one phonon, emit another, w_x - w_y = w_ZFS ~ 0) samples. The cut also shows the computed
pairs without the A x E pairs, which the sqrt(V_ii V_jj) fill cannot supply.

    python scripts/plot_spectral_comparison.py data/NV_64/pert_0.1/all_bands/20261021/NV_64_raw_zfs_data_all_bands_12d.npz \\
        --sigma 7.5 -o figures/sigma_scan/NV_64_spectral_2d_comparison.png
"""

import argparse
import logging
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from plot_t1_comparison import MODES, coupling_manager  # noqa: E402

from spinlabs.models import SpinPhononCouplingData  # noqa: E402
from spinlabs.utils import MathUtils  # noqa: E402

COUPLINGS = (("V2_0_pm", r"J_{0\pm}^{(2)}"), ("V2_p_m", r"J_{+-}^{(2)}"))
DYNAMIC_RANGE = 1e-6  # map colours span this fraction of the largest value


def second_order_couplings(manager):
    """V2_0_pm and V2_p_m (MHz) and the mode energies (meV) of a ZFSManager."""
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "coupling")
        manager.process_both_orders(output_filename=base)
        data = SpinPhononCouplingData.load(f"{base}.npz").to_unit("MHz").frequencies_to_unit("meV")
    couplings = {name: np.asarray(getattr(data, name), dtype=float) for name, _ in COUPLINGS}
    return np.asarray(data.frequencies, dtype=float), couplings


def drop_a_x_e(manager):
    """Leave out the computed A x E pairs, which the sqrt fill cannot supply."""
    a, e = {"A1", "A2"}, {"Ex", "Ey"}
    manager.zfs_tensors_2d = {
        k: v for k, v in manager.zfs_tensors_2d.items() if not (set(v["symmetry"]) & a and set(v["symmetry"]) & e)
    }
    return manager


def plot(w, sets, sigma, title, approx_label, out):
    """sets: {label: couplings} with the computed set first and the approximation second."""
    res = min(0.5, sigma / 4)  # the grid compute_rates uses
    labels = list(sets)
    computed, approx = labels[0], labels[1]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    for row, (name, tex) in enumerate(COUPLINGS):
        maps = {}
        for label, couplings in sets.items():
            X, Y, J = MathUtils.get_2d_spectral_density(w, couplings[name] ** 2, res, sigma)
            maps[label] = J
        vmax = max(float(maps[computed].max()), float(maps[approx].max()))
        norm = LogNorm(vmin=vmax * DYNAMIC_RANGE, vmax=vmax)
        for col, label in enumerate((computed, approx)):
            ax = axes[row, col]
            mesh = ax.pcolormesh(X, Y, np.clip(maps[label].T, vmax * DYNAMIC_RANGE, None), norm=norm,
                                 cmap="viridis", shading="auto", rasterized=True)
            ax.set_title(f"${tex}$, {label}", fontsize=11)
            fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label=r"MHz$^2$/meV$^2$")

        ax = axes[row, 2]
        floor = vmax * DYNAMIC_RANGE
        shown = maps[computed] > floor  # the ratio only where the computed map has weight
        ratio = np.full_like(maps[computed], np.nan)
        ratio[shown] = np.maximum(maps[approx][shown], floor) / maps[computed][shown]
        mesh = ax.pcolormesh(X, Y, ratio.T, norm=LogNorm(vmin=0.1, vmax=10), cmap="RdBu_r", shading="auto",
                             rasterized=True)
        ax.set_title(f"${tex}$: {approx_label} / computed", fontsize=11)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="ratio")

        for ax in axes[row, :3]:
            ax.plot([0, 200], [0, 200], color="white" if ax is not axes[row, 2] else "black", lw=0.8, ls="--")
            ax.set_xlim(0, 180)
            ax.set_ylim(0, 180)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$\omega_x$ (meV)")
            ax.set_ylabel(r"$\omega_y$ (meV)")

        ax = axes[row, 3]
        styles = ({"color": "#2a78d6", "ls": "-"}, {"color": "#eb6834", "ls": "--"}, {"color": "#888888", "ls": ":"})
        omega = X[0] if X.ndim == 2 else X
        for (label, J), style in zip(maps.items(), styles):
            ax.plot(omega, np.diag(J), lw=2, label=label, **style)
        ax.set_yscale("log")
        ax.set_ylim(vmax * 1e-4, vmax * 3)
        ax.set_xlim(0, 180)
        ax.set_xlabel(r"$\omega_x = \omega_y$ (meV)")
        ax.set_ylabel(r"MHz$^2$/meV$^2$")
        ax.set_title(f"${tex}$ along the diagonal (difference process)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, ls=":", alpha=0.5)
        for label, J in maps.items():
            print(f"  {name} {label}: total {J.sum() * res**2:.4g} MHz^2, diagonal {np.trace(J) * res:.4g} MHz^2/meV")

    fig.suptitle(f"{title}: 2D two-phonon spectral functions at $\\sigma$ = {sigma:g} meV, computed pairs against "
                 f"{approx_label}", fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D two-phonon spectral functions: computed pairs vs an approximation")
    parser.add_argument("raw_file", type=Path, help="Raw ZFS package (.npz) with the off-diagonal pairs computed")
    parser.add_argument("--approx", choices=[m for m in MODES if m != "computed"], default="diagonal-sqrt",
                        help="Approximation to compare with (default: diagonal-sqrt)")
    parser.add_argument("--displacement", choices=("normalized", "mass_weighted", "unweighted"),
                        help="Override the package's displacement convention")
    parser.add_argument("--sigma", type=float, default=7.5, help="Broadening in meV (default: 7.5)")
    parser.add_argument("--title", help="Figure title (default: <defect> <cell> step <pert>)")
    parser.add_argument("-o", "--output", type=Path, help="Figure path (default: <defect>_<cell>_spectral_2d_comparison.png)")
    args = parser.parse_args()
    logging.disable(logging.WARNING)  # the pipeline warnings would bury the summary

    manager, convention = coupling_manager(args.raw_file, "computed", args.displacement)
    raw = manager.raw_data
    w, computed = second_order_couplings(manager)
    _, approx = second_order_couplings(coupling_manager(args.raw_file, args.approx, args.displacement)[0])
    _, without_a_e = second_order_couplings(
        drop_a_x_e(coupling_manager(args.raw_file, "computed", args.displacement)[0])
    )
    approx_label = {"diagonal-sqrt": "diagonal + sqrt fill", "sqrt": "sqrt fill", "diagonal": "diagonal only"}[args.approx]
    print(f"{convention}; {len(w)} modes")
    plot(w, {"computed": computed, approx_label: approx, "computed without A x E": without_a_e}, args.sigma,
         args.title or f"{raw.defect} {raw.cell_size} step {raw.pert_scale:g}", approx_label,
         args.output or Path(f"{raw.defect}_{raw.cell_size}_spectral_2d_comparison.png"))
