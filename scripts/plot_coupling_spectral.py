"""Couplings and spectral functions of one raw ZFS package at one sigma, in six panels.

Top: first-order couplings and F(w); the low-energy tail of F(w) on a log scale, which the one-phonon
rate samples near w_ZFS; the diagonal second-order couplings and F(w) (as 'spinlabs plot coupling').
Bottom: the 2D two-phonon densities J_0pm(wx, wy) and J_pm(wx, wy) on the rate grid, and the share
of sum |V2|^2 against the pairs' energy difference.

MODE picks the second-order pairs as in plot_t1_comparison.py (computed, sqrt, diagonal, diagonal-sqrt).

    python scripts/plot_coupling_spectral.py data/NV_512/pert_0.1/all_bands/20261022/NV_512_raw_zfs_data_all_bands_12d.npz \\
        --mode sqrt --sigma 7.5 -o figures/sigma_scan/NV_512_coupling_spectral.png
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

from spinlabs.constants import CONSTANTS  # noqa: E402
from spinlabs.models import SpinPhononCouplingData  # noqa: E402
from spinlabs.plotter import plot_1d_spectral_functions  # noqa: E402
from spinlabs.utils import MathUtils  # noqa: E402

MODE_TITLES = {
    "computed": "all second-order pairs computed",
    "sqrt": "uncomputed pairs from sqrt(V_ii V_jj), symmetry-allowed types",
    "diagonal": "diagonal second-order pairs only",
    "diagonal-sqrt": "diagonal pairs + sqrt(V_ii V_jj), symmetry-allowed types",
}


def plot_coupling_spectral(data, zfs_mev, sigma, title, mode, out):
    """The six-panel figure for coupling data in MHz and meV."""
    w = np.asarray(data.frequencies, dtype=float)
    lowest = np.sort(w)[3]  # skip the three acoustic modes at ~0
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.6)

    ax = fig.add_subplot(gs[0, 0])
    plot_1d_spectral_functions(w, data.V_0_0, data.V_p_m, data.V_0_pm, sigma=sigma, res=0.1, order=1, ax=ax)
    ax.set_title(f"First order, $\\sigma$ = {sigma:g} meV", fontsize=11)

    ax = fig.add_subplot(gs[0, 1])
    for V, color, name in ((data.V_p_m, "red", "+-"), (data.V_0_pm, "blue", r"0\pm"), (data.V_0_0, "black", "00")):
        x, y = MathUtils.smear_data(w, np.asarray(V) ** 2, 0.02, sigma)
        keep = x <= 60
        ax.plot(x[keep], np.clip(y[keep], 1e-300, None), color=color, lw=2, label=f"$F_{{{name}}}^{{(1)}}$")
    # compute_transition_rates weighs F with a 1 meV Gaussian around w_ZFS, from w = 0.1 meV
    ax.axvspan(0.1, zfs_mev + 2.0, color="#f2c14e", alpha=0.35, lw=0, label="window of the one-phonon rate")
    ax.axvline(lowest, color="#666666", ls="--", lw=1, label=f"lowest mode {lowest:.1f} meV")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1e4)
    ax.set_xlim(0, 60)
    ax.set_xlabel("Vibration frequency (meV)")
    ax.set_ylabel(r"Spectral function (MHz$^2$/meV)")
    ax.set_title("First-order F(w), low-energy tail (log)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, ls=":", alpha=0.5)

    ax = fig.add_subplot(gs[0, 2])
    plot_1d_spectral_functions(w, np.diag(data.V2_0_0), np.diag(data.V2_p_m), np.diag(data.V2_0_pm), sigma=sigma,
                               res=0.1, order=2, ax=ax)
    ax.set_title(f"Second order, diagonal (i = i), $\\sigma$ = {sigma:g} meV", fontsize=11)

    res2 = min(0.5, sigma / 4)  # the grid compute_rates uses
    for col, (V2, name) in enumerate(((data.V2_0_pm, r"J_{0\pm}^{(2)}"), (data.V2_p_m, r"J_{+-}^{(2)}"))):
        X, Y, J = MathUtils.get_2d_spectral_density(w, np.asarray(V2) ** 2, res2, sigma)
        ax = fig.add_subplot(gs[1, col])
        vmax = float(J.max())
        mesh = ax.pcolormesh(X, Y, np.clip(J.T, vmax * 1e-6, None), norm=LogNorm(vmin=vmax * 1e-6, vmax=vmax),
                             cmap="viridis", shading="auto", rasterized=True)
        ax.plot([0, 200], [0, 200], color="white", lw=0.8, ls="--")
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 180)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\omega_x$ (meV)")
        ax.set_ylabel(r"$\omega_y$ (meV)")
        ax.set_title(f"${name}(\\omega_x, \\omega_y)$", fontsize=11)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label=r"MHz$^2$/meV$^2$")
        weights = np.abs(np.asarray(V2)) ** 2
        print(f"  {name}: diagonal share of sum |V2|^2 = {np.trace(weights) / weights.sum():.3g}")

    ax = fig.add_subplot(gs[1, 2])
    gap = np.abs(w[:, None] - w[None, :])
    bins = np.linspace(0, 160, 81)
    for V2, color, name in ((data.V2_0_pm, "blue", r"$|V_{0\pm}^{(2)}|^2$"),
                            (data.V2_p_m, "red", r"$|V_{+-}^{(2)}|^2$")):
        h, _ = np.histogram(gap.ravel(), bins=bins, weights=(np.abs(np.asarray(V2)) ** 2).ravel())
        ax.step(bins[:-1], h / h.sum(), where="post", color=color, lw=1.8, label=name)
    ax.set_yscale("log")
    ax.set_xlabel(r"$|\omega_i - \omega_j|$ (meV)")
    ax.set_ylabel("share of sum over pairs")
    ax.set_title("Two-phonon weight against pair energy difference", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)

    fig.suptitle(f"{title}: couplings and spectral functions at $\\sigma$ = {sigma:g} meV ({MODE_TITLES[mode]})",
                 fontsize=13, y=0.98)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Couplings and spectral functions of one raw ZFS package")
    parser.add_argument("raw_file", type=Path, help="Raw ZFS package (.npz) with first- and second-order data")
    parser.add_argument("--mode", choices=MODES, default="computed", help="Second-order pairs to use (default: computed)")
    parser.add_argument("--displacement", choices=("normalized", "mass_weighted", "unweighted"),
                        help="Override the package's displacement convention")
    parser.add_argument("--sigma", type=float, default=7.5, help="Broadening in meV (default: 7.5)")
    parser.add_argument("--title", help="Figure title (default: <defect> <cell> step <pert>)")
    parser.add_argument("-o", "--output", type=Path, help="Figure path (default: <defect>_<cell>_coupling_spectral.png)")
    args = parser.parse_args()
    logging.disable(logging.WARNING)  # the pipeline warnings would bury the summary

    manager, convention = coupling_manager(args.raw_file, args.mode, args.displacement)
    raw = manager.raw_data
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "coupling")
        manager.process_both_orders(output_filename=base)
        data = SpinPhononCouplingData.load(f"{base}.npz")
        zfs_mev = float(np.load(f"{base}.npz", allow_pickle=True)["zfs"]) / CONSTANTS["meV2J"]
    display = data.to_unit("MHz").frequencies_to_unit("meV")
    print(f"{convention}; {len(display.frequencies)} modes, ZFS {zfs_mev * 1e3:.2f} ueV, "
          f"{manager.n_offdiag_filled} pairs filled")
    title = args.title or f"{raw.defect} {raw.cell_size} step {raw.pert_scale:g}"
    out = args.output or Path(f"{raw.defect}_{raw.cell_size}_coupling_spectral.png")
    plot_coupling_spectral(display, zfs_mev, args.sigma, title, args.mode, out)
