"""2D two-phonon spectral functions of one raw ZFS package: all computed pairs against an approximation.

One row per spin-flip coupling (J_0pm, J_pm): the two maps on one log colour scale, their ratio
(approximation / computed), and both along the diagonal w_x = w_y, the line the difference process
(absorb one phonon, emit another, w_x - w_y = w_ZFS ~ 0) samples. The cut also shows the computed
pairs without the A x E pairs, which the sqrt(V_ii V_jj) fill cannot supply.

The approximation is kept only on the pairs that were computed (the computed pair keys plus any pair
with a nonzero computed coupling), so a package with only part of the (i, j) plane computed compares
like for like. --full-fill keeps the approximation on every pair.

    python scripts/plot_spectral_comparison.py data/NV_64/pert_0.1/all_bands/20261029/NV_64_raw_zfs_data_all_bands_12d.npz \\
        --sigma 3.5 -o figures/coupling/NV_64_spectral_2d_comparison.png

With --cuts, several packages are drawn side by side as diagonal cuts only (one column per package, rows
J_0pm and J_pm, panels (a), (b), ... row by row, one shared legend above): a single-column figure for the main
text, with the full 2D figures of each package as supplementary material.

    python scripts/plot_spectral_comparison.py --cuts --sigma 3.5 NV_64.npz DiV_128.npz -o figures/coupling/cuts.pdf
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
from spinlabs.transition_rate import TransitionRate  # noqa: E402
from spinlabs.utils import MathUtils  # noqa: E402

COUPLINGS = (("V2_0_pm", r"J_{0\pm}^{(2)}"), ("V2_p_m", r"J_{+-}^{(2)}"))
APPROX_LABELS = {"diagonal-sqrt": "geometric mean", "sqrt": "computed + geometric mean", "diagonal": "diagonal only"}
WITHOUT_A_X_E = r"computed without $A\times E$"
DEFECT_NAMES = {"NV": "NV$^{-}$", "DiV": "VV$^0$", "ClV": "ClV$^{+}$"}  # as in the text
CUT_STYLES = ({"color": "#2a78d6", "ls": "-"}, {"color": "#eb6834", "ls": "--"}, {"color": "#555555", "ls": ":"})
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


def computed_pair_mask(manager, couplings):
    """(N, N) bool: the computed (i, j) pairs, both orders, plus any pair with a nonzero computed coupling
    (pairs the pipeline adds from symmetry-degenerate partners)."""
    n = len(next(iter(couplings.values())))
    mask = np.zeros((n, n), dtype=bool)
    for i, j in manager.zfs_tensors_2d:
        mask[int(i), int(j)] = mask[int(j), int(i)] = True
    for values in couplings.values():
        mask |= values != 0
    return mask


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
        top = float(np.max(X))  # the grid ends a few sigma past the highest mode
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
            ax.plot([0, top], [0, top], color="white" if ax is not axes[row, 2] else "black", lw=0.8, ls="--")
            ax.set_xlim(0, top)
            ax.set_ylim(0, top)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$\omega_x$ (meV)")
            ax.set_ylabel(r"$\omega_y$ (meV)")

        ax = axes[row, 3]
        omega = X[0] if X.ndim == 2 else X
        for (label, J), style in zip(maps.items(), CUT_STYLES):
            ax.plot(omega, np.diag(J), lw=2, label=label, **style)
        ax.set_yscale("log")
        ax.set_ylim(vmax * 1e-4, vmax * 3)
        ax.set_xlim(0, top)
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


def coupling_sets(raw_file, approx, displacement=None, full_fill=False, exclude_degenerate=False):
    """Mode energies, {label: couplings} (computed, the approximation, computed without A x E) and the title.

    ``exclude_degenerate`` zeroes the pairs at zero frequency difference by symmetry, (i, i) and the partners
    of each degenerate group, in every set: what the two-phonon rate leaves out
    (TransitionRate.symmetry_zero_gap_pairs).
    """
    manager, convention = coupling_manager(raw_file, "computed", displacement)
    raw = manager.raw_data
    w, computed = second_order_couplings(manager)
    _, approximated = second_order_couplings(coupling_manager(raw_file, approx, displacement)[0])
    _, without_a_e = second_order_couplings(drop_a_x_e(coupling_manager(raw_file, "computed", displacement)[0]))
    approx_label = APPROX_LABELS[approx]
    mask = computed_pair_mask(manager, computed)
    print(f"{raw_file}: {convention}; {len(w)} modes; {np.count_nonzero(np.triu(mask))} of "
          f"{len(w) * (len(w) + 1) // 2} pairs computed")
    if not full_fill and not mask.all():  # like for like: the approximation only on the computed pairs
        approximated = {name: np.where(mask, values, 0.0) for name, values in approximated.items()}
    sets = {"computed": computed, approx_label: approximated, WITHOUT_A_X_E: without_a_e}
    if exclude_degenerate:
        zero_gap = TransitionRate.symmetry_zero_gap_pairs(len(w), manager.spectrum.deg_groups or [])
        sets = {label: {k: np.where(zero_gap, 0.0, v) for k, v in c.items()} for label, c in sets.items()}
        print(f"  left out {np.count_nonzero(np.triu(zero_gap))} pairs at zero gap by symmetry")
    name = f"{DEFECT_NAMES.get(raw.defect, raw.defect)} ({raw.cell_size})"
    return w, sets, approx_label, name, raw


def plot_cuts(panels, sigma, out, font=9):
    """Diagonal cuts only: one column per package, one row per coupling, sized for a single journal column."""
    res = min(0.5, sigma / 4)
    with plt.rc_context({"font.size": font, "axes.labelsize": font, "axes.titlesize": font,
                         "xtick.labelsize": font - 1, "ytick.labelsize": font - 1, "legend.fontsize": font - 1.5}):
        fig, axes = plt.subplots(len(COUPLINGS), len(panels), figsize=(3.4, 3.3), squeeze=False)
        for col, (w, sets, name) in enumerate(panels):
            for row, (key, tex) in enumerate(COUPLINGS):
                ax = axes[row, col]
                cuts = {}
                for label, couplings in sets.items():
                    X, _, J = MathUtils.get_2d_spectral_density(w, couplings[key] ** 2, res, sigma)
                    cuts[label] = np.diag(J)
                omega = X[0] if X.ndim == 2 else X
                for (label, cut), style in zip(cuts.items(), CUT_STYLES):
                    ax.plot(omega, cut, lw=1.2, label=label, **style)
                top = max(float(c.max()) for c in cuts.values())
                ax.set_yscale("log")
                ax.set_ylim(top * 1e-4, top * 20)  # headroom for the panel label in the upper left
                ax.set_xlim(0, float(np.max(omega)))
                ax.grid(True, ls=":", alpha=0.5, lw=0.5)
                panel = chr(ord("a") + row * len(panels) + col)
                ax.text(0.04, 0.95, f"({panel}) ${tex}$", transform=ax.transAxes, va="top")
                if row == 0:
                    ax.set_title(name)
                if row == len(COUPLINGS) - 1:
                    ax.set_xlabel(r"$\omega$ (meV)")
        fig.supylabel(r"$J^{(2)}(\omega, \omega)$ (MHz$^2$/meV$^2$)", fontsize=font, x=0.01)
        # one shared legend above the panels, short labels so its three entries fit the column width
        short = {WITHOUT_A_X_E: r"no $A\times E$"}
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, [short.get(lab, lab) for lab in labels], loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.53, 1.0), handlelength=1.8, columnspacing=1.0)
        fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.6, rect=(0.02, 0, 1, 0.94))
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"saved figure: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D two-phonon spectral functions: computed pairs vs an approximation")
    parser.add_argument("raw_files", nargs="+", type=Path,
                        help="Raw ZFS package(s) (.npz) with the off-diagonal pairs computed; several with --cuts")
    parser.add_argument("--approx", choices=[m for m in MODES if m != "computed"], default="diagonal-sqrt",
                        help="Approximation to compare with (default: diagonal-sqrt, the geometric mean)")
    parser.add_argument("--displacement", choices=("normalized", "mass_weighted", "unweighted"),
                        help="Override the package's displacement convention")
    parser.add_argument("--sigma", type=float, default=3.5, help="Broadening in meV (default: 3.5, the production value)")
    parser.add_argument("--full-fill", action="store_true",
                        help="Keep the approximation on every pair, not only on the computed ones")
    parser.add_argument("--cuts", action="store_true",
                        help="Only the diagonal cuts, one column per package (single-column main-text figure)")
    parser.add_argument("--exclude-degenerate", action="store_true",
                        help="Leave out the pairs the two-phonon rate excludes: (i, i) and degenerate partners")
    parser.add_argument("--title", help="Figure title of the 2D figure (default: <defect> <cell> step <pert>)")
    parser.add_argument("-o", "--output", type=Path, help="Figure path (default: <defect>_<cell>_spectral_2d_comparison.png)")
    args = parser.parse_args()
    logging.disable(logging.WARNING)  # the pipeline warnings would bury the summary

    if args.cuts:
        panels = []
        for raw_file in args.raw_files:
            w, sets, _, name, _ = coupling_sets(raw_file, args.approx, args.displacement, args.full_fill,
                                                args.exclude_degenerate)
            panels.append((w, sets, name))
        plot_cuts(panels, args.sigma, args.output or Path("spectral_cuts.pdf"))
    else:
        if len(args.raw_files) != 1:
            parser.error("the 2D figure takes one package; use --cuts for several")
        w, sets, approx_label, name, raw = coupling_sets(args.raw_files[0], args.approx, args.displacement,
                                                         args.full_fill, args.exclude_degenerate)
        title = args.title or f"{name}, step {raw.pert_scale:g}" + (
            ", without the (i, i) and degenerate pairs" if args.exclude_degenerate else "")
        plot(w, sets, args.sigma, title, approx_label,
             args.output or Path(f"{raw.defect}_{raw.cell_size}_spectral_2d_comparison.png"))
