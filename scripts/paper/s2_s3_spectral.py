"""S2 + S3: two-phonon spectral density J2 of NV- (64) and VV0 (128), AO, at the production sigma.

S2: along the diagonal J2(w, w), channels J_0+- and J_+-, 2 x 2 panels, one shared legend.
S3: the full 2D maps (all pairs computed, geometric-mean fill, fill / computed) for the same systems.
Sets (plot_spectral_comparison.coupling_sets): all computed pairs; diagonal + sqrt(V_ii V_jj) fill, kept on the
computed pairs only (like for like); computed without the A x E pairs. Zero-gap pairs ((l, l) and Ex-Ey partners)
are left out of every set, as in the production rates.
"""

import numpy as np
from matplotlib.colors import LogNorm

import config as C
from common import Record

from plot_spectral_comparison import WITHOUT_A_X_E, coupling_sets  # noqa: E402  (scripts/ on sys.path via common)
from spinlabs.utils import MathUtils

SYSTEMS = ("NV_64", "DiV_128")
CHANNELS = (("V2_0_pm", r"J^{(2)}_{0\pm}"), ("V2_p_m", r"J^{(2)}_{+-}"))
SET_KEYS = {"computed": "computed", "geometric mean": "fill", WITHOUT_A_X_E: "no_axe"}
DYNAMIC_RANGE = 1e-6


def maps(system):
    w, sets, _, _, raw = coupling_sets(C.package(system, "AO"), "diagonal-sqrt", exclude_degenerate=True)
    res = min(0.5, C.SIGMA / 4)
    out = {}
    for label, cpl in sets.items():
        for key, _ in CHANNELS:
            X, Y, J = MathUtils.get_2d_spectral_density(w, cpl[key] ** 2, res, C.SIGMA)
            out[(SET_KEYS[label], key)] = J
    omega = X[0] if X.ndim == 2 else X
    return omega, out, res


def s2(plt, data):
    rec = Record("S2", "Two-phonon spectral density along the diagonal", "scripts/paper/s2_s3_spectral.py", "sm")
    fig, axes = plt.subplots(2, 2, figsize=(C.SINGLE, 3.3), sharex="col")
    nums = {}
    for col, s in enumerate(SYSTEMS):
        omega, J, res = data[s]
        for row, (key, tex) in enumerate(CHANNELS):
            ax = axes[row, col]
            cuts = {k: np.diag(J[(k, key)]) for k in ("computed", "fill", "no_axe")}
            for k, cut in cuts.items():
                style = "fill-computed" if k == "fill" else k  # the fill is kept on the computed pairs (as in S4)
                ax.plot(omega, cut, color=C.COLORS[style], ls=C.PAIR_STYLES[style], lw=1.1, label=C.PAIR_LABELS[style])
            top = max(float(c.max()) for c in cuts.values())
            ax.set_yscale("log")
            ax.set_ylim(top * 1e-4, top * 20)
            ax.set_xlim(0, float(omega.max()))
            ax.grid(True, ls=":", alpha=0.5, lw=0.5)
            ax.text(0.04, 0.95, f"${tex}$", transform=ax.transAxes, va="top", fontsize=C.FONT - 1)
            C.panel_letter(ax, row * 2 + col, x=0.0, y=1.02, fontsize=C.FONT - 1)
            if row == 0:
                ax.set_title(C.SYSTEMS[s]["label"], pad=10)
            if row == 1:
                ax.set_xlabel(r"$\omega$ (meV)")
            nums[f"{s} {key}"] = {f"integral ratio fill/computed": float(cuts["fill"].sum() / cuts["computed"].sum()),
                                  f"integral ratio no AxE/computed": float(cuts["no_axe"].sum() / cuts["computed"].sum())}
    fig.supylabel(r"$J^{(2)}(\omega, \omega)$ (MHz$^2$/meV$^2$)", fontsize=C.FONT, x=0.01)
    h, l = axes[0, 0].get_legend_handles_labels()
    short = {C.PAIR_LABELS["no_axe"]: r"computed, no $A\times E$"}
    fig.legend(h, [short.get(x, x) for x in l], loc="upper center", ncol=2, bbox_to_anchor=(0.53, 1.04),
               handlelength=2.4, columnspacing=1.0, fontsize=C.FONT - 1.5)
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.8, rect=(0.02, 0, 1, 0.93))
    rec.data["files"] = C.save(fig, C.SM, "S2_spectral_diagonal_cuts")
    plt.close(fig)
    return rec, nums


def s3(plt, data):
    rec = Record("S3", "Two-phonon spectral density maps", "scripts/paper/s2_s3_spectral.py", "sm")
    fig, axes = plt.subplots(4, 3, figsize=(C.DOUBLE, 7.6))
    col_titles = ("computed pairs", "fill on computed pairs", "fill / computed")
    k = 0
    for i, s in enumerate(SYSTEMS):
        omega, J, res = data[s]
        top = float(omega.max())
        for j, (key, tex) in enumerate(CHANNELS):
            row = 2 * i + j
            comp, fill = J[("computed", key)], J[("fill", key)]
            vmax = max(comp.max(), fill.max())
            norm = LogNorm(vmin=vmax * DYNAMIC_RANGE, vmax=vmax)
            for col, (m, name) in enumerate(((comp, "computed pairs"), (fill, "fill on computed pairs"))):
                ax = axes[row, col]
                mesh = ax.pcolormesh(omega, omega, np.clip(m.T, vmax * DYNAMIC_RANGE, None), norm=norm,
                                     cmap="viridis", shading="auto", rasterized=True)
                cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
                cb.set_label(r"MHz$^2$/meV$^2$", fontsize=C.FONT - 1)
            ax = axes[row, 2]
            floor = vmax * DYNAMIC_RANGE
            shown = comp > floor
            ratio = np.full_like(comp, np.nan)
            ratio[shown] = np.maximum(fill[shown], floor) / comp[shown]
            mesh = ax.pcolormesh(omega, omega, ratio.T, norm=LogNorm(vmin=0.1, vmax=10), cmap="RdBu_r",
                                 shading="auto", rasterized=True)
            cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label("fill / computed", fontsize=C.FONT - 1)
            for ax in axes[row]:
                ax.plot([0, top], [0, top], color="white" if ax is not axes[row, 2] else "black", lw=0.6, ls="--")
                ax.set_xlim(0, top)
                ax.set_ylim(0, top)
                ax.set_aspect("equal")
                ax.set_xlabel(r"$\omega_x$ (meV)")
                ax.set_ylabel(r"$\omega_y$ (meV)")
                C.panel_letter(ax, k, x=-0.02, y=1.02, fontsize=C.FONT - 1)
                k += 1
                if row == 0:
                    ax.set_title(col_titles[list(axes[row]).index(ax)], pad=8)
                if row < 3:
                    ax.set_xlabel("")
                if ax is not axes[row, 0]:  # same omega_y axis as column 1: no labels next to the colour bars
                    ax.set_ylabel("")
                    ax.tick_params(labelleft=False)
                ax.text(0.04, 0.96, f"{C.SYSTEMS[s]['label']}\n${tex}$", transform=ax.transAxes, va="top",
                        fontsize=C.FONT - 1.5, color="white" if ax is not axes[row, 2] else "black")
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    rec.data["files"] = C.save(fig, C.SM, "S3_spectral_maps")
    plt.close(fig)
    return rec


def main():
    plt = C.style()
    data = {s: maps(s) for s in SYSTEMS}
    settings = {
        "systems": {s: f"{C.SYSTEMS[s]['label']}, step {C.SYSTEMS[s]['step']:g}, AO {C.SYSTEMS[s]['packages']['AO']}"
                    for s in SYSTEMS},
        "scheme": "AO", "sigma_meV": C.SIGMA, "grid": f"{min(0.5, C.SIGMA / 4):g} meV",
        "sets": "all computed pairs; diagonal + geometric-mean fill on the computed pairs only (like for like); "
                "computed without A x E pairs",
        "zero-gap pairs": "left out of every set (as in the production rates)",
        "DiV_128 computed pairs": "|gap| <~ 10 meV (pairs further apart were not run)",
    }
    rec, nums = s2(plt, data)
    rec.data["settings"] = settings
    rec.data["numbers"] = {"diagonal-cut integrals relative to all pairs": nums}
    rec.write()
    print(nums)
    rec3 = s3(plt, data)
    rec3.data["settings"] = settings | {"maps": "computed, fill, fill/computed (ratio where computed > 1e-6 max)"}
    rec3.write()


if __name__ == "__main__":
    main()
