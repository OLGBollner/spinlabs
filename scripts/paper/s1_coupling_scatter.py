"""S1: |V_UO| against |V_AO| per mode (first order) and per diagonal pair V^(ll) (second order), coloured by IPR.

Rows NV- (512), VV0 (128), ClV+ (576); columns V00, V+-, V0+-. No mode filter (as rho in M3); points with a zero
coupling in either scheme cannot be drawn on log axes and are counted in the record. Dashed: |V_UO| = |V_AO|;
solid (colour outside the IPR map): |V_UO| = rho |V_AO|.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import config as C
from common import Record, couplings
from m3_rho import KEYS, SYSTEMS, rho_sd

from spinlabs.parsers import parse_phonon_data


def figure(plt, order, rec):
    cmap = LinearSegmentedColormap.from_list("ipr", plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256)))
    fig, axes = plt.subplots(3, 3, figsize=(C.DOUBLE, 6.6))
    fig.subplots_adjust(wspace=0.42, hspace=0.32, right=0.88)
    info = {}
    for row, s in enumerate(SYSTEMS):
        ipr = np.asarray(parse_phonon_data(str(C.DATA / s / "phonon_data.npz")).get_ipr(), dtype=float)
        ao_c, uo_c = couplings(s, "AO"), couplings(s, "UO")
        vals = {}
        for ch in C.CHANNELS:
            a = np.abs(np.asarray(ao_c[KEYS[order][ch]], dtype=float))
            b = np.abs(np.asarray(uo_c[KEYS[order][ch]], dtype=float))
            if order == 2:
                a, b = np.diag(a), np.diag(b)
            vals[ch] = (a, b)
        shown_ipr = np.concatenate([ipr[(a > 0) & (b > 0)] for a, b in vals.values()])
        norm = LogNorm(vmin=shown_ipr.min(), vmax=shown_ipr.max())
        sc = None
        for col, ch in enumerate(C.CHANNELS):
            ax = axes[row, col]
            a, b = vals[ch]
            rho, sd, _ = rho_sd(a, b)
            m = (a > 0) & (b > 0)
            order_ipr = np.argsort(ipr[m])  # localized modes on top
            sc = ax.scatter(a[m][order_ipr], b[m][order_ipr], c=ipr[m][order_ipr], cmap=cmap, norm=norm, s=9,
                            lw=0.3, edgecolors="#ffffff", marker=C.CHANNEL_MARKERS[ch], zorder=2, rasterized=True)
            lo, hi = min(a[m].min(), b[m].min()) / 1.5, max(a[m].max(), b[m].max()) * 1.5
            line = np.array([lo, hi])
            ax.plot(line, line, color=C.COLORS["ref"], lw=0.7, ls="--", zorder=1)
            ax.plot(line, rho * line, color=C.COLORS["rho_line"], lw=1.1, zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, color="#e6e5e1", lw=0.4, zorder=0)
            ax.text(0.04, 0.96, f"{C.SYSTEMS[s]['label']}\n$\\rho = {rho:.2f}$", transform=ax.transAxes, va="top",
                    fontsize=C.FONT - 1.5)
            C.panel_letter(ax, row * 3 + col, x=-0.02, y=1.01, fontsize=C.FONT - 1)
            if row == 0:
                ax.set_title(C.CHANNEL_LABELS[ch] if order == 1 else C.CHANNEL_LABELS[ch].replace("V_", "V^{(ll)}_"), pad=10)
            if row == 2:
                ax.set_xlabel(r"$|V_\mathrm{AO}|$ (MHz)")
            if col == 0:
                ax.set_ylabel(r"$|V_\mathrm{UO}|$ (MHz)")
            info[f"{s} {ch}"] = {"rho": rho, "sd": sd, "drawn": int(m.sum()), "zero in AO or UO (not drawn)":
                                 int(((a > 0) | (b > 0)).sum() - m.sum()), "both zero": int(((a == 0) & (b == 0)).sum())}
        cax = fig.add_axes([0.9, axes[row, 2].get_position().y0, 0.012, axes[row, 2].get_position().height])
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label("mode IPR", fontsize=C.FONT - 1)
    return fig, info


def main():
    plt = C.style()
    for order, fid, name in ((1, "S1a", "S1a_coupling_scatter_first_order"),
                             (2, "S1b", "S1b_coupling_scatter_second_order_diagonal")):
        rec = Record(fid, f"Coupling scatter UO vs AO, {'first order' if order == 1 else 'second-order diagonal'}",
                     "scripts/paper/s1_coupling_scatter.py", "sm")
        fig, info = figure(plt, order, rec)
        rec.data["files"] = C.save(fig, C.SM, name)
        plt.close(fig)
        rec.data["settings"] = {
            "systems": {s: {sch: C.SYSTEMS[s]["packages"][sch] for sch in C.SCHEMES} for s in SYSTEMS},
            "couplings": "first order per mode" if order == 1 else "second order, diagonal pairs V^(ll)",
            "units": "|V| in MHz", "mode filter": "none (zeros cannot be drawn on log axes; counted)",
            "colour": "mode IPR (Blues, log, per-row scale)", "rho line": "|V_UO| = rho |V_AO|, rho as in M3",
            "axes": "equal x and y limits, equal aspect, log",
        }
        rec.data["numbers"] = info
        rec.write()
        for k, v in info.items():
            print(order, k, v)


if __name__ == "__main__":
    main()
