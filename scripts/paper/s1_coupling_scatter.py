"""S1: |V_UO| against |V_AO| per mode (first order) and per second-order pair V^(ll') (second order), coloured by IPR.

Second order: the pairs of M3(b), every pair computed in both AO and UO (diagonal only for all but VV0), coloured
by the geometric-mean IPR sqrt(IPR_l IPR_l') of the pair's two modes.

Rows NV- (512), VV0 (128), ClV+ (576), V_Si- (128) (m3_rho.SYSTEMS); columns V00, V+-, V0+-. No mode filter (as rho in M3); points with a zero
coupling in either scheme cannot be drawn on log axes and are counted in the record. Dashed: |V_UO| = |V_AO|;
solid (colour outside the IPR map): |V_UO| = rho |V_AO|.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import config as C
from common import Record
from m3_rho import SYSTEMS, channel_values, rho_sd, shared_pairs

from spinlabs.parsers import parse_phonon_data


def figure(plt, order, rec):
    cmap = LinearSegmentedColormap.from_list("ipr", plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256)))
    fig, axes = plt.subplots(len(SYSTEMS), 3, figsize=(C.DOUBLE, 2.2 * len(SYSTEMS)))
    fig.subplots_adjust(wspace=0.42, hspace=0.32, right=0.88)
    info = {}
    for row, s in enumerate(SYSTEMS):
        ipr = np.asarray(parse_phonon_data(str(C.DATA / s / "phonon_data.npz")).get_ipr(), dtype=float)
        if order == 2:  # a pair's IPR: the geometric mean of its two modes'
            rows, cols = shared_pairs(s)
            ipr = np.sqrt(ipr[rows] * ipr[cols])
        vals = {ch: (channel_values(s, "AO", order, ch), channel_values(s, "UO", order, ch)) for ch in C.CHANNELS}
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
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.plot(line, line, color=C.COLORS["ref"], lw=0.7, ls="--", zorder=1)
            ax.plot(line, rho * line, color=C.COLORS["rho_line"], lw=1.1, zorder=3)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, color="#e6e5e1", lw=0.4, zorder=0)
            ax.text(0.04, 0.96, f"{C.SYSTEMS[s]['label']}\n$\\rho = {rho:.2f}$", transform=ax.transAxes, va="top",
                    fontsize=C.FONT - 1.5)
            C.panel_letter(ax, row * 3 + col, x=-0.02, y=1.01, fontsize=C.FONT - 1)
            if row == 0:
                ax.set_title(C.CHANNEL_LABELS[ch] if order == 1 else C.CHANNEL_LABELS[ch].replace("V_", "V^{(ll')}_"), pad=10)
            if row == len(SYSTEMS) - 1:
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
                             (2, "S1b", "S1b_coupling_scatter_second_order")):
        rec = Record(fid, f"Coupling scatter UO vs AO, {'first order' if order == 1 else 'second order, shared computed pairs'}",
                     "scripts/paper/s1_coupling_scatter.py", "sm")
        fig, info = figure(plt, order, rec)
        rec.data["files"] = C.save(fig, C.SM, name)
        plt.close(fig)
        rec.data["settings"] = {
            "systems": {s: {sch: C.SYSTEMS[s]["packages"][sch] for sch in C.SCHEMES} for s in SYSTEMS},
            "couplings": "first order per mode" if order == 1 else "second order, every pair computed in both AO "
                         "and UO (M3(b)), computed couplings, no fill",
            "pairs drawn per system": {s: int(len(shared_pairs(s)[0])) for s in SYSTEMS} if order == 2 else None,
            "units": "|V| in MHz", "mode filter": "none (zeros cannot be drawn on log axes; counted)",
            "colour": "mode IPR (Blues, log, per-row scale); second order: sqrt(IPR_l IPR_l') of the pair",
            "rho line": "|V_UO| = rho |V_AO|, rho as in M3",
            "axes": "equal x and y limits, equal aspect, log",
        }
        rec.data["numbers"] = info
        rec.write()
        for k, v in info.items():
            print(order, k, v)


if __name__ == "__main__":
    main()
