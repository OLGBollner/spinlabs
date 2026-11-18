"""M3: coupling ratio rho = sqrt(sum V_UO^2 / sum V_AO^2) for NV- (512), VV0 (128), ClV+ (576), V_Si- (128).

(a) first order, over the modes; (b) second order, over every pair (l, l') computed in both AO and UO (each
unordered pair once; computed couplings, no fill; pairs excluded as inconsistent left out). Only VV0 (128) has
off-diagonal pairs (|gap| < ~20 meV in both schemes); NV- (512), ClV+ (576) and V_Si- (128) have the diagonal V^(ll)
only. Channels V00, V+-, V0+-. The weighted SD of the per-mode (per-pair) ratios r = |V_UO| / |V_AO| about rho,
weights V_AO^2 (modes with V_AO = 0 carry no weight), goes to the numbers file only; it is not drawn. No mode filter.
Couplings in MHz as they enter the rates (ZFSManager of the production packages): UO includes the spin
correction, x3/2 for the S = 1 defects and x2/3 for V_Si- (S = 3/2).
"""

import numpy as np
from matplotlib.ticker import MultipleLocator

import config as C
from functools import lru_cache

from common import Record, computed_pairs, couplings

SYSTEMS = ("NV_512", "DiV_128", "ClV_576", "V_Si_128")
KEYS = {1: {"V00": "V_0_0", "Vpm": "V_p_m", "V0pm": "V_0_pm"},
        2: {"V00": "V2_0_0", "Vpm": "V2_p_m", "V0pm": "V2_0_pm"}}


@lru_cache(maxsize=None)
def shared_pairs(system):
    """(rows, cols), rows <= cols: the second-order pairs computed in both AO and UO, each unordered pair once."""
    both = computed_pairs(system, "AO") & computed_pairs(system, "UO")
    return np.nonzero(np.triu(both))


def channel_values(system, scheme, order, ch):
    """|V| per mode (first order, production couplings) or per shared computed pair (second order, no fill)."""
    if order == 1:
        return np.abs(np.asarray(couplings(system, scheme)[KEYS[1][ch]], dtype=float))
    v = np.asarray(couplings(system, scheme, pairs="computed")[KEYS[2][ch]], dtype=float)
    return np.abs(v[shared_pairs(system)])


def rho_sd(ao, uo):
    w = ao**2
    if w.sum() == 0:
        return np.nan, np.nan, 0
    rho = float(np.sqrt((uo**2).sum() / w.sum()))
    m = ao > 0
    r = uo[m] / ao[m]
    sd = float(np.sqrt((w[m] * (r - rho) ** 2).sum() / w[m].sum()))
    return rho, sd, int(m.sum())


def compute():
    out = {}
    for order in (1, 2):
        for s in SYSTEMS:
            for ch in C.CHANNELS:
                ao, uo = (channel_values(s, sch, order, ch) for sch in ("AO", "UO"))
                if len(ao) == 0 or len(uo) == 0 or not np.any(ao):
                    raise SystemExit(f"{s}: no order-{order} {ch} couplings for AO or UO")
                rho, sd, n = rho_sd(ao, uo)
                out[(order, s, ch)] = dict(rho=rho, sd=sd, n=n)
    return out


def main():
    plt = C.style()
    rec = Record("M3", "Coupling ratio rho, first and second order", "scripts/paper/m3_rho.py", "main")
    res = compute()
    fig, axes = plt.subplots(1, 2, figsize=(C.SINGLE, 0.5 + 0.47 * len(SYSTEMS)), sharey=True)
    offsets = {"V00": 0.22, "Vpm": 0.0, "V0pm": -0.22}
    ys = {s: len(SYSTEMS) - 1 - k for k, s in enumerate(SYSTEMS)}
    for k, (ax, order) in enumerate(zip(axes, (1, 2))):
        for s in SYSTEMS:
            for ch in C.CHANNELS:
                v = res[(order, s, ch)]
                ax.plot(v["rho"], ys[s] + offsets[ch], ls="none", marker=C.CHANNEL_MARKERS[ch], ms=4,
                        color=C.COLORS[ch], label=C.CHANNEL_LABELS[ch] if (k == 0 and s == SYSTEMS[0]) else None)
        ax.axvline(1.0, color=C.COLORS["ref"], ls="--", lw=0.8, zorder=0)
        ax.set_xlabel(r"$\rho$")
        ax.set_xlim(0, 1.1 * max(res[(order, s, ch)]["rho"] for s in SYSTEMS for ch in C.CHANNELS))
        ax.xaxis.set_major_locator(MultipleLocator(0.5 if order == 1 else 1.0))
        ax.set_title("first order" if order == 1 else r"second order, $V^{(ll')}$", fontsize=C.FONT - 1)
        C.panel_letter(ax, k, x=0.0, y=1.02)
        ax.grid(True, axis="x", ls=":", lw=0.5, alpha=0.6)
    axes[0].set_yticks([ys[s] for s in SYSTEMS], [C.SYSTEMS[s]["label"] for s in SYSTEMS])
    axes[0].set_ylim(-0.5, len(SYSTEMS) - 0.5)
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.6, 1.1), fontsize=C.FONT - 1, handletextpad=0.2)
    fig.tight_layout()
    rec.data["files"] = C.save(fig, C.MAIN, "M3_rho")
    rec.data["settings"] = {
        "systems": {s: f"{C.SYSTEMS[s]['label']}, step {C.SYSTEMS[s]['step']:g}, AO "
                       f"{C.SYSTEMS[s]['packages']['AO']}, UO {C.SYSTEMS[s]['packages']['UO']}" for s in SYSTEMS},
        "rho": "sqrt(sum V_UO^2 / sum V_AO^2) over all modes (first order) or, second order, every pair (l, l') "
               "computed in both AO and UO, each unordered pair once (computed couplings, no fill; inconsistent "
               "pairs left out)",
        "second-order pairs used": {s: int(len(shared_pairs(s)[0])) for s in SYSTEMS},
        "weighted SD (numbers only, not drawn)": "sqrt(sum w (r - rho)^2 / sum w), r = |V_UO|/|V_AO| per "
                                                  "mode/pair, w = V_AO^2",
        "mode filter": "none",
        "ClV_576, V_Si_128 second order": "diagonal V^(ll) from central differences (f(+Q) + f(-Q) - 2 f(0)) / Q^2",
        "UO spin correction": "x3/2 for S = 1 (NV, VV0, ClV), x2/3 for S = 3/2 (V_Si), in the packages",
        "units": "couplings in MHz (rho is dimensionless)",
    }
    rec.data["numbers"] = {f"order {o}": {s: {ch: res[(o, s, ch)] for ch in C.CHANNELS} for s in SYSTEMS}
                           for o in (1, 2)}
    rec.write()
    for (o, s, ch), v in res.items():
        print(f"order {o} {s:8s} {ch:5s} rho = {v['rho']:.3f} +- {v['sd']:.3f} (n = {v['n']})")


if __name__ == "__main__":
    main()
