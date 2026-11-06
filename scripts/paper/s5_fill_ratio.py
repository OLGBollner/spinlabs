"""S5: T1(fill) / T1(all pairs computed) against temperature for NV- (64) and VV0 (128), AO, m_s = 0 and +-1.

fill: diagonal + sqrt(V_ii V_jj) geometric-mean fill, kept on the computed pairs only (the filter of S3: like for
like); all pairs: the computed pairs. Both with zero-gap pairs excluded, sigma 3.5 meV, Debye direct rate, no
first-order Raman. VV0 (128) was computed for |gap| <~ 10 meV only, so both curves cover that window.
"""

import numpy as np

import config as C
from common import Record, at, directional, rates, t1_details

SYSTEMS = ("NV_64", "DiV_128")
STATES = ("ms_0", "ms_1")


def main():
    plt = C.style()
    rec = Record("S5", "T1(fill)/T1(all pairs) vs temperature", "scripts/paper/s5_fill_ratio.py", "sm")
    T = C.TEMPERATURES
    ratio = {}
    for s in SYSTEMS:
        t1 = {}
        for pairs in ("fill-on-computed", "computed"):
            r = rates(s, "AO", pairs=pairs)
            for st in STATES:
                v, eig, src = t1_details(T, directional(r), st)
                rec.t1_curve(f"{s} AO {st} pairs={pairs}", T, v, eig, src)
                t1[(pairs, st)] = v
        for st in STATES:
            ratio[(s, st)] = t1[("fill-on-computed", st)] / t1[("computed", st)]
    rec.check_fallbacks()

    fig, axes = plt.subplots(2, 1, figsize=(C.SINGLE, 3.6), sharex=True)
    for k, (ax, s) in enumerate(zip(axes, SYSTEMS)):
        for st in STATES:
            y = ratio[(s, st)]
            ok = np.isfinite(y) & (T >= 20)
            ax.plot(T[ok], y[ok], color=C.COLORS["AO"], ls=C.STATE_STYLES[st], label=C.STATE_LABELS[st])
        ax.axhline(1.0, color=C.COLORS["ref"], lw=0.7, ls=":")
        ax.set_ylabel(r"$T_1^\mathrm{fill} / T_1^\mathrm{all\ pairs}$")
        ax.set_title(C.SYSTEMS[s]["label"], pad=4)
        C.panel_letter(ax, k, x=0.0, y=1.02)
        ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    axes[0].legend(loc="best")
    axes[-1].set_xlim(0, 600)
    axes[-1].set_xlabel("Temperature (K)")
    fig.tight_layout()
    rec.data["files"] = C.save(fig, C.SM, "S5_fill_over_all_pairs")

    rec.data["settings"] = {
        "systems": {s: f"{C.SYSTEMS[s]['label']}, step {C.SYSTEMS[s]['step']:g}, AO {C.SYSTEMS[s]['packages']['AO']}"
                    for s in SYSTEMS},
        "scheme": "AO", "sigma_meV": C.SIGMA, "direct": C.DIRECT, "zero-gap pairs": "excluded in both",
        "fill": "diagonal + geometric-mean fill, kept on the computed pairs only (S3 filter: computed pair keys "
                "plus pairs with a nonzero computed coupling); NV_64: 17954 of 17955 pairs, DiV_128: computed window",
        "all pairs": "computed pairs only (NV_64: 17954 of 17955; DiV_128: |gap| <~ 10 meV)",
        "initial states": "m_s = 0 and m_s = +1", "plotted from": "20 K (both T1 very large below)",
        "T1 extraction": "population fit, eigenvalue only as fallback",
    }
    nums = {f"T1(fill)/T1(all pairs) {s} {st}": at(T, ratio[(s, st)], C.QUOTE_T_RATIO) for s in SYSTEMS for st in STATES}
    rec.data["numbers"] = nums
    rec.write()
    for k, v in nums.items():
        print(k, {t: f"{x:.3f}" for t, x in v.items()})


if __name__ == "__main__":
    main()
