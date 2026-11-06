"""M4: T1 against temperature for VV0 (128) and ClV+ (576), AO and UO, m_s = 0 and m_s = +-1. No experiment.

Production settings (config.py). VV0: computed second-order pairs (|gap| up to ~10 meV) + geometric-mean fill for
the rest; ClV+ (576): diagonal pairs from central differences + fill for every off-diagonal pair.
"""

import numpy as np

import config as C
from common import Record, at, directional, rates, t1_details

SYSTEMS = ("DiV_128", "ClV_576")
STATES = ("ms_0", "ms_1")


def main():
    plt = C.style()
    rec = Record("M4", "VV0 (128) and ClV+ (576) T1 vs temperature", "scripts/paper/m4_t1_vv_clv.py", "main")
    T = C.TEMPERATURES
    res, debye = {}, {}
    for s in SYSTEMS:
        for sch in C.SCHEMES:
            r = rates(s, sch)
            debye[f"{s} {sch}"] = float(r["debye_C"])
            for st in STATES:
                t1, eig, src = t1_details(T, directional(r), st)
                rec.t1_curve(f"{s} {sch} {st}", T, t1, eig, src)
                res[(s, sch, st)] = t1
    rec.check_fallbacks()

    fig, axes = plt.subplots(2, 1, figsize=(C.SINGLE, 4.4), sharex=True)
    for k, (ax, s) in enumerate(zip(axes, SYSTEMS)):
        for sch in C.SCHEMES:
            for st in STATES:
                t1 = res[(s, sch, st)]
                ok = np.isfinite(t1) & (t1 > 0)
                ax.plot(T[ok], t1[ok], color=C.COLORS[sch], ls=C.STATE_STYLES[st], label=f"{sch}, {C.STATE_LABELS[st]}")
        C.phonon_limited_line(ax)
        ax.set_yscale("log")
        lo = min(np.nanmin(res[(s, sch, st)]) for sch in C.SCHEMES for st in STATES)
        ax.set_ylim(lo / 3, 1e3)
        ax.set_ylabel(r"$T_1$ (s)")
        ax.text(0.97, 0.75, C.SYSTEMS[s]["label"], transform=ax.transAxes, ha="right", va="top")
        C.panel_letter(ax, k, x=-0.16, y=0.98)
    axes[-1].set_xlim(0, 600)
    axes[-1].set_xlabel("Temperature (K)")
    axes[0].legend(loc="lower left", fontsize=C.FONT - 2, handlelength=2.2)
    fig.tight_layout()
    rec.data["files"] = C.save(fig, C.MAIN, "M4_VV128_ClV576_T1_vs_T")

    rec.data["settings"] = {
        "systems": {s: {"step": C.SYSTEMS[s]["step"], **{sch: C.SYSTEMS[s]["packages"][sch] for sch in C.SCHEMES}}
                    for s in SYSTEMS},
        "conventions": {f"{s} {sch}": str(rates(s, sch)["convention"]) for s in SYSTEMS for sch in C.SCHEMES},
        "sigma_meV": C.SIGMA, "direct": C.DIRECT, "zero-gap pairs": "excluded by symmetry",
        "pairs": {"DiV_128": "computed pairs (|gap| <~ 10 meV AO, 11 meV UO) + geometric-mean fill for the rest",
                  "ClV_576": "diagonal from central differences + geometric-mean fill for all off-diagonal pairs"},
        "first-order Raman": "excluded", "initial states": "m_s = 0 and m_s = +1 (= -1 at zero field)",
        "temperatures": f"{T[0]:g}-{T[-1]:g} K, step {T[1] - T[0]:g} K",
        "T1 extraction": "rate matrix, population fit a exp(-t/T1) + c (eigenvalue only as fallback)",
        "Debye C (1/meV^2)": debye,
    }
    rec.data["numbers"] = {f"T1 {s} {sch} {st} (s)": at(T, res[(s, sch, st)], C.QUOTE_T)
                           for s in SYSTEMS for sch in C.SCHEMES for st in STATES}
    rec.write()
    for k, v in rec.data["numbers"].items():
        print(k, {t: f"{x:.3g}" for t, x in v.items()})
    print("fallbacks:", rec.data["fallbacks"])


if __name__ == "__main__":
    main()
