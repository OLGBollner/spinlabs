"""S6: Debye fit of the direct rate: |V|^2 against E with the folded acoustic modes, both channels, every system.

Rows: V_0+- (the |dm| = 1 direct rate, used by the production rates) and V_+- (|dm| = 2); columns: NV- (64),
NV- (512), VV0 (128), ClV+ (576); AO couplings (MHz). Lines: |V|^2 = a_s E per acoustic branch s (spinlabs.debye).
C_0+- and C_+- (1/meV^2) for AO and UO and their ratio go to the record.
"""

import numpy as np
from scipy import constants as Cn

import config as C
from common import Record, couplings, rates

from spinlabs.debye import debye_coefficient
from spinlabs.parsers import parse_phonon_data

SYSTEMS = ("NV_64", "NV_512", "DiV_128", "ClV_576")
CHANNELS = (("V_0_pm", r"$|V_{0\pm}|^2$ (MHz$^2$)", "C_0pm"), ("V_p_m", r"$|V_{+-}|^2$ (MHz$^2$)", "C_pm"))
MHZ_PER_MEV = Cn.e * 1e-3 / Cn.h / 1e6


def fit(system, scheme, key):
    spec = parse_phonon_data(str(C.DATA / system / "phonon_data.npz"))
    c = couplings(system, scheme)
    return debye_coefficient(spec, c["frequencies"], np.abs(c[key]) / MHZ_PER_MEV), c


def main():
    plt = C.style()
    rec = Record("S6", "Debye fit of the acoustic modes", "scripts/paper/s6_debye.py", "sm")
    nums = {}
    for s in SYSTEMS:
        for sch in C.SCHEMES:
            f0, _ = fit(s, sch, "V_0_pm")
            fpm, _ = fit(s, sch, "V_p_m")
            used = float(rates(s, sch)["debye_C"])
            if not np.isclose(f0.C, used, rtol=1e-6):
                raise SystemExit(f"{s} {sch}: C_0pm {f0.C:.4e} differs from the rates' Debye C {used:.4e}")
            nums[f"{s} {sch}"] = {"C_0pm (1/meV^2)": f0.C, "C_pm (1/meV^2)": fpm.C, "C_pm / C_0pm": fpm.C / f0.C}

    fig, axes = plt.subplots(2, 4, figsize=(C.DOUBLE, 3.6))
    for col, s in enumerate(SYSTEMS):
        for row, (key, ylabel, cname) in enumerate(CHANNELS):
            ax = axes[row, col]
            dfit, c = fit(s, "AO", key)
            E = np.asarray(c["frequencies"], dtype=float)
            V2 = np.abs(np.asarray(c[key], dtype=float)) ** 2
            emax = 1.6 * max(b["energy_mev"] for b in dfit.branches.values())
            other = np.ones(len(E), dtype=bool)
            other[dfit.acoustic_modes] = False
            m = other & (E > 0) & (E <= emax)
            ax.plot(E[m], V2[m], ls="none", marker=".", ms=2.5, color=C.COLORS["other_modes"], label="other modes",
                    zorder=1)
            ymax = 0.0
            for name, b in dfit.branches.items():
                idx = b["modes"]
                ax.plot(E[idx], V2[idx], ls="none", marker="s" if name == "longitudinal" else "o", ms=3.5,
                        color=C.COLORS[name], mec="white", mew=0.4, label=f"{name} acoustic", zorder=3)
                x = np.array([0.0, emax])
                ax.plot(x, b["a_mev"] * MHZ_PER_MEV**2 * x, color=C.COLORS[name], lw=1.0, zorder=2,
                        label=fr"$a_s E$, {name[:5]}.")
                ymax = max(ymax, V2[idx].max(), b["a_mev"] * MHZ_PER_MEV**2 * emax)
            ax.set_xlim(0, emax)
            ax.set_ylim(0, 1.15 * ymax)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            ax.yaxis.get_offset_text().set_fontsize(C.FONT - 2)
            ax.grid(True, ls=":", lw=0.5, alpha=0.6)
            m_, e_ = f"{dfit.C:.1e}".split("e")
            ax.text(0.04, 0.95, rf"$C = {m_}\times10^{{{int(e_)}}}$ meV$^{{-2}}$", transform=ax.transAxes, va="top",
                    fontsize=C.FONT - 2)
            C.panel_letter(ax, row * 4 + col, x=0.0, y=1.02, fontsize=C.FONT - 1)
            if row == 0:
                ax.set_title(C.SYSTEMS[s]["label"], pad=10)
            if row == 1:
                ax.set_xlabel(r"$E$ (meV)")
            if col == 0:
                ax.set_ylabel(ylabel)
    h, l = axes[0, 0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.05),
               fontsize=C.FONT - 1)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=0.5)
    rec.data["files"] = C.save(fig, C.SM, "S6_debye_fit")

    rec.data["settings"] = {
        "systems": {s: {sch: C.SYSTEMS[s]["packages"][sch] for sch in C.SCHEMES} for s in SYSTEMS},
        "figure": "AO couplings; |V|^2 in MHz^2 against the mode energy",
        "model": "J(E) = C E^3, C = sum_s n_s V a_s / (2 pi^2 (hbar v_s)^3), a_s = <|V|^2/E> over the branch's "
                 "acoustic modes of the first reciprocal-lattice shell (spinlabs.debye, acoustic weight > 0.5)",
        "C_0pm": "used by the production rates (|dm| = 1); checked equal to the rates' Debye C",
        "C_pm": "|dm| = 2 channel; zero contribution at zero field (+1 and -1 degenerate)",
    }
    rec.data["numbers"] = {"Debye coefficients": nums}
    rec.write()
    for k, v in nums.items():
        print(k, {a: f"{b:.3g}" for a, b in v.items()})


if __name__ == "__main__":
    main()
