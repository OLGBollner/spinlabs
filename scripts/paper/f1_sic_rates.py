"""F1: transition rates of the SiC defects against temperature, AO and UO, production settings.

One row per defect, in the layout of the NV- figure sq_dq_rates_cambria_measured
(figures/cambria_comparison/sq_dq_rates.py), without experiment:
  VV0 (128), ClV+ (576), S = 1: (a) Omega (0 <-> +-1), (b) gamma (+1 <-> -1);
  V_Si- (128), S = 3/2: (a) dm = 1 (+3/2 <-> +1/2, equal to -3/2 <-> -1/2), (b) dm = 2 (+3/2 <-> -1/2, equal to
  -3/2 <-> +1/2). The within-doublet rates (+1/2 <-> -1/2, +3/2 <-> -3/2) are zero by symmetry; checked here.
Direction convention, as the NV- figure (Omega = 0 -> +1 there, upward since D > 0): each plotted rate is the
upward rate (lower to upper level) of the scheme's own ZFS D; gamma connects degenerate levels at zero field, so
both its directions are the same rate. Rates are directional (direct + two-phonon, C.PROCESSES), drawn from 50 K
as in the NV- figure. Phonon-limited line: none for VV0 and ClV+; for V_Si- the onset from the Simin et al. (2017)
fit (phonon_limited.py, as M5): 42.5 K, an upper limit since only A0 <= 0.1 1/s is given.
Outputs: figures/out/sm/F1_sic_rates.{pdf,png}, figures/out/sm/numbers_sic_rates.md, records/F1.json.
"""

import numpy as np

import config as C
from common import Record, directional, rates
from phonon_limited import SIMIN, onset, simin_dep

T = C.TEMPERATURES
QUOTE = (100.0, 300.0, 500.0)
T_MIN = 50.0  # K, first temperature drawn (as the NV- figure)
MHZ_PER_MEV = 241798.9242
ROWS = ("DiV_128", "ClV_576", "V_Si_128")
NAMES = {"DiV_128": "VV0 (128)", "ClV_576": "ClV+ (576)", "V_Si_128": "V_Si- (128)"}


def channels(spin, d_mev):
    """[(name, panel title, upward key, downward key)] of a defect: upward = lower level -> upper level of D."""
    if spin == 1.0:
        up, down = ("0_to_1", "1_to_0") if d_mev > 0 else ("1_to_0", "0_to_1")
        return [("Omega", r"$\Omega$ ($0\leftrightarrow\pm1$)", up, down),
                ("gamma", r"$\gamma$ ($+1\leftrightarrow-1$)", "1_to_-1", "-1_to_1")]
    # S = 3/2: E_m = D (m^2 - 5/4), the +-3/2 doublet lower for D < 0
    dm1, dm2 = ("1.5", "0.5"), ("1.5", "-0.5")  # (+3/2 doublet member, +-1/2 doublet member)
    if d_mev > 0:  # +-1/2 lower: upward is from the +-1/2 member
        dm1, dm2 = dm1[::-1], dm2[::-1]
    return [("dm1", r"$\Delta m_s=1$ ($\pm\frac{3}{2}\leftrightarrow\pm\frac{1}{2}$)", "_to_".join(dm1), "_to_".join(dm1[::-1])),
            ("dm2", r"$\Delta m_s=2$ ($\pm\frac{3}{2}\leftrightarrow\mp\frac{1}{2}$)", "_to_".join(dm2), "_to_".join(dm2[::-1]))]


def main():
    plt = C.style()
    rec = Record("F1", "SiC defects: transition rates vs temperature, AO and UO", "scripts/paper/f1_sic_rates.py", "sm")
    idx = [int(np.flatnonzero(np.isclose(T, t))[0]) for t in QUOTE]
    data, zero_check, conv = {}, {}, {}
    for system in ROWS:
        for scheme in C.SCHEMES:
            r = rates(system, scheme)
            spin, d = float(r["spin"]), float(r["zfs_mev"])
            tot = directional(r)
            chans = channels(spin, d)
            data[(system, scheme)] = {name: np.asarray(tot[up]) for name, _, up, _ in chans}
            conv[(system, scheme)] = {"spin": spin, "D_MHz": d * MHZ_PER_MEV, "channels": [
                {"name": n, "upward key": up, "downward key": dn,
                 "max |up - down| / up (T >= 50 K)": float(np.max(np.abs(tot[up] - tot[dn])[T >= T_MIN] / tot[up][T >= T_MIN]))}
                for n, _, up, dn in chans]}
            if spin == 1.5:
                for key in ("0.5_to_-0.5", "-0.5_to_0.5", "1.5_to_-1.5", "-1.5_to_1.5"):
                    vals = np.concatenate([np.asarray(r[f"{p}:{key}"], dtype=float)
                                           for p in ("first_order", "second_order", "two_phonon")])
                    zero_check[f"{scheme} {key}"] = (bool(np.all(vals == 0.0)), float(np.max(np.abs(vals))))
    for system in ROWS:
        for scheme in C.SCHEMES:
            for name, arr in data[(system, scheme)].items():
                if not np.all(arr[T >= T_MIN] > 0):
                    raise SystemExit(f"{system} {scheme} {name}: non-positive rate at T >= {T_MIN:g} K")
    spin_of = {s: conv[(s, "AO")]["spin"] for s in ROWS}
    t_c = onset(simin_dep, SIMIN["A0"])  # V_Si-, upper limit (as M5)

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(len(ROWS), 2, figsize=(C.DOUBLE, 2.5 * len(ROWS)), sharex=True, sharey="row")
    ok = T >= T_MIN
    for row, system in enumerate(ROWS):
        chans = channels(spin_of[system], conv[(system, "AO")]["D_MHz"])
        lo = min(data[(system, s)][n][ok].min() for s in C.SCHEMES for n, *_ in chans)
        hi = max(data[(system, s)][n][ok].max() for s in C.SCHEMES for n, *_ in chans)
        for col, (name, title, _, _) in enumerate(chans):
            ax = axes[row, col]
            for scheme in C.SCHEMES:
                ax.semilogy(T[ok], data[(system, scheme)][name][ok], color=C.COLORS[scheme], lw=1.4,
                            label=f"{C.SYSTEMS[system]['label']} {scheme}")
            ax.set_xlim(0, 620)
            ax.set_ylim(10 ** np.floor(np.log10(lo)), 10 ** np.ceil(np.log10(hi)))
            ax.set_title(title, pad=3)
            if system == "V_Si_128":
                C.phonon_limited_line(ax, t_c, "phonon-limited\n(upper limit)")
            C.panel_letter(ax, 2 * row + col)
            if row == len(ROWS) - 1:
                ax.set_xlabel("Temperature (K)")
        axes[row, 0].set_ylabel(f"{C.SYSTEMS[system]['label']}\n" + r"Rate (s$^{-1}$)")
        axes[row, 1].legend(loc="lower right", fontsize=C.FONT - 2)
    fig.tight_layout()
    rec.data["files"] = C.save(fig, C.SM, "F1_sic_rates")
    plt.close(fig)

    # ------------------------------------------------------------------ numbers
    hdr = " | ".join(f"{t:g} K" for t in QUOTE)
    L = ["# Transition rates of the SiC defects (F1)", "",
         "Generated by scripts/paper/f1_sic_rates.py. Production settings (scripts/paper/config.py): sigma "
         f"{C.SIGMA:g} meV, Debye direct rate, direct + two-phonon, zero-gap pairs excluded, second-order pairs "
         "computed where the package has them and geometric-mean fill for the rest. Rates in 1/s.", "",
         "## Direction convention", "",
         "Each plotted rate is the upward rate (lower level -> upper level, by the sign of the scheme's own computed "
         "D), the convention of the NV- figure sq_dq_rates_cambria_measured (Omega = 0 -> +1, upward since D > 0 "
         "for NV-). gamma (+1 <-> -1) connects levels degenerate at zero field: both directions are the same rate. "
         "Upward and downward differ by the detailed-balance factor exp(-gap/kT), gap <= 0.0074 meV, so by < 2e-3 "
         "at and above 50 K (column below).", "",
         "| defect, scheme | spin | D (MHz) | channel | upward key | downward key | max abs(up - down)/up, T >= 50 K |",
         "|---|---|---|---|---|---|---|"]
    for (system, scheme), c in conv.items():
        for ch in c["channels"]:
            L.append(f"| {NAMES[system]} {scheme} | {c['spin']:g} | {c['D_MHz']:.1f} | {ch['name']} | {ch['upward key']} "
                     f"| {ch['downward key']} | {ch['max |up - down| / up (T >= 50 K)']:.1e} |")
    L += ["", "V_Si- channel names: dm1 = +3/2 <-> +1/2 (equal to -3/2 <-> -1/2), dm2 = +3/2 <-> -1/2 (equal to "
          "-3/2 <-> +1/2). The two V_Si- schemes have opposite signs of D (AO < 0: +-3/2 doublet lower; UO > 0: "
          "+-1/2 lower), so their upward keys differ.", "", f"## Rates (1/s)", "",
          f"| defect | scheme | channel | {hdr} |", "|---|---|---|" + "---|" * len(QUOTE)]
    nums = {}
    for (system, scheme), d in data.items():
        for name, arr in d.items():
            L.append(f"| {NAMES[system]} | {scheme} | {name} | " + " | ".join(f"{arr[i]:.4g}" for i in idx) + " |")
            nums[f"{system} {scheme} {name} (1/s)"] = {f"{t:g} K": float(arr[i]) for t, i in zip(QUOTE, idx)}
    L += ["", "## V_Si- within-doublet rates", "",
          "Every process (first_order, second_order, two_phonon), every direction, every temperature of the grid "
          f"({T[0]:g}-{T[-1]:g} K):", "", "| scheme, transition | exactly zero | max abs value (1/s) |", "|---|---|---|"]
    for k, (zero, mx) in zero_check.items():
        L.append(f"| {k} | {'yes' if zero else 'NO'} | {mx:g} |")
    L += ["", "## Phonon-limited line", "",
          "VV0 and ClV+: none. V_Si-: dashed line at the onset from the Simin et al. (2017) fit, "
          f"{t_c:.1f} K, an upper limit since only A0 <= 0.1 1/s is given (scripts/paper/phonon_limited.py, as M5)."]
    path = C.SM / "numbers_sic_rates.md"
    path.write_text("\n".join(L) + "\n")
    print("written", path)

    rec.data["settings"] = {
        "systems": {s: {sc: C.SYSTEMS[s]["packages"][sc] for sc in C.SCHEMES} for s in ROWS},
        "sigma_meV": C.SIGMA, "direct": C.DIRECT, "pairs": C.PAIRS, "processes": list(C.PROCESSES),
        "zero-gap pairs": "excluded by symmetry", "temperatures drawn": f"{T_MIN:g}-{T[-1]:g} K",
        "direction convention": "upward rate (lower -> upper level by the scheme's computed D), as the NV- figure; "
                                "gamma: degenerate levels, both directions equal",
        "conventions": {f"{s} {sc}": v for (s, sc), v in conv.items()},
        "V_Si- within-doublet zero check": {k: {"exactly zero": z, "max abs": m} for k, (z, m) in zero_check.items()},
        "phonon-limited line": f"VV0, ClV+: none; V_Si-: {t_c:.1f} K (Simin et al. 2017 fit, upper limit; phonon_limited.py)",
        "numbers file": "sm/numbers_sic_rates.md",
    }
    rec.data["numbers"] = nums
    rec.write()
    for k, (zero, mx) in zero_check.items():
        print(f"zero check {k}: {'exactly zero' if zero else f'NOT zero (max {mx:g})'}")


if __name__ == "__main__":
    main()
