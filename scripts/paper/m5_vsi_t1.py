"""M5: T1 against temperature for V_Si^- (V2, k site, S = 3/2; 128 atoms), AO and UO, against experiment.

Production settings (config.py), as M2 and M4. The initial state is the {+-1/2} doublet (optical pumping polarizes
V2 into it), T1 fitted to the doublet's summed population; the {+-3/2} doublet gives the same T1 (only the population
difference between the two doublets relaxes, the Kramers-partner rates are zero), so it is not drawn.

Experiment, only datasets in experimental_data/V_Si, never interpolated: Simin et al. (2017), 1/T1 (Hz) against
temperature, drawn as T1 = 1/rate. The dashed line marks the onset of the phonon-limited regime from their fit
(phonon_limited.py): 42.5 K, an upper limit since their temperature-independent term is given only as A0 <= 0.1 1/s.
"""

import numpy as np

import config as C
from common import Record, at, directional, rates, t1_details
from phonon_limited import SIMIN, onset, simin_dep

from spinlabs.experimental import read_experimental_csv

SYSTEM = "V_Si_128"
STATE = "ms_pm1_2"
STATE_LABEL = r"$m_s=\pm1/2$"
EXPERIMENT = (("Simin_2017", "Simin et al. (2017)"),)  # (source, legend text)


def experiment():
    """[(source, series label, legend text, T (K), T1 (s))], as measured."""
    out = []
    for source, text in EXPERIMENT:
        for s in read_experimental_csv(C.EXPERIMENT_DIR / "V_Si" / f"{source}.csv"):
            e = s.to_units(x_unit="K", y_unit="s")  # 1/T1 (Hz) -> T1 (s)
            out.append((source, s.label, text, e.x, e.y))
    return out


def main():
    plt = C.style()
    rec = Record("M5", "V_Si- (128) T1 vs temperature against experiment", "scripts/paper/m5_vsi_t1.py", "main")
    T = C.TEMPERATURES
    res, debye = {}, {}
    for sch in C.SCHEMES:
        r = rates(SYSTEM, sch)
        spin = float(r["spin"])
        if spin != 1.5:
            raise SystemExit(f"{SYSTEM} {sch}: rates computed for S = {spin:g}, expected 3/2")
        debye[sch] = {"C_0pm": float(r["debye_C"]), "C_pm": float(r["debye_C_pm"])}
        t1, eig, src = t1_details(T, directional(r), STATE, spin=spin)
        rec.t1_curve(f"{SYSTEM} {sch} {STATE}", T, t1, eig, src)
        res[sch] = t1
    rec.check_fallbacks()
    exp = experiment()

    fig, ax = plt.subplots(figsize=(C.SINGLE, 2.6))
    for sch in C.SCHEMES:
        t1 = res[sch]
        ok = np.isfinite(t1) & (t1 > 0)
        ax.plot(T[ok], t1[ok], color=C.COLORS[sch], label=f"{sch}, {STATE_LABEL}")
    for source, label, text, x, y in exp:
        color, marker = C.EXPERIMENT_STYLES[(source, label)]
        ax.plot(x, y, ls="none", marker=marker, ms=3.5, mfc=color, mec=C.EXP_EDGE, mew=0.5, label=text, zorder=3)
    t_c = onset(simin_dep, SIMIN["A0"])  # upper limit: Simin et al. give only A0 <= 0.1 1/s
    C.phonon_limited_line(ax, t_c, "phonon-limited\n(upper limit)")
    ax.set_yscale("log")
    ax.set_xlim(0, 605)
    lo = min(np.nanmin(res[sch][np.isfinite(res[sch]) & (res[sch] > 0)]) for sch in C.SCHEMES)
    ax.set_ylim(lo / 3, 1e3)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$T_1$ (s)")
    ax.text(0.97, 0.96, C.SYSTEMS[SYSTEM]["label"], transform=ax.transAxes, ha="right", va="top")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88), fontsize=C.FONT - 2, handlelength=2.2)
    fig.tight_layout()
    rec.data["files"] = C.save(fig, C.MAIN, "M5_VSi128_T1_vs_T")

    rec.data["settings"] = {
        "system": {"step": C.SYSTEMS[SYSTEM]["step"], **{sch: C.SYSTEMS[SYSTEM]["packages"][sch] for sch in C.SCHEMES}},
        "conventions": {sch: str(rates(SYSTEM, sch)["convention"]) for sch in C.SCHEMES},
        "spin": "S = 3/2 (NUPDOWN = 3); UO tensors x2/3 (spin correction)",
        "sigma_meV": C.SIGMA, "direct": C.DIRECT, "zero-gap pairs": "excluded by symmetry",
        "pairs": "diagonal from central differences + geometric-mean fill for all off-diagonal pairs",
        "first-order Raman": "excluded",
        "initial state": "{+-1/2} doublet, T1 fitted to its summed population ({+-3/2} gives the same T1)",
        "temperatures": f"{T[0]:g}-{T[-1]:g} K, step {T[1] - T[0]:g} K",
        "T1 extraction": "rate matrix, population fit a exp(-t/T1) + c (eigenvalue only as fallback)",
        "Debye C (1/meV^2)": debye,
        "experiment": {"found": [f"{source} ({label})" for source, label, *_ in exp],
                       "used as": "1/T1 in Hz, drawn as T1 = 1/rate"},
    }
    nums = {f"T1 {sch} {STATE} (s)": at(T, res[sch], C.QUOTE_T) for sch in C.SCHEMES}
    # model / experiment at each measured temperature (model on its 5 K grid, log-interpolated to the point)
    for source, label, text, x, y in exp:
        nums[f"experiment {text}, all points (T K, T1 s)"] = [[float(a), float(b)] for a, b in zip(x, y)]
        for sch in C.SCHEMES:
            ok = np.isfinite(res[sch]) & (res[sch] > 0)
            model = np.exp(np.interp(x, T[ok], np.log(res[sch][ok])))
            nums[f"T1 {sch} / {text} at its points"] = [[float(a), float(m / e)] for a, m, e in zip(x, model, y)]
    rec.data["numbers"] = nums
    rec.data["settings"]["phonon-limited line"] = (
        f"{t_c:.1f} K: Simin et al. (2017) Eq. 2 / Table I, the constant A0 below 10% of the fitted 1/T1 "
        "(criterion of Cambria et al. 2023); an upper limit since only A0 <= 0.1 1/s is given (phonon_limited.py)")
    rec.write()
    for k, v in nums.items():
        if isinstance(v, dict):
            print(k, {t: f"{x:.3g}" for t, x in v.items()})
        else:
            print(k, [[f"{a:.4g}", f"{b:.3g}"] for a, b in v])
    print("fallbacks:", rec.data["fallbacks"])


if __name__ == "__main__":
    main()
