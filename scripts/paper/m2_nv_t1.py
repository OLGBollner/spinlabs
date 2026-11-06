"""M2: NV- (512) T1 against temperature, AO and UO (m_s = 0), the Cambria et al. (2023) model, and experiment.

Experiment, only datasets in experimental_data/NV, never interpolated:
  Jarmola et al. 2012, samples S3 and S8 (1/T1 of an m_s = 0 polarization, as measured; S2 left out),
  Toyli et al. 2012 (T1 at 600 K),
  Lin et al. 2021 (NV ensemble): T1 of m_s = 0 from the measured Omega at its own temperatures, through the same
  rate-matrix solution and fit as the computed curves. At zero field p_0 obeys dp_0/dt = Omega - 3 Omega p_0 for
  any gamma, so T1(m_s = 0) = 1/(3 Omega) and the gamma series (measured at other temperatures) is not needed.
"""

import numpy as np

import config as C
from common import Record, at, cambria_rates, directional, rates, t1_details

from spinlabs.experimental import read_experimental_csv

SYSTEM, STATE = "NV_512", "ms_0"
JARMOLA_SERIES = ("S3", "S8")
QUOTE_TOL_K = 10.0  # an experimental point is quoted at 100/300/600 K only if measured within this distance


def experiment():
    """[(source, series label, legend text, T (K), T1 (s))], as measured (Lin: T1 = 1/(3 Omega) via the solver)."""
    out = []
    folder = C.EXPERIMENT_DIR / "NV"
    for s in read_experimental_csv(folder / "Jarmola_2012.csv"):
        if s.label in JARMOLA_SERIES:
            e = s.to_units(x_unit="K", y_unit="s")  # 1/T1 (Hz) -> T1 (s)
            out.append(("Jarmola_2012", s.label, f"Jarmola 2012 ({s.label})", e.x, e.y))
    for s in read_experimental_csv(folder / "Toyli_2012.csv"):
        e = s.to_units(x_unit="K", y_unit="s")
        out.append(("Toyli_2012", s.label, "Toyli 2012", e.x, e.y))
    omega = next(s for s in read_experimental_csv(folder / "Lin_2021.csv") if s.label == "omega").to_units("K", "Hz")
    w = omega.y
    zero = np.zeros_like(w)
    rates_lin = {"0_to_1": w, "0_to_-1": w, "1_to_0": w, "-1_to_0": w, "1_to_-1": zero, "-1_to_1": zero}
    t1, _, src = t1_details(omega.x, rates_lin, STATE)
    if not np.allclose(t1, 1.0 / (3.0 * w), rtol=1e-4) or set(src) != {"fit"}:
        raise SystemExit("Lin 2021: the solver's T1 differs from 1/(3 Omega)")
    out.append(("Lin_2021", "omega", "Lin 2021", omega.x, t1))
    return out


def main():
    plt = C.style()
    rec = Record("M2", "NV- (512) T1 vs temperature against experiment", "scripts/paper/m2_nv_t1.py", "main")
    T = C.TEMPERATURES
    curves = {}
    for scheme in C.SCHEMES:
        r = rates(SYSTEM, scheme)
        t1, t1_eig, src = t1_details(T, directional(r), STATE)
        rec.t1_curve(f"{SYSTEM} {scheme} {STATE}", T, t1, t1_eig, src)
        curves[scheme] = (t1, r)
    cam, omega, gamma = cambria_rates(T)
    t1_cam, eig_cam, src_cam = t1_details(T, cam, STATE)
    rec.t1_curve(f"Cambria model {STATE}", T, t1_cam, eig_cam, src_cam)
    rec.check_fallbacks()
    exp = experiment()

    fig, ax = plt.subplots(figsize=(C.SINGLE, 2.7))
    for scheme, (t1, _) in curves.items():
        ok = np.isfinite(t1) & (t1 > 0)
        ax.plot(T[ok], t1[ok], color=C.COLORS[scheme], label=f"{scheme}")
    ok = np.isfinite(t1_cam) & (t1_cam > 0)
    ax.plot(T[ok], t1_cam[ok], color=C.COLORS["cambria"], ls="-.", label="Cambria model")
    for source, label, text, x, y in exp:
        color, marker = C.EXPERIMENT_STYLES[(source, label)]
        ax.plot(x, y, ls="none", marker=marker, ms=3.5, mfc=color, mec=C.EXP_EDGE, mew=0.5, label=text, zorder=3)
    C.phonon_limited_line(ax)
    ax.set_yscale("log")
    ax.set_xlim(0, 605)  # room for the full markers of the 600 K points
    lo = min(np.nanmin(c[0][(T >= 100)]) for c in curves.values())
    ax.set_ylim(lo / 3, 1e3)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$T_1$ (s)")
    ax.legend(loc="lower left", fontsize=C.FONT - 2, handlelength=1.8, ncol=1)
    fig.tight_layout()
    rec.data["files"] = C.save(fig, C.MAIN, "M2_NV512_T1_vs_T")

    rec.data["settings"] = {
        "system": "NV- (512), step 0.05",
        "packages": {s: C.SYSTEMS[SYSTEM]["packages"][s] for s in C.SCHEMES},
        "conventions": {s: str(curves[s][1]["convention"]) for s in C.SCHEMES},
        "sigma_meV": C.SIGMA, "direct": f"{C.DIRECT} (C from the folded acoustic modes, V_0pm)",
        "pairs": "diagonal (computed) + geometric-mean fill for every off-diagonal pair (no off-diagonal runs)",
        "zero-gap pairs": "excluded by symmetry", "first-order Raman": "excluded",
        "processes": list(C.PROCESSES), "initial state": STATE,
        "temperatures": f"{T[0]:g}-{T[-1]:g} K, step {T[1] - T[0]:g} K",
        "T1 extraction": "rate matrix -> dp/dt = A p from the polarized state, a exp(-t/T1) + c fitted to p_init "
                         "(t up to 10 T1_eig + 1/T, 10000 points); eigenvalue only as fallback",
        "Cambria model": dict(C.CAMBRIA, reference="Cambria et al., PRL 130, 256903 (2023)",
                              mapping="Omega on 0<->+1 and 0<->-1 (both directions), gamma on +1<->-1"),
        "experiment": {"found": ["Jarmola_2012 (S3, S8; S2 present but not used)", "Lin_2021 (Omega, gamma)",
                                 "Toyli_2012 (600 K)"],
                       "not in repo": ["Bar-Gill 2013", "Drigo 2026"],
                       "Lin 2021 formula": "T1(m_s=0) = 1/(3 Omega) (rate-matrix solution; gamma drops out of p_0)"},
        "Debye C (1/meV^2)": {s: float(curves[s][1]["debye_C"]) for s in C.SCHEMES},
    }
    nums = {f"T1 {s} (s)": at(T, curves[s][0], C.QUOTE_T) for s in C.SCHEMES}
    nums["T1 Cambria model (s)"] = at(T, t1_cam, C.QUOTE_T)
    nums["Cambria Omega, gamma (1/s)"] = {"Omega": at(T, omega, C.QUOTE_T), "gamma": at(T, gamma, C.QUOTE_T)}
    quoted = {}
    for source, label, text, x, y in exp:
        for Tq in C.QUOTE_T:
            k = np.flatnonzero(np.abs(x - Tq) <= QUOTE_TOL_K)
            for i in k:
                quoted.setdefault(f"{Tq:g} K", []).append({"dataset": text, "T_measured_K": float(x[i]),
                                                           "T1_s": float(y[i])})
    nums["experiment near 100/300/600 K (measured points within 10 K, T1 in s)"] = quoted
    nums["experiment, all points used (T K, T1 s)"] = {text: [[float(a), float(b)] for a, b in zip(x, y)]
                                                       for _, _, text, x, y in exp}
    rec.data["numbers"] = nums
    rec.write()
    for k, v in nums.items():
        if "all points" not in k:
            print(k, v)


if __name__ == "__main__":
    main()
