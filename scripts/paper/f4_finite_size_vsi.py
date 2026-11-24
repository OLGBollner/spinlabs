"""F4: finite-size effects of V_Si^- (V2, k site, S = 3/2): the 128- and 576-atom cells, AO and UO.

Both cells with the production settings (config.py): central differences at step 0.05, diagonal second order with
the geometric-mean fill for every other pair, zero-gap pairs excluded, direct (Debye) + two-phonon rates.
(a), (b): the two-phonon spectral density along its diagonal, J2(w, w), of the Delta1 (V2_0pm) and Delta2 (V2_pm)
channels: the production couplings (spinlabs.pipeline.coupling_manager, as `spinlabs run`) with the pair weights
the two-phonon rate integrates (TransitionRate.pair_weights) and its zero-gap pairs left out, as in S2.
(c): T1 of the {+-1/2} doublet with the Simin et al. (2017) points of M5. (d), (e): the Delta1 and Delta2 rates,
+3/2 -> +1/2 and +3/2 -> -1/2 (each equal to its reverse and to its mirror, -3/2 -> -1/2 and -3/2 -> +1/2, at these
temperatures: checked); a fixed pair of levels for both cells, since D changes sign between them (128 AO: -19.4 MHz,
576 AO: +35.7 MHz). (f): T1(128) / T1(576).
The 576-atom UO package lacks the -Q runs of modes 1712 (A1, 113.7 meV) and 1717 (Ex, 114.1 meV), so has no second
difference for them; their share of the 576 AO spectral weight is recorded as an estimate of what UO misses.
The cells also differ in the phonon modes run along: unaligned (128) and mirror-aligned (576) degenerate pairs.
Shaded in (c) and (f): the effect of the mirror projection, from each curve to the one without it (dotted;
config.projection_band).
Outputs: figures/out/sm/F4_finite_size_VSi.{pdf,png}, figures/out/sm/numbers_finite_size_vsi.md, records/F4.json.
"""

import tempfile
from pathlib import Path

import numpy as np

import config as C
from common import Record, _key, directional, manager_for, package_key, rates, t1_details
from m5_vsi_t1 import STATE, STATE_LABEL, experiment

from spinlabs.models import SpinPhononCouplingData
from spinlabs.transition_rate import TransitionRate
from spinlabs.utils import MathUtils

T = C.TEMPERATURES
CELLS = ("V_Si_128", "V_Si_576")
STYLE = {"V_Si_576": "-", "V_Si_128": "--"}
CELL_LABEL = {"V_Si_576": "576", "V_Si_128": "128"}
QUOTE = (50.0, 100.0, 300.0, 500.0)
T_MIN = 25.0  # K, first temperature of the rates and T1 (as F1 for V_Si-, left of the phonon-limited line)
T_MIN_RATIO = 50.0  # K, first temperature of (f)
CHANNELS = (("V2_0_pm", r"J^{(2)}_{\Delta 1}"), ("V2_p_m", r"J^{(2)}_{\Delta 2}"))
RATES = (  # (name, panel title, key, keys equal to it)
    ("dm1", r"$\Gamma_{\Delta 1}$ ($+\frac{3}{2}\to+\frac{1}{2}$)", "1.5_to_0.5", ("0.5_to_1.5", "-1.5_to_-0.5")),
    ("dm2", r"$\Gamma_{\Delta 2}$ ($+\frac{3}{2}\to-\frac{1}{2}$)", "1.5_to_-0.5", ("-0.5_to_1.5", "-1.5_to_0.5")),
)
UO_MISSING_576 = (1712, 1717)  # 1-based modes without their -Q run in the 576 UO package
MHZ_PER_MEV = 241798.9242


def spectral(system, scheme):
    """J2(w, w) per channel of the production couplings: {"omega": grid (meV), "<channel>": cut (MHz^2/meV^2),
    "<channel>_weight": summed pair weight, "frequencies", "share_<mode>": a mode's share of the diagonal weight}."""
    cache = C.CACHE / f"j2_diagonal_{system}_{scheme}_{C.PAIRS}_s{C.SIGMA:g}_{_key(package_key(system, scheme))}.npz"
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    print(f"  computing J2: {system} {scheme}")
    manager, _ = manager_for(system, scheme, C.PAIRS)
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "coupling")
        manager.process_both_orders(output_filename=base)
        data = SpinPhononCouplingData.load(f"{base}.npz").to_unit("MHz").frequencies_to_unit("meV")
    w = np.asarray(data.frequencies, dtype=float)
    zero_gap = TransitionRate.symmetry_zero_gap_pairs(len(w), manager.spectrum.deg_groups or [])
    res = min(0.5, C.SIGMA / 4)
    out = {"frequencies": w}
    for key, _ in CHANNELS:
        weights = TransitionRate.pair_weights(np.where(zero_gap, 0.0, np.asarray(getattr(data, key), dtype=float)))
        X, _, J = MathUtils.get_2d_spectral_density(w, weights, res, C.SIGMA)
        out["omega"] = X[0] if X.ndim == 2 else X
        out[key] = np.diag(J)
        out[f"{key}_weight"] = np.array(weights.sum())
        diag = np.abs(np.diag(np.asarray(getattr(data, key), dtype=float))) ** 2
        for mode in UO_MISSING_576:  # what these two modes carry where they were run (576 AO)
            if mode <= len(w):
                out[f"{key}_share_{mode}"] = np.array(diag[mode - 1] / diag.sum() if diag.sum() else 0.0)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def main():
    plt = C.style()
    rec = Record("F4", "V_Si- finite-size effects: the 128- and 576-atom cells, spectral functions, rates and T1",
                 "scripts/paper/f4_finite_size_vsi.py", "sm")
    J, gamma, gamma_raw, t1, t1_raw, zfs, symmetric = {}, {}, {}, {}, {}, {}, {}
    for system in CELLS:
        for scheme in C.SCHEMES:
            key = (system, scheme)
            J[key] = spectral(system, scheme)
            r, r_raw = rates(system, scheme), rates(system, scheme, mirror_projection=False)
            spin = float(r["spin"])
            if spin != 1.5:
                raise SystemExit(f"{system} {scheme}: rates computed for S = {spin:g}, expected 3/2")
            zfs[key] = float(r["zfs_mev"]) * MHZ_PER_MEV
            tot, tot_raw = directional(r), directional(r_raw)
            ok = T >= 50.0
            for name, _, k, same in RATES:
                gamma[key + (name,)] = np.asarray(tot[k])
                gamma_raw[key + (name,)] = np.asarray(tot_raw[k])
                symmetric[f"{system} {scheme} {name}"] = float(max(
                    np.max(np.abs(np.asarray(tot[s])[ok] / np.asarray(tot[k])[ok] - 1)) for s in same))
            t, t_eig, src = t1_details(T, tot, STATE, spin=spin)
            rec.t1_curve(f"{system} {scheme} {STATE}", T, t, t_eig, src)
            t1[key] = t
            t1_raw[key] = t1_details(T, tot_raw, STATE, spin=spin)[0]
    rec.check_fallbacks()
    if max(symmetric.values()) > 1e-3:
        raise SystemExit(f"rates of one channel differ between its transitions: {symmetric}")
    exp = experiment()

    fig, axes = plt.subplots(2, 3, figsize=(C.DOUBLE, 4.6))
    (a, b, c), (d, e, f) = axes
    for ax, (key, tex) in zip((a, b), CHANNELS):
        for (system, scheme), data in J.items():
            ax.plot(data["omega"], data[key], color=C.COLORS[scheme], ls=STYLE[system], lw=1.0,
                    label=f"{scheme}, {CELL_LABEL[system]} atoms")
        top = max(float(data[key].max()) for data in J.values())
        ax.set_yscale("log")
        ax.set_ylim(top * 1e-4, top * 20)
        ax.set_xlim(0, max(float(data["omega"].max()) for data in J.values()))
        ax.set_xlabel(r"$\omega$ (meV)")
        ax.text(0.04, 0.95, f"${tex}(\\omega,\\omega)$", transform=ax.transAxes, va="top", fontsize=C.FONT - 1)
    a.set_ylabel(r"MHz$^2$/meV$^2$")  # colours and line styles as in the legend of (c)

    okT = T >= T_MIN
    for (system, scheme), y in t1.items():
        c.plot(T[okT], y[okT], color=C.COLORS[scheme], ls=STYLE[system], zorder=2,
               label=f"{scheme}, {CELL_LABEL[system]}")
        C.projection_band(c, T, y, t1_raw[(system, scheme)], C.COLORS[scheme], ok=okT)
    for source, label, text, x, y in exp:
        color, marker = C.EXPERIMENT_STYLES[(source, label)]
        c.plot(x, y, ls="none", marker=marker, ms=3, mfc=color, mec=C.EXP_EDGE, mew=0.5, label=text, zorder=3)
    c.set_yscale("log")
    c.set_xlim(0, 605)
    c.set_ylim(min(np.nanmin(v[T >= 100]) for d_ in (t1, t1_raw) for v in d_.values()) / 3, 1e3)
    c.set_xlabel("Temperature (K)")
    c.set_ylabel(f"$T_1$ ({STATE_LABEL}) (s)")
    c.legend(loc="upper right", fontsize=C.FONT - 2.5, handlelength=2.2)  # the key of every panel

    for ax, (name, title, _, _) in zip((d, e), RATES):
        for system in CELLS:
            for scheme in C.SCHEMES:
                y = gamma[(system, scheme, name)]
                ax.plot(T[okT], y[okT], color=C.COLORS[scheme], ls=STYLE[system])
                C.projection_band(ax, T, y, gamma_raw[(system, scheme, name)], C.COLORS[scheme], ok=okT)
        ax.set_yscale("log")
        ax.set_xlim(0, 605)
        ax.set_xlabel("Temperature (K)")
        ax.set_title(title, fontsize=C.FONT)
    d.set_ylabel(r"rate (s$^{-1}$)")

    okr = T >= T_MIN_RATIO
    ratio, ratio_raw = {}, {}
    for scheme in C.SCHEMES:
        ratio[scheme] = t1[("V_Si_128", scheme)] / t1[("V_Si_576", scheme)]
        ratio_raw[scheme] = t1_raw[("V_Si_128", scheme)] / t1_raw[("V_Si_576", scheme)]
        f.plot(T[okr], ratio[scheme][okr], color=C.COLORS[scheme], label=scheme)
        C.projection_band(f, T, ratio[scheme], ratio_raw[scheme], C.COLORS[scheme], ok=okr)
    f.axhline(1.0, color=C.COLORS["ref"], lw=0.6, ls=":")
    f.set_xlim(0, 605)
    f.set_xlabel("Temperature (K)")
    f.set_ylabel(r"$T_1(128)\,/\,T_1(576)$")
    f.legend(loc="best", fontsize=C.FONT - 2)
    for k, ax in enumerate(axes.flat):
        C.panel_letter(ax, k, x=-0.16 if ax is c else -0.02)  # (c): clear of its 10^3 tick label
    fig.tight_layout()
    rec.data["files"] = C.save(fig, C.SM, "F4_finite_size_VSi")
    plt.close(fig)

    # ------------------------------------------------------------------ numbers
    idx = [int(np.flatnonzero(np.isclose(T, t))[0]) for t in QUOTE]
    hdr = " | ".join(f"{t:g} K" for t in QUOTE)
    rows = lambda vals, fmt: " | ".join(format(vals[i], fmt) for i in idx)  # noqa: E731
    L = ["# Finite-size effects, V_Si- (F4)", "",
         "Generated by scripts/paper/f4_finite_size_vsi.py. Production settings (sigma "
         f"{C.SIGMA:g} meV, Debye direct rate, direct + two-phonon, zero-gap pairs excluded, central differences at "
         "step 0.05, diagonal + geometric-mean fill), T1 of the {+-1/2} doublet. The 576-atom UO package lacks the -Q "
         "runs of modes 1712 (A1) and 1717 (Ex).", "",
         "## Zero-field splitting D (MHz)", "", "| cell | AO | UO |", "|---|---|---|"]
    for system in CELLS:
        L.append(f"| {CELL_LABEL[system]} | {zfs[(system, 'AO')]:.1f} | {zfs[(system, 'UO')]:.1f} |")
    L += ["", "## T1 (s)", "", f"| cell, scheme | {hdr} |", "|---|" + "---|" * len(QUOTE)]
    for (system, scheme), v in t1.items():
        L.append(f"| {CELL_LABEL[system]} {scheme} | {rows(v, '.4g')} |")
    L += ["", "## T1(128) / T1(576)", "", f"| scheme | {hdr} |", "|---|" + "---|" * len(QUOTE)]
    for scheme in C.SCHEMES:
        L.append(f"| {scheme} | {rows(ratio[scheme], '.3f')} |")
    for name, title, k, _ in RATES:
        L += ["", f"## {name} rate {k} (1/s)", "", f"| cell, scheme | {hdr} |", "|---|" + "---|" * len(QUOTE)]
        for system in CELLS:
            for scheme in C.SCHEMES:
                L.append(f"| {CELL_LABEL[system]} {scheme} | {rows(gamma[(system, scheme, name)], '.4g')} |")
        L.append(f"| ratio 128/576 AO | {rows(gamma[('V_Si_128', 'AO', name)] / gamma[('V_Si_576', 'AO', name)], '.3f')} |")
        L.append(f"| ratio 128/576 UO | {rows(gamma[('V_Si_128', 'UO', name)] / gamma[('V_Si_576', 'UO', name)], '.3f')} |")
    L += ["", "## Two-phonon spectral weight (summed pair weights, MHz^2)", "",
          "| channel | 128 AO | 576 AO | 576/128 AO | 128 UO | 576 UO | 576/128 UO |", "|---|---|---|---|---|---|---|"]
    weight = {}
    for key, _ in CHANNELS:
        w = {(s, sc): float(J[(s, sc)][f"{key}_weight"]) for s in CELLS for sc in C.SCHEMES}
        weight[key] = w
        L.append(f"| {key} | {w[('V_Si_128', 'AO')]:.4g} | {w[('V_Si_576', 'AO')]:.4g} | "
                 f"{w[('V_Si_576', 'AO')] / w[('V_Si_128', 'AO')]:.3f} | {w[('V_Si_128', 'UO')]:.4g} | "
                 f"{w[('V_Si_576', 'UO')]:.4g} | {w[('V_Si_576', 'UO')] / w[('V_Si_128', 'UO')]:.3f} |")
    share = {f"{key} mode {m}": float(J[("V_Si_576", "AO")][f"{key}_share_{m}"]) for key, _ in CHANNELS
             for m in UO_MISSING_576}
    L += ["", "## Modes missing from 576 UO: their share of the 576 AO diagonal weight |V2_ii|^2", "",
          "| channel, mode | share |", "|---|---|"] + [f"| {k} | {v:.2%} |" for k, v in share.items()]
    L += ["", "## Check: each rate equals its reverse and mirror transitions (max rel diff, T >= 50 K)", ""]
    L += [f"- {k}: {v:.1e}" for k, v in symmetric.items()] + [""]
    path = C.SM / "numbers_finite_size_vsi.md"
    path.write_text("\n".join(L) + "\n")
    print("written", path)

    rec.data["settings"] = {
        "packages": {f"{s} {sc}": C.SYSTEMS[s]["packages"][sc] for s in CELLS for sc in C.SCHEMES},
        "sigma_meV": C.SIGMA, "direct": C.DIRECT, "processes": list(C.PROCESSES), "initial state": STATE,
        "pairs": "diagonal from central differences + geometric-mean fill, zero-gap pairs excluded (production)",
        "spectral density": "J2(w, w) of the production couplings, TransitionRate.pair_weights, zero-gap pairs out, "
                            f"grid {min(0.5, C.SIGMA / 4):g} meV",
        "rates drawn": {name: k for name, _, k, _ in RATES},
        "phonon modes": "128: unaligned (phonon_data.npz); 576: mirror-aligned (phonon_data_aligned.npz)",
        "576 UO": "no -Q runs of modes 1712 (A1, 113.7 meV) and 1717 (Ex, 114.1 meV): no second difference for them",
        "experiment": "Simin et al. (2017), as M5", "mirror projection": C.MIRROR_PROJECTION, "band": C.BAND_NOTE,
        "numbers file": "sm/numbers_finite_size_vsi.md",
    }
    rec.data["numbers"] = {
        "D (MHz)": {f"{s} {sc}": v for (s, sc), v in zfs.items()},
        "T1 (s)": {f"{s} {sc}": {f"{t:g} K": float(v[i]) for t, i in zip(QUOTE, idx)} for (s, sc), v in t1.items()},
        "T1(128)/T1(576)": {sc: {f"{t:g} K": float(ratio[sc][i]) for t, i in zip(QUOTE, idx)} for sc in C.SCHEMES},
        "T1(128)/T1(576), no mirror projection": {sc: {f"{t:g} K": float(ratio_raw[sc][i]) for t, i in zip(QUOTE, idx)}
                                                   for sc in C.SCHEMES},
        "rates (1/s)": {f"{s} {sc} {n}": {f"{t:g} K": float(v[i]) for t, i in zip(QUOTE, idx)}
                        for (s, sc, n), v in gamma.items()},
        "two-phonon spectral weight (MHz^2)": {k: {f"{s} {sc}": v for (s, sc), v in w.items()} for k, w in weight.items()},
        "576 UO missing modes, share of the 576 AO diagonal weight": share,
        "rate symmetry check (max rel diff)": symmetric,
    }
    rec.write()
    print("\n".join(L))


if __name__ == "__main__":
    main()
