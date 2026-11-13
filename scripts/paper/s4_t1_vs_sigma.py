"""S4: T1 at 300 K (m_s = 0) against the broadening sigma, for NV- (64), NV- (512), VV0 (128); one figure for AO
(S4_T1_vs_sigma) and one for UO (S4_UO_T1_vs_sigma), the same layout.

Curves (second-order pairs):
  diagonal only        (l, l) pairs, zero-gap pairs kept (every diagonal pair is a zero-gap pair)
  diagonal + fill      diagonal + sqrt fill on every off-diagonal pair, zero-gap pairs kept
  fill, no zero gap    the same, zero-gap pairs excluded by symmetry (production fill)
  computed pairs       the explicitly computed pairs, zero-gap pairs excluded (systems with off-diagonal pairs
                       computed: AO NV- (64) and VV0 (128); UO VV0 (128) only, NV- (64) UO has the diagonal only)
  fill on computed     the fill kept on the computed pairs only (the S3 filter), zero-gap pairs excluded (same systems)
The first three curves are unfiltered.
Row (a-c): Debye direct rate; row (d-f): Gaussian-smeared direct rate J(D). Vertical line: production sigma.
Rates: direct + two-phonon (no first-order Raman); T1 by the population fit as everywhere else.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

import config as C
from common import DIRECTIONS, Record, _key, manager_for, mask_to_computed, pair_coverage, t1_details

from spinlabs.constants import CONSTANTS
from spinlabs.transition_rate import TransitionRate

SYSTEMS = ("NV_64", "NV_512", "DiV_128")
SIGMAS = [0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 3.5, 5.0, 6.0, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0]
T = 300.0
STATE = "ms_0"
CURVES = {  # key: (pairs, nogap, label)
    "diagonal": ("diagonal", False, "diagonal only"),
    "diagonal-sqrt": ("diagonal-sqrt", False, "diagonal + fill"),
    "fill": ("diagonal-sqrt", True, "fill, no zero-gap pairs"),
    "computed": ("computed", True, C.PAIR_LABELS["computed"]),
    "fill-computed": ("fill-on-computed", True, C.PAIR_LABELS["fill-computed"]),
}
ONLY_WITH_COMPUTED = ("computed", "fill-computed")
HAS_COMPUTED = {"AO": ("NV_64", "DiV_128"), "UO": ("DiV_128",)}  # off-diagonal pairs computed
OUTPUT = {"AO": ("S4", "S4_T1_vs_sigma"), "UO": ("S4_UO", "S4_UO_T1_vs_sigma")}  # record id, file name
DIRECTS = ("debye", "gaussian")


def scan(system, scheme, pairs, nogap):
    """{direct: [directional rates at T per sigma]}"""
    key = _key(str(C.package(system, scheme)), pairs, nogap, T, SIGMAS, DIRECTS, list(DIRECTIONS), list(C.PROCESSES))
    cache = C.CACHE / f"s4_{system}_{scheme}_{pairs}_{'nogap' if nogap else 'gap'}_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print(f"  sigma scan: {system} {scheme} {pairs} nogap={nogap}")
    manager, _ = manager_for(system, scheme, pairs)
    out = {d: [] for d in DIRECTS}
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "coupling")
        manager.process_both_orders(output_filename=base)
        if pairs == "fill-on-computed":
            mask_to_computed(system, scheme, f"{base}.npz")
        calc = TransitionRate(f"{base}.npz")
        calc.data_2ph = calc.data
        calc.set_debye_from_spectrum(manager.spectrum)
        zfs = calc.data["zfs"] / CONSTANTS["meV2J"]
        groups = (manager.spectrum.deg_groups or []) if nogap else None
        for sigma in SIGMAS:
            omega, j0pm, jpm, j00 = calc.get_spectral_density(res=0.02, sigma=sigma)
            lines = calc.get_line_spectral_density(sigma, degenerate_groups=groups)
            for d in DIRECTS:
                calc.direct = d
                calc.compute_transition_rates(T, omega, j0pm, jpm, j00, zfs)
                calc.compute_two_phonon_rates(T, lines)
                out[d].append({k: float(sum(calc.transition_rate[p].get(k, 0.0) for p in C.PROCESSES))
                               for k in DIRECTIONS})
    C.CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def figure(plt, scheme):
    fid, name = OUTPUT[scheme]
    rec = Record(fid, f"T1 at 300 K against sigma, {scheme}", "scripts/paper/s4_t1_vs_sigma.py", "sm")
    res = {}
    for s in SYSTEMS:
        for key, (pairs, nogap, _) in CURVES.items():
            if key in ONLY_WITH_COMPUTED and s not in HAS_COMPUTED[scheme]:
                continue
            sc = scan(s, scheme, pairs, nogap)
            for d in DIRECTS:
                t1s, eigs, srcs = [], [], []
                for sigma, r in zip(SIGMAS, sc[d], strict=True):
                    t1, eig, src = t1_details([T], {k: np.array([v]) for k, v in r.items()}, STATE)
                    t1s.append(t1[0]), eigs.append(eig[0]), srcs.append(src[0])
                    if src[0] == "eigenvalue":
                        rec.data["fallbacks"].append({"curve": f"{s} {scheme} {STATE} {key} {d} sigma={sigma:g}", "T_K": T})
                t1s, eigs = np.array(t1s), np.array(eigs)
                fit = np.array([x == "fit" for x in srcs])
                if fit.any():
                    rel = np.abs(t1s[fit] - eigs[fit]) / eigs[fit]
                    rec.data["fit_vs_eigen"].append({"curve": f"{s} {scheme} {STATE} {key} {d} (over sigma)",
                                                     "max_rel_diff": float(rel.max()), "at_T_K": T,
                                                     "at_sigma_meV": float(np.array(SIGMAS)[fit][np.argmax(rel)]),
                                                     "n_points": int(fit.sum())})
                res[(s, key, d)] = t1s
    rec.check_fallbacks()

    fig, axes = plt.subplots(2, 3, figsize=(C.DOUBLE, 4.2), sharex=True)
    for row, d in enumerate(DIRECTS):
        for col, s in enumerate(SYSTEMS):
            ax = axes[row, col]
            for key, (_, _, label) in CURVES.items():
                if (s, key, d) not in res:
                    continue
                ax.plot(SIGMAS, res[(s, key, d)], color=C.COLORS[key], ls=C.PAIR_STYLES[key], marker=".", ms=3,
                        label=label)
            ax.axvline(C.SIGMA, color=C.COLORS["ref"], ls="--", lw=0.8, zorder=0)
            ax.set_yscale("log")
            ax.grid(True, ls=":", lw=0.5, alpha=0.6)
            if row == 0:
                ax.set_title(C.SYSTEMS[s]["label"], pad=8)
            C.panel_letter(ax, row * 3 + col, x=0.0, y=1.02)
            if row == 1:
                ax.set_xlabel(r"$\sigma$ (meV)")
            if col == 0:
                ax.set_ylabel(r"$T_1$(300 K) (s)" + "\n" + ("Debye direct" if d == "debye" else "Gaussian direct"))
    axes[0, 0].set_xlim(0, 12.5)
    entries = {}  # every curve once, in CURVES order, from whichever panel has it
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(l, h)
    labels = [c[2] for c in CURVES.values() if c[2] in entries]
    h, l = [entries[x] for x in labels], labels
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04), columnspacing=1.0, handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    rec.data["files"] = C.save(fig, C.SM, name)
    plt.close(fig)

    rec.data["settings"] = {
        "systems": {s: f"{C.SYSTEMS[s]['label']}, step {C.SYSTEMS[s]['step']:g}, {scheme} "
                       f"{C.SYSTEMS[s]['packages'][scheme]}" for s in SYSTEMS},
        "scheme": scheme, "temperature_K": T, "initial state": STATE, "sigmas_meV": SIGMAS,
        "production sigma (vertical line)": C.SIGMA,
        "curves": {k: f"pairs={p}, zero-gap pairs {'excluded' if g else 'kept'}" for k, (p, g, _) in CURVES.items()},
        f"computed pairs ({scheme})": {s: pair_coverage(s, scheme) for s in SYSTEMS},
        "computed-pair curves drawn for": list(HAS_COMPUTED[scheme]),
        "filtered curve": "fill on computed pairs: diagonal + geometric-mean fill, zeroed outside the computed pairs "
                          "(computed pair keys plus pairs with a nonzero computed coupling, as S3); "
                          "diagonal only, diagonal + fill and the production fill are unfiltered",
        "direct": "row 1 Debye (C from the folded acoustic modes, sigma independent); row 2 Gaussian-smeared J(D)",
        "processes": "direct + two-phonon (first-order Raman excluded)",
        "T1 extraction": "population fit, eigenvalue only as fallback",
    }
    i = SIGMAS.index(C.SIGMA)
    rec.data["numbers"] = {"T1(300 K) at production sigma (s)": {f"{s} {k} {d}": float(v[i])
                                                                  for (s, k, d), v in res.items()}}
    rec.write()
    for k, v in rec.data["numbers"]["T1(300 K) at production sigma (s)"].items():
        print(f"{scheme} {k}: {v:.3g}")


def main():
    plt = C.style()
    for scheme in ("AO", "UO"):
        figure(plt, scheme)


if __name__ == "__main__":
    main()
