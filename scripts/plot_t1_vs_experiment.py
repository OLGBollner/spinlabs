"""T1 of point defects against temperature for the cases in SETTINGS below, with the digitized experimental data.

Edit the settings block and run from the repo root:

    python scripts/plot_t1_vs_experiment.py

Each case is one raw ZFS package (data/<defect>_<cell>/pert_<step>/<method>/<date>/) and how its rates are
computed. The experiment (experimental_data/<defect>/*.csv for every defect in the cases, read by spinlabs.experimental) gives T1 or 1/T1
directly (for NV: Jarmola et al. 2012, an m_s = 0 polarization: decay rate Gamma(+-1 -> 0) + 2 Gamma(0 -> +-1),
about 3 Omega, whatever gamma is; so compare it with INIT_STATE = "ms_0"), or Omega and gamma
(Lin et al. 2021), from which T1 of INIT_STATE is solved as for the computed rates.
The computed numbers go to <OUTPUT>.json next to the figure.
"""

import json
import logging
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
import numpy as np  # noqa: E402
from plot_t1_comparison import coupling_manager  # noqa: E402

from spinlabs.constants import CONSTANTS  # noqa: E402
from spinlabs.experimental import load_experimental_data, relaxation_series  # noqa: E402
from spinlabs.pipeline import solve_t1  # noqa: E402
from spinlabs.transition_rate import TransitionRate  # noqa: E402

# ============================== settings: edit these ==============================
DATA = Path("data")  # raw ZFS packages, as data/<defect>_<cell>/pert_<step>/<method>/<date>/
EXPERIMENT_DIR = Path("experimental_data")  # one subfolder per defect: <EXPERIMENT_DIR>/<defect>/*.csv
EXPERIMENT_SOURCES = None  # e.g. ["Jarmola_2012"]; None: every CSV in the subfolders of the cases' defects
EXPERIMENT_SERIES = None  # e.g. ["S3", "S8", "T1 ms_0 from omega and gamma"]; None: every series of those sources

# One curve per case:
#   defect               "NV", "DiV", ...: the package and the experimental data subfolder
#   cell, step, method   the package: <defect>_<cell>, step <step>, "all_bands" or "defect_band_approx"
#   date                 package folder (YYYYMMDD); None: the newest one with a 12d package
#   pairs                second-order pairs: "computed" (as computed), "diagonal" (only (i, i)),
#                        "diagonal-sqrt" (diagonal + sqrt(V_ii V_jj) fill: "fill"),
#                        "sqrt" (computed pairs + fill for the uncomputed ones)
#   nogap                leave out the zero-gap pairs (diagonal, degenerate E partners), see the sigma scans
#   sigma                broadening in meV
#   displacement         None: the package's own; or "normalized", "mass_weighted", "unweighted"
#   label                legend text; None: the method, "AE" (all_bands) or "UE" (defect_band_approx), with the
#                        cell size and the pairs ("explicit", "geometric mean") when the cases differ in them

CASES = [
    dict(defect="NV", cell=64, step=0.1, method="all_bands", pairs="diagonal-sqrt", nogap=True, sigma=3.5,
         displacement=None, label=None),
    dict(defect="NV", cell=64, step=0.1, method="all_bands", pairs="computed", nogap=True, sigma=3.5,
         displacement=None, label=None),
    dict(defect="NV", cell=512, step=0.05, method="all_bands", pairs="diagonal-sqrt", nogap=True, sigma=0.5,
         displacement=None, label=None),
]
# CASES = [
#     dict(defect="DiV", cell=128, step=0.05, method="all_bands", pairs="diagonal-sqrt", nogap=True, sigma=3.5,
#          displacement=None, label=None),
#     dict(defect="DiV", cell=128, step=0.05, method="all_bands", pairs="computed", nogap=True, sigma=3.5,
#          displacement=None, label=None),
# ]

TEMPERATURES = np.arange(0.0, 601.0, 10.0)  # K
INIT_STATE = "ms_0"  # "ms_0", "ms_1" or "ms_-1"; the experiment is an m_s = 0 polarization
INCLUDE_DIRECT = True  # one-phonon rate: a smearing-tail artifact without acoustic phonons, off by default
# How the one-phonon rate is computed when INCLUDE_DIRECT: "debye" (J = C E^3 from the supercell's folded
# acoustic modes, spinlabs.debye; an order-of-magnitude estimate) or "gaussian" (the smeared tail at D)
DIRECT_MODEL = "debye"
INCLUDE_VIRTUAL_RAMAN = True  # first-order coupling at second order; ~1e-7 of the two-phonon rate
PLOT = "T1"  # "T1" (seconds) or "rate" (1/T1 in Hz, as the experiment is reported)
LOG_X = False
FONT_SIZE = 20  # axis labels and ticks; the legend is two points smaller
OUTPUT = Path("figures/t1/NV_t1_vs_experiment_600K.png")
# ==================================================================================

ZERO_GAP_TOL_MEV = 1e-3  # symmetry-degenerate partners; accidental near-degeneracies start near 0.01 meV
STATE_TITLES = {"ms_0": "$m_s=0$", "ms_1": "$m_s=+1$", "ms_-1": "$m_s=-1$"}
DIRECTIONS = ("0_to_1", "0_to_-1", "1_to_0", "-1_to_0", "1_to_-1", "-1_to_1")
COLORS = ["#2a78d6", "#1baf7a", "#eb6834", "#8e5bd0", "#c9a227", "#555555"]
# Experimental data: one colour per source (paper), one marker per series within it. Dark red, yellow and
# near-black, validated all-pairs together with the first two curves (blue, aqua): worst pair Delta E 22.9
# normal, 9.1 colour-blind. Dark red sits near the third curve colour (orange), so with three or more cases
# tell them apart by form (line vs marker). A source keeps its colour when others are filtered out (its place
# among all CSVs); sources past the third are drawn in grey, told apart by marker and legend.
EXP_COLORS = ["#a8201a", "#eda100", "#2b2b2b"]
EXP_OVERFLOW_COLOR = "#898781"
EXP_EDGE = "#52514e"  # dark outline: yellow is below 3:1 against the white background
EXP_MARKERS = "sD^vPX"  # no plain circles


def package(case):
    """The raw second-order package of a case."""
    folder = DATA / f"{case['defect']}_{case['cell']}" / f"pert_{case['step']:g}" / case["method"]
    name = f"{case['defect']}_{case['cell']}_raw_zfs_data_{case['method']}_12d.npz"
    if case.get("date"):
        path = folder / case["date"] / name
    else:
        found = sorted(folder.glob(f"*/{name}"))
        if not found:
            raise SystemExit(f"no {name} under {folder}")
        path = found[-1]
    if not path.exists():
        raise SystemExit(f"no such package: {path}")
    return path


# How the legend names the second-order pairs, shown only when the cases differ in them
PAIR_NAMES = {"computed": "explicit", "diagonal-sqrt": "geometric mean", "diagonal": "diagonal only",
              "sqrt": "explicit + geometric mean"}


def legend_label(case):
    """The method as the paper names it, AE (all_bands) or UE (defect_band_approx), with the supercell size and
    the second-order pairs (explicit or geometric mean) when they differ between the cases."""
    if case.get("label"):
        return case["label"]
    label = {"all_bands": "AE", "defect_band_approx": "UE"}[case["method"]]
    extras = []
    if len({c["cell"] for c in CASES}) > 1:
        extras.append(f"{case['cell']} atoms")
    if len({c["pairs"] for c in CASES}) > 1:
        extras.append(PAIR_NAMES[case["pairs"]])
    return label + (f" ({', '.join(extras)})" if extras else "")


def case_description(case):
    """Every setting of a case, for the printout and the JSON."""
    pairs = {"computed": "all pairs", "diagonal": "diagonal only", "diagonal-sqrt": "diagonal + fill",
             "sqrt": "computed + fill"}[case["pairs"]]
    method = "all bands" if case["method"] == "all_bands" else "defect band approx"
    gap = ", no zero-gap pairs" if case["nogap"] else ""
    return f"{case['defect']} {case['cell']} step {case['step']:g}, {method}, {pairs}{gap}, $\\sigma$ = {case['sigma']:g} meV"


def t1_curve(case):
    """T1 (s) at TEMPERATURES for INIT_STATE, from the processes the settings include."""
    manager, convention = coupling_manager(package(case), case["pairs"], case.get("displacement"))
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "coupling")
        manager.process_both_orders(output_filename=base)
        calc = TransitionRate(f"{base}.npz")
        if INCLUDE_DIRECT and DIRECT_MODEL == "debye":
            fit = calc.set_debye_from_spectrum(manager.spectrum)
            branches = ", ".join(f"{len(b['modes'])} {name} at {b['energy_mev']:.0f} meV" for name, b in fit.branches.items())
            print(f"  Debye fit: C = {fit.C:.3e} 1/meV^2 from {branches}")
        else:
            calc.direct = DIRECT_MODEL if INCLUDE_DIRECT else "none"
        omega, j0pm, jpm, j00 = calc.get_spectral_density(res=0.02, sigma=case["sigma"])
        lines = calc.get_line_spectral_density(case["sigma"], zero_gap_tol=ZERO_GAP_TOL_MEV if case["nogap"] else None)
        zfs = calc.data["zfs"] / CONSTANTS["meV2J"]
        processes = ["two_phonon"] + (["first_order"] if INCLUDE_DIRECT else []) + (
            ["second_order"] if INCLUDE_VIRTUAL_RAMAN else [])
        directional = {d: [] for d in DIRECTIONS}
        for T in TEMPERATURES:
            calc.compute_transition_rates(T, omega, j0pm, jpm, j00, zfs)  # first and second order
            calc.compute_two_phonon_rates(T, lines)
            for d in DIRECTIONS:
                directional[d].append(sum(calc.transition_rate[p].get(d, 0.0) for p in processes))
        rates_file = Path(tmp) / "directional.npz"
        np.savez(rates_file, temperatures=TEMPERATURES, **directional)
        t1, _ = solve_t1(rates_file, INIT_STATE)
    return np.asarray(t1, dtype=float), convention


def main():
    logging.disable(logging.WARNING)  # the per-run pipeline warnings would bury the summary
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 2})
    fig, ax = plt.subplots(figsize=(8.5, 6))
    result = {"temperature_K": TEMPERATURES.tolist(), "init_state": INIT_STATE, "include_direct": INCLUDE_DIRECT,
              "direct_model": DIRECT_MODEL, "include_virtual_raman": INCLUDE_VIRTUAL_RAMAN, "cases": []}

    for n, case in enumerate(CASES):
        t1, convention = t1_curve(case)
        label, description = legend_label(case), case_description(case)
        y = 1.0 / t1 if PLOT == "rate" else t1
        ok = np.isfinite(y) & (y > 0)
        ax.plot(TEMPERATURES[ok], y[ok], color=COLORS[n % len(COLORS)], lw=2.2, label=label)
        result["cases"].append({**case, "package": str(package(case)), "convention": convention, "label": label,
                                "description": description,
                                "t1_s": [float(v) for v in t1]})
        i300 = int(np.argmin(np.abs(TEMPERATURES - 300)))
        print(f"{description} ({convention}): T1({TEMPERATURES[i300]:g} K) = {t1[i300] * 1e3:.3g} ms")

    defects = list(dict.fromkeys(case["defect"] for case in CASES))
    experiments = [(defect, source, series_list) for defect in defects  # each folder sorted by source name
                   for source, series_list in load_experimental_data(EXPERIMENT_DIR / defect).items()]
    entries = []  # (colour, marker, defect, series) per plotted experimental series
    for k, (defect, source, series_list) in enumerate(experiments):
        if EXPERIMENT_SOURCES is not None and source not in EXPERIMENT_SOURCES:
            continue
        color = EXP_COLORS[k] if k < len(EXP_COLORS) else EXP_OVERFLOW_COLOR
        # T1 / 1/T1 as measured, and T1 of INIT_STATE solved from a source's Omega and gamma
        for m, s in enumerate(relaxation_series(series_list, INIT_STATE)):
            if EXPERIMENT_SERIES is not None and s.label not in EXPERIMENT_SERIES:
                continue
            entries.append((color, EXP_MARKERS[m % len(EXP_MARKERS)], defect, s))
    # Legend: one entry per citation, showing the markers of all its series side by side
    cited = {}  # citation -> the marker handles of its series, in plotting order
    for color, marker, defect, s in entries:
        text = (f"{defect}: " if len(defects) > 1 else "") + (s.citation or s.source.replace("_", " "))
        # T1 in s or 1/T1 in Hz, whichever the source reports (a time or a rate): to_units inverts as needed
        exp = s.to_units(x_unit="K", y_unit="Hz" if PLOT == "rate" else "s")
        (handle,) = ax.plot(exp.x, exp.y, ls="none", marker=marker, ms=7, mfc=color, mec=EXP_EDGE, mew=0.8, zorder=3)
        cited.setdefault(text, []).append(handle)

    ax.set_yscale("log")
    if LOG_X:
        ax.set_xscale("log")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$1/T_1$ (s$^{-1}$)" if PLOT == "rate" else r"$T_1$ (s)")
    # included = "two-phonon" + (f" + direct ({DIRECT_MODEL})" if INCLUDE_DIRECT else "") + (" + virtual Raman" if INCLUDE_VIRTUAL_RAMAN else "")
    # ax.set_title(f"{', '.join(defects)} in diamond, {STATE_TITLES.get(INIT_STATE, INIT_STATE)} ({included})", fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()  # the computed curves
    handles += [tuple(h) for h in cited.values()]
    labels += list(cited)
    ax.legend(handles, labels, loc="upper right", handler_map={tuple: HandlerTuple(ndivide=None)})
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    OUTPUT.with_suffix(".json").write_text(json.dumps(result, indent=1))
    print(f"saved figure: {OUTPUT}")


if __name__ == "__main__":
    main()
