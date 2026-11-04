"""T1 of NV against temperature for the cases in SETTINGS below, with the digitized experimental data.

Edit the settings block and run from the repo root:

    python scripts/plot_t1_vs_experiment.py

Each case is one raw ZFS package (data/NV_<cell>/pert_<step>/<method>/<date>/) and how its rates are
computed. The experiment (experimental_data/*.csv, read by spinlabs.experimental) gives T1 or 1/T1
directly (Jarmola et al. 2012, an m_s = 0 polarization: decay rate Gamma(+-1 -> 0) + 2 Gamma(0 -> +-1),
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
import numpy as np  # noqa: E402
from plot_t1_comparison import coupling_manager  # noqa: E402

from spinlabs.constants import CONSTANTS  # noqa: E402
from spinlabs.experimental import load_experimental_data, relaxation_series  # noqa: E402
from spinlabs.pipeline import solve_t1  # noqa: E402
from spinlabs.transition_rate import TransitionRate  # noqa: E402

# ============================== settings: edit these ==============================
DATA = Path("data")  # raw ZFS packages, as data/NV_<cell>/pert_<step>/<method>/<date>/
EXPERIMENT_DIR = Path("experimental_data")
EXPERIMENT_SOURCES = None  # e.g. ["Jarmola_2012"]; None: every CSV in EXPERIMENT_DIR
EXPERIMENT_SERIES = None  # e.g. ["S3", "S8", "T1 ms_0 from omega and gamma"]; None: every series of those sources

# One curve per case:
#   cell, step, method   the package: NV_<cell>, step <step>, "all_bands" or "defect_band_approx"
#   date                 package folder (YYYYMMDD); None: the newest one with a 12d package
#   pairs                second-order pairs: "computed" (as computed), "diagonal" (only (i, i)),
#                        "diagonal-sqrt" (diagonal + sqrt(V_ii V_jj) fill: "fill"),
#                        "sqrt" (computed pairs + fill for the uncomputed ones)
#   nogap                leave out the zero-gap pairs (diagonal, degenerate E partners), see the sigma scans
#   sigma                broadening in meV
#   displacement         None: the package's own; or "normalized", "mass_weighted", "unweighted"
#   label                legend text; None: built from the settings
CASES = [
    dict(cell=64, step=0.1, method="all_bands", date="20261029", pairs="computed", nogap=True, sigma=3.5,
         displacement=None, label=None),
    dict(cell=512, step=0.05, method="all_bands", date="20261030", pairs="diagonal-sqrt", nogap=True, sigma=0.5,
         displacement=None, label=None),
]

TEMPERATURES = np.arange(10.0, 501.0, 10.0)  # K
INIT_STATE = "ms_0"  # "ms_0", "ms_1" or "ms_-1"; the experiment is an m_s = 0 polarization
INCLUDE_DIRECT = False  # one-phonon rate: a smearing-tail artifact without acoustic phonons, off by default
INCLUDE_VIRTUAL_RAMAN = True  # first-order coupling at second order; ~1e-7 of the two-phonon rate
PLOT = "T1"  # "T1" (seconds) or "rate" (1/T1 in Hz, as the experiment is reported)
LOG_X = False
OUTPUT = Path("figures/t1/NV_t1_vs_experiment.png")
# ==================================================================================

ZERO_GAP_TOL_MEV = 1e-3  # symmetry-degenerate partners; accidental near-degeneracies start near 0.01 meV
STATE_TITLES = {"ms_0": "$m_s=0$", "ms_1": "$m_s=+1$", "ms_-1": "$m_s=-1$"}
DIRECTIONS = ("0_to_1", "0_to_-1", "1_to_0", "-1_to_0", "1_to_-1", "-1_to_1")
COLORS = ["#2a78d6", "#1baf7a", "#eb6834", "#8e5bd0", "#c9a227", "#555555"]
EXP_MARKERS = "osD^vP"


def package(case):
    """The raw second-order package of a case."""
    folder = DATA / f"NV_{case['cell']}" / f"pert_{case['step']:g}" / case["method"]
    name = f"NV_{case['cell']}_raw_zfs_data_{case['method']}_12d.npz"
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


def case_label(case, convention):
    if case.get("label"):
        return case["label"]
    pairs = {"computed": "all pairs", "diagonal": "diagonal only", "diagonal-sqrt": "diagonal + fill",
             "sqrt": "computed + fill"}[case["pairs"]]
    method = "all bands" if case["method"] == "all_bands" else "defect band approx"
    gap = ", no zero-gap pairs" if case["nogap"] else ""
    return f"NV {case['cell']} step {case['step']:g}, {method}, {pairs}{gap}, $\\sigma$ = {case['sigma']:g} meV"


def t1_curve(case):
    """T1 (s) at TEMPERATURES for INIT_STATE, from the processes the settings include."""
    manager, convention = coupling_manager(package(case), case["pairs"], case.get("displacement"))
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "coupling")
        manager.process_both_orders(output_filename=base)
        calc = TransitionRate(f"{base}.npz")
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
    fig, ax = plt.subplots(figsize=(8.5, 6))
    result = {"temperature_K": TEMPERATURES.tolist(), "init_state": INIT_STATE, "include_direct": INCLUDE_DIRECT,
              "include_virtual_raman": INCLUDE_VIRTUAL_RAMAN, "cases": []}

    for n, case in enumerate(CASES):
        t1, convention = t1_curve(case)
        label = case_label(case, convention)
        y = 1.0 / t1 if PLOT == "rate" else t1
        ok = np.isfinite(y) & (y > 0)
        ax.plot(TEMPERATURES[ok], y[ok], color=COLORS[n % len(COLORS)], lw=2.2, label=label)
        result["cases"].append({**case, "package": str(package(case)), "convention": convention, "label": label,
                                "t1_s": [float(v) for v in t1]})
        i300 = int(np.argmin(np.abs(TEMPERATURES - 300)))
        print(f"{label} ({convention}): T1({TEMPERATURES[i300]:g} K) = {t1[i300] * 1e3:.3g} ms")

    experiments = load_experimental_data(EXPERIMENT_DIR)
    m = 0
    for source, series_list in experiments.items():
        if EXPERIMENT_SOURCES is not None and source not in EXPERIMENT_SOURCES:
            continue
        # T1 / 1/T1 as measured, and T1 of INIT_STATE solved from a source's Omega and gamma
        for s in relaxation_series(series_list, INIT_STATE):
            if EXPERIMENT_SERIES is not None and s.label not in EXPERIMENT_SERIES:
                continue
            # T1 in s or 1/T1 in Hz, whichever the source reports (a time or a rate): to_units inverts as needed
            exp = s.to_units(x_unit="K", y_unit="Hz" if PLOT == "rate" else "s")
            ax.plot(exp.x, exp.y, ls="none", marker=EXP_MARKERS[m % len(EXP_MARKERS)], ms=6, mfc="white", mew=1.4,
                    color="black", label=f"{source.replace('_', ' ')}, {s.label}")
            m += 1

    ax.set_yscale("log")
    if LOG_X:
        ax.set_xscale("log")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$1/T_1$ (s$^{-1}$)" if PLOT == "rate" else r"$T_1$ (s)")
    included = "two-phonon" + (" + direct" if INCLUDE_DIRECT else "") + (" + virtual Raman" if INCLUDE_VIRTUAL_RAMAN else "")
    ax.set_title(f"NV in diamond, {STATE_TITLES.get(INIT_STATE, INIT_STATE)} ({included})", fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    OUTPUT.with_suffix(".json").write_text(json.dumps(result, indent=1))
    print(f"saved figure: {OUTPUT}")


if __name__ == "__main__":
    main()
