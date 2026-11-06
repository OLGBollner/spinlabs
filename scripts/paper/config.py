"""Production configuration of the paper figures: every figure script in scripts/paper imports it.

Run the figure scripts from the repo root, e.g. ``python scripts/paper/m2_nv_t1.py``. Outputs go to
figures/out/main/ and figures/out/sm/ (PDF + 300 dpi PNG), one JSON record per figure to figures/out/records/
(settings, numbers, T1 fit fallbacks), from which build_reports.py writes figures/out/manifest.md and
figures/out/numbers.md. Computed rates are cached in figures/out/cache/ (delete it after a code or data change).

V_Si^- (S = 3/2) is left out of every figure here.
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EXPERIMENT_DIR = ROOT / "experimental_data"
OUT = ROOT / "figures" / "out"
MAIN, SM = OUT / "main", OUT / "sm"
RECORDS, CACHE = OUT / "records", OUT / "cache"

# ------------------------------------------------------------------ physics settings
SIGMA = 3.5  # meV, Gaussian broadening of the spectral functions, all systems
DIRECT = "debye"  # one-phonon rate: J(E) = C E^3, C fitted to the folded acoustic modes (spinlabs.debye)
NOGAP = True  # two-phonon: zero-gap pairs left out by symmetry ((l, l) and Ex-Ey partners), no energy tolerance
# Second-order pairs: computed where a package has them, sqrt(V_ii V_jj) fill (E x E: V_T; A x E:
# sqrt(V00(a,a) V_T(e,e))) for the rest. "sqrt" in plot_t1_comparison.coupling_manager.
PAIRS = "sqrt"
PROCESSES = ("first_order", "two_phonon")  # direct + two-phonon; the first-order Raman term (second_order) is out
TEMPERATURES = np.arange(0.0, 601.0, 5.0)  # K
PHONON_LIMITED_K = 125.0  # onset of the phonon-limited regime, dashed line in T1-vs-T figures
FALLBACK_LIMIT_K = 50.0  # eigenvalue-fallback T1 points above this stop a figure before plotting
QUOTE_T = (100.0, 300.0, 600.0)  # K, T1 values quoted in the text
QUOTE_T_RATIO = (100.0, 300.0, 500.0)  # K, T1(fill)/T1(computed) quoted

# Cambria et al., PRL 130, 256903 (2023): phonon-limited part (A3 = B3 = 0). Rates in s^-1, gaps in meV.
# Omega: each 0 <-> +-1 transition; gamma: +1 <-> -1.
CAMBRIA = dict(A1=0.0700e3, B1=0.910e3, Delta1=62.5, A2=0.169e3, B2=2.94e3, Delta2=139.0, A3=0.0, B3=0.0)

# ------------------------------------------------------------------ systems and packages
SCHEMES = {"AO": "all_bands", "UO": "defect_band_approx"}  # all orbitals / unpaired orbitals


def _pkg(system, step, method, date, suffix="12d"):
    return f"{system}/pert_{step:g}/{method}/{date}/{system}_raw_zfs_data_{method}_{suffix}.npz"


SYSTEMS = {
    # NV_64 all bands: the 20261029 repackage (unweighted); defect band approx: 20261021 (diagonal pairs only)
    "NV_64": dict(defect="NV", cell=64, step=0.1, label=r"NV$^-$ (64)", packages={
        "AO": _pkg("NV_64", 0.1, "all_bands", "20261029"),
        "UO": _pkg("NV_64", 0.1, "defect_band_approx", "20261021")}),
    "NV_512": dict(defect="NV", cell=512, step=0.05, label=r"NV$^-$ (512)", packages={
        "AO": _pkg("NV_512", 0.05, "all_bands", "20261030"),
        "UO": _pkg("NV_512", 0.05, "defect_band_approx", "20261030")}),
    "DiV_128": dict(defect="DiV", cell=128, step=0.05, label=r"VV$^0$ (128)", packages={
        "AO": _pkg("DiV_128", 0.05, "all_bands", "20261105"),
        "UO": _pkg("DiV_128", 0.05, "defect_band_approx", "20261105")}),
    # central differences (+-Q first-order runs): diagonal second order (f+ + f- - 2 f0)/Q^2, no off-diagonal runs
    "ClV_576": dict(defect="ClV", cell=576, step=0.05, label=r"ClV$^+$ (576)", packages={
        "AO": _pkg("ClV_576", 0.05, "all_bands", "20261103", "central"),
        "UO": _pkg("ClV_576", 0.05, "defect_band_approx", "20261103", "central")}),
}


def package(system: str, scheme: str) -> Path:
    path = DATA / SYSTEMS[system]["packages"][scheme]
    if not path.exists():
        raise SystemExit(f"missing package: {path}")
    return path


# ------------------------------------------------------------------ names
STATE_LABELS = {"ms_0": r"$m_s=0$", "ms_1": r"$m_s=\pm1$"}  # +1 and -1 relax alike at zero field
CHANNELS = ("V00", "Vpm", "V0pm")
CHANNEL_LABELS = {"V00": r"$V_{00}$", "Vpm": r"$V_{+-}$", "V0pm": r"$V_{0\pm}$"}
PAIR_LABELS = {  # second-order pair treatments of S2, S4, S5
    "computed": "computed pairs",
    "fill-computed": "fill on computed pairs",
    "fill": "geometric-mean fill",
    "diagonal": "diagonal only",
    "diagonal-sqrt": "diagonal + fill",
    "no_axe": r"computed without $A\times E$",
}

# ------------------------------------------------------------------ style: one colour per quantity, everywhere
COLORS = {
    "AO": "#2a78d6", "UO": "#eb6834",  # ZFS schemes
    "cambria": "#2b2b2b",  # analytical model
    "V00": "#0f766e", "Vpm": "#7048e8", "V0pm": "#c9a227",  # coupling channels
    "computed": "#0b0b0b", "fill": "#1baf7a", "diagonal-sqrt": "#8a5a2b", "diagonal": "#898781",
    "no_axe": "#d6336c", "fill-computed": "#4dabf7",  # pair treatments
    "rho_line": "#d6336c",  # rho = 1 reference / rho line in scatter (not in the IPR Blues map)
    "longitudinal": "#c9a227", "transverse": "#0f766e", "other_modes": "#b8b6b0",  # Debye fit
    "ref": "#52514e",  # reference lines (125 K, rho = 1)
}
STATE_STYLES = {"ms_0": "-", "ms_1": "--"}
PAIR_STYLES = {"computed": "-", "fill": "--", "fill-computed": (0, (5, 1.5, 1, 1.5)), "diagonal-sqrt": "-.", "diagonal": ":", "no_axe": ":"}
# Experimental datasets: (colour, marker), the same in every figure
EXPERIMENT_STYLES = {
    ("Jarmola_2012", "S3"): ("#a8201a", "s"),
    ("Jarmola_2012", "S8"): ("#a8201a", "D"),
    ("Lin_2021", "omega"): ("#eda100", "^"),
    ("Toyli_2012", "toyli"): ("#5f3dc4", "v"),
    ("Cambria_2023_NVA", "omega"): ("#2f9e44", "P"),
    ("Cambria_2023_NVB", "omega"): ("#2f9e44", "X"),
}
EXP_EDGE = "#52514e"
CHANNEL_MARKERS = {"V00": "o", "Vpm": "s", "V0pm": "^"}

SINGLE, DOUBLE = 3.4, 7.0  # inch
FONT = 8


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": FONT, "axes.labelsize": FONT, "axes.titlesize": FONT, "legend.fontsize": FONT - 1,
        "xtick.labelsize": FONT - 1, "ytick.labelsize": FONT - 1, "lines.linewidth": 1.3,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })
    return plt


def panel_letter(ax, k, x=-0.02, y=1.02, **kw):
    """(a), (b), ... above the top-left corner of the axes."""
    ax.text(x, y, f"({'abcdefghijklmnop'[k]})", transform=ax.transAxes, ha="right", va="bottom",
            fontweight="bold", **kw)


def phonon_limited_line(ax):
    """Dashed line at PHONON_LIMITED_K with a short label to its right."""
    ax.axvline(PHONON_LIMITED_K, color=COLORS["ref"], ls="--", lw=0.8, zorder=1)
    ax.text(PHONON_LIMITED_K + 8, 0.97, "phonon-limited", transform=ax.get_xaxis_transform(), ha="left", va="top",
            fontsize=FONT - 2, color=COLORS["ref"])


def save(fig, folder: Path, name: str) -> list[str]:
    """PDF (vector) and 300 dpi PNG; returns the paths relative to figures/out."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = folder / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path.relative_to(OUT)))
    print("saved", ", ".join(paths))
    return paths
