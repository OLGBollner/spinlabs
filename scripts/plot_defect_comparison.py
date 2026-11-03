"""Couplings, transition rates and T1 of several defects, for both ZFS methods (all bands, defect band).

Each dataset is a ``--dataset LABEL ALL_BANDS_FILE DEFECT_BAND_FILE [SIGMA]`` (paths relative to --data
unless they exist as given). Each defect is broadened with its own sigma (meV): above the plateau of its
two-phonon T1, below the onset of the one-phonon rate from the broadening tail, which scales with the
cell's lowest mode (7.5 meV for NV_512, 4.5 meV for the 128-atom SiC cells); --sigma is used for a
dataset given without one. Second-order pairs that were not run are filled from the diagonal (the sqrt
fill of --offdiag-from-diagonal, with the A x E mixed root). Without --dataset, DiV_128 and ClV_128 are
compared, with NV_512 as the reference. Writes, next to --output:

  <output>_couplings.png  columns = method; rows = F(1) and sqrt(F(2)) of V_0pm and V_pm, one line per dataset
  <output>_rates.png      columns = method; rows = 0 <-> +-1 and +1 <-> -1; rate against T per process
  <output>_t1.png         columns = method; rows = m_s = 0 and +-1; T1 against T
  <output>.json           the numbers, and the 300 K summary that is also printed

All bands is the left column, the defect band (spin-contamination corrected) the right one.

    python scripts/plot_defect_comparison.py -o figures/defect_comparison/DiV_vs_ClV
"""

import argparse
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
from spinlabs.models import SpinPhononCouplingData  # noqa: E402
from spinlabs.pipeline import compute_rates, solve_t1  # noqa: E402
from spinlabs.utils import MathUtils  # noqa: E402

METHODS = (("all_bands", "all bands"), ("defect_band", "defect band (spin-contamination corrected)"))
STATES = (("ms_0", "$m_s=0$"), ("ms_1", "$m_s=\\pm1$"))
TRANSITIONS = (("0_1", r"$0 \leftrightarrow \pm1$"), ("1_-1", r"$+1 \leftrightarrow -1$"))
PROCESSES = (("first_order", "direct (1 phonon)", ":"), ("second_order", "1st-order Raman", "-."),
             ("two_phonon", "2nd-order Raman", "-"))
COLORS = ["#2a78d6", "#1baf7a", "#eb6834", "#555555", "#8e5bd0", "#c9a227"]


def _pair(defect, cell, step, date_all, date_db):
    base = f"{defect}_{cell}/pert_{step}"
    name = f"{defect}_{cell}_raw_zfs_data"
    return (f"{base}/all_bands/{date_all}/{name}_all_bands_12d.npz",
            f"{base}/defect_band_approx/{date_db}/{name}_defect_band_approx_12d.npz")


DEFAULT_DATASETS = [  # label, all-bands file, defect-band file, sigma (meV)
    ("DiV_128 step 0.05", *_pair("DiV", 128, "0.05", "20261030", "20261030"), 4.5),
    ("ClV_128 step 0.025", *_pair("ClV", 128, "0.025", "20261021", "20261021"), 4.5),
    ("NV_512 step 0.05 (reference)", *_pair("NV", 512, "0.05", "20261030", "20261030"), 7.5),  # least noisy NV set
]


def _label(ds):
    return f"{ds['label']}, $\\sigma$ = {ds['sigma_mev']:g} meV"


def analyse(raw_file, temps, sigma):
    """Couplings, D, rates per process and T1 of one raw package, the sqrt fill on."""
    manager, convention = coupling_manager(raw_file, "sqrt")
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "coupling")
        manager.process_both_orders(output_filename=base)
        coupling = Path(f"{base}.npz")
        data = SpinPhononCouplingData.load(coupling).to_unit("MHz").frequencies_to_unit("meV")
        zfs_mhz = float(np.load(coupling, allow_pickle=True)["zfs"]) / CONSTANTS["MHz2J"]
        valid, rates, directional = compute_rates(coupling, temps, sigma=sigma, has_second_order=True)
        if len(valid) != len(temps):
            raise SystemExit(f"{raw_file}: rates missing at some temperatures")
        path = Path(tmp) / "directional.npz"
        np.savez(path, temperatures=valid, **directional)
        t1 = {state: solve_t1(path, state)[0].tolist() for state, _ in STATES}
    freqs = np.asarray(data.frequencies, dtype=float)
    return {
        "file": str(raw_file),
        "convention": convention,
        "D_MHz": zfs_mhz,
        "lowest_mode_meV": float(np.sort(freqs)[3]),
        "freqs_meV": freqs.tolist(),
        "V1": {k: np.abs(np.asarray(getattr(data, k), dtype=float)).tolist() for k in ("V_0_pm", "V_p_m")},
        "V2_diag": {k: np.abs(np.diag(getattr(data, k))).tolist() for k in ("V2_0_pm", "V2_p_m")},
        "rates_per_s": {p: {tr: [float(r) for r in rates[p][tr]] for tr, _ in TRANSITIONS} for p, _, _ in PROCESSES},
        "t1_s": t1,
    }


def _grid(nitems, rowheight=3.8, colwidth=5.6, share_rows=False):
    """One column per method (all bands left, defect band right), one row per item; indexed [method, item].

    ``share_rows``: both methods' panels of a row share their y axis, so the methods compare at a glance.
    """
    fig, axes = plt.subplots(nitems, len(METHODS), figsize=(colwidth * len(METHODS) + 1, rowheight * nitems),
                             squeeze=False, sharey="row" if share_rows else False)
    return fig, axes.T


def _legend(fig, handles, labels, ncol):
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=ncol, fontsize=9, frameon=False)


def plot_couplings(result, out):
    columns = (("V1", "V_0_pm", r"$F^{(1)}_{0\pm}$ (MHz$^2$/meV)", False),
               ("V1", "V_p_m", r"$F^{(1)}_{+-}$ (MHz$^2$/meV)", False),
               ("V2_diag", "V2_0_pm", r"$\sqrt{F^{(2)}_{0\pm}}$ (MHz/meV$^{1/2}$)", True),
               ("V2_diag", "V2_p_m", r"$\sqrt{F^{(2)}_{+-}}$ (MHz/meV$^{1/2}$)", True))
    fig, axes = _grid(len(columns))
    for row, (method, method_label) in enumerate(METHODS):
        for col, (group, key, ylabel, root) in enumerate(columns):
            ax = axes[row, col]
            for k, ds in enumerate(result["datasets"]):
                d = ds[method]
                x, y = MathUtils.smear_data(np.asarray(d["freqs_meV"]), np.asarray(d[group][key]) ** 2, 0.1,
                                            ds["sigma_mev"])
                ax.plot(x, np.sqrt(np.clip(y, 0, None)) if root else y, color=COLORS[k % len(COLORS)], lw=1.8,
                        label=_label(ds))
            ax.set_xlim(0, 200)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Vibration frequency (meV)")
            ax.set_ylabel(ylabel)
            ax.grid(True, ls=":", alpha=0.5)
            ax.set_title(("first order" if group == "V1" else "second order, diagonal") + f" - {method_label}",
                         fontsize=9)
    _legend(fig, *axes[0, 0].get_legend_handles_labels(), ncol=len(result["datasets"]))
    fig.suptitle("Spin-flip spectral functions, each defect at its own $\\sigma$", y=1.08, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out}")


def plot_rates(result, out):
    temps = np.asarray(result["temperature_K"])
    fig, axes = _grid(len(TRANSITIONS), share_rows=True)
    for row, (method, method_label) in enumerate(METHODS):
        for col, (tr, tr_label) in enumerate(TRANSITIONS):
            ax = axes[row, col]
            for k, ds in enumerate(result["datasets"]):
                for p, _, ls in PROCESSES:
                    y = np.asarray(ds[method]["rates_per_s"][p][tr])
                    ax.plot(temps, np.where(y > 0, y, np.nan), ls=ls, lw=1.7, color=COLORS[k % len(COLORS)])
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel("Rate (1/s)")
            ax.set_title(f"{tr_label} - {method_label}", fontsize=9)
            ax.grid(True, which="major", ls=":", alpha=0.5)
    handles = [plt.Line2D([], [], color=COLORS[k % len(COLORS)], lw=2) for k in range(len(result["datasets"]))]
    handles += [plt.Line2D([], [], color="#333333", ls=ls, lw=1.7) for _, _, ls in PROCESSES]
    labels = [_label(ds) for ds in result["datasets"]] + [name for _, name, _ in PROCESSES]
    _legend(fig, handles, labels, ncol=4)
    fig.suptitle("Transition rates per process, each defect at its own $\\sigma$", y=1.1, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out}")


def plot_t1(result, out):
    temps = np.asarray(result["temperature_K"])
    fig, axes = _grid(len(STATES), share_rows=True)
    for row, (method, method_label) in enumerate(METHODS):
        for col, (state, state_label) in enumerate(STATES):
            ax = axes[row, col]
            for k, ds in enumerate(result["datasets"]):
                y = np.asarray(ds[method]["t1_s"][state], dtype=float)
                ax.plot(temps, np.where(np.isfinite(y) & (y > 0), y, np.nan), lw=2, color=COLORS[k % len(COLORS)],
                        label=_label(ds))
            ax.axvline(300, color="#888888", lw=1, ls=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel(r"$T_1$ (s)")
            ax.set_title(f"{state_label} - {method_label}", fontsize=9)
            ax.grid(True, which="major", ls=":", alpha=0.5)
    _legend(fig, *axes[0, 0].get_legend_handles_labels(), ncol=len(result["datasets"]))
    fig.suptitle("$T_1$, each defect at its own $\\sigma$, diagonal runs + sqrt fill", y=1.08, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out}")


def summary(result, T=300.0):
    """D, lowest mode, T1 and the process shares of the 0 <-> +-1 rate at T, per dataset and method."""
    i = result["temperature_K"].index(T)
    lines = [f"At {T:g} K (rate shares of 0 <-> +-1: direct / 1st Raman / 2nd Raman)"]
    for ds in result["datasets"]:
        for method, _ in METHODS:
            d = ds[method]
            r = [d["rates_per_s"][p]["0_1"][i] for p, _, _ in PROCESSES]
            shares = " / ".join(f"{100 * x / sum(r):.1f}%" for x in r)
            lines.append(f"  {ds['label']:31} sigma {ds['sigma_mev']:3g} {method:12} D {d['D_MHz']:8.1f} MHz  lowest mode {d['lowest_mode_meV']:5.1f} meV"
                         f"  T1 {d['t1_s']['ms_0'][i] * 1e3:9.3g} / {d['t1_s']['ms_1'][i] * 1e3:9.3g} ms  shares {shares}")
    return "\n".join(lines)


def _path(p, data):
    return Path(p) if Path(p).exists() else Path(data) / p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Couplings, rates and T1 of several defects, both ZFS methods")
    parser.add_argument("--data", default="data", help="Folder the dataset files are relative to (default: data)")
    parser.add_argument("--dataset", nargs="+", action="append", metavar="ARG",
                        help="LABEL ALL_BANDS DEFECT_BAND [SIGMA]: a dataset, its two raw packages and its sigma "
                        "in meV (default: DiV_128 and ClV_128 at 4.5, NV_512 at 7.5)")
    parser.add_argument("--sigma", type=float, default=7.5,
                        help="Broadening in meV for a --dataset given without one (default: 7.5)")
    parser.add_argument("--t-range", nargs=3, type=float, default=[10.0, 500.0, 10.0],
                        metavar=("START", "END", "STEP"), help="Temperatures in K (default: 10 500 10)")
    parser.add_argument("-o", "--output", type=Path, default=Path("defect_comparison"),
                        help="Output stem: <stem>_couplings.png, _rates.png, _t1.png and .json")
    parser.add_argument("--replot", action="store_true", help="Redraw from <output>.json without recomputing")
    args = parser.parse_args()
    logging.disable(logging.WARNING)  # the per-run pipeline warnings would bury the summary

    json_path = args.output.with_suffix(".json")
    if args.replot:
        result = json.loads(json_path.read_text())
    else:
        start, end, step = args.t_range
        temps = np.arange(start, end + step / 2, step).tolist()
        result = {"temperature_K": temps, "datasets": []}
        for given in args.dataset or DEFAULT_DATASETS:
            if len(given) not in (3, 4):
                raise SystemExit(f"--dataset wants LABEL ALL_BANDS DEFECT_BAND [SIGMA]: {given}")
            label, all_bands, defect_band = given[:3]
            sigma = float(given[3]) if len(given) == 4 else args.sigma
            entry = {"label": label, "sigma_mev": sigma}
            for (method, _), raw in zip(METHODS, (all_bands, defect_band)):
                entry[method] = analyse(_path(raw, args.data), temps, sigma)
                print(f"{label}, {method}: {entry[method]['convention']}", flush=True)
            result["datasets"].append(entry)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result))
    stem = str(args.output)
    plot_couplings(result, Path(f"{stem}_couplings.png"))
    plot_rates(result, Path(f"{stem}_rates.png"))
    plot_t1(result, Path(f"{stem}_t1.png"))
    print(summary(result))
