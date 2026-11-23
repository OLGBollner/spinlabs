"""Couplings, rates and T1 of the full spin decontamination and the unpaired orbitals, each against all orbitals.

All orbitals (AO) is the spin-contaminated reference. The unpaired orbitals (UO, defect band with the
spin-contamination correction) and the full decontamination (decont, Biktagirov: one package per spin
sublevel, combined on load) are each compared with AO, not with each other. Every method uses the same
second-order treatment: the diagonal (i, i) terms only, the off-diagonal pairs filled as their geometric
mean (``diagonal-sqrt``), because the decont set has no pair runs. AO with all of its computed pairs
(production) is drawn as a reference where it matters (T1).

Production rate settings (as `spinlabs run` and the paper figures): Gaussian broadening ``--sigma`` (meV),
two-phonon rates with the zero-gap pairs left out by symmetry. The direct (one-phonon) rate is left out:
it changes T1 by < 0.3 % above 80 K and only sets T1 below ~50 K, hence the 50 K start of the grid. Channels as in the paper figures: V_Delta0 (A1, no spin flip), V_Delta1 (Delta m_s = 1),
V_Delta2 (Delta m_s = 2), first and second order (the diagonal (l, l) terms); rates Omega (0 <-> +-1) and
gamma (+1 <-> -1). rho = sqrt(sum V^2 / sum V_AO^2) per channel, with the V_AO^2-weighted SD of the
per-mode ratios (as the paper figure M3). Writes, to --outdir:

  couplings_first_order.png   smeared J of each channel per method; below, sqrt(J / J_AO)
  couplings_second_order.png  the same for the diagonal second-order terms
  couplings_per_mode.png      |V| of UO and decont against AO, mode by mode, every channel
  rho.png                     rho of UO and decont per channel, first and second order
  rates.png                   two-phonon Omega and gamma against T; below, the rate over AO's
  t1.png                      T1 of m_s = 0 and +-1 against T with experiment; below, T1 over AO's
  summary.json                D0, rms couplings, rho, rates and T1 at 100/300/500 K, and the ratios to AO

    python scripts/plot_decont_comparison.py                 # NV_64 defaults
    python scripts/plot_decont_comparison.py --decont NV_64/pert_0.2/full_decont/20261119 -o figures/x
"""

from __future__ import annotations

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
from spinlabs.experimental import read_experimental_csv  # noqa: E402
from spinlabs.models import SpinPhononCouplingData  # noqa: E402
from spinlabs.pipeline import compute_rates, solve_t1  # noqa: E402
from spinlabs.utils import MathUtils  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {  # relative to --data
    "AO": "NV_64/pert_0.1/all_bands/20261029/NV_64_raw_zfs_data_all_bands_12d.npz",
    "UO": "NV_64/pert_0.1/defect_band_approx/20261021/NV_64_raw_zfs_data_defect_band_approx_12d.npz",
    "decont": "NV_64/pert_0.2/full_decont/20261119",
}
METHODS = {  # key -> legend label, color (AO is the neutral reference)
    "AO": ("AO (spin-contaminated)", "#555555"),
    "UO": ("UO (spin-contamination corrected)", "#2a78d6"),
    "decont": ("full decontamination", "#eb6834"),
}
COMPARED = ("UO", "decont")  # each against AO
AO_PRODUCTION = ("AO, all computed pairs (production)", "#555555", "--")
DIRECT, PROCESSES = "debye", ("two_phonon",)  # two-phonon only (compute_rates still needs a direct model)
STATES = (("ms_0", "$m_s=0$"), ("ms_1", "$m_s=\\pm1$"))
TRANSITIONS = (("0_1", r"$\Omega$ ($0\leftrightarrow\pm1$)"), ("1_-1", r"$\gamma$ ($+1\leftrightarrow-1$)"))
RATE_PROCESSES = (("two_phonon", "two-phonon", "-"),)
CHANNELS = (  # name, key in the coupling data, order, coupling symbol, spectral-density symbol (as the paper figures)
    ("V_Delta0", "V_0_0", 1, r"V_{\Delta 0}", r"J_{\Delta 0}"),
    ("V_Delta1", "V_0_pm", 1, r"V_{\Delta 1}", r"J_{\Delta 1}"),
    ("V_Delta2", "V_p_m", 1, r"V_{\Delta 2}", r"J_{\Delta 2}"),
    ("V2_Delta0", "V2_0_0", 2, r"V^{(2)}_{\Delta 0}", r"J^{(2)}_{\Delta 0}"),
    ("V2_Delta1", "V2_0_pm", 2, r"V^{(2)}_{\Delta 1}", r"J^{(2)}_{\Delta 1}"),
    ("V2_Delta2", "V2_p_m", 2, r"V^{(2)}_{\Delta 2}", r"J^{(2)}_{\Delta 2}"),
)
JARMOLA_SERIES = ("S3", "S8")  # as the paper figure M2
QUOTE_T = (100.0, 300.0, 500.0)


def analyse(manager, temps, sigma):
    """D0, per-mode couplings, rates per process and T1 per state of one ZFSManager."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "coupling"
        manager.process_both_orders(output_filename=str(base))
        coupling = Path(f"{base}.npz")
        data = SpinPhononCouplingData.load(coupling).to_unit("MHz").frequencies_to_unit("meV")
        zfs_mhz = float(np.load(coupling, allow_pickle=True)["zfs"]) / CONSTANTS["MHz2J"]
        valid, rates, directional = compute_rates(
            coupling, temps, sigma=sigma, has_second_order=True, degenerate_groups=manager.spectrum.deg_groups or [],
            direct=DIRECT, spectrum=manager.spectrum, processes=PROCESSES)
        if len(valid) != len(temps):
            raise SystemExit(f"rates missing at some temperatures: got {list(valid)}")
        path = Path(tmp) / "directional.npz"
        np.savez(path, temperatures=valid, **directional)
        t1 = {state: solve_t1(path, state)[0].tolist() for state, _ in STATES}
    couplings = {}
    for name, key, order, _, _ in CHANNELS:
        values = np.abs(np.asarray(getattr(data, key)))
        couplings[name] = (values if order == 1 else np.diag(values)).astype(float).tolist()
    return {
        "D_MHz": zfs_mhz,
        "freqs_meV": np.asarray(data.frequencies, dtype=float).tolist(),
        "couplings_MHz": couplings,
        "rates_per_s": {p: {tr: [float(r) for r in rates[p][tr]] for tr, _ in TRANSITIONS} for p, _, _ in RATE_PROCESSES},
        "t1_s": t1,
    }


def spectral(res, name, sigma):
    """Smeared J(E) = sum_i |V_i|^2 delta(E - E_i) (MHz^2/meV) on the energy grid (second order: the diagonal)."""
    return MathUtils.smear_data(np.asarray(res["freqs_meV"]), np.asarray(res["couplings_MHz"][name]) ** 2, 0.1, sigma)


def rho_sd(ref, values):
    """rho = sqrt(sum V^2 / sum V_AO^2) and the V_AO^2-weighted SD of the per-mode |V|/|V_AO| about it (as M3)."""
    w = ref**2
    rho = float(np.sqrt(np.sum(values**2) / w.sum()))
    m = ref > 0
    sd = float(np.sqrt(np.sum(w[m] * (values[m] / ref[m] - rho) ** 2) / w[m].sum()))
    return rho, sd, int(m.sum())


def rho_summary(res, ref):
    """{channel: {rho, sd, n}} of one method against AO; second order over the diagonal (l, l) terms."""
    out = {}
    for name, _, _, _, _ in CHANNELS:
        rho, sd, n = rho_sd(np.asarray(ref["couplings_MHz"][name]), np.asarray(res["couplings_MHz"][name]))
        out[name] = {"rho": rho, "sd": sd, "n": n}
    return out


def at(temps, values, targets=QUOTE_T):
    temps = np.asarray(temps)
    return {f"{T:g}": float(np.asarray(values)[np.flatnonzero(np.isclose(temps, T))[0]])
            for T in targets if np.isclose(temps, T).any()}


def _ratio_axis(ax, ylabel):
    ax.axhline(1.0, color="0.6", lw=1, zorder=0)
    ax.set_ylabel(ylabel)
    ax.grid(True, ls=":", alpha=0.5)


def _legend(fig, axes):
    handles = {}
    for ax in np.ravel(axes):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(list(handles.values()), list(handles), loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=min(len(handles), 4), fontsize=9, frameon=False)


def _save(fig, out, title):
    fig.suptitle(title, y=1.06, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out}")


def plot_couplings(result, out, order):
    """Smeared J of the three channels of one order; below, sqrt(J / J_AO) (the rho of each energy)."""
    sigma = result["sigma_meV"]
    channels = [c for c in CHANNELS if c[2] == order]
    fig, axes = plt.subplots(2, len(channels), figsize=(4.6 * len(channels), 7.2), sharex=True,
                             gridspec_kw={"height_ratios": (3, 2)})
    for col, (name, _, _, vsym, jsym) in enumerate(channels):
        top, bottom = axes[0, col], axes[1, col]
        x, ref = spectral(result["methods"]["AO"], name, sigma)
        for m, (mlabel, color) in METHODS.items():
            _, y = spectral(result["methods"][m], name, sigma)
            top.plot(x, y, color=color, lw=1.8, label=mlabel)
            if m in COMPARED:
                keep = ref > 1e-2 * ref.max()  # no ratio where AO has (almost) no weight
                bottom.plot(x[keep], np.sqrt(y[keep] / ref[keep]), color=color, lw=1.8)
        top.set_ylim(bottom=0)
        top.set_ylabel(f"${jsym}$ (MHz$^2$/meV)")
        top.set_title(f"${vsym}$", fontsize=10)
        top.grid(True, ls=":", alpha=0.5)
        _ratio_axis(bottom, f"$\\sqrt{{{jsym} \\,/\\, {{{jsym}}}^{{\\mathrm{{AO}}}}}}$")
        bottom.set_ylim(0, min(3.0, bottom.get_ylim()[1]))
        bottom.set_xlabel("Vibration frequency (meV)")
        bottom.set_xlim(0, 175)
    _legend(fig, axes)
    what = "first order" if order == 1 else "second order, diagonal $(l, l)$ terms"
    _save(fig, out, f"{result['system']}: spin-phonon spectral densities, {what}, $\\sigma$ = {sigma:g} meV")


def plot_couplings_per_mode(result, out):
    """|V| of UO and decont against AO, mode by mode; first order on top, second order (diagonal) below."""
    fig, axes = plt.subplots(2, 3, figsize=(4.4 * 3, 8.6))
    ao = result["methods"]["AO"]["couplings_MHz"]
    for ax, (name, _, _, vsym, _) in zip(axes.ravel(), CHANNELS):
        x = np.asarray(ao[name])
        hi = lo = None
        for m in COMPARED:
            y = np.asarray(result["methods"][m]["couplings_MHz"][name])
            keep = (x > 0) & (y > 0)
            ax.scatter(x[keep], y[keep], s=16, color=METHODS[m][1], alpha=0.75, edgecolor="white", lw=0.4,
                       label=METHODS[m][0])
            both = np.concatenate([x[keep], y[keep]])
            hi = max(hi or 0, both.max())
            lo = min(lo or np.inf, np.percentile(both, 2))
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=1, zorder=0, label="equal to AO")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo / 1.5, hi * 1.5)
        ax.set_ylim(lo / 1.5, hi * 1.5)
        ax.set_xlabel(f"AO $|{vsym}|$ (MHz)")
        ax.set_ylabel(f"$|{vsym}|$ (MHz)")
        ax.grid(True, ls=":", alpha=0.5)
    _legend(fig, axes)
    _save(fig, out, f"{result['system']}: couplings per mode against AO (spin-contaminated); "
                    "second order: diagonal $(l, l)$ terms")


def plot_rho(result, out):
    """rho per channel of UO and decont against AO; first and second order side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))  # not sharey: each panel names its own channels
    for ax, order in zip(axes, (1, 2)):
        channels = [c for c in CHANNELS if c[2] == order]
        for k, m in enumerate(COMPARED):
            rho = result["summary"][m]["vs_AO"]["rho"]
            ys = np.arange(len(channels))[::-1] + (0.12 if k == 0 else -0.12)
            ax.plot([rho[c[0]]["rho"] for c in channels], ys, "o", ms=7, color=METHODS[m][1],
                    mec="white", mew=1, label=METHODS[m][0], ls="none")
            for (name, *_), y in zip(channels, ys):
                ax.annotate(f"{rho[name]['rho']:.2f}", (rho[name]["rho"], y), xytext=(7, 0),
                            textcoords="offset points", va="center", fontsize=8, color="0.25")
        ax.axvline(1.0, color="0.6", ls="--", lw=1, zorder=0)
        ax.set_yticks(np.arange(len(channels))[::-1], [f"${c[3]}$" for c in channels])
        ax.set_xlim(0, 1.6)
        ax.set_xlabel(r"$\rho = \sqrt{\sum V^2 / \sum V_{AO}^2}$")
        ax.set_title("first order" if order == 1 else "second order, diagonal $(l, l)$", fontsize=10)
        ax.grid(True, axis="x", ls=":", alpha=0.5)
    _legend(fig, axes)
    _save(fig, out, f"{result['system']}: coupling ratio $\\rho$ against AO")


def plot_rates(result, out):
    temps = np.asarray(result["temperature_K"])
    fig, axes = plt.subplots(2, len(TRANSITIONS), figsize=(6.2 * len(TRANSITIONS), 7.2), sharex=True,
                             gridspec_kw={"height_ratios": (3, 2)})
    for col, (tr, tr_label) in enumerate(TRANSITIONS):
        top, bottom = axes[0, col], axes[1, col]
        totals = {}
        for m, (mlabel, color) in METHODS.items():
            rates = result["methods"][m]["rates_per_s"]
            for p, plabel, ls in RATE_PROCESSES:
                y = np.asarray(rates[p][tr])
                keep = y > 0
                if keep.any():
                    top.plot(temps[keep], y[keep], color=color, ls=ls, lw=1.8, label=mlabel)
            totals[m] = sum(np.asarray(rates[p][tr]) for p, _, _ in RATE_PROCESSES)
        for m in COMPARED:
            keep = totals["AO"] > 0
            bottom.plot(temps[keep], totals[m][keep] / totals["AO"][keep], color=METHODS[m][1], lw=1.8)
        top.set_yscale("log")
        top.set_ylim(bottom=1e-6)
        top.set_ylabel(f"{tr_label} (1/s)")
        top.grid(True, ls=":", alpha=0.5)
        _ratio_axis(bottom, "rate / AO")
        bottom.set_ylim(0, max(1.2, bottom.get_ylim()[1]))
        bottom.set_xlabel("Temperature (K)")
    _legend(fig, axes)
    _save(fig, out, f"{result['system']}: two-phonon transition rates, $\\sigma$ = {result['sigma_meV']:g} meV")


def experiment(folder):
    """[(label, T (K), T1 (s))] of Jarmola et al. (2012), the samples of the paper figure M2."""
    path = folder / "Jarmola_2012.csv"
    if not path.exists():
        return []
    out = []
    for s in read_experimental_csv(path):
        if s.label in JARMOLA_SERIES:
            e = s.to_units(x_unit="K", y_unit="s")  # 1/T1 (Hz) -> T1 (s)
            out.append((f"Jarmola 2012 ({s.label})", e.x, e.y))
    return out


def plot_t1(result, out, exp):
    temps = np.asarray(result["temperature_K"])
    fig, axes = plt.subplots(2, len(STATES), figsize=(6.2 * len(STATES), 7.2), sharex=True,
                             gridspec_kw={"height_ratios": (3, 2)})
    for col, (state, slabel) in enumerate(STATES):
        top, bottom = axes[0, col], axes[1, col]
        ref = np.asarray(result["methods"]["AO"]["t1_s"][state])
        for m, (mlabel, color) in METHODS.items():
            t1 = np.asarray(result["methods"][m]["t1_s"][state])
            top.plot(temps, t1, color=color, lw=1.8, label=mlabel)
            if m in COMPARED:
                bottom.plot(temps, t1 / ref, color=color, lw=1.8)
        label, color, ls = AO_PRODUCTION
        top.plot(temps, result["ao_production"]["t1_s"][state], color=color, ls=ls, lw=1.2, label=label)
        if state == "ms_0":
            for (elabel, x, y), marker in zip(exp, "os^"):
                top.plot(x, y, marker, ms=5, mfc="white", mec="black", mew=1, ls="", label=elabel)
        top.set_yscale("log")
        top.set_ylabel(f"$T_1$ of {slabel} (s)")
        top.grid(True, ls=":", alpha=0.5)
        top.set_xlim(temps.min(), temps.max())
        top.set_ylim(1e-4, max(1e2, top.get_ylim()[0]))
        _ratio_axis(bottom, "$T_1$ / AO")
        bottom.set_ylim(0, max(1.2, bottom.get_ylim()[1]))
        bottom.set_xlabel("Temperature (K)")
    _legend(fig, axes)
    _save(fig, out, f"{result['system']}: $T_1$ from the two-phonon rates, $\\sigma$ = {result['sigma_meV']:g} meV")


def summary(result):
    temps = result["temperature_K"]
    ref = result["methods"]["AO"]
    out = {}
    for m, res in [*result["methods"].items(), ("AO_production", result["ao_production"])]:
        entry = {
            "D_MHz": res["D_MHz"],
            "rms_coupling_MHz": {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in res["couplings_MHz"].items()},
            "rate_per_s": {f"{p}:{tr}": at(temps, res["rates_per_s"][p][tr]) for p, _, _ in RATE_PROCESSES
                           for tr, _ in TRANSITIONS},
            "t1_s": {state: at(temps, res["t1_s"][state]) for state, _ in STATES},
        }
        if m in COMPARED:
            entry["vs_AO"] = {
                "D": res["D_MHz"] / ref["D_MHz"],
                "rho": rho_summary(res, ref),
                "t1": {state: at(temps, np.asarray(res["t1_s"][state]) / np.asarray(ref["t1_s"][state]))
                       for state, _ in STATES},
            }
        out[m] = entry
    return out


def print_summary(s):
    names = [c[0] for c in CHANNELS]
    print(f"\n{'':14s} {'D (MHz)':>8s} " + " ".join(f"{'rms ' + n:>14s}" for n in names)
          + f" {'T1 ms0 300K':>12s} {'T1 ms1 300K':>12s}")
    for m, e in s.items():
        r = e["rms_coupling_MHz"]
        print(f"{m:14s} {e['D_MHz']:8.1f} " + " ".join(f"{r[n]:14.4g}" for n in names)
              + f" {e['t1_s']['ms_0'].get('300', np.nan):12.4g} {e['t1_s']['ms_1'].get('300', np.nan):12.4g}")
    print("\nrho = sqrt(sum V^2 / sum V_AO^2) against AO (V_AO^2-weighted SD of the per-mode ratios); "
          "second order over the diagonal (l, l)")
    print(f"{'':10s}" + "".join(f"{n:>17s}" for n in names))
    for m in COMPARED:
        v = s[m]["vs_AO"]
        print(f"{m:10s}" + "".join(f"{v['rho'][n]['rho']:9.2f} ({v['rho'][n]['sd']:4.2f})" for n in names))
    for m in COMPARED:
        v = s[m]["vs_AO"]
        t1 = "  ".join(f"{T} K {x:.2f}" for T, x in v["t1"]["ms_0"].items())
        print(f"{m:10s} D/AO {v['D']:.3f}   T1(ms0)/AO: {t1}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    for m, path in DEFAULTS.items():
        ap.add_argument(f"--{m.lower()}", default=path, help=f"{m} package (folder for decont), relative to --data")
    ap.add_argument("--system", default="NV$^-$ (64)", help="name in the titles")
    ap.add_argument("--sigma", type=float, default=3.5, help="Gaussian broadening (meV), the production 3.5")
    ap.add_argument("--temps", type=float, nargs=3, default=(50.0, 600.0, 10.0), metavar=("MIN", "MAX", "STEP"))
    ap.add_argument("-o", "--outdir", type=Path, default=ROOT / "figures" / "decont_comparison")
    args = ap.parse_args()
    logging.disable(logging.WARNING)  # the per-run symmetry checks would bury the summary

    def resolve(p):
        return Path(p) if Path(p).exists() else args.data / p

    temps = np.arange(args.temps[0], args.temps[1] + args.temps[2] / 2, args.temps[2])
    paths = {m: resolve(getattr(args, m.lower())) for m in METHODS}
    managers = {"AO": lambda: coupling_manager(paths["AO"], "diagonal-sqrt"),
                "UO": lambda: coupling_manager(paths["UO"], "diagonal-sqrt"),
                "decont": lambda: coupling_manager(paths["decont"], "diagonal-sqrt")}
    result = {"system": args.system, "sigma_meV": args.sigma, "temperature_K": temps.tolist(),
              "settings": {"direct": DIRECT, "processes": PROCESSES, "zero_gap_pairs": "excluded",
                           "second_order": "diagonal + geometric mean (diagonal-sqrt)"},
              "packages": {m: str(p) for m, p in paths.items()}, "methods": {}}
    for m, make in managers.items():
        print(f"computing {m}: {paths[m]}")
        manager, convention = make()
        result["methods"][m] = {"convention": convention, **analyse(manager, temps, args.sigma)}
    print("computing AO production (all computed pairs)")
    manager, convention = coupling_manager(paths["AO"], "sqrt")
    result["ao_production"] = {"convention": convention, **analyse(manager, temps, args.sigma)}

    args.outdir.mkdir(parents=True, exist_ok=True)
    result["summary"] = summary(result)
    (args.outdir / "summary.json").write_text(json.dumps(result, indent=1))
    plot_couplings(result, args.outdir / "couplings_first_order.png", 1)
    plot_couplings(result, args.outdir / "couplings_second_order.png", 2)
    plot_couplings_per_mode(result, args.outdir / "couplings_per_mode.png")
    plot_rho(result, args.outdir / "rho.png")
    plot_rates(result, args.outdir / "rates.png")
    plot_t1(result, args.outdir / "t1.png", experiment(ROOT / "experimental_data" / "NV"))
    print_summary(result["summary"])


if __name__ == "__main__":
    main()
