"""Compare the spin-phonon couplings of the two ZFS methods mode by mode.

"contaminated": all occupied bands enter the spin density (spin-contaminated
valence bands included). "corrected": only the unpaired defect bands
(defect_band_approx). For every first-order mode the three couplings are
computed from dD = D(Q) - D0 in the contaminated ground state's principal frame,
with the formulas of ZFSManager.calculate_first_order_derivatives:

    V_00  = |dD_zz - (dD_xx + dD_yy)/2| / 3           (A1 modes)
    V_+-  = sqrt((dD_xx - dD_yy)^2 + 4 dD_xy^2) / 2   (E modes, in-plane)
    V_0+- = sqrt(dD_xz^2 + dD_yz^2) / sqrt 2          (E modes, tilt)

and set against the mode's localization from the IPR (PhononSpectrum.get_ipr,
N_eff = 1/IPR atoms). Writes, to figures/method_comparison/:

    coupling_scatter.pdf    corrected vs contaminated per mode, colored by IPR
    ratio_vs_ipr.pdf        per-mode ratio vs N_eff; running median and quartiles, binned strength ratios
    unpaired_fraction.pdf   share of the coupling strength the unpaired bands carry
    e_direction.pdf         alignment of the E responses of the two methods
    method_comparison.tex   summary table (booktabs, siunitx)
    method_comparison.csv   per-mode values

Usage (from the repo root): python scripts/plot_method_comparison.py [--defects NV_512 ...] [--pert 0.025]
"""

from __future__ import annotations

import argparse
import glob
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from spinlabs.convergence import principal_frame
from spinlabs.models import RawZFSData, ZFSTensor
from spinlabs.parsers import parse_phonon_data

DEFECTS = ("NV_512", "DiV_128", "ClV_128", "ClV_576")
TEX_NAMES = {"NV_512": r"NV$^-$ (512)", "DiV_128": r"V$_\text{Si}$V$_\text{C}$ (128)",
             "ClV_128": r"Cl$_\text{C}$V$_\text{Si}$ (128)", "ClV_576": r"Cl$_\text{C}$V$_\text{Si}$ (576)"}
PLOT_NAMES = {"NV_512": "NV (512)", "DiV_128": "DiV (128)", "ClV_128": "ClV (128)", "ClV_576": "ClV (576)"}
CHANNELS = ("V00", "Vpm", "V0pm")
CH_LABEL = {"V00": r"$V_{00}$", "Vpm": r"$V_{+-}$", "V0pm": r"$V_{0\pm}$"}
CH_COLOR = {"V00": "#2a78d6", "Vpm": "#eb6834", "V0pm": "#1baf7a"}  # validated, all pairs (light)
CH_MARKER = {"V00": "o", "Vpm": "s", "V0pm": "^"}  # second encoding: aqua is below 3:1 contrast
SNR_MIN = 10.0  # a mode enters the ratios only when its contaminated coupling is this far above noise
# Runs known to have failed (0-based mode index): ClV_576 all bands runs/1 gives dD = -1245 MHz
# for an acoustic mode.
BAD = {("ClV_576", "all_bands"): {0}}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
# one hue, light -> dark, starting visibly above the white surface
IPR_CMAP = LinearSegmentedColormap.from_list("ipr", plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256)))


def _tensor(entry) -> np.ndarray:
    return np.asarray(entry.zfs_tensor.matrix if hasattr(entry, "zfs_tensor") else entry["tensor"], dtype=float)


def couplings(dd: np.ndarray) -> dict[str, float]:
    """V_00, V_+-, V_0+- (MHz per unit Q) of a ZFS change per unit Q in the D0 frame."""
    return {
        "V00": abs(dd[2, 2] - 0.5 * (dd[0, 0] + dd[1, 1])) / 3.0,
        "Vpm": 0.5 * np.sqrt((dd[0, 0] - dd[1, 1]) ** 2 + 4.0 * dd[0, 1] ** 2),
        "V0pm": np.sqrt(dd[0, 2] ** 2 + dd[1, 2] ** 2) / np.sqrt(2.0),
    }


def load(defect: str, pert: str, method: str) -> RawZFSData:
    files = glob.glob(f"data/{defect}/pert_{pert}/{method}/*/*.npz")
    if not files:
        raise SystemExit(f"no {method} data for {defect} at pert {pert}")
    return RawZFSData.load(sorted(files)[-1]).to_unit("MHz")


def noise_per_q(raw: RawZFSData, symmetries: list[str], R: np.ndarray) -> float:
    """Noise (MHz per unit Q) of one tensor element of dD/Q, from the E parts of A-mode runs.

    An A1 or A2 displacement keeps C3, so the E parts of its D change vanish at every
    order; their scatter about the median (robust sigma) is numerical noise.
    """
    d0 = R.T @ raw.ground_state_zfs.matrix @ R
    vals = []
    for i, e in raw.first_order.items():
        if i < len(symmetries) and symmetries[i] in ("A1", "A2"):
            d = R.T @ _tensor(e) @ R - d0
            vals += [(d[0, 0] - d[1, 1]) / np.sqrt(2), d[0, 1], d[0, 2], d[1, 2]]
    vals = np.asarray(vals)
    sigma = 1.4826 * np.median(np.abs(vals - np.median(vals)))
    return float(np.sqrt(2) * sigma / abs(raw.pert_scale))


def analyse(defect: str, pert: str) -> dict:
    spec = parse_phonon_data(f"data/{defect}/phonon_data.npz")
    symmetries = [str(s) for s in spec.symmetries]
    ipr = np.asarray(spec.get_ipr())
    cont, corr = load(defect, pert, "all_bands"), load(defect, pert, "defect_band_approx")
    R = principal_frame(cont.ground_state_zfs.matrix)
    noise = noise_per_q(cont, symmetries, R)
    rows = []
    for i, e in cont.first_order.items():
        if i >= len(symmetries) or symmetries[i] == "A2" or i not in corr.first_order:
            continue
        if i in BAD.get((defect, "all_bands"), ()) or spec.frequencies_mev[i] < 1.0:
            continue
        va = couplings((R.T @ (_tensor(e) - cont.ground_state_zfs.matrix) @ R) / cont.pert_scale)
        vb = couplings((R.T @ (_tensor(corr.first_order[i]) - corr.ground_state_zfs.matrix) @ R) / corr.pert_scale)
        da = R.T @ (_tensor(e) - cont.ground_state_zfs.matrix) @ R
        db = R.T @ (_tensor(corr.first_order[i]) - corr.ground_state_zfs.matrix) @ R
        ea = np.array([(da[0, 0] - da[1, 1]) / np.sqrt(2), da[0, 1], da[0, 2], da[1, 2]])
        eb = np.array([(db[0, 0] - db[1, 1]) / np.sqrt(2), db[0, 1], db[0, 2], db[1, 2]])
        cos = float(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb))) if symmetries[i] != "A1" else np.nan
        rows.append(dict(mode=i + 1, sym=symmetries[i], freq=float(spec.frequencies_mev[i]), ipr=float(ipr[i]),
                         cont=va, corr=vb, cos=cos))
    d_cont = 1.5 * ZFSTensor(matrix=cont.ground_state_zfs.matrix, unit="MHz").principal_components()[2]
    d_corr = 1.5 * ZFSTensor(matrix=corr.ground_state_zfs.matrix, unit="MHz").principal_components()[2]
    return dict(defect=defect, rows=rows, noise=noise, d_cont=d_cont, d_corr=d_corr, n_atoms=len(spec.atom_symbols))


def channel_rows(res: dict, ch: str) -> list[dict]:
    """Modes that carry channel ch by symmetry and are resolved above noise in the contaminated run."""
    want = "A1" if ch == "V00" else "E"
    return [r for r in res["rows"] if (r["sym"] == "A1") == (want == "A1") and r["cont"][ch] > SNR_MIN * res["noise"]]


def strength_ratio(rows: list[dict], ch: str) -> float:
    """sqrt(sum V_corr^2 / sum V_cont^2): rates go as V^2, so this weighs modes as T1 does."""
    a = np.array([r["cont"][ch] for r in rows])
    b = np.array([r["corr"][ch] for r in rows])
    return float(np.sqrt((b**2).sum() / (a**2).sum())) if len(a) else np.nan


def ipr_bins(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """IPR-rank groups: lowest 50 %, 50-90 %, 90-99 %, top 1 %."""
    if not rows:
        return []
    iprs = np.array([r["ipr"] for r in rows])
    q = np.quantile(iprs, [0.5, 0.9, 0.99])
    edges = [(-np.inf, q[0], "0-50 %"), (q[0], q[1], "50-90 %"), (q[1], q[2], "90-99 %"), (q[2], np.inf, "top 1 %")]
    return [(lab, [r for r in rows if lo < r["ipr"] <= hi]) for lo, hi, lab in edges]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def plot_scatter(results: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(len(results), 3, figsize=(7.0, 2.1 * len(results)), squeeze=False)
    iprs = [r["ipr"] for res in results for r in res["rows"]]
    norm = LogNorm(vmin=min(iprs), vmax=max(iprs))
    for row, res in enumerate(results):
        for col, ch in enumerate(CHANNELS):
            ax = axes[row, col]
            rows = channel_rows(res, ch)
            if not rows:
                ax.set_axis_off()
                continue
            a = np.array([r["cont"][ch] for r in rows])
            b = np.array([r["corr"][ch] for r in rows])
            c = np.array([r["ipr"] for r in rows])
            order = np.argsort(c)  # localized modes on top
            sc = ax.scatter(a[order], np.maximum(b[order], 1e-3 * a.min()), c=c[order], cmap=IPR_CMAP, norm=norm,
                            s=9, lw=0.3, edgecolors="#ffffff", marker=CH_MARKER[ch], zorder=2)
            lo, hi = min(a.min(), b[b > 0].min()), max(a.max(), b.max())
            line = np.array([lo, hi])
            ax.plot(line, line, color=INK2, lw=0.8, ls="--", zorder=1)
            ratio = strength_ratio(rows, ch)
            ax.plot(line, ratio * line, color=CH_COLOR[ch], lw=1.4, zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo / 1.5, hi * 1.5)
            ax.set_ylim(lo / 1.5, hi * 1.5)
            ax.grid(True, color=GRID, lw=0.5, zorder=0)
            ax.text(0.04, 0.95, f"{PLOT_NAMES[res['defect']]}\nratio {ratio:.2f}", transform=ax.transAxes,
                    va="top", fontsize=7.5, color=INK)
            if row == 0:
                ax.set_title(CH_LABEL[ch])
            if row == len(results) - 1:
                ax.set_xlabel("contaminated (MHz/Q)")
            if col == 0:
                ax.set_ylabel("corrected (MHz/Q)")
    cbar = fig.colorbar(sc, ax=axes, shrink=0.5, pad=0.02)
    cbar.set_label("mode IPR (localization)")
    fig.savefig(out)
    plt.close(fig)


N_BINS = 10  # log-spaced N_eff bins per defect
MIN_BIN = 3  # fewest modes in a bin; sparser bins are merged with their neighbour


SMOOTH_DEX = 0.35  # half-width of the running window in decades of N_eff
SMOOTH_MIN = 20  # fewest modes in the running window; widened to the nearest ones where sparse


def running_quartiles(neff: np.ndarray, ratio: np.ndarray, n_grid: int = 60) -> tuple[np.ndarray, ...]:
    """25/50/75 % of ratio on a log grid of N_eff from the most delocalized to the most localized mode.

    Each grid point takes the modes within SMOOTH_DEX decades, or the SMOOTH_MIN modes
    nearest in log N_eff where fewer lie that close, so the curve reaches the most
    localized mode while every point stays a median of several modes. The last array
    marks the points whose window held SMOOTH_MIN modes without widening.
    """
    lx = np.log10(neff)
    grid = np.linspace(lx.max(), lx.min(), n_grid)
    q = np.empty((n_grid, 3))
    dense = np.zeros(n_grid, dtype=bool)
    for k, g in enumerate(grid):
        d = np.abs(lx - g)
        sel = d <= SMOOTH_DEX
        dense[k] = sel.sum() >= SMOOTH_MIN
        if not dense[k]:
            sel = d <= np.sort(d)[min(SMOOTH_MIN, len(d)) - 1]
        q[k] = np.percentile(ratio[sel], [25, 50, 75])
    return 10**grid, q[:, 0], q[:, 1], q[:, 2], dense


def merged_bins(neff: np.ndarray, edges: np.ndarray) -> list[np.ndarray]:
    """Index groups of log-spaced bins, each merged with its neighbour until it holds MIN_BIN modes.

    Merging starts at the localized end, so the last group holds the most localized
    modes; a sparse remainder at the delocalized end joins the first full group there.
    """
    which = np.clip(np.digitize(neff, edges) - 1, 0, len(edges) - 2)
    groups, current = [], []
    for k in range(len(edges) - 1):  # bins from small (localized) to large N_eff
        current += list(np.where(which == k)[0])
        if len(current) >= MIN_BIN:
            groups.append(np.array(current))
            current = []
    if current:
        if groups:
            groups[-1] = np.concatenate([groups[-1], current])
        else:
            groups.append(np.array(current))
    return groups[::-1]  # delocalized first, as the axis reads


def plot_ratio_vs_ipr(results: list[dict], out: Path) -> None:
    """Per-mode ratio against N_eff, with medians, quartiles and strength ratios in log-spaced bins."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ncols = 2
    nrows = -(-len(results) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6, 2.5 * nrows + 0.5), sharey=True, squeeze=False)
    for ax, res in zip(axes.flat, results):
        per_ch = {ch: channel_rows(res, ch) for ch in CHANNELS}
        neff_all = np.array([1 / r["ipr"] for rows in per_ch.values() for r in rows])
        edges = np.geomspace(neff_all.min() * 0.999, neff_all.max() * 1.001, N_BINS + 1)
        for ch, rows in per_ch.items():
            if not rows:
                continue
            neff = np.array([1 / r["ipr"] for r in rows])
            ratio = np.array([r["corr"][ch] / r["cont"][ch] for r in rows])
            ax.scatter(neff, np.maximum(ratio, 1e-3), s=5, color=CH_COLOR[ch], alpha=0.25, lw=0,
                       marker=CH_MARKER[ch], zorder=1)
            xr, lo, med, hi, dense = running_quartiles(neff, ratio)
            xs, strength = [], []
            for idx in merged_bins(neff, edges):
                xs.append(np.median(neff[idx]))
                strength.append(strength_ratio([rows[i] for i in idx], ch))
            # band and solid median only where the window held SMOOTH_MIN modes by itself;
            # the widened (sparse, localized) end continues as a dashed median
            ax.fill_between(xr, lo, hi, where=dense, color=CH_COLOR[ch], alpha=0.18, lw=0, zorder=2)
            ax.plot(xr, med, color=CH_COLOR[ch], lw=1.3, ls=(0, (3, 2)), zorder=3)
            ax.plot(xr, np.where(dense, med, np.nan), color=CH_COLOR[ch], lw=1.6, zorder=3)
            ax.plot(xs, strength, ls="none", marker=CH_MARKER[ch], ms=5, mfc="#ffffff", mec=CH_COLOR[ch],
                    mew=1.2, zorder=4)
        ax.axhline(1.0, color=INK2, lw=0.8, ls="--", zorder=2.5)  # above bands and dots, below the medians
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.02, 6)
        ax.set_xlim(edges[-1] * 1.15, edges[0] / 1.15)  # localized modes (few atoms) to the right
        ax.set_title(PLOT_NAMES[res["defect"]])
        ax.grid(True, color=GRID, lw=0.5, zorder=0)
    for ax in axes[-1]:
        ax.set_xlabel("$N_\\mathrm{eff}$ = 1/IPR  (→ localized)")
    for ax in axes[:, 0]:
        ax.set_ylabel("$V_\\mathrm{corr}/V_\\mathrm{cont}$")
    for ax in list(axes.flat)[len(results):]:
        ax.set_axis_off()
    handles = [Patch(color=CH_COLOR[ch], label=CH_LABEL[ch]) for ch in CHANNELS] + [
        Line2D([], [], color=INK2, lw=1.6, label="running median"),
        Line2D([], [], color=INK2, lw=1.3, ls=(0, (3, 2)), label=f"median, under {SMOOTH_MIN} modes in window"),
        Patch(color=INK2, alpha=0.25, label="25-75 %"),
        Line2D([], [], ls="none", marker="o", mfc="#ffffff", mec=INK2, mew=1.2, label="strength ratio per bin"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_unpaired_fraction(results: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 0.55 * len(results) + 1.0))
    y = np.arange(len(results))[::-1]
    offsets = {"V00": 0.2, "Vpm": 0.0, "V0pm": -0.2}
    for ch in CHANNELS:
        f = np.array([strength_ratio(channel_rows(res, ch), ch) ** 2 for res in results])
        ax.scatter(f, y + offsets[ch], color=CH_COLOR[ch], marker=CH_MARKER[ch], s=36, lw=0.8,
                   edgecolors="#ffffff", label=CH_LABEL[ch], zorder=3)
        for fi, yi in zip(f, y + offsets[ch]):
            ax.text(fi * 1.12, yi, f"{fi:.2f}", va="center", fontsize=7, color=INK2)
    ax.axvline(1.0, color=INK2, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 5)
    ax.set_yticks(y)
    ax.set_yticklabels([PLOT_NAMES[r["defect"]] for r in results])
    ax.set_xlabel("$\\Sigma V_\\mathrm{corr}^2 / \\Sigma V_\\mathrm{cont}^2$  (coupling strength of the unpaired bands)")
    ax.grid(True, axis="x", color=GRID, lw=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_e_direction(results: list[dict], out: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(1.7 * n + 0.4, 2.1), sharey=True, squeeze=False)
    bins = np.linspace(-1, 1, 21)
    for ax, res in zip(axes[0], results):
        cos = np.array([r["cos"] for r in channel_rows(res, "Vpm")])
        ax.hist(cos, bins=bins, color="#2a78d6", edgecolor="#ffffff", lw=0.6, weights=np.ones(len(cos)) / len(cos))
        ax.axvline(np.median(cos), color=INK, lw=1.0)
        ax.text(0.04, 0.95, f"median {np.median(cos):.2f}\nn = {len(cos)}", transform=ax.transAxes, va="top",
                fontsize=7.5)
        ax.set_title(PLOT_NAMES[res["defect"]])
        ax.set_xlabel("cos(dD$_E^\\mathrm{corr}$, dD$_E^\\mathrm{cont}$)")
        ax.grid(True, axis="y", color=GRID, lw=0.5)
    axes[0, 0].set_ylabel("fraction of E modes")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_table(results: list[dict], out: Path) -> None:
    lines = [
        "% generated by scripts/plot_method_comparison.py",
        r"% requires \usepackage{booktabs} and \usepackage{siunitx}",
        r"\begin{table*}[htbp]",
        r"  \centering",
        r"  \begin{tabular}{lcccccccc}",
        r"    \toprule",
        r"    Defect & $D_\text{cont}$ & $D_\text{corr}$ & \multicolumn{3}{c}{$\Sigma V_\text{corr}^2/\Sigma V_\text{cont}^2$}"
        r" & \multicolumn{2}{c}{$\rho_\text{IPR}$} & $\langle\cos\rangle_E$ \\",
        r"    \cmidrule(lr){4-6} \cmidrule(lr){7-8}",
        r"    & (\si{\mega\hertz}) & (\si{\mega\hertz}) & $V_{00}$ & $V_{+-}$ & $V_{0\pm}$ & $V_{+-}$ & $V_{0\pm}$ & \\",
        r"    \midrule",
    ]
    for res in results:
        cells = [TEX_NAMES[res["defect"]], f"{res['d_cont']:.0f}", f"{res['d_corr']:.0f}"]
        cells += [f"{strength_ratio(channel_rows(res, ch), ch) ** 2:.2f}" for ch in CHANNELS]
        for ch in ("Vpm", "V0pm"):
            rows = channel_rows(res, ch)
            rho = spearman(np.array([r["ipr"] for r in rows]), np.array([r["corr"][ch] / r["cont"][ch] for r in rows]))
            cells.append(f"{rho:+.2f}")
        cells.append(f"{np.median([r['cos'] for r in channel_rows(res, 'Vpm')]):.2f}")
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{First-order spin-phonon couplings with only the unpaired defect bands (corrected) vs"
        r" all bands (contaminated), at $Q = 0.025$. $\Sigma V_\text{corr}^2/\Sigma V_\text{cont}^2$: coupling"
        r" strength carried by the unpaired bands, summed over modes resolved above " + f"{SNR_MIN:g}"
        r"$\times$ the noise. $\rho_\text{IPR}$: rank correlation of the per-mode ratio"
        r" $V_\text{corr}/V_\text{cont}$ with the mode IPR. $\langle\cos\rangle_E$: median alignment of the"
        r" E-type ZFS changes of the two methods.}",
        r"  \label{tab:method_comparison}",
        r"\end{table*}",
    ]
    out.write_text("\n".join(lines) + "\n")


def write_ipr_table(results: list[dict]) -> str:
    text = ["strength ratio sqrt(sum V_corr^2 / sum V_cont^2) by IPR group (n modes, share of contaminated V^2)"]
    for res in results:
        text.append(f"{res['defect']}:")
        for ch in CHANNELS:
            rows = channel_rows(res, ch)
            total = sum(r["cont"][ch] ** 2 for r in rows)
            parts = [f"{lab} {strength_ratio(g, ch):.2f} (n {len(g)}, {sum(r['cont'][ch] ** 2 for r in g) / total:.0%})"
                     for lab, g in ipr_bins(rows) if g]
            text.append(f"  {ch:5}: overall {strength_ratio(rows, ch):.2f}; " + "; ".join(parts))
    return "\n".join(text)


def write_csv(results: list[dict], out: Path) -> None:
    with out.open("w") as fh:
        fh.write("defect,mode,sym,freq_meV,ipr,V00_cont,V00_corr,Vpm_cont,Vpm_corr,V0pm_cont,V0pm_corr,cos_E\n")
        for res in results:
            for r in res["rows"]:
                fh.write(",".join(str(x) for x in (
                    res["defect"], r["mode"], r["sym"], f"{r['freq']:.3f}", f"{r['ipr']:.5g}",
                    *(f"{r[m][ch]:.6g}" for ch in CHANNELS for m in ("cont", "corr")), f"{r['cos']:.4f}",
                )) + "\n")


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--defects", nargs="+", default=list(DEFECTS))
    ap.add_argument("--pert", default="0.025")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("figures/method_comparison"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    _style()
    results = [analyse(d, args.pert) for d in args.defects]
    plot_scatter(results, args.outdir / "coupling_scatter.pdf")
    plot_ratio_vs_ipr(results, args.outdir / "ratio_vs_ipr.pdf")
    plot_unpaired_fraction(results, args.outdir / "unpaired_fraction.pdf")
    plot_e_direction(results, args.outdir / "e_direction.pdf")
    write_table(results, args.outdir / "method_comparison.tex")
    write_csv(results, args.outdir / "method_comparison.csv")
    summary = write_ipr_table(results)
    (args.outdir / "ipr_groups.txt").write_text(summary + "\n")
    print(summary)
    print(f"-> {args.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
