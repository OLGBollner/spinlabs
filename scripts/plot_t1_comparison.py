"""T1 of several coupling datasets against the broadening sigma or the temperature, in one figure.

Each curve is a ``--case LABEL FILE MODE [DISPLACEMENT]``: a raw ZFS package (relative to --data
unless the path exists as given) and which second-order pairs to use:

  computed        the pairs as computed
  sqrt            uncomputed pairs i != j filled with sqrt(V_ii V_jj), symmetry-allowed types only
  diagonal        only the diagonal pairs (i, i)
  diagonal-sqrt   only the diagonal pairs, the rest filled with sqrt(V_ii V_jj)

DISPLACEMENT (normalized, mass_weighted, unweighted) overrides the package's displacement convention.
Inconsistent second-order entries are excluded. Without --case, the NV_64 / NV_512 comparison of
2026-10-28 is drawn. The numbers go to <output>.json; --replot redraws from it without recomputing.

    python scripts/plot_t1_comparison.py sigma -o figures/sigma_scan/t1_vs_sigma.png
    python scripts/plot_t1_comparison.py temperature --sigma 7.5 -o figures/sigma_scan/t1_vs_temperature.png
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

from spinlabs.pipeline import compute_rates, load_raw_inputs, solve_t1  # noqa: E402
from spinlabs.zfs_manager import ZFSManager  # noqa: E402

MODES = ("computed", "sqrt", "diagonal", "diagonal-sqrt")
STATES = (("ms_0", "$m_s=0$"), ("ms_1", "$m_s=\\pm1$"))
DEFAULT_SIGMAS = [0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 3.5, 5.0, 6.0, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0]
NV64 = "NV_64/pert_0.1/all_bands/20261021/NV_64_raw_zfs_data_all_bands_12d.npz"
NV512_STEP_0025 = "NV_512/pert_0.025/all_bands/20261020/NV_512_raw_zfs_data_all_bands_12d.npz"
NV512_STEP_01 = "NV_512/pert_0.1/all_bands/20261022/NV_512_raw_zfs_data_all_bands_12d.npz"
DEFAULT_CASES = [
    ("NV 64 step 0.1, all pairs computed", NV64, "computed"),
    ("NV 64 step 0.1, diagonal + sqrt(V_ii V_jj)", NV64, "diagonal-sqrt"),
    ("NV 512 step 0.025, diagonal + sqrt(V_ii V_jj)", NV512_STEP_0025, "sqrt"),
    ("NV 512 step 0.1, diagonal + sqrt(V_ii V_jj)", NV512_STEP_01, "sqrt"),
    ("NV 512 step 0.025, diagonal only", NV512_STEP_0025, "diagonal"),
    ("NV 512 step 0.1, diagonal only", NV512_STEP_01, "diagonal"),
]
DEFAULT_CASES_TEMPERATURE = DEFAULT_CASES[:4]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#c9a227", "#555555"]  # one per raw file
MARKERS = "osD^vP"
MODE_STYLES = {
    "computed": dict(ls="-"),
    "sqrt": dict(ls="--", mfc="white"),
    "diagonal-sqrt": dict(ls="--", mfc="white"),
    "diagonal": dict(ls=":"),
}


def coupling_manager(raw_file, mode, displacement=None):
    """ZFSManager for one raw package with the second-order pairs of ``mode`` (one of MODES)."""
    method = "defect_band_approx" if "defect_band_approx" in Path(raw_file).name else "all_bands"
    raw, spectrum, _ = load_raw_inputs(raw_zfs_file=raw_file, calc_method=method, zfs_folder="ZFS_hyp", order=2)
    if displacement:
        raw.displacement = displacement
    manager = ZFSManager(
        raw_data=raw,
        spectrum=spectrum,
        exclude_inconsistent_pairs=True,
        offdiag_from_diagonal=mode in ("sqrt", "diagonal-sqrt"),
    )
    manager.calc_method = method
    if mode.startswith("diagonal"):  # as if only the (i, i) pairs had been run
        manager.zfs_tensors_2d = {k: v for k, v in manager.zfs_tensors_2d.items() if k[0] == k[1]}
    return manager, f"{raw.defect}_{raw.cell_size} step {raw.pert_scale:g}: {manager.displacement}"


def t1_grid(manager, sigmas, temps):
    """T1 (s) per initial state and the 0<->1 one- and two-phonon rates (1/s), as [sigma][temperature]."""
    t1 = {state: [] for state, _ in STATES}
    rates_0_1 = {"first_order": [], "two_phonon": []}
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "coupling")
        manager.process_both_orders(output_filename=base)
        for n, sigma in enumerate(sigmas):
            valid, rates, directional = compute_rates(Path(f"{base}.npz"), temps, sigma=sigma, has_second_order=True)
            if len(valid) != len(temps):
                raise SystemExit(f"rates missing at some temperatures for sigma {sigma:g}: got {list(valid)}")
            path = Path(tmp) / f"directional_{n}.npz"
            np.savez(path, temperatures=valid, **directional)
            for state in t1:
                t1[state].append(solve_t1(path, state)[0].tolist())
            for process, values in rates_0_1.items():
                values.append([float(r) for r in rates[process]["0_1"]])
    return t1, rates_0_1


def compute(kind, cases, sigmas, temps):
    """Run every case; the result dict that plot() draws and the JSON stores."""
    result = {"kind": kind, "sigma_mev": sigmas, "temperature_K": temps, "cases": []}
    for label, raw_file, mode, displacement in cases:
        manager, convention = coupling_manager(raw_file, mode, displacement)
        t1, rates_0_1 = t1_grid(manager, sigmas, temps)
        result["cases"].append(
            {
                "label": label,
                "file": str(raw_file),
                "mode": mode,
                "convention": convention,
                "offdiag_filled": manager.n_offdiag_filled,
                "t1_s": t1,
                "rates_0_1_per_s": rates_0_1,
            }
        )
        share = [f / (f + t) for f, t in zip(np.ravel(rates_0_1["first_order"]), np.ravel(rates_0_1["two_phonon"]))]
        print(f"{label} ({convention}, {mode}, {manager.n_offdiag_filled} pairs filled)", flush=True)
        print(f"  one-phonon share of the 0<->1 rate: {[f'{s:.2g}' for s in share]}", flush=True)
    return result


def plot(result, out, mark):
    """Two panels (m_s = 0 and +-1), one curve per case; a dotted line at ``mark`` on the x axis."""
    kind = result["kind"]
    files = list(dict.fromkeys(case["file"] for case in result["cases"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, (state, title) in zip(axes, STATES):
        for case in result["cases"]:
            values = np.asarray(case["t1_s"][state], dtype=float)  # [sigma][temperature]
            if kind == "sigma":
                x, y, every = result["sigma_mev"], values[:, 0], None
            else:
                x, y, every = result["temperature_K"], values[0], 3
            i = files.index(case["file"])
            style = dict(color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)], **MODE_STYLES[case["mode"]])
            ax.plot(x, y, lw=2, ms=5, mew=1.3, markevery=every, label=case["label"], **style)
        ax.axvline(mark, color="#888888", lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if kind == "sigma":
            ax.set_xlabel(r"Broadening $\sigma$ (meV)")
            ax.set_title(f"{title}, {result['temperature_K'][0]:g} K", fontsize=11)
        else:
            ax.set_xlabel("Temperature (K)")
            ax.set_title(f"{title}, $\\sigma$ = {result['sigma_mev'][0]:g} meV", fontsize=11)
        ax.grid(True, which="major", ls=":", alpha=0.6)
    axes[0].set_ylabel(r"$T_1$ (s)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.12),
               ncol=3 if len(result["cases"]) > 4 else 2, fontsize=8.5, frameon=False)
    conventions = "; ".join(dict.fromkeys(case["convention"] for case in result["cases"]))
    mark_text = f"sigma = {mark:g} meV" if kind == "sigma" else f"{mark:g} K"
    fig.text(0.5, -0.05, f"displacement: {conventions}\ninconsistent second-order entries excluded; sqrt fill only "
             f"for symmetry-allowed coupling types; vertical dotted line: {mark_text}",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out}")


def parse_cases(raw_cases, defaults, data):
    """(label, path, mode, displacement) per --case; the defaults when none is given."""
    cases = []
    for case in raw_cases or [list(c) for c in defaults]:
        if len(case) not in (3, 4) or case[2] not in MODES:
            raise SystemExit(f"--case wants LABEL FILE MODE [DISPLACEMENT] with MODE one of {MODES}: {case}")
        path = Path(case[1]) if Path(case[1]).exists() else Path(data) / case[1]
        if not path.exists():
            raise SystemExit(f"no such raw package: {case[1]} (also looked in {data})")
        cases.append((case[0], path, case[2], case[3] if len(case) == 4 else None))
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T1 of several datasets against sigma or temperature in one figure")
    kinds = parser.add_subparsers(dest="kind", required=True)
    for kind, help_text in (("sigma", "T1 against the broadening sigma at one temperature"),
                            ("temperature", "T1 against temperature at one sigma")):
        p = kinds.add_parser(kind, help=help_text)
        p.add_argument("--data", default="data", help="Folder the --case files are relative to (default: data)")
        p.add_argument("--case", nargs="+", action="append", metavar="ARG",
                       help="LABEL FILE MODE [DISPLACEMENT]; repeat per curve (default: the NV_64/NV_512 comparison)")
        p.add_argument("-o", "--output", default=f"t1_vs_{kind}.png", help="Figure path; the numbers go next to it")
        p.add_argument("--replot", action="store_true", help="Redraw from <output>.json without recomputing")
        if kind == "sigma":
            p.add_argument("--sigmas", nargs="+", type=float, default=DEFAULT_SIGMAS, help="Broadenings in meV")
            p.add_argument("--temperature", type=float, default=300.0, help="Temperature in K (default: 300)")
            p.add_argument("--mark", type=float, default=7.5, help="Sigma to mark with a dotted line (default: 7.5)")
        else:
            p.add_argument("--sigma", type=float, default=7.5, help="Broadening in meV (default: 7.5)")
            p.add_argument("--t-range", nargs=3, type=float, default=[10.0, 500.0, 10.0],
                           metavar=("START", "END", "STEP"), help="Temperatures in K (default: 10 500 10)")
            p.add_argument("--mark", type=float, default=300.0, help="Temperature to mark (default: 300)")
    args = parser.parse_args()
    logging.disable(logging.WARNING)  # the per-run pipeline warnings would bury the summary

    out = Path(args.output)
    if args.replot:
        result = json.loads(out.with_suffix(".json").read_text())
    else:
        if args.kind == "sigma":
            sigmas, temps, defaults = args.sigmas, [args.temperature], DEFAULT_CASES
        else:
            start, end, step = args.t_range
            sigmas, temps = [args.sigma], np.arange(start, end + step / 2, step).tolist()
            defaults = DEFAULT_CASES_TEMPERATURE
        result = compute(args.kind, parse_cases(args.case, defaults, args.data), sigmas, temps)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps(result, indent=1))
    plot(result, out, args.mark)
