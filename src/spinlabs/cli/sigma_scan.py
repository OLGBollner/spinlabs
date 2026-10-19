"""``spinlabs run sigma-scan``: T1 against the spectral broadening sigma for one dataset.

sigma broadens every mode in the spectral functions so the supercell's discrete
modes stand in for the continuous phonon bands of a macroscopic crystal. The
scan computes the couplings once, then the rates and T1 at each sigma, so the
dependence on that choice can be seen and justified. Nothing is written to the
registry or the run folders.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np

from spinlabs.cli import add_verbosity_args
from spinlabs.cli.coupling import add_input_args, load_manager
from spinlabs.logging_utils import get_logger
from spinlabs.relaxation_dynamics import INIT_STATES, normalize_init_state

LOGGER = get_logger(__name__)

DEFAULT_SIGMAS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0)


def build_sigma_scan_parser(parser: argparse.ArgumentParser) -> None:
    """Configure ``parser`` as the ``run sigma-scan`` subcommand."""
    opt = add_input_args(parser)
    opt.add_argument(
        "--sigma",
        nargs="+",
        type=float,
        default=list(DEFAULT_SIGMAS),
        metavar="MEV",
        help=f"Broadenings to scan in meV (default: {' '.join(f'{s:g}' for s in DEFAULT_SIGMAS)})",
    )
    opt.add_argument(
        "--temperature",
        nargs="+",
        type=float,
        default=[300.0],
        metavar="K",
        help="Temperatures to report T1 at (default: 300)",
    )
    opt.add_argument(
        "--init-state",
        nargs="+",
        type=normalize_init_state,
        choices=INIT_STATES,
        default=list(INIT_STATES),
        help="Initial spin state(s) (default: all; 1 0 -1 also accepted)",
    )
    opt.add_argument("--plot", type=Path, metavar="FILE", help="Also save a T1-vs-sigma figure to FILE")
    add_verbosity_args(opt)
    parser.set_defaults(func=_run_sigma_scan)


def _run_sigma_scan(args: argparse.Namespace) -> None:
    from spinlabs.models import SpinPhononCouplingData
    from spinlabs.pipeline import DEFAULT_SIGMA_MEV, compute_rates, solve_t1

    sigmas = sorted(set(args.sigma))
    if any(s <= 0 for s in sigmas):
        raise SystemExit("--sigma values must be positive")
    temps = np.array(sorted(set(args.temperature)), dtype=float)
    manager, ph_path, orders = load_manager(args, "spinlabs run sigma-scan")

    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "spin_phonon_coupling")
        if 2 in orders and manager.zfs_tensors_2d:
            manager.process_both_orders(output_filename=base)
        else:
            manager.process_first_order_perturbations(output_filename=base)
        coupling = Path(f"{base}.npz")
        has_second = SpinPhononCouplingData.load(coupling).has_second_order

        # t1[sigma][state] -> T1 (s) at each temperature
        t1: dict[float, dict[str, np.ndarray]] = {}
        for n, sigma in enumerate(sigmas, start=1):
            print(f"[{n}/{len(sigmas)}] sigma {sigma:g} meV", flush=True)
            valid, _, directional = compute_rates(coupling, temps, sigma=sigma, has_second_order=has_second)
            dir_path = Path(tmp) / f"directional_rates_{n}.npz"
            np.savez(dir_path, temperatures=valid, **directional)
            t1[sigma] = {state: _on_grid(valid, solve_t1(dir_path, state)[0], temps) for state in args.init_state}

    print(_table(manager, t1, temps, args.init_state, has_second, DEFAULT_SIGMA_MEV))
    if args.plot:
        if not any(np.any(np.isfinite(v) & (v > 0)) for by_state in t1.values() for v in by_state.values()):
            print(f"No finite T1 to plot; {args.plot} not written")
        else:
            _plot(manager, t1, temps, args.init_state, args.plot, DEFAULT_SIGMA_MEV)
            print(f"Saved {args.plot}")


def _on_grid(valid: np.ndarray, values: np.ndarray, temps: np.ndarray) -> np.ndarray:
    """``values`` at ``temps``; NaN where a temperature gave no result."""
    lookup = dict(zip(np.round(valid, 9), values))
    return np.array([lookup.get(round(float(T), 9), np.nan) for T in temps])


def _label(manager) -> str:
    weighting = "" if manager.mass_weighted else ", not mass-weighted (corrected)"
    return f"{manager.defect}_{manager.cell_size}  {manager.calc_method}  pert {manager.pert_scale:g}{weighting}"


def _table(manager, t1, temps, states, has_second, default_sigma) -> str:
    """T1 (ms) per sigma, state and temperature, plus the spread over the scan."""
    cols = [(state, T) for T in temps for state in states]
    head = f"{'sigma (meV)':>12s} " + " ".join(f"{state + f' @{T:g}K':>14s}" for state, T in cols)
    lines = [
        _label(manager),
        f"  second order: {'yes' if has_second else 'no (one-phonon processes only)'}; "
        f"inconsistent second-order entries excluded: {len(manager.inconsistent_pairs)}",
        "  T1 in ms:",
        head,
    ]
    for sigma, by_state in t1.items():
        mark = "*" if np.isclose(sigma, default_sigma) else " "
        vals = [by_state[state][list(temps).index(T)] * 1e3 for state, T in cols]
        lines.append(f"{sigma:>11g}{mark} " + " ".join(f"{v:>14.4g}" for v in vals))
    spread = []
    for state, T in cols:
        v = np.array([t1[s][state][list(temps).index(T)] for s in t1])
        v = v[np.isfinite(v) & (v > 0)]
        spread.append(f"{v.max() / v.min():>14.3g}" if v.size else f"{'---':>14s}")
    lines.append(f"{'max/min':>12s} " + " ".join(spread))
    lines.append(f"  * default sigma ({default_sigma:g} meV)")
    return "\n".join(lines)


def _plot(manager, t1, temps, states, path: Path, default_sigma) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spinlabs.plotter import _INIT_STATE_LABELS

    sigmas = np.array(list(t1))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.tab10.colors
    styles = ["-", "--", ":", "-."]
    # ms=+1 and ms=-1 relax identically (symmetric rates); draw such curves once.
    curves: list[tuple[list[str], int, np.ndarray]] = []
    for state in states:
        for m in range(len(temps)):
            y = np.array([t1[s][state][m] for s in sigmas])
            match = next((c for c in curves if c[1] == m and np.allclose(c[2], y, rtol=1e-9, equal_nan=True)), None)
            if match:
                match[0].append(state)
            else:
                curves.append(([state], m, y))
    for k, (group, m, y) in enumerate(curves):
        name = "$m_s=\\pm1$" if sorted(group) == ["ms_-1", "ms_1"] else ", ".join(_INIT_STATE_LABELS.get(g, g) for g in group)
        label = name + (f", {temps[m]:g} K" if len(temps) > 1 else "")
        ax.plot(sigmas, y, styles[m % len(styles)], marker="o", ms=5, lw=2, color=colors[k % len(colors)], label=label)
    ax.axvline(default_sigma, color="#888888", lw=1, ls=":", label=f"default ({default_sigma:g} meV)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Broadening $\sigma$ (meV)")
    ax.set_ylabel(r"$T_1$ (s)" + (f" at {temps[0]:g} K" if len(temps) == 1 else ""))
    ax.set_title(_label(manager).replace("  ", ", "), fontsize=10)
    ax.grid(True, which="major", ls=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
