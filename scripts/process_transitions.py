#!/usr/bin/env python3
"""
Calculate and plot spin transition rates versus temperature.

Modes:
  --calc      compute transition rates from a ZFS data file and save to .npz
  --plot      read previously saved .npz files and produce a line or stacked-area plot

The two modes can be combined (e.g. --calc --plot) to compute and immediately visualise.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# -----------------------------------------------------------------------------
#  Constants and configuration
# -----------------------------------------------------------------------------
COLORS = {
    "direct": "#2176AE",  # steel blue
    "first-order-raman": "#D64B20",  # burnt orange-red
    "second-order-raman": "#F4C416",  # gold
}
FILL_ALPHA = 0.60
EDGE_ALPHA = 0.95
EDGE_LW = 0.7
TOTAL_COLOR = "black"
TOTAL_LW = 1.4
FIG_SIZE = (6.0, 5.0)  # for stacked-area plots
LINE_FIG_SIZE = (6, 5)  # for line plots

# Process labels for the stacked area legend (in order: first, second, two-phonon)
PROCESS_LABELS = [
    "Direct (1‑phonon)",
    "1st‑order Raman",
    "2nd‑order Raman",
]

# Labels for individual transitions (used in line plots)
TRANSITION_LABELS = {
    "first_order": {
        "0_1": r"$\Gamma_{1, 0\pm}^{(1)}$",
        "1_-1": r"$\Gamma_{1, +-}^{(1)}$",
        "0_0": r"$\Gamma_{1, 00}^{(1)}$",
    },
    "second_order": {
        "0_1": r"$\Gamma_{2, 0\pm}^{(1)}$",
        "1_-1": r"$\Gamma_{2, +-}^{(1)}$",
        "0_0": r"$\Gamma_{2, 00}^{(1)}$",
    },
    "two_phonon": {
        "0_1": r"$\Gamma_{1, 0\pm}^{(2)}$",
        "1_-1": r"$\Gamma_{1, +-}^{(2)}$",
        "0_0": r"$\Gamma_{1, 00}^{(2)}$",
    },
}

LINE_STYLES = ["-", "--", "-.", ":"]

SPIN_FORMALISM_LABELS = {
    "all_bands": r"$D_{\mathrm{cont}}$",
    "defect_band_approx": r"$D_{\mathrm{corr}}$",
}


# -----------------------------------------------------------------------------
#  Plot style (shared by both plot types)
# -----------------------------------------------------------------------------
def set_plot_style():
    """Apply rcParams so that line and stacked plots have the same appearance."""
    plt.rcParams.update(
        {
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 14,
        }
    )


# -----------------------------------------------------------------------------
#  Stacked filled area plot on log‑log axes
# -----------------------------------------------------------------------------
def plot_stacked_rates(T, rates, labels, ax=None):
    """
    Draw a stacked filled-area plot on log-log axes.

    Parameters
    ----------
    T      : 1-D array-like, temperature values (K)
    rates  : list of 1-D arrays, one per process, in bottom-to-top order
    labels : list of str, one per process
    ax     : optional existing Axes; if None a new figure is created

    Returns
    -------
    fig, ax
    """
    process_colors = [COLORS["direct"], COLORS["first-order-raman"], COLORS["second-order-raman"]]

    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
    else:
        fig = ax.get_figure()

    T = np.asarray(T, dtype=float)

    # --- stacked fill_between ------------------------------------------------
    cumulative = np.zeros_like(T)
    for rate, label, color in zip(rates, labels, process_colors):
        rate = np.asarray(rate, dtype=float)
        upper = cumulative + rate

        ax.fill_between(
            T,
            cumulative,
            upper,
            color=color,
            alpha=FILL_ALPHA,
            label=label,
            linewidth=0,
            zorder=2,
        )
        # thin border on top edge of each band for visual clarity
        ax.plot(T, upper, color=color, lw=EDGE_LW, alpha=EDGE_ALPHA, zorder=3)

        cumulative = upper.copy()

    # --- total rate line -----------------------------------------------------
    ax.plot(
        T,
        cumulative,
        color=TOTAL_COLOR,
        lw=TOTAL_LW,
        label="Total",
        zorder=4,
        linestyle="--",
    )

    # --- axes formatting -----------------------------------------------------
    ax.set_xlim(1e-1, T[-1])
    ax.set_ylim(1e-9, 1e5)

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Relaxation rate $\ (\mathrm{s^{-1}})$")

    ax.legend(loc="upper left", framealpha=0.88, edgecolor="0.7")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.30)

    fig.tight_layout()
    return fig, ax


# =============================================================================
#  Calculation helpers
# =============================================================================
def compute_rates(data_path, two_phonon_path, t_start, t_end, t_step):
    """
    Compute transition rates for a range of temperatures and return results
    together with metadata and temperature array.
    """
    try:
        from spinlabs.transition_rate import TransitionRate
        from spinlabs.constants import CONSTANTS
    except ImportError as exc:
        raise ImportError(
            "Could not import TransitionRate or CONSTANTS. Ensure the spinlabs package is in the Python path."
        ) from exc

    calculator = TransitionRate(str(data_path), two_phonon_path)

    # Build temperature grid: fine near 0 K, coarser above
    low_range = np.arange(0, 1, t_step / 100)
    high_range = np.arange(1 + t_step, t_end + t_step, t_step)
    temperatures = np.concatenate([low_range, high_range])

    # Pre‑compute spectral densities
    omega, J_0_pm, J_p_m, J_0_0 = calculator.get_spectral_density(res=0.01, sigma=7.5)
    if two_phonon_path:
        omega_x, omega_y, J2_0_pm, J2_p_m, J2_0_0 = calculator.get_2d_spectral_density(res=0.5, sigma=7.5)
    else:
        omega_x = omega_y = J2_0_pm = J2_p_m = J2_0_0 = None

    zfs = calculator.data["zfs"] / CONSTANTS["meV2J"]

    results = {
        "first_order": {"0_1": [], "1_-1": []},
        "second_order": {"0_1": [], "1_-1": []},
        "two_phonon": {"0_1": [], "1_-1": []},
    }
    directional_results = {
        "0_to_1": [],
        "0_to_-1": [],
        "1_to_0": [],
        "-1_to_0": [],
        "1_to_-1": [],
        "-1_to_1": [],
    }
    valid_temps = []

    print(f"Computing transition rates for {len(temperatures)} temperature points...")

    for T in temperatures:
        calculator.compute_transition_rates(T, omega, J_0_pm, J_p_m, J_0_0, zfs)
        if two_phonon_path:
            calculator.compute_two_phonon_rates(T, omega_x, omega_y, J2_0_pm, J2_p_m, J2_0_0, zfs)
        total_rates = calculator.get_total_rates()
        directional_rates = calculator.get_directional_rates()

        if total_rates and directional_rates:
            for direction, rate in directional_rates.items():
                directional_results[direction].append(rate)

            for order in ["first_order", "second_order", "two_phonon"]:
                for transition, rate in total_rates[order].items():
                    results[order][transition].append(rate)

            valid_temps.append(T)

    meta_data = {
        "cell_size": calculator.data["cell_size"],
        "defect": calculator.data["defect"],
        "calc_method": calculator.data["calc_method"],
        "pert_scale": calculator.data["pert_scale"],
    }

    return results, directional_results, valid_temps, meta_data


def save_rates(results, directional_results, valid_temps, meta_data):
    """Write computed rates and directional rates to .npz files."""
    save_dir = Path("rates")
    save_dir.mkdir(exist_ok=True)

    base_name = (
        f"{meta_data['defect']}_{meta_data['cell_size']}_rates_{meta_data['calc_method']}_{meta_data['pert_scale']}"
    )
    total_path = save_dir / base_name
    np.savez(total_path.with_suffix(".npz"), **meta_data, **results, temperatures=valid_temps)
    print(f"Saved transition rates -> {total_path}.npz")

    dir_name = base_name.replace("_rates_", "_directional_rates_")
    dir_path = save_dir / dir_name
    np.savez(dir_path.with_suffix(".npz"), **meta_data, **directional_results, temperatures=valid_temps)
    print(f"Saved directional rates -> {dir_path}.npz")


# =============================================================================
#  Line plot
# =============================================================================
def plot_line_rates(data_files, orders_to_plot, log_scale, output_arg, show):
    """
    Create a line plot of relaxation rates from one or more .npz data files.

    Each file is drawn with a distinct line style; transitions share a common colour
    across files. An extra legend shows the data source (defect, cell size, formalism).
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=LINE_FIG_SIZE)

    if log_scale:
        ax.set_yscale("log")
        ax.set_xscale("log")

    color_for_pair = {}  # (order, transition) -> colour
    first_file = True
    data_arrays = [np.load(file, allow_pickle=True) for file in data_files]

    for i, data in enumerate(data_arrays):
        valid_temps = data["temperatures"]
        results = {k: item[()] for k, item in data.items() if "order" in k or "phonon" in k}

        for order in orders_to_plot:
            if order not in results:
                continue
            for transition, rate in results[order].items():
                label = None
                if first_file:
                    label = TRANSITION_LABELS.get(order, {}).get(transition, f"{order} {transition}")

                ls = LINE_STYLES[i % len(LINE_STYLES)]
                pair = (order, transition)

                if pair in color_for_pair:
                    ax.plot(valid_temps, rate, color=color_for_pair[pair], linestyle=ls, linewidth=2, label=label)
                else:
                    (line,) = ax.plot(valid_temps, rate, linestyle=ls, linewidth=2, label=label)
                    color_for_pair[pair] = line.get_color()

        first_file = False

    # ---- Build a second legend for line styles (data source) -----------------
    from matplotlib.lines import Line2D

    all_defects = [str(d["defect"]) for d in data_arrays]
    all_sizes = [int(d["cell_size"]) for d in data_arrays]
    all_formalisms = [str(d["calc_method"]) for d in data_arrays]

    show_defect = len(set(all_defects)) > 1
    show_size = len(set(all_sizes)) > 1
    show_formalism = len(set(all_formalisms)) > 1

    style_handles = []
    for i, data in enumerate(data_arrays):
        parts = []
        if show_defect:
            parts.append(str(data["defect"]))
        if show_size:
            parts.append(str(data["cell_size"]))
        if show_formalism:
            fm = str(data["calc_method"])
            parts.append(f"({SPIN_FORMALISM_LABELS.get(fm, fm)})")

        if not parts:  # fallback when all sources are identical
            fm = str(data["calc_method"])
            parts.append(f"{data['defect']} {data['cell_size']} ({SPIN_FORMALISM_LABELS.get(fm, fm)})")
        label = " ".join(parts)

        style_handles.append(Line2D([0], [0], color="black", linestyle=LINE_STYLES[i % len(LINE_STYLES)], label=label))

    leg1 = ax.legend(loc="best")
    _ = ax.legend(handles=style_handles, loc="upper center")
    ax.add_artist(leg1)  # keep both legends

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Transition Rate (s$^{-1}$)")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_ylim(1e-3, 1e5)
    ax.set_xlim(left=1e-1)
    fig.tight_layout()

    # ---- Save or show -------------------------------------------------------
    if output_arg or not show:
        save_dir = Path("figures")
        save_dir.mkdir(exist_ok=True)
        filename = build_plot_filename(data_arrays[-1], orders_to_plot, log_scale, output_arg)
        full_path = save_dir / filename
        fig.savefig(full_path, dpi=300)
        print(f"Plot saved to {full_path}")
    else:
        plt.show()


def build_plot_filename(data, orders_to_plot, log_scale, output_arg):
    """Construct a sensible filename for saving the figure."""
    if output_arg:
        return output_arg

    defect = data["defect"]
    calc_method = data["calc_method"]
    pert_scale = data["pert_scale"]
    t_min = int(np.min(data["temperatures"]))
    t_max = int(np.max(data["temperatures"]))
    scale_str = "_log" if log_scale else ""

    filename = f"{defect}_{'_'.join(orders_to_plot)}_rates_{calc_method}_{t_min}-{t_max}K{scale_str}_{pert_scale}.png"
    return filename


# =============================================================================
#  Stacked area plot (single file, two figures)
# =============================================================================
def plot_stacked_area_from_file(data_file, output_base, show, log_scale):
    """
    Read a single .npz file and produce two stacked area plots:
      - Figure 1: transition 0_1
      - Figure 2: transition 1_-1

    Parameters
    ----------
    data_file   : str or Path
    output_base : str or None
        Base name for saved files; if None, a name is auto‑generated.
    show        : bool
        If True, display the figures interactively.
    log_scale   : bool
    """
    set_plot_style()

    data = np.load(data_file, allow_pickle=True)
    T = data["temperatures"]
    results = {k: item[()] for k, item in data.items() if "order" in k or "phonon" in k}

    # Order of processes: first_order, second_order, two_phonon
    order_keys = ["first_order", "second_order", "two_phonon"]

    # Build the two lists of rates
    rates_single = [results[order]["0_1"] for order in order_keys]
    rates_double = [results[order]["1_-1"] for order in order_keys]

    # Figure 1 – single transition
    fig1, ax1 = plot_stacked_rates(T, rates_single, PROCESS_LABELS)

    # Figure 2 – double transition
    fig2, ax2 = plot_stacked_rates(T, rates_double, PROCESS_LABELS)

    # Construct base filename for saving
    if output_base is None:
        defect = str(data["defect"])
        calc_method = str(data["calc_method"])
        pert_scale = str(data["pert_scale"])
        cell_size = str(data["cell_size"])
        t_min = int(np.min(T))
        t_max = int(np.max(T))
        output_base = f"{defect}_{cell_size}_stacked_rates_{calc_method}_{t_min}-{t_max}K_{pert_scale}"
    else:
        output_base = Path(output_base).stem  # strip extension if given

    save_dir = Path("figures")
    save_dir.mkdir(exist_ok=True)
    if log_scale:
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_xscale("log")
        ax2.set_yscale("log")

    if show:
        plt.show()
    else:
        # Save both figures
        fn1 = save_dir / f"{output_base}_single.png"
        fn2 = save_dir / f"{output_base}_double.png"
        fig1.savefig(fn1, dpi=300)
        fig2.savefig(fn2, dpi=300)
        print(f"Plots saved to:\n  {fn1}\n  {fn2}")


# =============================================================================
#  Argument parsing
# =============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate and plot spin transition rates vs Temperature.")
    parser.add_argument(
        "data_file",
        type=str,
        nargs="+",
        help="Path to data file(s) (.npz). For --calc provide exactly one; for --plot one or more.",
    )

    calc_group = parser.add_argument_group("Calculation options")
    calc_group.add_argument("--calc", action="store_true", help="Calculate transition rates.")
    calc_group.add_argument("--two-phonon", type=str, metavar="FILE", help="Path to two‑phonon ZFS data (.npz).")
    calc_group.add_argument("--t-start", type=float, default=0.0, help="Start temperature (K) (default: 0).")
    calc_group.add_argument("--t-end", type=float, default=300.0, help="End temperature (K) (default: 300).")
    calc_group.add_argument("--t-step", type=float, default=10.0, help="Temperature step (K) (default: 10).")

    plot_group = parser.add_argument_group("Plotting options")
    plot_group.add_argument("--plot", action="store_true", help="Plot the rates (from .npz files).")
    plot_group.add_argument("--save", action="store_true", help="Save the plot to file (implied if --output is given).")
    plot_group.add_argument("-o", "--output", type=str, help="Output filename base (e.g., rates.png).")
    plot_group.add_argument("--log", action="store_true", help="Use logarithmic scale for both axes.")
    plot_group.add_argument("--first-order", action="store_true", help="Plot first‑order Raman contributions.")
    plot_group.add_argument("--second-order", action="store_true", help="Plot second‑order Raman contributions.")
    plot_group.add_argument("--second-phonon", action="store_true", help="Plot two‑phonon (Raman) contributions.")

    return parser.parse_args()


def determine_orders_to_plot(args):
    """Return list of order keys to plot based on command‑line flags."""
    orders = []
    if args.first_order:
        orders.append("first_order")
    if args.second_order:
        orders.append("second_order")
    if args.second_phonon:
        orders.append("two_phonon")
    if not orders:
        raise ValueError("No rate type specified. Use --first-order, --second-order, and/or --second-phonon.")
    return orders


# =============================================================================
#  Main dispatcher
# =============================================================================
def main():
    args = parse_arguments()

    # ---------- Calculation mode ----------
    if args.calc:
        if len(args.data_file) != 1:
            print("Error: --calc requires exactly one data file.")
            sys.exit(1)

        data_path = Path(args.data_file[0])
        if not data_path.exists():
            print(f"Error: Data file '{data_path}' not found.")
            sys.exit(1)

        two_phonon_path = None
        if args.two_phonon:
            two_phonon_path = Path(args.two_phonon)
            if not two_phonon_path.exists():
                print(f"Error: Two‑phonon data file '{two_phonon_path}' not found.")
                sys.exit(1)

        results, directional, temps, meta = compute_rates(
            data_path, two_phonon_path, args.t_start, args.t_end, args.t_step
        )
        save_rates(results, directional, temps, meta)

    # ---------- Plotting mode ----------
    if args.plot:
        orders = determine_orders_to_plot(args)

        # Decide between stacked area (only when all three orders are present
        # and exactly one data file is given) and line plot otherwise.
        if set(orders) == {"first_order", "second_order", "two_phonon"} and len(args.data_file) == 1:
            plot_stacked_area_from_file(
                data_file=args.data_file[0],
                output_base=args.output,
                show=not (args.save or args.output),
                log_scale=args.log,
            )
        else:
            plot_line_rates(
                data_files=args.data_file,
                orders_to_plot=orders,
                log_scale=args.log,
                output_arg=args.output if args.save else None,
                show=not (args.save or args.output),
            )


if __name__ == "__main__":
    main()
