#!/usr/bin/env python3
"""
Dedicated Plotting Script for Defect Spin-Phonon Pipeline Results.

Reads all results stored in a run directory and easily creates publication-quality plots:
  - Coupling coefficients & spectral functions F(w)
  - Transition rates vs temperature (stacked area charts & lines)
  - T_1 relaxation times vs temperature

Usage examples:
  # Plot everything from a run folder:
  python scripts/plot_results.py runs/NV_64_all_bands_1d_20260903_120000/

  # Plot only spin-phonon coupling:
  python scripts/plot_results.py runs/NV_64_all_bands_1d_20260903_120000/ --coupling

  # Plot only T_1 relaxation times:
  python scripts/plot_results.py runs/NV_64_all_bands_1d_20260903_120000/ --t1

  # Compare T_1 across multiple runs:
  python scripts/plot_results.py runs/run_NV_64 runs/run_NV_128 --t1 --show
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from spinlabs.plotter import (
    plot_t1,
    plot_run_coupling,
    plot_run_rates,
    plot_run_t1,
)


def main():
    parser = argparse.ArgumentParser(description="Plot results from spin-phonon pipeline run folders.")
    parser.add_argument(
        "run_dirs",
        type=str,
        nargs="+",
        help="Path to one or more run directories (e.g. runs/NV_64_all_bands_1d_...).",
    )

    # Plot selectors
    parser.add_argument(
        "-c",
        "--coupling",
        action="store_true",
        help="Plot spin-phonon coupling coefficients.",
    )
    parser.add_argument(
        "-r",
        "--rates",
        action="store_true",
        help="Plot transition rates vs temperature.",
    )
    parser.add_argument(
        "-t",
        "--t1",
        action="store_true",
        help="Plot T1 relaxation times vs temperature.",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Plot all graph types (default if none selected).",
    )

    # Output options
    parser.add_argument("-o", "--out_dir", type=str, help="Custom output directory for figures.")
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Image format.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Resolution for raster images.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")

    args = parser.parse_args()

    # Default to all plots if no specific flag chosen
    do_all = args.all or (not args.coupling and not args.rates and not args.t1)
    plot_coupling = do_all or args.coupling
    plot_rates = do_all or args.rates
    do_t1 = do_all or args.t1

    run_paths = [Path(p) for p in args.run_dirs]

    # If multiple run directories provided and T1 is requested, also create a comparison plot
    if len(run_paths) > 1 and do_t1:
        comp_out = Path(args.out_dir) if args.out_dir else run_paths[0].parent
        comp_out.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Generating Comparison Plots Across {len(run_paths)} Runs ---")
        plot_t1(run_paths, output_path=comp_out / f"t1_comparison.{args.format}")

    for run_dir in run_paths:
        if not run_dir.is_dir():
            print(f"Warning: {run_dir} is not a valid directory – skipping.")
            continue

        fig_out = Path(args.out_dir) if args.out_dir else (run_dir / "figures")
        fig_out.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Generating Plots for Run: {run_dir.name} ---")
        if plot_coupling:
            plot_run_coupling(run_dir, fig_out, args.format, args.dpi, args.show)
        if plot_rates:
            plot_run_rates(run_dir, fig_out, args.format, args.dpi, args.show)
        if do_t1:
            plot_run_t1(run_dir, fig_out, args.format, args.dpi, args.show)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
