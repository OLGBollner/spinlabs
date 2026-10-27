#!/usr/bin/env python3
"""plot_pert_convergence.py -- thin wrapper around spinlabs.plot convergence.

The implementation moved into the package (src/spinlabs/convergence.py); this
script stays for compatibility with existing cluster workflows. Same flags.

Usage:
    python plot_pert_convergence.py pert_runs.npz [-m MODE] [-o out.png] [--quad] [--deviation] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running from anywhere: find the spinlabs package on the repo's src/
_repo = Path(__file__).resolve().parents[2]
if (_repo / "src").is_dir():
    sys.path.insert(0, str(_repo / "src"))

from spinlabs.convergence import plot_pert_convergence  # noqa: E402
from spinlabs.logging_utils import configure_logging  # noqa: E402


def main() -> int:
    configure_logging()
    ap = argparse.ArgumentParser(
        description="Plot ZFS principal components vs perturbation scale with linear-fit residuals"
    )
    ap.add_argument("npz", type=Path, help="npz from pack_perturbation_runs.py")
    ap.add_argument("-m", "--mode", type=int, action="append", help="plot only this mode (repeatable; default: all)")
    ap.add_argument("-o", "--output", type=Path, default=None, help="output png (default: <npz stem>_convergence.png)")
    ap.add_argument(
        "--quad", action="store_true", help="also fit y = a + b x + c x^2 and report the second-order derivative 2c"
    )
    ap.add_argument(
        "--deviation",
        action="store_true",
        help="plot D(Q) - D(Q=0) instead of raw D, so the response fills the y-range",
    )
    ap.add_argument(
        "--csv", type=Path, default=None, help="write slopes (and quad coefficients with --quad) to this csv"
    )
    ap.add_argument("--no-show", action="store_true", help="don't call plt.show() (useful on headless cluster)")
    ap.add_argument(
        "--drop-anchor",
        action="store_true",
        help="exclude the Q=0 anchor from fits and residuals (curves still extrapolate to Q=0)",
    )
    args = ap.parse_args()
    plot_pert_convergence(
        args.npz,
        output_path=args.output,
        mode=args.mode,
        quad=args.quad,
        deviation=args.deviation,
        csv_path=args.csv,
        show=not args.no_show,
        drop_anchor=args.drop_anchor,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
