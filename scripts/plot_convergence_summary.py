"""One convergence summary figure per defect from the packs in a data folder.

Thin wrapper; the figure is spinlabs.convergence_summary.plot_convergence_summary, which
``spinlabs plot convergence`` makes by default. This picks the newest pack per defect and
method by file name (``<defect>_<date>_conv_test_<all|occup>.npz``) instead of the registry.
"""

import re
from argparse import ArgumentParser as Parser
from pathlib import Path

from spinlabs.convergence_summary import plot_convergence_summary
from spinlabs.logging_utils import configure_logging

PACK = re.compile(r"(?P<defect>.+)_(?P<date>\d{8})_conv_test_(?P<method>all|occup)\.npz")


def find_packs(data_dir):
    """Newest pack per (defect, method) under data_dir, as {defect: {method: path}}."""
    packs = {}
    for path in sorted(Path(data_dir).rglob("*_conv_test_*.npz"), key=lambda p: p.name):
        m = PACK.fullmatch(path.name)
        if m:  # sorted by name, so a later date replaces an earlier one
            packs.setdefault(m["defect"], {})[m["method"]] = path
    return packs


if __name__ == "__main__":
    parser = Parser("Plots one convergence figure per defect: finite differences and SNR against step size.")
    parser.add_argument("--data", default="data", help="Folder with the <defect>_<date>_conv_test_*.npz packs")
    parser.add_argument("--defect", nargs="+", help="Defects to plot, e.g. NV_512 (default: every pack found)")
    parser.add_argument("-o", "--output-dir", default=".", help="Folder for <defect>_convergence_summary.png")
    args = parser.parse_args()
    configure_logging()

    packs = find_packs(args.data)
    if not packs:
        raise SystemExit(f"No *_conv_test_*.npz packs in {args.data}")
    for defect in args.defect or sorted(packs):
        if defect not in packs:
            raise SystemExit(f"No convergence packs for {defect} in {args.data} (found: {', '.join(sorted(packs))})")
        by_method = packs[defect]
        plot_convergence_summary(
            [by_method[m] for m in ("all", "occup") if m in by_method],
            Path(args.output_dir) / f"{defect}_convergence_summary.png",
        )
