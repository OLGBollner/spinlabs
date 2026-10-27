#!/usr/bin/env python3
"""create_zfs_tree.py -- create the VASP directory tree for a point defect.

Thin CLI wrapper; the logic lives in spinlabs.vasp.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spinlabs.vasp import build_zfs_tree
from spinlabs.logging_utils import configure_logging


def main() -> int:
    configure_logging()
    ap = argparse.ArgumentParser(description="Create the VASP directory tree for a point-defect calculation.")
    ap.add_argument("input", type=Path, help="input folder with data/ and template/relax/")
    ap.add_argument(
        "-o", "--output", type=Path, default=None, help="output directory (default: name of the input folder)"
    )
    args = ap.parse_args()
    try:
        build_zfs_tree(args.input, out=args.output)
    except (FileNotFoundError, FileExistsError) as e:
        sys.exit(f"Error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
