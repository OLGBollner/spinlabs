#!/usr/bin/env python3
"""verify_perturbation_setup.py -- check a run folder before sbatch.

Copied into every perturbation-run folder by create_perturbation_tree.py.
Run it from the all_bands folder; it verifies the INCAR and the prefilled
SLURM script so that problems surface before the job queues, not hours
into the scheduler.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spinlabs.vasp.verify import verify_setup


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the INCAR and SLURM script of this run folder")
    ap.add_argument("--script", default=None, help="run script name (default: the single run_*.sh here)")
    args = ap.parse_args()

    folder = Path(__file__).resolve().parent
    script = args.script
    if script is None:
        candidates = sorted(folder.glob("run_*.sh"))
        if len(candidates) != 1:
            sys.exit("Error: cannot decide which run script to check; pass --script")
        script = candidates[0].name

    problems = verify_setup(folder, script)
    for p in problems:
        print(p)
    if not problems:
        print("OK: setup looks consistent -- ready for sbatch")
    return 1 if any(p.startswith("FAIL") for p in problems) else 0


if __name__ == "__main__":
    sys.exit(main())
