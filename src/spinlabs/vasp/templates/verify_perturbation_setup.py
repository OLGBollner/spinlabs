#!/usr/bin/env python3
"""verify_perturbation_setup.py -- check a run folder before sbatch.

Copied into every perturbation-run folder by create_perturbation_tree.py.
Run it from the all_bands folder; it verifies the INCAR and the prefilled
SLURM script so that problems surface before the job queues, not hours
into the scheduler. Then it counts the finished runs and offers to fit the
script's time limit to the runs that are left. All of it lives in
spinlabs.vasp.verify.main, so changes there reach every tree without
copying this file again.
"""

import sys
from pathlib import Path

from spinlabs.vasp.verify import main

if __name__ == "__main__":
    sys.exit(main(Path(__file__).resolve().parent))
