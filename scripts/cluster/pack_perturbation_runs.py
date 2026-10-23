#!/usr/bin/env python3
"""pack_perturbation_runs.py -- collect perturbation runs into one .npz.

Thin wrapper around spinlabs.pack_convergence. See that module for the
packing logic and the npz layout (modes, perts, tensors, energies and a
JSON-string ``metadata`` key with defect/cell).
"""

import sys

from spinlabs.pack_convergence import main

if __name__ == "__main__":
    sys.exit(main())
