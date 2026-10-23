#!/usr/bin/env python3
"""get_n_modes.py <phonon.npz> [--sym] -- the modes a tree runs, as its run folders number them.

Thin wrapper around spinlabs.vasp.modes; see that module for the numbering.
"""

import sys

from spinlabs.vasp.modes import main

if __name__ == "__main__":
    sys.exit(main())
