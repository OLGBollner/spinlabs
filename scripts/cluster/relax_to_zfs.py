#!/usr/bin/env python3
"""relax_to_zfs.py -- pipeline step: relaxation output -> ZFS run directory.

Thin CLI wrapper; the logic lives in spinlabs.vasp (prepare_relax_to_zfs).
"""

from __future__ import annotations

import sys
from pathlib import Path

from spinlabs.vasp import prepare_relax_to_zfs
from spinlabs.logging_utils import configure_logging


def main() -> int:
    configure_logging()
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit(__doc__)
    failures: list[str] = []
    for zfs_dir in args[1:]:
        failures.extend(prepare_relax_to_zfs(Path(args[0]), Path(zfs_dir)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
