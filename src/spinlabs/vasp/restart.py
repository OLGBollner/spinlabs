"""Is a run's own CHGCAR usable to restart it from?

The perturbation INCAR reads a CHGCAR (ICHARG = 1). A run that did not finish can start again from
the CHGCAR it left, instead of the ground-state one in input/, but only if that file is whole: VASP
writes CHGCAR at the end of the electronic minimization, so a run cut off mid-SCF leaves none, an
empty one, or one cut short. Reading such a file would stop VASP.

    python -m spinlabs.vasp.restart <CHGCAR>     exit 0: complete, 1: not (the reason on stdout)
"""

from __future__ import annotations

import sys
from pathlib import Path


def chgcar_problem(path: str | Path) -> str | None:
    """None if ``path`` is a CHGCAR whose total-density block is complete, else why it is not.

    Layout: comment, scale, three lattice vectors, (element names,) atom counts, (Selective
    dynamics,) Direct/Cartesian, one line per atom, a blank line, the grid NGX NGY NGZ, then
    NGX*NGY*NGZ values. Only that first block is needed to restart.
    """
    path = Path(path)
    try:
        if path.stat().st_size == 0:
            return "empty"
        with open(path, encoding="utf-8", errors="replace") as fh:
            header = [fh.readline() for _ in range(7)]  # comment, scale, 3 lattice vectors, 2 more
            if not header[6]:
                return "header cut short"
            line6 = header[5].split()
            if line6 and line6[0].isdigit():  # VASP 4: atom counts, then the coordinate mode
                counts, mode = line6, header[6]
            else:  # VASP 5: element names, atom counts, then the coordinate mode
                counts, mode = header[6].split(), fh.readline()
            try:
                n_atoms = sum(int(c) for c in counts)
            except ValueError:
                return "no atom counts in the header"
            if mode.strip().lower().startswith("s"):  # Selective dynamics comes before Direct/Cartesian
                mode = fh.readline()
            if not mode.strip():
                return "header cut short"
            for _ in range(n_atoms):
                if not fh.readline():
                    return "positions cut short"
            grid_line = fh.readline()
            while grid_line and not grid_line.strip():
                grid_line = fh.readline()
            try:
                n_points = 1
                for g in grid_line.split()[:3]:
                    n_points *= int(g)
                if len(grid_line.split()) < 3:
                    raise ValueError
            except ValueError:
                return "no grid line"
            seen = 0
            for line in fh:
                seen += len(line.split())
                if seen >= n_points:
                    return None
            return f"density cut short ({seen} of {n_points} values)"
    except OSError as exc:
        return f"unreadable ({exc.strerror or exc})"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m spinlabs.vasp.restart <CHGCAR>")
        return 2
    problem = chgcar_problem(args[0])
    print("complete" if problem is None else problem)
    return 0 if problem is None else 1


if __name__ == "__main__":
    sys.exit(main())
