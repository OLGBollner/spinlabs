"""The INCAR of a spin-sublevel run, derived at run time from the SCF INCAR of its run directory.

A full-decontamination run directory (a tree's run, or the ground state ZFS_full) runs the SCF of the
m_s = S state, then one ALGO = None ZFS run per spin sublevel from its WAVECAR. The runner writes each
sublevel's INCAR from the SCF INCAR just before running it:

    python -m spinlabs.vasp.sublevels INCAR ms_0_127 > ms_0_127/INCAR

so a setting changed in the SCF INCAR (e.g. the HSE06 tags) reaches every sublevel. The sublevel's own
settings (trees.decont_sublevel_incar: ALGO = None, ISTART = 1, ICHARG = 0, its NUPDOWN and occupations,
LDMATRIX on, LWAVE/LCHARG off) are set over it; its occupations come from the SCF's m_s = S ones
(occupations.spin_sublevels).
"""

from __future__ import annotations

import sys
from pathlib import Path


def sublevel_incar(scf_incar: str, name: str) -> str:
    """The INCAR of spin sublevel ``name`` (e.g. ms_1, ms_0_127) of the SCF INCAR ``scf_incar``.

    ValueError when the SCF's occupations give no such sublevel (they were changed since the run
    directory's sublevel folders were set up).
    """
    from spinlabs.vasp.occupations import parse_occupations, spin_sublevels
    from spinlabs.vasp.trees import decont_sublevel_incar

    levels = {level.name: level for level in spin_sublevels(*parse_occupations(scf_incar))}
    if name not in levels:
        raise ValueError(f"the SCF INCAR's occupations give the sublevels {', '.join(levels) or 'none'}, not {name}")
    return decont_sublevel_incar(scf_incar, levels[name])


def main(argv: list[str] | None = None) -> int:
    """``python -m spinlabs.vasp.sublevels SCF_INCAR NAME``: the sublevel's INCAR on stdout; 1 on an error."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: python -m spinlabs.vasp.sublevels SCF_INCAR NAME", file=sys.stderr)
        return 2
    try:
        sys.stdout.write(sublevel_incar(Path(args[0]).read_text(), args[1]))
    except (OSError, ValueError) as error:
        print(f"sublevel INCAR of {args[1]} from {args[0]}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
