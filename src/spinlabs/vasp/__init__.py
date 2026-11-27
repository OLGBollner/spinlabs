"""VASP simulation setup: directory trees, perturbed structures, occupations.

High-level functions build ready-to-run simulation folders; thin scripts in
scripts/cluster/ wrap them with argparse CLIs.

The names below are loaded on first use (PEP 562), not when the package is imported: the cluster
runner starts many small ``python -m spinlabs.vasp.<module>`` processes (restart, sublevels, wavecar,
settings), and each used to load pymatgen's VASP I/O, scipy and the tree builder (~1.5 s) for nothing.
"""

from __future__ import annotations

import importlib

# public name -> the submodule that defines it
_LAZY = {
    "init_tree": "init",
    "init_zfs": "init",
    "prepare_relax_to_zfs": "occupations",
    "apply_combined_perturbation": "structures",
    "apply_perturbation": "structures",
    "compare_displacement": "structures",
    "compare_displacement_cli": "structures",
    "load_phonon_data": "structures",
    "load_poscar": "structures",
    "write_perturbed_poscar": "structures",
    "build_perturbation_tree": "trees",
    "default_phonon": "trees",
    "n_modes_from_phonon": "trees",
    "n_tasks_for": "trees",
    "read_outcar_time": "trees",
    "resolve_sym_phonon": "trees",
    "sbatch_time": "trees",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value  # loaded once
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
