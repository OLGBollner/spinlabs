"""VASP simulation setup: directory trees, perturbed structures, occupations.

High-level functions build ready-to-run simulation folders; thin scripts in
scripts/cluster/ wrap them with argparse CLIs.
"""

from spinlabs.vasp.occupations import prepare_relax_to_zfs
from spinlabs.vasp.structures import (
    apply_combined_perturbation,
    apply_perturbation,
    compare_displacement,
    compare_displacement_cli,
    load_phonon_data,
    load_poscar,
    write_perturbed_poscar,
)
from spinlabs.vasp.trees import (
    build_perturbation_tree,
    build_zfs_tree,
    default_phonon,
    n_modes_from_phonon,
    n_tasks_for,
    read_outcar_time,
    resolve_sym_phonon,
    sbatch_time,
)

__all__ = [
    "apply_combined_perturbation",
    "apply_perturbation",
    "compare_displacement",
    "compare_displacement_cli",
    "build_perturbation_tree",
    "build_zfs_tree",
    "default_phonon",
    "load_phonon_data",
    "load_poscar",
    "n_modes_from_phonon",
    "n_tasks_for",
    "prepare_relax_to_zfs",
    "read_outcar_time",
    "resolve_sym_phonon",
    "sbatch_time",
    "write_perturbed_poscar",
]
