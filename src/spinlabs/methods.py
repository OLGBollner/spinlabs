"""The ZFS calculation methods and every name each has been stored under.

Three methods compute the ground-state ZFS and its changes:

- ``AO`` (all orbitals): the full spin density of every band (formerly ``all_bands``; ground state
  in ``ZFS_AO``, formerly ``ZFS_hyp``).
- ``UO`` (unpaired orbitals): only the unpaired defect bands occupied, spin-corrected by
  spin_correction_scale (formerly ``defect_band_approx``; ground state ``ZFS_UO``, formerly
  ``ZFS_occup``).
- ``full_decont``: the full spin decontamination from the spin sublevels (ground state ``ZFS_full``).

New data are written under the canonical names below; the legacy names of folders, packages,
pack metadata and registry rows are still read (normalize_calc_method, aliases and the *_dir
helpers), so nothing written before has to move.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Method:
    key: str  # canonical calc_method: stored in packages, the registry, folder and file names
    cli: tuple[str, ...]  # --method values; the first is the default spelling
    legacy: tuple[str, ...]  # earlier calc_method values and CLI/metadata spellings, still read
    label: str  # what plots and tables call it
    ground_state: str  # <defect>/<ground_state>: the unperturbed ZFS run
    legacy_ground_state: tuple[str, ...]
    convergence: str  # <defect>/<convergence>: the step-size convergence folder
    legacy_convergence: tuple[str, ...]
    spin_corrected: bool  # the defect-band spin correction (models.spin_correction_scale) applies

    @property
    def basis(self) -> str:
        """Tree basis folder (<order>/pert_<p>/<basis>) and package folder name."""
        return self.key

    @property
    def legacy_basis(self) -> tuple[str, ...]:
        return tuple(name for name in self.legacy if "_" in name)  # all_bands, defect_band_approx

    @property
    def names(self) -> tuple[str, ...]:
        """Every name the method is known by: canonical, CLI and legacy."""
        return (self.key, *self.cli, *self.legacy)


METHODS = {
    "AO": Method(
        key="AO", cli=("all", "AO"), legacy=("all_bands",), label="AO",
        ground_state="ZFS_AO", legacy_ground_state=("ZFS_hyp",),
        convergence="convergence_test_AO", legacy_convergence=("convergence_test_all",),
        spin_corrected=False,
    ),
    "UO": Method(
        key="UO", cli=("unpaired", "UO"), legacy=("defect_band_approx", "approx", "defect_band"), label="UO",
        ground_state="ZFS_UO", legacy_ground_state=("ZFS_occup",),
        convergence="convergence_test_UO", legacy_convergence=("convergence_test_occup",),
        spin_corrected=True,
    ),
    "full_decont": Method(
        key="full_decont", cli=("decont",), legacy=(), label="FD",
        ground_state="ZFS_full", legacy_ground_state=(),
        convergence="convergence_test_decont", legacy_convergence=(),
        spin_corrected=False,
    ),
}
CLI_CHOICES = tuple(name for m in METHODS.values() for name in m.cli)  # all, AO, unpaired, UO, decont

RELAX_DIR = "relax"
LEGACY_RELAX_DIRS = ("relaxation_data",)

_BY_NAME = {name.lower(): m for m in METHODS.values() for name in m.names}


def method(name: str) -> Method:
    """The Method of any of its names (case-blind); ValueError for an unknown one."""
    found = _BY_NAME.get(str(name).strip().lower())
    if found is None:
        raise ValueError(f"unknown calculation method {name!r} (known: {', '.join(CLI_CHOICES)})")
    return found


def normalize_calc_method(name: str | None) -> str | None:
    """The canonical calc_method (AO, UO, full_decont) of any name; None for none, unknown names as given."""
    if name is None or str(name) in ("None", ""):
        return None
    found = _BY_NAME.get(str(name).strip().lower())
    return found.key if found else str(name)


def aliases(name: str) -> tuple[str, ...]:
    """Every calc_method a method has been stored under (for registry queries): canonical and legacy."""
    m = method(name)
    return (m.key, *m.legacy_basis)


def spin_corrected(name: str | None) -> bool:
    """Whether the defect-band spin correction applies to data of this method."""
    found = _BY_NAME.get(str(name).strip().lower()) if name else None
    return bool(found and found.spin_corrected)


def _existing(parent: Path, names: tuple[str, ...]) -> Path:
    """parent/<first name that exists>, else parent/<names[0]> (the name to write)."""
    for name in names:
        if (parent / name).exists():
            return parent / name
    return parent / names[0]


def ground_state_dir(defect: str | Path, name: str) -> Path:
    """<defect>/ZFS_AO (or the legacy ZFS_hyp that exists), ZFS_UO / ZFS_occup, ZFS_full."""
    m = method(name)
    return _existing(Path(defect), (m.ground_state, *m.legacy_ground_state))


def basis_dir(pert_dir: str | Path, name: str) -> Path:
    """<order>/pert_<p>/AO (or the legacy all_bands that exists), UO / defect_band_approx, full_decont."""
    m = method(name)
    return _existing(Path(pert_dir), (m.basis, *m.legacy_basis))


def convergence_dir(defect: str | Path, name: str) -> Path:
    """<defect>/convergence_test_AO (or the legacy convergence_test_all that exists), ..."""
    m = method(name)
    return _existing(Path(defect), (m.convergence, *m.legacy_convergence))


def relax_dir(defect: str | Path) -> Path:
    """<defect>/relax, or the legacy relaxation_data that exists."""
    return _existing(Path(defect), (RELAX_DIR, *LEGACY_RELAX_DIRS))


def is_ground_state_dir(name: str) -> bool:
    """Whether a folder name is a ground-state folder of some method (new or legacy name)."""
    return any(name in (m.ground_state, *m.legacy_ground_state) for m in METHODS.values())


def defect_folder_of(path: str | Path) -> Path | None:
    """The defect folder above ``path``: the first parent holding a ground-state folder of any method."""
    for parent in Path(path).resolve().parents:
        if any(p.is_dir() and is_ground_state_dir(p.name) for p in parent.iterdir()):
            return parent
    return None


def method_of_folder(name: str) -> str | None:
    """The method a ground-state, basis or convergence folder name belongs to (new or legacy), else None."""
    for m in METHODS.values():
        if name in (m.ground_state, *m.legacy_ground_state, m.basis, *m.legacy_basis,
                    m.convergence, *m.legacy_convergence):
            return m.key
    return None


# substrings that name a method in a path, most specific first (legacy names included)
_PATH_PATTERNS = (
    ("full_decont", r"full_decont|zfs_full|test_decont"),
    ("UO", r"defect_band_approx|zfs_occup|zfs_uo|test_occup|test_uo|(?<![a-z])uo(?![a-z])"),
    ("AO", r"all_bands|zfs_hyp|zfs_ao|test_all|test_ao|(?<![a-z])ao(?![a-z])"),
)


def method_from_path(path: str | Path) -> str | None:
    """The method a file or folder path names (package file, tree, ground-state or convergence folder),
    new or legacy spelling; None when it names none or more than one."""
    text = str(path).lower()
    found = {key for key, pattern in _PATH_PATTERNS if re.search(pattern, text)}
    if "full_decont" in found:
        return "full_decont"
    return found.pop() if len(found) == 1 else None
