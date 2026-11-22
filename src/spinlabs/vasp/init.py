"""Set up a new defect: its folder tree (``spinlabs init tree``) and its ground-state ZFS runs (``init zfs``).

``init_tree`` makes ``<defect>/`` from a folder of inputs, searched recursively (the shallowest file of
each name wins):

    <defect>/
        relax/     POSCAR (or structure.vasp as POSCAR), INCAR, KPOINTS, POTCAR: the relaxation's inputs
                   (no job script: add the relaxation's own)
        data/      <inputs>/data/ and any phonopy / phonon file found
        ZFS_AO/    seeded with the relaxation's inputs; filled by init_zfs once the relaxation ran
        ZFS_UO/    the same

``init_zfs`` fills a method's ground-state folder from the finished relaxation (CONTCAR, CHGCAR,
EIGENVAL, ...; occupations.prepare_relax_to_zfs): ZFS_AO with the full occupations, ZFS_UO with only
the unpaired orbitals occupied, ZFS_full (the full spin decontamination) with the SCF and one
ALGO=None run per spin sublevel. Each gets run_ground_state.sh to submit on its own.
"""

from __future__ import annotations

import shutil
from fnmatch import fnmatch
from pathlib import Path

from spinlabs.logging_utils import get_logger, section
from spinlabs.methods import METHODS, RELAX_DIR, ground_state_dir, normalize_calc_method
from spinlabs.utils import resolve_path

LOGGER = get_logger(__name__)

REQUIRED_INPUTS = ("INCAR", "KPOINTS", "POTCAR")  # plus POSCAR or structure.vasp
# phonon data in the inputs folder: copied to data/, not to relax/
PHONON_PATTERNS = ("phonopy*.yaml", "phonon*.yaml", "phonons*.yaml", "phonon*.npz", "phonons*.npz",
                   "FORCE_SETS", "FORCE_CONSTANTS", "force_constants.hdf5")
SEED_FILES = ("INCAR", "POSCAR", "KPOINTS", "POTCAR")  # what ZFS_AO / ZFS_UO start with


def _is_phonon_file(name: str) -> bool:
    return any(fnmatch(name, pattern) for pattern in PHONON_PATTERNS)


def init_tree(defect: str, inputs: str | Path, parent: str | Path = ".", force: bool = False) -> Path:
    """Create ``<parent>/<defect>/`` from the input folder ``inputs`` (see the module docstring).

    FileNotFoundError when an input is missing, FileExistsError when the folder exists and not ``force``
    (``force`` replaces it). Returns the defect folder.
    """
    inputs = resolve_path(inputs)
    if not inputs.is_dir():
        raise FileNotFoundError(f"input folder not found: {inputs}")

    found: dict[str, Path] = {}
    for p in sorted(inputs.rglob("*"), key=lambda p: len(p.parts)):  # shallowest wins on duplicates
        if p.is_file():
            found.setdefault(p.name, p)

    missing = [name for name in REQUIRED_INPUTS if name not in found]
    structure = found.get("POSCAR") or found.get("structure.vasp")
    if structure is None:
        missing.insert(0, "POSCAR (or structure.vasp)")
    if missing:
        raise FileNotFoundError(f"missing in {inputs}: {', '.join(missing)}")

    out = resolve_path(parent) / defect
    if out.exists():
        if not force:
            raise FileExistsError(f"{out} already exists (--force replaces it)")
        LOGGER.info(f"replacing {out}")
        shutil.rmtree(out)

    section(LOGGER, f"New defect {out} from {inputs}")
    relax = out / RELAX_DIR
    relax.mkdir(parents=True)
    phonons: list[Path] = []
    for name, path in sorted(found.items()):
        if _is_phonon_file(name):
            phonons.append(path)
        elif name in REQUIRED_INPUTS or name in SEED_FILES or name == "POSCAR":
            shutil.copy2(path, relax / name)
    if not (relax / "POSCAR").is_file():
        shutil.copy2(structure, relax / "POSCAR")
    LOGGER.info(f"{relax}: {', '.join(sorted(p.name for p in relax.iterdir()))}")

    data = out / "data"
    if (inputs / "data").is_dir():
        shutil.copytree(inputs / "data", data)
    data.mkdir(exist_ok=True)
    for path in phonons:
        shutil.copy2(path, data / path.name)
    LOGGER.info(f"{data}: {', '.join(sorted(p.name for p in data.iterdir())) or '(empty)'}")
    if not any(_is_phonon_file(p.name) for p in data.rglob("*")):
        LOGGER.warning(f"no phonon file (phonopy.yaml, phonon_data.npz, ...) in {inputs} or its data/")

    for key in ("AO", "UO"):
        zfs = out / METHODS[key].ground_state
        zfs.mkdir()
        for name in SEED_FILES:
            shutil.copy2(relax / name, zfs / name)
        LOGGER.info(f"{zfs}: seeded with the relaxation's inputs; fill it with: spinlabs init zfs {relax} --method {key}")
    return out


def _filled(zfs: Path) -> bool:
    """Whether a ground-state folder was already filled from the relaxation (or has run)."""
    return any((zfs / name).exists() for name in ("CHGCAR", "OUTCAR", "run_ground_state.sh"))


def init_zfs(
    relax: str | Path, method: str = "AO", *, vasp_binary: str | Path | None = None,
    max_hours: float | None = None, force: bool = False,
) -> list[str]:
    """Fill the ground-state folder of ``method`` (AO, UO, full_decont; any spelling) next to the
    finished relaxation ``relax``. Returns the failure messages (empty on success).

    AO: ZFS_AO with the EIGENVAL occupations. UO: ZFS_UO with only the unpaired orbitals occupied.
    Both run as one self-consistent ZFS calculation (run_ground_state.sh, the runner's plain
    ground_state mode). full_decont: ZFS_full, the SCF that writes the WAVECAR and one ALGO=None
    ms_*/INCAR per spin sublevel. A folder already filled is kept unless ``force``.
    """
    from spinlabs.vasp.occupations import prepare_relax_to_zfs
    from spinlabs.vasp.trees import (
        GROUND_STATE_SCRIPT, TEMPLATES, guarded_incar, read_outcar_time, script_ranks, write_decont_inputs,
        write_ground_state_script, SUBLEVEL_TIME_FRACTION,
    )

    relax = resolve_path(relax)
    key = normalize_calc_method(method)
    if key not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    defect = relax.parent
    dst = ground_state_dir(defect, key)
    if dst.exists() and _filled(dst) and not force:
        LOGGER.info(f"exists, kept (--force refills it): {dst}")
        return []
    if dst.exists() and force:
        shutil.rmtree(dst)
    # NBANDS is fitted to the job run_ground_state.sh submits (its template's node and task counts), so
    # VASP keeps it and the occupations fit; the trees' jobs use the same counts.
    ranks = script_ranks((TEMPLATES / GROUND_STATE_SCRIPT).read_text())
    failures = prepare_relax_to_zfs(relax, dst, unpaired_only=METHODS[key].spin_corrected, ranks=ranks)
    if failures:
        return failures
    seconds = read_outcar_time(relax)  # an upper bound: the relaxation ran several ionic steps
    if key == "full_decont":
        try:
            levels = write_decont_inputs(dst, guarded_incar((dst / "INCAR").read_text(), dst), levels_dir=dst)
        except ValueError as error:
            return [f"{dst}: cannot set up the spin sublevels: {error}"]
        if seconds is not None:
            seconds *= 1 + SUBLEVEL_TIME_FRACTION * len(levels)
        LOGGER.info(f"{dst}: SCF + spin sublevels {', '.join(levels)}")
    write_ground_state_script(
        dst, defect.name, "ground_state" if key == "full_decont" else f"ground_state_{key}", seconds,
        vasp_binary=resolve_path(vasp_binary) if vasp_binary else None, max_hours=max_hours,
    )
    LOGGER.info(f"{dst} ready: submit with: cd {dst} && sh run_ground_state.sh")
    return []
