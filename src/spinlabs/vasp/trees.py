"""Build ready-to-run VASP directory trees on the cluster."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path


from spinlabs.models import reduced_spectrum_reason
from spinlabs.parsers import parse_phonon_data, save_phonon_npz
from spinlabs.vasp.modes import mode_numbers
from spinlabs.symmetry import (
    SymmetryDetectionError,
    align_degenerate_pairs,
    classify_and_pair,
    filter_degenerate_partners,
    symmetrize_degenerate_groups,
)

from spinlabs.logging_utils import get_logger, section
from spinlabs.utils import resolve_path
from spinlabs.methods import METHODS, basis_dir, convergence_dir, defect_folder_of, ground_state_dir

LOGGER = get_logger(__name__)


FIRST_ORDER_SCRIPT = "run_perturbation_first_order.sh"
SECOND_ORDER_SCRIPT = "run_perturbation_second_order.sh"
CONVERGENCE_SCRIPT = "run_convergence.sh"
COPIED_FILES = ("INCAR", "POSCAR", "KPOINTS", "POTCAR", "CHGCAR")
REQUIRED_FILES = COPIED_FILES[:4]  # CHGCAR optional

# The methods a defect's trees are built for (spinlabs.methods): the AO and UO trees take their input
# from ZFS_AO and ZFS_UO, the full-decontamination trees from ZFS_AO. Folder names come from the
# method table; an existing folder under a legacy name (ZFS_hyp, all_bands/, convergence_test_all,
# ...) is used where it is, never duplicated under the new one.
TREE_METHODS = ("AO", "UO")
# full spin decontamination (Biktagirov): one tree basis, from the all-orbital ZFS_AO run
DECONT_BASIS = METHODS["full_decont"].basis
SUBLEVELS_DIR = "sublevels"  # input/sublevels/<name>/INCAR: the ALGO=None ZFS run of each spin sublevel
# <defect>/ZFS_full: the unperturbed ground state through the same steps, once per defect, submitted
# on its own (build_decont_ground_state); spinlabs package reads the ground-state tensors there
DECONT_GROUND_STATE = METHODS["full_decont"].ground_state
SUBLEVEL_TIME_FRACTION = 0.1  # time of one ALGO=None sublevel run, as a fraction of the reference ZFS run

_DISPLACEMENT_LINE = re.compile(r"^\s*DISPLACEMENT\s*=\s*(\S+)", re.M)
_MASS_WEIGHTED_LINE = re.compile(r"^\s*MASS_WEIGHTED\s*=\s*(\S+)", re.M)
_PHONON_PATH_LINE = re.compile(r"^\s*PHONON_PATH\s*=\s*(\S+)", re.M)
# A symmetry-reduced phonon file: <stem>_sym<n>[_aligned].npz (older files: <stem>_sym_n<n>.npz).
_SYM_PART = re.compile(r"_sym(?:_n)?\d+")


def tree_displacement(folder: Path) -> str | None:
    """A tree's displacement convention (spinlabs.models.DISPLACEMENTS) from its job script in ``folder``.

    DISPLACEMENT if the script sets it (a ${DISPLACEMENT:-x} default counts as x), else the
    older MASS_WEIGHTED (0 or ${MASS_WEIGHTED:-0}: unweighted, 1 or ${MASS_WEIGHTED:-1}: mass_weighted);
    None when the script says neither (trees from before both were mass-weighted).
    """
    from spinlabs.models import displacement_convention

    for script in sorted(Path(folder).glob("run_*.sh")):
        text = script.read_text(errors="replace")
        match = _DISPLACEMENT_LINE.search(text)
        if match:
            value = re.sub(r"^\$\{DISPLACEMENT:-(.*)\}$", r"\1", match.group(1).strip("'\""))
            return displacement_convention(value)
        match = _MASS_WEIGHTED_LINE.search(text)
        if match:  # a ${MASS_WEIGHTED:-0} default counts as its value, 0
            value = re.sub(r"^\$\{MASS_WEIGHTED:?-(.*)\}$", r"\1", match.group(1).strip("'\""))
            return "unweighted" if value.strip("'\"") == "0" else "mass_weighted"
    return None


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def replace_tag(incar: str, tag: str, value: str) -> str:
    """Replace the value of every occurrence of a tag.

    Keeps any trailing comment on its own terms: the original spacing before
    an inline comment is preserved, and a comment that sat flush against the
    old value gets a single space so it does not merge with the new value.
    """

    def _repl(m: re.Match) -> str:
        spacing, comment = m.group(3), m.group(4)
        out = f"{m.group(1)}{value}"
        if comment:
            out += spacing if spacing else " "
            out += comment
        return out

    return re.sub(
        rf"^(\s*{tag}\s*=\s*)(.*?)([ \t]*)([!#].*)?$",
        _repl,
        incar,
        flags=re.M | re.I,
    )


RESTART_TAGS = ("ISTART", "ICHARG", "LWAVE", "LCHARG")
RESTART_VALUES = {"ISTART": "0", "ICHARG": "1", "LWAVE": ".TRUE.", "LCHARG": ".TRUE."}


def patch_restart_incar(incar: str) -> str:
    """Make an INCAR restart-ready from the CHGCAR.

    Sets ISTART=0 (fresh SCF cycle) and ICHARG=1 (read CHGCAR), and turns
    on LWAVE/LCHARG so a failed run leaves a usable CHGCAR behind.  Any
    missing tag is appended.
    """
    out = incar
    for tag, value in RESTART_VALUES.items():
        out = replace_tag(out, tag, value)
        if not re.search(rf"^\s*{tag}\s*=", out, re.M | re.I):
            out = out.rstrip("\n") + f"\n{tag} = {value}\n"
    return out


def forbid_algo_none(incar: str) -> tuple[str, str | None]:
    """Guard against ALGO=None in a perturbation INCAR.

    ALGO=None does no electronic minimization at all: the orbitals stay as
    they start, and the density is never updated. Without a WAVECAR
    (ISTART=0) the orbitals start random and the ZFS is noise; with one,
    they are the WAVECAR's orbitals of another structure. Either way the
    D tensor is silently invalid.  If ALGO=None is found, correct it to
    ALGO=Normal and return a warning message.
    """
    if not re.search(r"^\s*ALGO\s*=\s*None\b", incar, re.M | re.I):
        return incar, None
    out = replace_tag(incar, "ALGO", "Normal")
    warning = (
        "INCAR had ALGO=None (no orbital or density update, so the ZFS "
        "would come from the starting orbitals); corrected to ALGO=Normal "
        "so the perturbed structure gets its own SCF solution"
    )
    return out, warning


def symmetry_off(incar: str) -> tuple[str, str | None]:
    """Set ISYM = 0 in a perturbation INCAR (appended when missing).

    A perturbed structure has lower symmetry than the ground state, and with ISYM = 3 VASP
    regularly stopped on perturbed runs because it could not find an inverse rotation
    (INVGRP); those runs had to be redone with ISYM = 0. Returns a note when ISYM changed.
    """
    isym = re.search(r"^\s*ISYM\s*=\s*(-?\d+)", incar, re.M | re.I)
    if isym is not None and isym.group(1) == "0":
        return incar, None
    if isym is None:
        return incar.rstrip("\n") + "\nISYM = 0\n", "INCAR had no ISYM; set ISYM=0 (symmetry off for perturbed structures)"
    return replace_tag(incar, "ISYM", "0"), (
        f"INCAR had ISYM={isym.group(1)}; set to ISYM=0 (symmetry off: perturbed structures made VASP "
        "fail to find inverse rotations)"
    )


def forbid_relaxation(incar: str) -> tuple[str, str | None]:
    """Guard against ionic relaxation in a ZFS/perturbation INCAR.

    The ZFS must be evaluated on the exact perturbed geometry, so no ion
    may move: with IBRION>=0 (or NSW>0) VASP would relax the structure
    before writing LDMATRIX results, silently mixing relaxation and
    perturbation.  Correct IBRION to -1 (static run) and return a
    warning message.
    """
    if not re.search(r"^\s*LDMATRIX\s*=\s*\.TRUE\.", incar, re.M | re.I):
        return incar, None
    ibrion = re.search(r"^\s*IBRION\s*=\s*([0-8]+)\b", incar, re.M | re.I)
    nsw = re.search(r"^\s*NSW\s*=\s*(\d+)\b", incar, re.M | re.I)
    relaxes = (ibrion is not None and int(ibrion.group(1)) >= 0) or (
        ibrion is None and nsw is not None and int(nsw.group(1)) > 0
    )
    if not relaxes:
        return incar, None
    if ibrion is not None:
        out = replace_tag(incar, "IBRION", "-1")
    else:
        out = incar.rstrip("\n") + "\nIBRION = -1\n"
    return out, (
        "INCAR had ionic relaxation enabled together with "
        "LDMATRIX; corrected to IBRION=-1 (static run) so the "
        "ZFS is evaluated on the exact perturbed geometry"
    )


def _set_tags(incar: str, values: dict[str, str]) -> str:
    """Set each tag to its value, appending tags the INCAR does not have."""
    out = incar
    for tag, value in values.items():
        out = replace_tag(out, tag, value)
        if not re.search(rf"^\s*{tag}\s*=", out, re.M | re.I):
            out = out.rstrip("\n") + f"\n{tag} = {value}\n"
    return out


def decont_scf_incar(incar: str) -> str:
    """Step 2 of a full-decontamination run: the SCF of the ms = S state that writes the WAVECAR.

    The ZFS run's INCAR (its ms = S occupations) with LDMATRIX off (the sublevel runs evaluate
    it) and LWAVE on, restart-ready from the CHGCAR like every perturbation INCAR.
    """
    return patch_restart_incar(_set_tags(incar, {"LDMATRIX": ".FALSE."}))


def decont_sublevel_incar(incar: str, level) -> str:
    """Step 4: the ZFS of one spin sublevel (an occupations.SpinSublevel) from the step-2 WAVECAR.

    Both occupation sets describe ``level``'s determinant: DOCCUP/DOCCDO, which LDMATRIX evaluates
    in the decontamination build of VASP, and FERWE/FERDO with ISMEAR = -2 and NUPDOWN = 2 ms,
    VASP's own occupations (with smearing instead, degenerate levels such as the NV e pair get
    half occupations). VASP takes the spin of its 1/(S(2S-1)) prefactor from the FERWE/FERDO
    magnetization, so the ms = S tensor comes out scaled and the ms = S - 1 ones as that
    determinant gives them. ALGO = None keeps the WAVECAR's orbitals. Nothing is written back
    (LWAVE/LCHARG off), so the WAVECAR can be a symlink.
    """
    return _set_tags(
        incar,
        {
            "ALGO": "None",
            "ISTART": "1",
            "ICHARG": "0",
            "LDMATRIX": ".TRUE.",
            "ISMEAR": "-2",
            "NUPDOWN": str(level.nupdown),
            "FERWE": level.doccup(),
            "FERDO": level.doccdo(),
            "DOCCUP": level.doccup(),
            "DOCCDO": level.doccdo(),
            "LWAVE": ".FALSE.",
            "LCHARG": ".FALSE.",
            "NSW": "0",
            "IBRION": "-1",
        },
    )


def decont_ground_state(tree: Path) -> Path | None:
    """The ground state ZFS_full of the defect a tree belongs to: next to the ground-state folders
    (ZFS_AO, ZFS_UO, or their legacy names) of the first parent that has them."""
    defect = defect_folder_of(tree)
    return defect / DECONT_GROUND_STATE if defect is not None else None


def guarded_incar(incar: str, where: Path) -> str:
    """A ZFS INCAR made fit for the perturbation runs: no ALGO=None, no relaxation, ISYM=0 (logged)."""
    warnings = []
    for guard in (forbid_algo_none, forbid_relaxation):
        incar, warning = guard(incar)
        if warning:
            warnings.append(warning)
    incar, note = symmetry_off(incar)
    if note:
        LOGGER.info(f"{where}: {note}")
    for warning in warnings:
        LOGGER.warning(f"{where}: {warning}")
    return incar


def write_decont_inputs(inp: Path, incar: str, levels_dir: Path | None = None) -> list[str]:
    """Write inp/INCAR (step 2) and <levels_dir>/<name>/INCAR (step 4) from a guarded ZFS INCAR.

    ``levels_dir`` defaults to inp/sublevels (a tree's input); the ground state ZFS_full puts the
    sublevel folders next to its SCF instead. Returns the sublevel names.
    """
    from spinlabs.vasp.occupations import parse_occupations, spin_sublevels

    levels = spin_sublevels(*parse_occupations(incar))
    (inp / "INCAR").write_text(decont_scf_incar(incar))
    for level in levels:
        d = (levels_dir if levels_dir is not None else inp / SUBLEVELS_DIR) / level.name
        d.mkdir(parents=True, exist_ok=True)
        (d / "INCAR").write_text(decont_sublevel_incar(incar, level))
    return [level.name for level in levels]


GROUND_STATE_SCRIPT = "run_ground_state.sh"


def build_decont_ground_state(
    out: Path, *, vasp_binary: Path | None = None, max_hours: float | None = None, force: bool = False
) -> list[str]:
    """Fill <out>/ZFS_full, the ground state of the full spin decontamination, ready to submit on its own.

    From ZFS_AO (or the legacy ZFS_hyp): INCAR (the SCF, guarded like the trees' input and restart-ready), POSCAR, KPOINTS,
    POTCAR, CHGCAR; one ms_*/INCAR per spin sublevel (ALGO=None from ../WAVECAR); and
    run_ground_state.sh (one job: the runner's ground_state mode). Its time limit is the reference
    ZFS run's wall time with the sublevels' share (SUBLEVEL_TIME_FRACTION), capped at ``max_hours``.
    A ZFS_full that already holds an INCAR is kept unless ``force``. Returns the failure messages.
    """
    src = zfs_sources(out, decont=True)[DECONT_BASIS]
    dst = out / DECONT_GROUND_STATE
    if (dst / "INCAR").is_file() and not force:
        LOGGER.info(f"exists, kept (--force rebuilds it): {dst}")
        return []
    if dst.exists() and force:
        LOGGER.info(f"rebuilding: {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in COPIED_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
    try:
        levels = write_decont_inputs(dst, guarded_incar((src / "INCAR").read_text(), dst), levels_dir=dst)
    except ValueError as error:
        LOGGER.error(f"{dst}: cannot set up the spin sublevels: {error}")
        return [f"{dst}: cannot set up the spin sublevels: {error}"]
    per_sim = read_outcar_time(src)
    seconds = per_sim * (1 + SUBLEVEL_TIME_FRACTION * len(levels)) if per_sim is not None else None
    write_ground_state_script(dst, out.name, "ground_state", seconds, vasp_binary=vasp_binary, max_hours=max_hours)
    LOGGER.info(f"ground state {dst}: SCF + sublevels {', '.join(levels)}; submit with: sh {GROUND_STATE_SCRIPT}")
    return []


def write_ground_state_script(
    dst: Path, defect: str, job: str, seconds: float | None, *, vasp_binary: Path | None = None,
    max_hours: float | None = None,
) -> Path:
    """run_ground_state.sh in ``dst`` (the runner's ground_state mode: ZFS_full with its sublevels, or one
    plain ZFS run in ZFS_AO / ZFS_UO): -J <defect>_<job>, -t from ``seconds`` x 1.5 (one job: margin for
    a slow node), capped at ``max_hours``."""
    text = (TEMPLATES / GROUND_STATE_SCRIPT).read_text()
    text = replace_tag(text, "DEFECT", defect)
    if vasp_binary:
        text = replace_tag(text, "BINARY", str(resolve_path(vasp_binary)))
    text = replace_tag(text, "SPINLABS_REPO", str(repo_root()))
    text = replace_sbatch(text, "-J", f"{defect}_{job}")
    if seconds is not None:
        seconds *= 1.5
        if max_hours is not None:
            seconds = min(seconds, max_hours * 3600)
        text = replace_sbatch(text, "-t", sbatch_time(seconds))
    (dst / GROUND_STATE_SCRIPT).write_text(text)
    return dst / GROUND_STATE_SCRIPT


# job-script templates copied into every tree (package data, next to this module)
TEMPLATES = Path(__file__).resolve().parent / "templates"


def repo_root(path: Path = TEMPLATES) -> Path:
    """The git checkout holding ``path`` (by default this package's templates), resolved.

    Outside git, the nearest parent holding scripts/cluster/, where the job's runner lives.
    """
    path = Path(path).resolve()
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(out.stdout.strip()).resolve()
    except (OSError, subprocess.CalledProcessError):
        return next((d for d in path.parents if (d / "scripts" / "cluster").is_dir()), path.parents[-1])


def script_ranks(text: str) -> int | None:
    """MPI ranks a SLURM job script runs VASP on: -n/--ntasks, else nodes (-N/--nodes) x --ntasks-per-node.

    None when the script says neither (VASP's band count cannot be fitted to it).
    """
    def sbatch(*names: str) -> int | None:
        for name in names:
            m = re.search(rf"^#SBATCH\s+{re.escape(name)}(?:\s+|=)(\d+)\b", text, re.M)
            if m:
                return int(m.group(1))
        return None

    ntasks = sbatch("-n", "--ntasks")
    if ntasks:
        return ntasks
    nodes, per_node = sbatch("-N", "--nodes"), sbatch("--ntasks-per-node")
    return nodes * per_node if nodes and per_node else None


def replace_sbatch(text: str, directive: str, value: str) -> str:
    """Replace the value of an #SBATCH directive (e.g. -J, -t, -a)."""
    return re.sub(rf"(#SBATCH\s+{re.escape(directive)}\s+).*", rf"\g<1>{value}", text, flags=re.M)


def read_outcar_time(src: Path) -> float | None:
    """Elapsed VASP wall time (sec) from the top-level OUTCAR under src.

    Continued runs append one `Elapsed time` line per invocation; sum them.
    """
    outcar = src / "OUTCAR"
    if not outcar.is_file():
        return None
    times = re.findall(r"Elapsed time \(sec\):\s*([\d\.]+)", outcar.read_text(errors="replace"))
    return sum(float(m) for m in times) if times else None


def sbatch_time(seconds: float) -> str:
    """Format seconds as a SLURM time limit, rounded up to whole minutes."""
    minutes = max(1, math.ceil(seconds / 60))
    h, m = divmod(minutes, 60)
    return f"{h}:{m:02d}:00"


# --------------------------------------------------------------------------- #
# phonon resolution
# --------------------------------------------------------------------------- #
def default_phonon(out: Path) -> Path:
    """Default phonon file: <defect>/data/phonon_data.npz."""
    return out / "data" / "phonon_data.npz"


def resolve_sym_phonon(out: Path, phonon: Path, align: bool = True) -> Path:
    """Return the symmetrised phonon file for the perturbation runs.

    Classifies, symmetrises and filters `phonon` (one mode per degenerate set). With ``align``
    (the default) every degenerate E set is first rotated onto one mirror plane
    (symmetry.align_degenerate_pairs): the runs use <defect>/data/phonon_data_sym<n>_aligned.npz
    and the analysis the aligned full spectrum phonon_data_aligned.npz next to it (``spinlabs
    package`` pairs each dataset with the full spectrum of the file its job script names). A
    structure that cannot be aligned (no C3v, or a set off +-1) falls back to the classified-only
    phonon_data_sym<n>.npz. An existing file of that name is reused.
    """
    data = out / "data"
    spectrum = parse_phonon_data(phonon)
    reduced = reduced_spectrum_reason(spectrum)
    if reduced is not None:
        # Filtering and aligning a reduced spectrum find no partners: it would pass through as its own
        # "full" spectrum (phonon_data[_aligned].npz) and its unaligned E modes be tagged aligned.
        raise ValueError(
            f"{phonon} is a symmetry-reduced phonon spectrum ({reduced}); pass the full spectrum "
            "(the phonon code's, e.g. phonons.yaml or phonon_data.npz with every mode)"
        )
    classify_and_pair(spectrum)
    symmetrized = symmetrize_degenerate_groups(spectrum)
    symmetrized.aligned = False  # classified only, unless aligned below
    if align:
        try:
            aligned = align_degenerate_pairs(symmetrized)
        except (SymmetryDetectionError, NotImplementedError, ValueError) as error:
            LOGGER.info(f"phonon modes not aligned to a mirror plane ({error}); the runs use the symmetrised file")
        else:
            if aligned.aligned:
                symmetrized = aligned
            else:
                LOGGER.warning("not every degenerate E set could be aligned; the runs use the symmetrised file")
    filtered = filter_degenerate_partners(symmetrized)
    stub = "_aligned" if symmetrized.aligned else ""
    target = data / f"phonon_data_sym{filtered.n_modes}{stub}.npz"
    full = data / f"phonon_data{stub}.npz"
    if symmetrized.aligned and not full.is_file():
        save_phonon_npz(symmetrized, full)
        LOGGER.info(f"saved the aligned full phonon spectrum: {full} ({symmetrized.n_modes} modes)")
    if target.is_file():
        LOGGER.info(f"symmetrised phonon data (existing file): {target}")
        return target
    save_phonon_npz(filtered, target)
    LOGGER.info(f"saved symmetrised phonon data: {target} ({filtered.n_modes} of {spectrum.n_modes} modes)")
    return target


def tree_phonon(folder: Path) -> Path | None:
    """The phonon file a tree's job script in ``folder`` displaces along (its PHONON_PATH), or None.

    A ${PHONON_PATH:-x} default counts as x; None when no run_*.sh sets it (older scripts).
    """
    for script in sorted(Path(folder).glob("run_*.sh")):
        match = _PHONON_PATH_LINE.search(script.read_text(errors="replace"))
        if match:
            value = re.sub(r"^\$\{PHONON_PATH:?-(.*)\}$", r"\1", match.group(1).strip("'\""))
            return Path(value.strip("'\""))
    return None


def full_spectrum_of(phonon: Path) -> Path:
    """The full phonon spectrum (every mode) that belongs to the phonon file a tree displaced along.

    A full spectrum is its own. A symmetry-reduced <stem>_sym<n>[_aligned].npz belongs to
    <stem>[_aligned].npz next to it (spinlabs symmetrize and resolve_sym_phonon write that pair); it
    counts only when it has the reduced file's n_full modes and holds each kept vector at the kept
    mode's index, so data displaced along aligned vectors is never paired with the original spectrum.
    Raises ValueError (with the reason) otherwise.
    """
    import numpy as np

    phonon = Path(phonon)
    if not phonon.is_file():
        raise ValueError(f"phonon file not found: {phonon}")
    reduced = parse_phonon_data(phonon)
    idx = None if reduced.original_indices is None else np.asarray(reduced.original_indices, dtype=int)
    if idx is None or np.array_equal(idx, np.arange(reduced.n_modes)):
        # every mode in its own place: a full spectrum, unless modes are missing (a reduced file without indices)
        reason = reduced_spectrum_reason(reduced)
        if reason is not None:
            raise ValueError(f"{phonon} is symmetry-reduced ({reason}) and does not say which full spectrum it is from")
        return phonon
    full_path = phonon.with_name(_SYM_PART.sub("", phonon.stem, count=1) + phonon.suffix)
    if full_path == phonon or not full_path.is_file():
        raise ValueError(f"no full spectrum {full_path.name} next to the symmetry-reduced {phonon}")
    full = parse_phonon_data(full_path)
    n_full = reduced.n_full if reduced.n_full is not None else full.n_modes
    if full.n_modes != n_full:
        raise ValueError(f"{full_path} has {full.n_modes} modes, {phonon.name} was reduced from {n_full}")
    if idx.max() >= full.n_modes or not np.allclose(
        np.asarray(full.eigenvectors)[idx], np.asarray(reduced.eigenvectors), atol=1e-8
    ):
        raise ValueError(f"the modes of {phonon.name} are not those of {full_path.name} (different vectors)")
    return full_path


def n_modes_from_phonon(phonon: Path) -> int:
    """Number of phonon modes to perturb along (the modes the runner loops over)."""
    return len(mode_numbers(phonon))


def n_tasks_for(order: str, pair_mode: str | None, n_modes: int) -> int:
    """Number of simulations a full perturbation sweep requires."""
    if order == "first_order":
        return n_modes
    if pair_mode == "all":
        return n_modes * (n_modes + 1) // 2
    return n_modes  # diag


# --------------------------------------------------------------------------- #
# perturbation tree
# --------------------------------------------------------------------------- #
def prepare_basis(
    src: Path,
    dst: Path,
    script_name: str,
    phonon: Path,
    defect: str,
    pert: float | None,
    order: str,
    n_array_jobs: int,
    n_modes: int,
    basis: str,
    vasp_binary: Path | None = None,
    pair_mode: str | None = None,
    max_hours: float | None = None,
    displacement: str = "normalized",
    conv_modes: list[int] | None = None,
    conv_perts: list[float] | None = None,
    decont: bool = False,
) -> list[str]:
    """Fill dst/input/ and write the prefilled SLURM script.

    The script comes from the package's templates/ folder.  ``order`` is first_order,
    second_order (one step, ``pert``) or convergence (``conv_modes`` at -p and +p for each
    step size p in ``conv_perts``).  With ``decont`` the tree runs the full spin decontamination:
    input/INCAR is the SCF that writes the WAVECAR, input/sublevels/<name>/INCAR the ALGO=None
    ZFS run of each spin sublevel; the defect's ground state ZFS_full is its own job
    (build_decont_ground_state, see scripts/cluster/README.md).
    Returns the list of failure messages.
    """
    failures: list[str] = []

    def fail(msg: str) -> None:
        failures.append(msg)
        LOGGER.error(f"{dst}: {msg}")

    inp = dst / "input"
    inp.mkdir(parents=True, exist_ok=True)

    for name in COPIED_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, inp / name)

    tmpl = TEMPLATES / script_name
    if not tmpl.is_file():
        fail(f"missing template script {tmpl}")
        return failures

    # the perturbation INCAR is the ZFS run's INCAR, verbatim apart from
    # the restart flags and ISYM = 0 (symmetry off): same setup, only the
    # structure differs, and the CHGCAR lets a failed run pick up where it left off
    incar_path = inp / "INCAR"
    if incar_path.is_file():
        incar = guarded_incar(incar_path.read_text(), dst)
        if decont:
            try:
                levels = write_decont_inputs(inp, incar)
                LOGGER.info(f"{dst.name}: spin sublevels {', '.join(levels)}")
            except ValueError as error:
                fail(f"cannot set up the spin sublevels: {error}")
                levels = []
        else:
            incar_path.write_text(patch_restart_incar(incar))
    else:
        fail("no INCAR in the ZFS source folder -- cannot make it restart-ready")
        levels = []

    # timing: per-sim wall time from the reference ZFS run's OUTCAR
    per_sim = read_outcar_time(src)
    if order == "convergence":
        n_tasks = len(conv_modes or []) * 2 * len(step_sizes(conv_perts or []))  # -p and +p
    else:
        n_tasks = n_tasks_for(order, pair_mode, n_modes)
    if decont:
        if per_sim is not None:
            per_sim *= 1 + SUBLEVEL_TIME_FRACTION * len(levels)
    if per_sim is None:
        fail(
            f"no Elapsed time found in OUTCARs under {src} -- leaving the template time limit and array size untouched"
        )
    total = per_sim * n_tasks if per_sim is not None else None
    per_job = total / n_array_jobs if total is not None else None
    if per_job is not None and max_hours is not None:
        cap = max_hours * 3600
        if per_job > cap:
            n_array_jobs = max(n_array_jobs, math.ceil(total / cap))
            per_job = total / n_array_jobs

    text = tmpl.read_text()
    text = replace_tag(text, "DEFECT", defect)
    text = replace_tag(text, "PHONON_PATH", str(phonon.resolve()))
    if order == "convergence":
        text = replace_tag(text, "MODES", f'"{" ".join(str(m) for m in conv_modes or [])}"')
        text = replace_tag(text, "PERTS", f'"{" ".join(f"{p:g}" for p in step_sizes(conv_perts or []))}"')
    else:
        text = replace_tag(text, "PERT", str(pert))
    if vasp_binary:
        text = replace_tag(text, "BINARY", str(resolve_path(vasp_binary)))
    if pair_mode:
        text = replace_tag(text, "PAIR_MODE", pair_mode)
    text = replace_tag(text, "DISPLACEMENT", displacement)
    # The loop and the structure scripts run from a snapshot of this repo taken at
    # submission (see the script header), so the script only names the checkout.
    text = replace_tag(text, "SPINLABS_REPO", str(repo_root()))

    # job name reflects defect, order and basis; time limit and array size
    # only change when the reference OUTCAR gave us a timing to compute from
    text = replace_sbatch(text, "-J", f"{defect}_{order}_{basis}")
    if per_job is not None:
        text = replace_sbatch(text, "-t", sbatch_time(per_job))
        text = replace_sbatch(text, "-a", f"0-{n_array_jobs - 1}")

    (dst / script_name).write_text(text)
    shutil.copy2(TEMPLATES / "verify_perturbation_setup.py", dst / "verify_perturbation_setup.py")

    if per_sim is not None:
        LOGGER.info(
            f"{dst.name} wall time: {per_sim:.0f} s/sim x {n_tasks} tasks / "
            f"{n_array_jobs} array jobs = {per_job:.0f} s/job "
            f"-> {sbatch_time(per_job)}"
        )
    return failures


def build_perturbation_tree(
    out: Path,
    pert: list[float],
    phonon: Path,
    *,
    displacement: str,
    conv_modes: list[int] | None,
    conv_perts: list[float] | None = None,
    pair_mode: str = "diag",
    array_jobs: int = 10,
    conv_array_jobs: int = 5,
    max_hours: float | None = None,
    vasp_binary: Path | None = None,
    force: bool = False,
    central: bool = False,
    decont: bool = False,
) -> list[str]:
    """Build ready-to-go perturbed ZFS run folders under a defect directory.

    `out` is the defect folder (its basename is the defect name); it must
    already contain ZFS_AO/ and ZFS_UO/ (or the legacy ZFS_hyp/, ZFS_occup/) with INCAR, POSCAR, KPOINTS and
    POTCAR.  `phonon` should be the raw phonon npz (default
    <out>/data/phonon_data.npz) -- the tree always runs from a symmetrised
    file in <out>/data, created by resolve_sym_phonon if missing.
    Every step in `pert` gets first_order/ and second_order/ trees displaced by
    `displacement` (spinlabs.models.DISPLACEMENTS); with `conv_modes` the defect also
    gets its step-size convergence folders (see build_convergence_tree), at
    `conv_perts` or, by default, convergence_steps(pert).
    With `central`, every step p instead gets first_order/ trees at +p and -p and no second_order/
    tree: the analysis takes central differences from them (``spinlabs package --central``).
    With `decont` the trees (and convergence folder) run the full spin decontamination instead,
    one basis ``full_decont`` from ZFS_AO (see prepare_basis).
    Returns the list of failure messages (empty on success).
    """
    if not pert:
        raise ValueError("no perturbation step given: the step (--pert) is required")
    if central and any(p <= 0 for p in pert):
        raise ValueError(f"central differences need positive steps p (run at +p and -p); got {pert}")
    if central:
        jobs = [("first_order", s * p) for p in pert for s in (1, -1)]
    else:
        jobs = [(order, p) for order in ("first_order", "second_order") for p in pert]
    failures: list[str] = []
    defect = out.name
    sources = zfs_sources(out, decont=decont)
    if decont:
        failures.extend(build_decont_ground_state(out, vasp_binary=vasp_binary, max_hours=max_hours, force=force))

    phonon = resolve_sym_phonon(out, phonon)
    n_modes = n_modes_from_phonon(phonon)
    section(LOGGER, f"Perturbation trees {defect}: {n_modes} modes, {displacement} displacements")
    LOGGER.info(f"phonon data: {phonon}")
    for order, p in jobs:
        for basis, src in sources.items():
            dst = basis_dir(out / order / f"pert_{p}", basis)  # an existing legacy-named tree is kept
            if not _make_room(dst, force):
                continue
            script_name = FIRST_ORDER_SCRIPT if order == "first_order" else SECOND_ORDER_SCRIPT
            failures.extend(
                prepare_basis(
                    src,
                    dst,
                    script_name,
                    phonon,
                    defect=defect,
                    pert=p,
                    order=order,
                    n_array_jobs=array_jobs,
                    n_modes=n_modes,
                    basis=dst.name,
                    vasp_binary=vasp_binary,
                    displacement=displacement,
                    pair_mode=pair_mode if order == "second_order" else None,
                    max_hours=max_hours,
                    decont=decont,
                )
            )
    if conv_modes:
        failures.extend(
            build_convergence_tree(
                out,
                phonon,
                conv_modes,
                conv_perts if conv_perts else convergence_steps(pert),
                displacement=displacement,
                array_jobs=conv_array_jobs,
                max_hours=max_hours,
                vasp_binary=vasp_binary,
                force=force,
                decont=decont,
                ground_state=False,  # built above
            )
        )
    return failures


def zfs_sources(out: Path, decont: bool = False, methods: list[str] | None = None) -> dict[str, Path]:
    """method -> the defect's ground-state ZFS folder its trees take their input from: AO -> ZFS_AO and
    UO -> ZFS_UO (or their legacy names ZFS_hyp, ZFS_occup); with ``decont`` only full_decont -> ZFS_AO.
    ``methods`` (AO, UO) limits the first to those. Each must hold the VASP input.
    """
    sources: dict[str, Path] = {}
    wanted = {DECONT_BASIS: "AO"} if decont else {m: m for m in (methods or TREE_METHODS)}
    for basis, source in wanted.items():
        src = ground_state_dir(out, source)
        if not src.is_dir():
            needed = " and ".join(METHODS[m].ground_state for m in set(wanted.values()))
            raise FileNotFoundError(f"missing {src} -- {needed} must exist before creating perturbation folders")
        for name in REQUIRED_FILES:
            if not (src / name).is_file():
                raise FileNotFoundError(f"missing {name} in {src}")
        sources[basis] = src
    return sources


def _make_room(dst: Path, force: bool) -> bool:
    """Whether to (re)create dst: an existing folder is kept unless ``force``."""
    if dst.exists():
        if not force:
            LOGGER.info(f"exists, kept (--force rebuilds it): {dst}")
            return False
        LOGGER.info(f"rebuilding: {dst}")
        shutil.rmtree(dst)
    LOGGER.info(f"building: {dst}")
    return True


def defect_trees(out: Path) -> list[Path]:
    """A defect's perturbation tree folders (first_order/pert_*/<basis>, then second_order)."""
    return sorted(out.glob("first_order/pert_*/*")) + sorted(out.glob("second_order/pert_*/*"))


def defect_steps(out: Path) -> list[float]:
    """The steps p of a defect's perturbation trees (<order>/pert_<p>/), sorted by size; empty without trees."""
    steps: set[float] = set()
    for tree in defect_trees(out):
        try:
            steps.add(abs(float(tree.parent.name.removeprefix("pert_"))))
        except ValueError:
            continue
    return sorted(steps)


def defect_displacement(out: Path) -> str | None:
    """The displacement convention a defect's existing trees use: None if it has no trees.

    Trees whose job scripts name none predate both settings and were mass-weighted.
    Raises ValueError when the trees disagree.
    """
    trees = defect_trees(out)
    if not trees:
        return None
    found = {tree_displacement(t) or "mass_weighted" for t in trees}
    if len(found) > 1:
        raise ValueError(f"the trees of {out.name} use different displacements {sorted(found)}")
    return found.pop()


def defect_vasp_binary(out: Path) -> Path | None:
    """The VASP binary a defect's existing trees' job scripts name, if any."""
    for tree in defect_trees(out):
        for script in sorted(tree.glob("run_*.sh")):
            match = re.search(r"^\s*binary\s*=\s*(\S+)", script.read_text(errors="replace"), re.M)
            value = re.sub(r"^\$\{BINARY:-(.*)\}$", r"\1", match.group(1).strip("'\"")) if match else ""
            if value and "<" not in value:
                return Path(value)
    return None


def step_sizes(steps: list[float]) -> list[float]:
    """The convergence step sizes in `steps`: sorted, without sign, zero or repeats.

    The runner runs each size at -p and +p, so a signed list and its magnitudes give the same runs.
    """
    return sorted({round(abs(s), 10) for s in steps if round(abs(s), 10) > 0})


def convergence_steps(pert: list[float]) -> list[float]:
    """Convergence step sizes for a tree's steps: p/2, p, 2p and 4p for each p (each run at both
    signs). The largest, 4p, is the 2Q helper of 2p (convergence.helper_step): it completes the
    forward second-order difference at 2p and is no fit point of its own, which leaves p/2, p and
    2p for the noise and the Taylor fit, and second-order points up to twice the tree's step."""
    return step_sizes([f * p for p in pert for f in (0.5, 1.0, 2.0, 4.0)])


def build_convergence_tree(
    out: Path,
    phonon: Path,
    modes: list[int],
    steps: list[float],
    *,
    displacement: str,
    array_jobs: int = 5,
    max_hours: float | None = None,
    vasp_binary: Path | None = None,
    force: bool = False,
    decont: bool = False,
    ground_state: bool = True,
    methods: list[str] | None = None,
) -> list[str]:
    """Step-size convergence folders of a defect: <out>/convergence_test_<AO|UO>/ (or the legacy test_all/occup).

    Each holds input/ (from ZFS_AO, ZFS_UO) and run_convergence.sh, which runs every
    1-based mode in `modes` at -p and +p for every step size p in `steps` (a sign there is
    ignored) into runs/<mode>_pert_<step>/,
    displaced by `displacement`, through the shared perturbation runner. Works on a new
    defect folder and on one that already has its perturbation trees; existing convergence
    folders are kept unless `force`. With `decont`, one folder convergence_test_decont/ that runs
    the full spin decontamination (see prepare_basis), plus the defect's ground state ZFS_full
    unless ``ground_state`` is False. ``methods`` (AO, UO) builds only those folders. Returns the list of
    failure messages.
    """
    steps = step_sizes(steps)
    if not modes or not steps:
        raise ValueError("convergence folders need modes and nonzero step sizes")
    known = set(mode_numbers(phonon))
    missing = [m for m in modes if m not in known]
    if missing:
        raise ValueError(f"modes {missing} are not in {phonon} (its modes: {min(known)}..{max(known)})")
    failures: list[str] = []
    n_modes = n_modes_from_phonon(phonon)
    section(LOGGER, f"Convergence folders {out.name}: modes {modes}, steps {steps}, {displacement} displacements")
    if decont and ground_state:
        failures.extend(build_decont_ground_state(out, vasp_binary=vasp_binary, max_hours=max_hours, force=force))
    for basis, src in zfs_sources(out, decont=decont, methods=methods).items():
        dst = convergence_dir(out, basis)  # an existing legacy-named folder is kept
        if not _make_room(dst, force):
            continue
        failures.extend(
            prepare_basis(
                src,
                dst,
                CONVERGENCE_SCRIPT,
                phonon,
                defect=out.name,
                pert=None,
                order="convergence",
                n_array_jobs=array_jobs,
                n_modes=n_modes,
                basis=dst.name,
                vasp_binary=vasp_binary,
                displacement=displacement,
                max_hours=max_hours,
                conv_modes=list(modes),
                conv_perts=list(steps),
                decont=decont,
            )
        )
    return failures
