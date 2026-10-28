"""Build ready-to-run VASP directory trees on the cluster."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path


from spinlabs.parsers import parse_phonon_data, save_phonon_npz
from spinlabs.vasp.modes import mode_numbers
from spinlabs.symmetry import classify_and_pair, filter_degenerate_partners, symmetrize_degenerate_groups

from spinlabs.logging_utils import get_logger, section

LOGGER = get_logger(__name__)

RUN_STAGES = ("relaxation_data", "ZFS_hyp", "ZFS_occup")

FIRST_ORDER_SCRIPT = "run_perturbation_first_order.sh"
SECOND_ORDER_SCRIPT = "run_perturbation_second_order.sh"
CONVERGENCE_SCRIPT = "run_convergence.sh"
COPIED_FILES = ("INCAR", "POSCAR", "KPOINTS", "POTCAR", "CHGCAR")
REQUIRED_FILES = COPIED_FILES[:4]  # CHGCAR optional

# basis -> source ZFS folder (relative to the defect/output folder)
BASIS_SOURCE = {
    "all_bands": "ZFS_hyp",
    "defect_band_approx": "ZFS_occup",
}
# basis -> step-size convergence folder of the defect (`spinlabs package --conv` finds convergence*,
# and tells the methods apart by "all"/"occup" in the name)
CONVERGENCE_FOLDER = {
    "all_bands": "convergence_test_all",
    "defect_band_approx": "convergence_test_occup",
}

_DISPLACEMENT_LINE = re.compile(r"^\s*DISPLACEMENT\s*=\s*(\S+)", re.M)
_MASS_WEIGHTED_LINE = re.compile(r"^\s*MASS_WEIGHTED\s*=\s*(\S+)", re.M)


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


def forbid_sym_ldmatrix(incar: str) -> tuple[str, str | None]:
    """Guard against ISYM=1/2 combined with LDMATRIX.

    VASP's LDMATRIX is not supported with symmetry handling ISYM=1 or 2:
    the density symmetrization conflicts with the D-matrix evaluation.
    If such a combination is found, correct ISYM to 3 (symmetry applied
    to the detected structure only) and return a warning message.
    """
    if not re.search(r"^\s*LDMATRIX\s*=\s*\.TRUE\.", incar, re.M | re.I):
        return incar, None
    isym = re.search(r"^\s*ISYM\s*=\s*([12])\b", incar, re.M | re.I)
    if isym is None:
        return incar, None
    out = replace_tag(incar, "ISYM", "3")
    warning = (
        f"INCAR had ISYM={isym.group(1)} together with LDMATRIX; "
        "corrected to ISYM=3 (ISYM=1/2 is not supported with "
        "LDMATRIX)"
    )
    return out, warning


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


def resolve_sym_phonon(out: Path, phonon: Path) -> Path:
    """Return the symmetrised phonon file for the perturbation runs.

    Looks in <defect>/data for phonon_data_sym<n>.npz (n = filtered mode
    count); if missing, classifies, symmetrises and filters `phonon` and
    saves the reduced set there first.
    """
    data = out / "data"
    spectrum = parse_phonon_data(phonon)
    classify_and_pair(spectrum)
    symmetrized = symmetrize_degenerate_groups(spectrum)
    filtered = filter_degenerate_partners(symmetrized)
    target = data / f"phonon_data_sym{filtered.n_modes}.npz"
    if target.is_file():
        LOGGER.info(f"symmetrised phonon data (existing file): {target}")
        return target
    save_phonon_npz(filtered, target)
    LOGGER.info(f"saved symmetrised phonon data: {target} ({filtered.n_modes} of {spectrum.n_modes} modes)")
    return target


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
# ZFS tree
# --------------------------------------------------------------------------- #
def build_zfs_tree(inp: Path, run_dirs: list[str] | tuple[str, ...] = RUN_STAGES, out: Path | None = None) -> Path:
    """Create the VASP directory tree for a point-defect calculation.

    `inp` holds data/ (POSCAR or structure.vasp) and template/relax/
    (INCAR, KPOINTS, POTCAR, run_vasp, ...).  The output tree gets a copy
    of data/ plus one directory per run stage, each seeded from
    template/relax and given the pristine POSCAR.  Refuses to overwrite
    an existing output directory.  Returns the output path.
    """
    inp = inp.resolve()
    data_src = inp / "data"
    relax_src = inp / "template" / "relax"
    if not data_src.is_dir() or not relax_src.is_dir():
        raise FileNotFoundError(f"{inp} must contain 'data/' and 'template/relax/'.")

    poscar = data_src / "POSCAR"
    struct_file = poscar if poscar.is_file() else data_src / "structure.vasp"
    if not struct_file.is_file():
        raise FileNotFoundError(f"no POSCAR or structure.vasp in {data_src}.")

    out = out if out is not None else Path(inp.name)
    if out.exists():
        raise FileExistsError(f"output directory {out} already exists, refusing to overwrite")

    section(LOGGER, f"Create ZFS tree {out} from {inp}")
    shutil.copytree(data_src, out / "data")

    for stage in run_dirs:
        dst = out / stage
        shutil.copytree(relax_src, dst)
        shutil.copy2(struct_file, dst / "POSCAR")
        LOGGER.info(f"created {dst} (template/relax + POSCAR)")
    return out


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
) -> list[str]:
    """Fill dst/input/ and write the prefilled SLURM script.

    The script comes from the package's templates/ folder.  ``order`` is first_order,
    second_order (one step, ``pert``) or convergence (``conv_modes`` at -p and +p for each
    step size p in ``conv_perts``).  Returns the list of failure messages.
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
    # the restart flags: same setup, only the structure differs, and the
    # CHGCAR lets a failed run pick up where it left off
    incar_path = inp / "INCAR"
    if incar_path.is_file():
        incar = incar_path.read_text()
        warnings = []
        for guard in (forbid_algo_none, forbid_sym_ldmatrix, forbid_relaxation):
            incar, warning = guard(incar)
            if warning:
                warnings.append(warning)
        incar_path.write_text(patch_restart_incar(incar))
        for warning in warnings:
            LOGGER.warning(f"{dst}: {warning}")
    else:
        fail("no INCAR in the ZFS source folder -- cannot make it restart-ready")

    # timing: per-sim wall time from the reference ZFS run's OUTCAR
    per_sim = read_outcar_time(src)
    if order == "convergence":
        n_tasks = len(conv_modes or []) * 2 * len(step_sizes(conv_perts or []))  # -p and +p
    else:
        n_tasks = n_tasks_for(order, pair_mode, n_modes)
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
        text = replace_tag(text, "BINARY", str(vasp_binary))
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
) -> list[str]:
    """Build ready-to-go perturbed ZFS run folders under a defect directory.

    `out` is the defect folder (its basename is the defect name); it must
    already contain ZFS_hyp/ and ZFS_occup/ with INCAR, POSCAR, KPOINTS and
    POTCAR.  `phonon` should be the raw phonon npz (default
    <out>/data/phonon_data.npz) -- the tree always runs from a symmetrised
    file in <out>/data, created by resolve_sym_phonon if missing.
    Every step in `pert` gets first_order/ and second_order/ trees displaced by
    `displacement` (spinlabs.models.DISPLACEMENTS); with `conv_modes` the defect also
    gets its step-size convergence folders (see build_convergence_tree), at
    `conv_perts` or, by default, convergence_steps(pert).
    Returns the list of failure messages (empty on success).
    """
    if not pert:
        raise ValueError("no perturbation step given: the step (--pert) is required")
    failures: list[str] = []
    defect = out.name
    sources = zfs_sources(out)

    phonon = resolve_sym_phonon(out, phonon)
    n_modes = n_modes_from_phonon(phonon)
    section(LOGGER, f"Perturbation trees {defect}: {n_modes} modes, {displacement} displacements")
    LOGGER.info(f"phonon data: {phonon}")
    for order in ("first_order", "second_order"):
        for basis, src in sources.items():
            for p in pert:
                dst = out / order / f"pert_{p}" / basis
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
            )
        )
    return failures


def zfs_sources(out: Path) -> dict[str, Path]:
    """basis -> the defect's ground-state ZFS folder (ZFS_hyp, ZFS_occup); both must hold the VASP input."""
    sources: dict[str, Path] = {}
    for basis, folder in BASIS_SOURCE.items():
        src = out / folder
        if not src.is_dir():
            raise FileNotFoundError(
                f"missing {src} -- both ZFS_hyp and ZFS_occup must exist before creating perturbation folders"
            )
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
) -> list[str]:
    """Step-size convergence folders of a defect: <out>/convergence_test_<all|occup>/.

    Each holds input/ (from ZFS_hyp, ZFS_occup) and run_convergence.sh, which runs every
    1-based mode in `modes` at -p and +p for every step size p in `steps` (a sign there is
    ignored) into runs/<mode>_pert_<step>/,
    displaced by `displacement`, through the shared perturbation runner. Works on a new
    defect folder and on one that already has its perturbation trees; existing convergence
    folders are kept unless `force`. Returns the list of failure messages.
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
    for basis, src in zfs_sources(out).items():
        dst = out / CONVERGENCE_FOLDER[basis]
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
            )
        )
    return failures
