"""Build ready-to-run VASP directory trees on the cluster."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from spinlabs.parsers import parse_phonon_data, save_phonon_npz
from spinlabs.symmetry import classify_and_pair, filter_degenerate_partners, symmetrize_degenerate_groups

from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)

RUN_STAGES = ("relaxation_data", "ZFS_hyp", "ZFS_occup")

FIRST_ORDER_SCRIPT = "run_perturbation_first_order.sh"
SECOND_ORDER_SCRIPT = "run_perturbation_second_order.sh"
COPIED_FILES = ("INCAR", "POSCAR", "KPOINTS", "POTCAR", "CHGCAR")
REQUIRED_FILES = COPIED_FILES[:4]  # CHGCAR optional

# basis -> source ZFS folder (relative to the defect/output folder)
BASIS_SOURCE = {
    "all_bands": "ZFS_hyp",
    "defect_band_approx": "ZFS_occup",
}


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

    ALGO=None runs a single diagonalization on whatever CHGCAR is present,
    so a perturbed structure would be evaluated on the relaxed structure's
    (stale) charge density -- silently invalid ZFS.  If ALGO=None is found,
    correct it to ALGO=Normal and return a warning message.
    """
    if not re.search(r"^\s*ALGO\s*=\s*None\b", incar, re.M | re.I):
        return incar, None
    out = replace_tag(incar, "ALGO", "Normal")
    warning = (
        "INCAR had ALGO=None (non-self-consistent single "
        "diagonalization on the CHGCAR); corrected to ALGO=Normal "
        "so the perturbed structure gets its own SCF density"
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


def repo_root(scripts_dir: Path) -> Path:
    """The git checkout holding ``scripts_dir`` (repo/scripts/cluster), resolved."""
    scripts_dir = Path(scripts_dir).resolve()
    try:
        out = subprocess.run(
            ["git", "-C", str(scripts_dir), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(out.stdout.strip()).resolve()
    except (OSError, subprocess.CalledProcessError):
        return scripts_dir.parents[1]


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
        LOGGER.info(f"  using existing symmetrised phonon file {target}")
        return target
    save_phonon_npz(filtered, target)
    LOGGER.info(f"  symmetrised phonon data saved to {target} ({filtered.n_modes} of {spectrum.n_modes} modes)")
    return target


def n_modes_from_phonon(phonon: Path) -> int:
    """Number of phonon modes to perturb along (same logic as get_n_modes)."""
    with np.load(phonon) as data:
        if "idx" in data:
            return len(data["idx"])
        return int(data["freqs"].shape[0])


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

    LOGGER.info(f"Creating {out}/ from {inp}")
    shutil.copytree(data_src, out / "data")

    for stage in run_dirs:
        dst = out / stage
        shutil.copytree(relax_src, dst)
        shutil.copy2(struct_file, dst / "POSCAR")
        LOGGER.info(f"  {stage}/  <- template/relax + POSCAR")
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
    pert: float,
    order: str,
    n_array_jobs: int,
    n_modes: int,
    basis: str,
    scripts_dir: Path,
    vasp_binary: Path | None = None,
    pair_mode: str | None = None,
    max_hours: float | None = None,
    mass_weighted: bool = True,
) -> list[str]:
    """Fill dst/input/ and write the prefilled SLURM script.

    `scripts_dir` is the folder holding the .sh templates and the helper
    scripts referenced by them.  Returns the list of failure messages.
    """
    failures: list[str] = []

    def fail(msg: str) -> None:
        failures.append(msg)
        LOGGER.error(f"  FAIL: {msg}")

    inp = dst / "input"
    inp.mkdir(parents=True, exist_ok=True)

    for name in COPIED_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, inp / name)

    tmpl = scripts_dir / script_name
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
            LOGGER.warning(f"  WARN: {warning}")
    else:
        fail("no INCAR in the ZFS source folder -- cannot make it restart-ready")

    # timing: per-sim wall time from the reference ZFS run's OUTCAR
    per_sim = read_outcar_time(src)
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
    text = replace_tag(text, "PERT", str(pert))
    if vasp_binary:
        text = replace_tag(text, "BINARY", str(vasp_binary))
    if pair_mode:
        text = replace_tag(text, "PAIR_MODE", pair_mode)
    # 0 keeps a tree displaced by Q*e (as NV_64's first trees were) consistent with its old runs
    text = replace_tag(text, "MASS_WEIGHTED", "${MASS_WEIGHTED:-1}" if mass_weighted else "0")
    # The loop and the structure scripts run from a snapshot of this repo taken at
    # submission (see the script header), so the script only names the checkout.
    text = replace_tag(text, "SPINLABS_REPO", str(repo_root(scripts_dir)))

    # job name reflects defect, order and basis; time limit and array size
    # only change when the reference OUTCAR gave us a timing to compute from
    text = replace_sbatch(text, "-J", f"{defect}_{order}_{basis}")
    if per_job is not None:
        text = replace_sbatch(text, "-t", sbatch_time(per_job))
        text = replace_sbatch(text, "-a", f"0-{n_array_jobs - 1}")

    (dst / script_name).write_text(text)
    shutil.copy2(scripts_dir / "verify_perturbation_setup.py", dst / "verify_perturbation_setup.py")

    if per_sim is not None:
        LOGGER.info(
            f"    time: {per_sim:.0f} s/sim x {n_tasks} tasks / "
            f"{n_array_jobs} array jobs = {per_job:.0f} s/job "
            f"-> {sbatch_time(per_job)}"
        )
    return failures


def build_perturbation_tree(
    out: Path,
    pert: list[float],
    phonon: Path,
    scripts_dir: Path,
    pair_mode: str = "diag",
    array_jobs: int = 10,
    max_hours: float | None = None,
    vasp_binary: Path | None = None,
    force: bool = False,
    mass_weighted: bool = True,
) -> list[str]:
    """Build ready-to-go perturbed ZFS run folders under a defect directory.

    `out` is the defect folder (its basename is the defect name); it must
    already contain ZFS_hyp/ and ZFS_occup/ with INCAR, POSCAR, KPOINTS and
    POTCAR.  `phonon` should be the raw phonon npz (default
    <out>/data/phonon_data.npz) -- the tree always runs from a symmetrised
    file in <out>/data, created by resolve_sym_phonon if missing.
    Returns the list of failure messages (empty on success).
    """
    failures: list[str] = []
    defect = out.name

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

    phonon = resolve_sym_phonon(out, phonon)
    n_modes = n_modes_from_phonon(phonon)
    LOGGER.info(f"=== {defect}: phonon {phonon} ({n_modes} modes) -> {out} ===")
    for order in ("first_order", "second_order"):
        for basis, src in sources.items():
            for p in pert:
                dst = out / order / f"pert_{p}" / basis
                if dst.exists():
                    if force:
                        LOGGER.info(f"  overwriting {dst}")
                        shutil.rmtree(dst)
                    else:
                        LOGGER.info(f"  note: {dst} already exists -- skipping")
                        continue
                LOGGER.info(f"  creating {dst}")
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
                        scripts_dir=scripts_dir,
                        vasp_binary=vasp_binary,
                        mass_weighted=mass_weighted,
                        pair_mode=pair_mode if order == "second_order" else None,
                        max_hours=max_hours,
                    )
                )
    return failures
