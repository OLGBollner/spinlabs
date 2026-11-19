"""Derive fixed defect occupations for ZFS runs from a finished relaxation."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from spinlabs.logging_utils import get_logger, section

LOGGER = get_logger(__name__)

GAMMA_KPOINTS = """Gamma point only (written by spinlabs init zfs)
0
Gamma
1 1 1
0 0 0
"""

REMOVE_TAGS = (
    "NSW",
    "IBRION",
    "LDMATRIX",
    "DOCCUP",
    "DOCCDO",
    "NUPDOWN",
    "DOCC",
    "LDAPMINUS",
    "NBANDS",
    "KPAR",
    "ISYM",
)  # never kept verbatim; rebuilt below


def parse_tag(incar: str, tag: str) -> str | None:
    """Return the LAST value of a tag in the INCAR text (VASP semantics)."""
    vals = re.findall(rf"^\s*{tag}\s*=\s*(.+?)\s*(?:!|#|$)", incar, re.M | re.I)
    return vals[-1] if vals else None


def all_tag_occurrences(incar: str, tag: str) -> list[str]:
    return re.findall(rf"^\s*{tag}\s*=\s*(.+?)\s*(?:!|#|$)", incar, re.M | re.I)


def occupied_bands_from_eigenval(eigenval: Path) -> tuple[int, int, int, set[int], set[int]]:
    """Parse EIGENVAL (ISPIN=2).

    Returns (n_up, n_down, nbands, unpaired_up, unpaired_dn).  Occupied means
    occ > 0.5 in the respective spin column.  Assumes Gamma-only (one k-point)
    or takes the first k-point.  Unpaired bands are occupied in one spin
    channel only (the defect states the UO runs keep).
    """
    lines = eigenval.read_text().splitlines()
    header = lines[5].split()
    nkpts, nbands = int(header[1]), int(header[2])
    data = lines[7:] if nkpts == 1 else lines[7 : 7 + nbands + 1]
    n_up = n_dn = 0
    unpaired_up: set[int] = set()
    unpaired_dn: set[int] = set()
    for line in data[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        occ_up = float(parts[3]) > 0.5
        occ_dn = len(parts) >= 5 and float(parts[4]) > 0.5
        n_up += occ_up
        n_dn += occ_dn
        if occ_up and not occ_dn:
            unpaired_up.add(int(parts[0]))
        elif occ_dn and not occ_up:
            unpaired_dn.add(int(parts[0]))
    return n_up, n_dn, nbands, unpaired_up, unpaired_dn


def kpoints_is_gamma_only(text: str) -> bool:
    """True if a KPOINTS file describes a Gamma-only mesh."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return False
    if lines[1].startswith("0"):
        return lines[2].lower().startswith("g")
    div = lines[3].split()
    shift = lines[4].split() if len(lines) > 4 else ["0"] * 3
    gamma = lines[2].lower().startswith("g")
    return gamma and all(int(d) == 1 for d in div[:3]) and all(float(s) == 0 for s in shift[:3])


def patch_incar(
    text: str,
    n_up: int,
    n_dn: int,
    nbands: int,
    unpaired_up: set[int] | None = None,
    unpaired_dn: set[int] | None = None,
    unpaired_only: bool = False,
) -> str:
    """Turn a relaxation INCAR into a single-point ZFS INCAR: the self-consistent ZFS run (ALGO as in the
    relaxation, never None), no ionic steps, LDMATRIX on.

    The occupations (DOCCUP/DOCCDO) are the EIGENVAL ones, the AO scheme. With ``unpaired_only`` (the UO
    scheme) they are 0 everywhere except the unpaired orbitals (``unpaired_up``/``unpaired_dn``, which
    must be given); NUPDOWN is the same either way. ISMEAR and FERWE/FERDO stay as in the relaxation.
    """
    from spinlabs.vasp.trees import forbid_algo_none

    text, algo_note = forbid_algo_none(text)
    if algo_note:
        LOGGER.warning(algo_note)
    body_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        tag = stripped.split("=")[0].strip().upper() if "=" in stripped else ""
        if tag in REMOVE_TAGS:
            continue  # dropped, re-added below
        if tag == "ICHARG" and parse_tag(text, "ICHARG") == "2":
            line = re.sub(r"ICHARG\s*=\s*2", "ICHARG = 1", line)  # 2 -> 1
        body_lines.append(line)

    if unpaired_only:
        if unpaired_up is None or unpaired_dn is None:
            raise ValueError("the UO occupations need the unpaired bands (unpaired_up, unpaired_dn)")
        doccup = format_occupations([1.0 if b in unpaired_up else 0.0 for b in range(1, nbands + 1)])
        doccdo = format_occupations([1.0 if b in unpaired_dn else 0.0 for b in range(1, nbands + 1)])
    else:
        doccup = f"{n_up}*1.0 {nbands - n_up}*0.0"
        doccdo = f"{n_dn}*1.0 {nbands - n_dn}*0.0"
    additions = f"""
### ZFS stage (added by spinlabs.vasp)
NSW = 0                  # single point: no ionic relaxation
IBRION = -1              # no ionic degrees of freedom
ISYM = 0                 # symmetry off, as in the perturbation runs differenced against this ground state
LDMATRIX = .TRUE.
NUPDOWN = {int(n_up - n_dn)}
NBANDS = {nbands}
DOCCUP = {doccup}
DOCCDO = {doccdo}
"""
    if unpaired_up is not None and unpaired_dn is not None:
        # machine-readable record of the defect bands, derived here where the
        # occupations are first known; downstream tooling reads these comments
        additions += (
            f"# UNPAIRED_UP = {' '.join(str(i) for i in sorted(unpaired_up)) or 'none'}\n"
            f"# UNPAIRED_DN = {' '.join(str(i) for i in sorted(unpaired_dn)) or 'none'}\n"
        )
    return "\n".join(body_lines).rstrip() + "\n" + additions


def prepare_relax_to_zfs(relax: Path, zfs: Path, unpaired_only: bool = False) -> list[str]:
    """Prepare one ZFS run directory from a finished relaxation directory.

    ``unpaired_only``: the UO scheme's occupations, only the unpaired orbitals (see patch_incar).

    Copies CONTCAR -> POSCAR, CHGCAR, POTCAR and (Gamma-only) KPOINTS,
    derives DOCCUP/DOCCDO from the EIGENVAL and patches the INCAR.  Nothing
    in `relax` is modified.  Returns the list of failure messages (empty
    when all sanity checks passed).
    """
    failures: list[str] = []

    def fail(msg: str) -> None:
        failures.append(msg)
        LOGGER.error(f"{zfs}: {msg}")

    def warn(msg: str) -> None:
        LOGGER.warning(f"{zfs}: {msg}")

    section(LOGGER, f"ZFS run {zfs} from relaxation {relax}")
    if not relax.is_dir():
        fail(f"relaxation dir {relax} does not exist")
        return failures
    zfs.mkdir(parents=True, exist_ok=True)

    required = ["CONTCAR", "CHGCAR", "EIGENVAL", "INCAR", "POTCAR", "KPOINTS"]
    for name in required:
        if not (relax / name).is_file():
            fail(f"missing {name} in {relax}")

    shutil.copy2(relax / "CONTCAR", zfs / "POSCAR")
    shutil.copy2(relax / "CHGCAR", zfs / "CHGCAR")
    shutil.copy2(relax / "POTCAR", zfs / "POTCAR")
    kpts_text = (relax / "KPOINTS").read_text()
    if kpoints_is_gamma_only(kpts_text):
        shutil.copy2(relax / "KPOINTS", zfs / "KPOINTS")
        LOGGER.debug("KPOINTS already Gamma-only; copied as-is")
    else:
        (zfs / "KPOINTS").write_text(GAMMA_KPOINTS)
        warn(
            "KPOINTS was not Gamma-only; ZFS run needs the full k-point grid "
            "of the relaxation -- wrote a Gamma-only KPOINTS into the ZFS dir "
            "instead of copying"
        )
    LOGGER.info("copied CONTCAR -> POSCAR, CHGCAR, POTCAR, KPOINTS (Gamma-only)")

    if (zfs / "WAVECAR").exists():
        warn(
            "WAVECAR present in ZFS dir; it is incompatible with the "
            "vasp_gam -> vasp_std switch and will NOT be used -- remove it"
        )

    incar_text = (relax / "INCAR").read_text()
    for tag in ("NUPDOWN", "NELECT", "ISPIN", "NBANDS", "NSW", "IBRION"):
        occ = all_tag_occurrences(incar_text, tag)
        if len(occ) > 1:
            fail(f"INCAR defines {tag} {len(occ)} times: {occ} (last one wins in VASP -- fix before running)")

    n_up, n_dn, nbands, unp_up, unp_dn = occupied_bands_from_eigenval(relax / "EIGENVAL")
    LOGGER.info(
        f"EIGENVAL: {n_up} occupied (up), {n_dn} (down), NBANDS = {nbands}; "
        f"unpaired up {sorted(unp_up)}, down {sorted(unp_dn)}"
    )

    nelect = parse_tag(incar_text, "NELECT")
    isp = parse_tag(incar_text, "ISPIN") or "2"
    if int(isp) == 2 and n_up + n_dn != int(nelect):
        fail(f"occupied bands {n_up}+{n_dn} = {n_up + n_dn} != NELECT = {nelect}")
    if n_up <= n_dn:
        fail(
            f"majority channel not larger than minority ({n_up} <= {n_dn}); "
            f"spin columns may be swapped in this EIGENVAL"
        )
    nupdown = parse_tag(incar_text, "NUPDOWN")
    if nupdown and int(float(nupdown)) != n_up - n_dn:
        fail(f"NUPDOWN = {nupdown} inconsistent with occupied difference {n_up} - {n_dn} = {n_up - n_dn}")

    incar = patch_incar(incar_text, n_up, n_dn, nbands, unp_up, unp_dn, unpaired_only=unpaired_only)
    (zfs / "INCAR").write_text(incar)
    LOGGER.info(
        f"wrote INCAR with NUPDOWN={n_up - n_dn}, DOCCUP={parse_tag(incar, 'DOCCUP')}, "
        f"DOCCDO={parse_tag(incar, 'DOCCDO')}" + (" (unpaired orbitals only)" if unpaired_only else "")
    )

    if failures:
        LOGGER.error(f"{zfs}: {len(failures)} problem(s) found; do NOT launch until fixed")
    else:
        LOGGER.info("all checks passed; ready to launch")
    return failures


# --------------------------------------------------------------------------- #
# spin sublevels for the full (Biktagirov) spin decontamination
# --------------------------------------------------------------------------- #
def expand_occupations(value: str) -> list[float]:
    """Expand a DOCCUP/DOCCDO value such as ``125*1.0 1*0.0 128*0.0`` into one entry per band."""
    occ: list[float] = []
    for item in value.split():
        count, star, occupation = item.partition("*")
        if star:
            occ.extend([float(occupation)] * int(count))
        else:
            occ.append(float(item))
    return occ


def parse_occupations(incar: str) -> tuple[list[float], list[float]]:
    """The (up, down) band occupations of an INCAR: DOCCUP/DOCCDO, else FERWE/FERDO (the last of each wins)."""
    up, dn = parse_tag(incar, "DOCCUP"), parse_tag(incar, "DOCCDO")
    if up is None or dn is None:
        up, dn = parse_tag(incar, "FERWE"), parse_tag(incar, "FERDO")
    if up is None or dn is None:
        raise ValueError("INCAR has no DOCCUP/DOCCDO or FERWE/FERDO occupations")
    up_occ, dn_occ = expand_occupations(up), expand_occupations(dn)
    if len(up_occ) != len(dn_occ):
        raise ValueError(f"DOCCUP has {len(up_occ)} bands, DOCCDO {len(dn_occ)}")
    return up_occ, dn_occ


def _runs(occ: list[float]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(occ):
        j = i
        while j < len(occ) and occ[j] == occ[i]:
            j += 1
        out.append(f"{j - i}*{occ[i]:.1f}")
        i = j
    return out


def format_occupations(occ: list[float], window: tuple[int, int] | None = None) -> str:
    """A DOCCUP/DOCCDO value: run-length ``n*x`` entries; the bands of ``window``
    (1-based, inclusive) are written one by one (``1*1.0 1*0.0``) so the defect levels stand out."""
    if window is None:
        return " ".join(_runs(occ))
    lo, hi = max(window[0], 1), min(window[1], len(occ))
    parts = _runs(occ[: lo - 1]) + [f"1*{x:.1f}" for x in occ[lo - 1 : hi]] + _runs(occ[hi:])
    return " ".join(parts)


@dataclass(frozen=True)
class SpinSublevel:
    """One determinant of the decontamination: its run folder name, occupations and NUPDOWN."""

    name: str
    up: tuple[float, ...]
    dn: tuple[float, ...]
    nupdown: int
    window: tuple[int, int]

    def doccup(self) -> str:
        return format_occupations(list(self.up), self.window)

    def doccdo(self) -> str:
        return format_occupations(list(self.dn), self.window)


def spin_sublevels(up: list[float], dn: list[float]) -> list[SpinSublevel]:
    """The ms = S determinant and the 2S spin-flip ms = S-1 determinants of a high-spin state.

    The 2S unpaired bands are the ones occupied in the up channel and empty in the down
    channel; S = (number of unpaired bands)/2. ``ms_<S>`` keeps the occupations; for each
    unpaired band k, ``ms_<S-1>_<k>`` moves k's electron to the down channel at the same band
    index (up loses k, down gains k), NUPDOWN = 2S - 2. For the NV centre (S = 1, unpaired 127
    and 128) that is ms_1, ms_0_127 and ms_0_128.
    """
    if len(up) != len(dn):
        raise ValueError(f"up has {len(up)} bands, down {len(dn)}")
    unpaired = [i + 1 for i, (u, d) in enumerate(zip(up, dn)) if u > 0.5 and d < 0.5]
    minority = [i + 1 for i, (u, d) in enumerate(zip(up, dn)) if d > 0.5 and u < 0.5]
    if minority:
        raise ValueError(f"bands {minority} are occupied in the down channel only; the majority spin must be up")
    if not unpaired:
        raise ValueError("no unpaired band (occupied up, empty down): not a high-spin state")
    two_s = len(unpaired)
    window = (unpaired[0] - 1, unpaired[-1])

    def label(two_ms: int) -> str:
        return f"ms_{two_ms / 2:g}"

    levels = [SpinSublevel(label(two_s), tuple(up), tuple(dn), two_s, window)]
    for k in unpaired:
        flipped_up, flipped_dn = list(up), list(dn)
        flipped_up[k - 1], flipped_dn[k - 1] = 0.0, 1.0
        levels.append(
            SpinSublevel(f"{label(two_s - 2)}_{k}", tuple(flipped_up), tuple(flipped_dn), two_s - 2, window)
        )
    return levels
