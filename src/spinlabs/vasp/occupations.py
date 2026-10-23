"""Derive fixed defect occupations for ZFS runs from a finished relaxation."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)

GAMMA_KPOINTS = """Gamma point only (rewritten by relax_to_zfs)
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
    channel only (the defect states used by the defect_band_approx runs).
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
) -> str:
    """Turn a relaxation INCAR into a single-point ZFS INCAR."""
    body_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        tag = stripped.split("=")[0].strip().upper() if "=" in stripped else ""
        if tag in REMOVE_TAGS:
            continue  # dropped, re-added below
        if tag == "ISYM" and parse_tag(text, "ISYM") == "2":
            line = re.sub(r"ISYM\s*=\s*2", "ISYM = 3", line)  # 2 -> 3
        if tag == "ICHARG" and parse_tag(text, "ICHARG") == "2":
            line = re.sub(r"ICHARG\s*=\s*2", "ICHARG = 1", line)  # 2 -> 1
        body_lines.append(line)

    additions = f"""
### ZFS stage (added by spinlabs.vasp)
NSW = 0                  # single point: no ionic relaxation
IBRION = -1              # no ionic degrees of freedom
LDMATRIX = .TRUE.
NUPDOWN = {int(n_up - n_dn)}
NBANDS = {nbands}
DOCCUP = {n_up}*1.0 {nbands - n_up}*0.0
DOCCDO = {n_dn}*1.0 {nbands - n_dn}*0.0
"""
    if unpaired_up is not None and unpaired_dn is not None:
        # machine-readable record of the defect bands, derived here where the
        # occupations are first known; downstream tooling reads these comments
        additions += (
            f"# UNPAIRED_UP = {' '.join(str(i) for i in sorted(unpaired_up)) or 'none'}\n"
            f"# UNPAIRED_DN = {' '.join(str(i) for i in sorted(unpaired_dn)) or 'none'}\n"
        )
    return "\n".join(body_lines).rstrip() + "\n" + additions


def prepare_relax_to_zfs(relax: Path, zfs: Path) -> list[str]:
    """Prepare one ZFS run directory from a finished relaxation directory.

    Copies CONTCAR -> POSCAR, CHGCAR, POTCAR and (Gamma-only) KPOINTS,
    derives DOCCUP/DOCCDO from the EIGENVAL and patches the INCAR.  Nothing
    in `relax` is modified.  Returns the list of failure messages (empty
    when all sanity checks passed).
    """
    failures: list[str] = []

    def fail(msg: str) -> None:
        failures.append(msg)
        LOGGER.error(f"  FAIL: {msg}")

    def warn(msg: str) -> None:
        LOGGER.warning(f"  warn: {msg}")

    LOGGER.info(f"=== {relax}  ->  {zfs} ===")
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
        LOGGER.info("  KPOINTS already Gamma-only; copied as-is")
    else:
        (zfs / "KPOINTS").write_text(GAMMA_KPOINTS)
        warn(
            "KPOINTS was not Gamma-only; ZFS run needs the full k-point grid "
            "of the relaxation -- wrote a Gamma-only KPOINTS into the ZFS dir "
            "instead of copying"
        )
    LOGGER.info("  copied CONTCAR->POSCAR, CHGCAR, POTCAR (+ KPOINTS as Gamma-only)")

    if (zfs / "WAVECAR").exists():
        warn(
            "WAVECAR present in ZFS dir; it is incompatible with the "
            "vasp_gam -> vasp_std switch and will NOT be used -- remove it"
        )

    incar_text = (relax / "INCAR").read_text()
    for tag in ("NUPDOWN", "NELECT", "ISPIN", "NBANDS", "ISYM", "NSW", "IBRION"):
        occ = all_tag_occurrences(incar_text, tag)
        if len(occ) > 1:
            fail(f"INCAR defines {tag} {len(occ)} times: {occ} (last one wins in VASP -- fix before running)")

    n_up, n_dn, nbands, unp_up, unp_dn = occupied_bands_from_eigenval(relax / "EIGENVAL")
    LOGGER.info(
        f"  EIGENVAL: {n_up} occupied (up), {n_dn} (down), NBANDS = {nbands}; "
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

    (zfs / "INCAR").write_text(patch_incar(incar_text, n_up, n_dn, nbands, unp_up, unp_dn))
    LOGGER.info(
        f"  wrote INCAR with NUPDOWN={n_up - n_dn}, "
        f"DOCCUP={n_up}*1.0 {nbands - n_up}*0.0, DOCCDO={n_dn}*1.0 {nbands - n_dn}*0.0"
    )

    if failures:
        LOGGER.error(f"  {len(failures)} problem(s) found -- do NOT launch until fixed")
    else:
        LOGGER.info("  all checks passed; ready to launch")
    return failures
