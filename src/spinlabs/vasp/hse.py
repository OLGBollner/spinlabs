"""HSE06 ZFS runs: the INCAR tags that turn a PBE ZFS run into a hybrid-functional one.

``spinlabs init zfs --hse`` writes them into the ground-state ZFS INCAR (before NBANDS is fitted to the job,
since NCORE = 1 sets the band groups), and every perturbation tree built from that folder copies them. The
runs go the same way as PBE ones (from the ground state's CHGCAR, Gamma point, e.g. vasp_gam); the trees of
an HSE INCAR keep each finished run's WAVECAR (KEEP_WAVECAR=1 in their job scripts).
"""

from __future__ import annotations

import re

# tag -> value, in the order they are appended to an INCAR that lacks them
HSE_TAGS = {
    "LHFCALC": ".TRUE.",  # hybrid functional
    "HFSCREEN": "0.2",  # HSE06 screening
    "AEXX": "0.25",
    "AGGAX": "0.75",
    "AGGAC": "1.00",
    "LASPH": ".TRUE.",  # non-spherical contributions in the PAW spheres (important for hybrids)
    "ALGO": "All",
    "PREC": "Accurate",
    "PRECFOCK": "Fast",
    "NCORE": "1",  # one core per band: hybrid runs parallelise over bands
}
HSE_HEADER = "### HSE06 (spinlabs init zfs --hse)"


def _tag_pattern(tag: str) -> re.Pattern:
    return re.compile(rf"^(\s*{tag}\s*=\s*)(.*?)([ \t]*[!#].*)?$", re.M | re.I)


def hse_incar(incar: str) -> str:
    """``incar`` with the HSE06 tags: a tag it sets already gets the HSE value in place (its comment kept),
    the others are appended under a header. Applying it twice changes nothing."""
    out = incar
    missing = []
    for tag, value in HSE_TAGS.items():
        pattern = _tag_pattern(tag)
        if pattern.search(out):
            out = pattern.sub(lambda m, v=value: f"{m.group(1)}{v}{m.group(3) or ''}", out)
        else:
            missing.append(f"{tag} = {value}")
    if missing:
        out = out.rstrip("\n") + "\n\n" + HSE_HEADER + "\n" + "\n".join(missing) + "\n"
    return out


def is_hse(incar: str) -> bool:
    """Whether an INCAR runs a hybrid functional (LHFCALC = .TRUE., the last setting wins as in VASP)."""
    values = re.findall(r"^\s*LHFCALC\s*=\s*(\S+)", incar, re.M | re.I)
    return bool(values) and values[-1].strip(".").upper().startswith("T")
