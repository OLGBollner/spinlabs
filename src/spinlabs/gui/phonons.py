"""Choose the phonon spectrum a convergence pack belongs to.

The main pipeline pairs data with either ``phonon_data.npz`` (the spectrum
phonopy produced) or ``phonon_data_aligned.npz`` (the mirror-aligned full
spectrum that data displaced along aligned modes needs).  A convergence pack
can be packaged before that association is recorded, so the GUI looks in the
same places as the pipeline as a fallback: the registry, the packaged data
folder next to the pack, and the defect folder named by the stored ground-state
path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from spinlabs.models import ALIGNED_PHONON_NAME, PACKAGED_PHONON_NAME
from spinlabs.parsers import parse_phonon_data
from spinlabs.registry import ensure_db, find_phonon
from spinlabs.vasp.trees import full_spectrum_of


@dataclass
class PhononChoice:
    path: Path | None = None
    aligned: bool | None = None
    notes: list[str] = field(default_factory=list)
    warning: str | None = None  # a choice to show even when a spectrum was found (the wanted kind is missing)


def _candidate_paths(pack_path: Path, meta: dict, notes: list[str]) -> tuple[list[Path], Path | None]:
    """Full spectra that may belong to the pack, and the one it records (its full spectrum), if found.

    Looks at the recorded file, the registry, and the defect's own folders only: the pack's ancestor named
    <defect>_<cell> and the defect folder of the stored ground state (without a defect and cell, the pack's
    own folder and its parent), so another defect's spectrum is never picked up. A symmetry-reduced file is
    replaced by its full spectrum; one that cannot be paired with it is left out, with a note.
    """
    pack_path = Path(pack_path)
    out: list[Path] = []
    recorded = meta.get("phonon_file")
    if recorded:
        out.append(Path(str(recorded)))
    defect = meta.get("defect")
    cell = meta.get("cell")
    if defect is not None and cell is not None:
        try:
            out.extend(find_phonon(ensure_db(), str(defect), int(cell)))
        except (TypeError, ValueError):
            pass
    folders: list[Path] = []
    ground = meta.get("ground_state")
    if ground:
        ground_path = Path(str(ground))
        folders.append(ground_path.parent.parent)
    if defect is not None and cell is not None:
        folders += [parent for parent in pack_path.parents if parent.name == f"{defect}_{cell}"]
    else:
        folders += list(pack_path.parents)[:2]
    for folder in folders:
        for base in (folder / "data", folder):
            if not base.is_dir():
                continue
            for name in (ALIGNED_PHONON_NAME, PACKAGED_PHONON_NAME, "phonons.npz", "phonons_aligned.npz"):
                candidate = base / name
                if candidate.is_file():
                    out.append(candidate)
            out.extend(sorted(base.glob("*phonon*.npz")))
    # Resolve symmetry-reduced files to their full spectrum and drop duplicates.
    resolved: list[Path] = []
    seen: set[Path] = set()
    recorded_full: Path | None = None
    for k, candidate in enumerate(out):
        candidate = Path(candidate)
        if not candidate.is_file():
            continue
        try:
            full = full_spectrum_of(candidate)
        except ValueError as error:  # never a reduced file in place of its full spectrum
            notes.append(f"{candidate}: {error}; not used")
            continue
        if k == 0 and recorded:
            recorded_full = full
        if not full.is_file() or full in seen:
            continue
        seen.add(full)
        resolved.append(full)
    return resolved, recorded_full


def _is_aligned(path: Path, spectrum) -> bool:
    flag = getattr(spectrum, "aligned", None)
    if flag is not None:
        return bool(flag)
    return path.stem.endswith("_aligned")


def resolve_phonon(
    pack_path: Path, meta: dict, max_mode: int, override_aligned: bool | None = None,
    cache: dict | None = None,
) -> PhononChoice:
    """The best full phonon spectrum for a pack, and why if none covers its modes.

    ``cache`` (shared by the GUI's packs) stores parsed spectra, so several
    columns or tabs using one phonon file pay for parsing it only once.
    """
    cache = {} if cache is None else cache
    pack_path = Path(pack_path)
    wanted_aligned = override_aligned if override_aligned is not None else meta.get("aligned")
    if isinstance(wanted_aligned, str):
        wanted_aligned = wanted_aligned.lower() in ("1", "true", "yes")
    notes: list[str] = []
    best = None
    best_score = -10**9
    candidates, recorded = _candidate_paths(pack_path, meta, notes)
    for path in candidates:
        spectrum_key = ("phonon-spectrum", str(path))
        if spectrum_key in cache:
            spectrum = cache[spectrum_key]
        else:
            try:
                spectrum = parse_phonon_data(path)
            except Exception as exc:
                notes.append(f"{path}: cannot parse ({exc})")
                continue
            cache[spectrum_key] = spectrum
        if spectrum is None:
            notes.append(f"{path}: cannot parse")
            continue
        if spectrum.n_modes < max_mode:
            notes.append(f"{path}: {spectrum.n_modes} modes, need {max_mode}")
            continue
        aligned = _is_aligned(path, spectrum)
        score = 10
        if wanted_aligned is not None:
            score += 100 if bool(wanted_aligned) == aligned else -100
        elif aligned:  # aligned spectra are the safer display default for old packs
            score += 5
        if recorded is not None and path == recorded:
            score += 50  # the spectrum the pack's runs were displaced along: first within its kind
        if path.name == ALIGNED_PHONON_NAME:
            score += 4
        elif path.name == PACKAGED_PHONON_NAME:
            score += 3
        if score > best_score:
            best, best_score = (path, aligned), score
    if best is None:
        if not notes:
            notes.append("no phonon .npz found in the registry, the pack's data folder or the ground-state folder")
        return PhononChoice(path=None, aligned=None, notes=notes)
    warning = None
    if wanted_aligned is not None and best[1] != bool(wanted_aligned):
        warning = (f"no {'aligned' if wanted_aligned else 'unaligned'} full spectrum covers the modes; mode "
                   f"symmetry and frequency from the {'aligned' if best[1] else 'unaligned'} {best[0]}")
        notes.append(warning)
    return PhononChoice(path=best[0], aligned=best[1], notes=notes, warning=warning)
