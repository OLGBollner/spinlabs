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


def _candidate_paths(pack_path: Path, meta: dict) -> list[Path]:
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
    for parent in list(pack_path.parents)[:5]:
        folders.append(parent)
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
    for candidate in out:
        candidate = Path(candidate)
        if not candidate.is_file():
            continue
        try:
            full = full_spectrum_of(candidate)
        except ValueError:
            full = candidate
        if not full.is_file() or full in seen:
            continue
        seen.add(full)
        resolved.append(full)
    return resolved


def _is_aligned(path: Path, spectrum) -> bool:
    flag = getattr(spectrum, "aligned", None)
    if flag is not None:
        return bool(flag)
    return path.stem.endswith("_aligned")


def resolve_phonon(
    pack_path: Path, meta: dict, max_mode: int, override_aligned: bool | None = None
) -> PhononChoice:
    """The best full phonon spectrum for a pack, and why if none covers its modes."""
    pack_path = Path(pack_path)
    wanted_aligned = override_aligned if override_aligned is not None else meta.get("aligned")
    if isinstance(wanted_aligned, str):
        wanted_aligned = wanted_aligned.lower() in ("1", "true", "yes")
    notes: list[str] = []
    best = None
    best_score = -10**9
    for path in _candidate_paths(pack_path, meta):
        try:
            spectrum = parse_phonon_data(path)
        except Exception as exc:
            notes.append(f"{path}: cannot parse ({exc})")
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
    return PhononChoice(path=best[0], aligned=best[1], notes=notes)
