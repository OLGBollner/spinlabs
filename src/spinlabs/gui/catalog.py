"""Registry-backed catalogue of convergence packs for the GUI.

A *scheme* is one convergence folder: AO, UO, full decont, or a non-standard
folder key.  A *group* is what becomes one top-level tab: defect + cell +
displacement convention.  A scheme may have several dated files; the GUI
preselects the newest and lets the tab's dropdown switch date.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sqlite3

import numpy as np

from spinlabs.convergence_summary import STANDARD_KEYS
from spinlabs.methods import METHODS, method_from_path, normalize_calc_method
from spinlabs.registry import ensure_db, find_convergence

_DATE = re.compile(r"_(\d{8})_conv_")
_DISPLAY = {"AO": "AO", "UO": "UO", "full_decont": "full decont"}
_METHOD_ORDER = {key: i for i, key in enumerate(METHODS)}
# Folder suffixes whose method name is already shown; no need to append [key].
_QUIET_KEYS = set(STANDARD_KEYS) | {"all", "occup", "AO", "UO", "decont"}


def read_pack_metadata(path: Path) -> dict:
    """Metadata of one pack without loading its arrays (a decont folder's first sublevel)."""
    path = Path(path)
    candidates = sorted(path.glob("*.npz")) if path.is_dir() else [path]
    if not candidates:
        return {}
    try:
        with np.load(candidates[0], allow_pickle=True) as data:
            if "metadata" not in data.files:
                return {}
            return json.loads(str(data["metadata"]))
    except Exception:  # a non-npz or truncated file is simply unregistered metadata
        return {}


def pack_date(path: Path) -> str:
    """The YYYYMMDD date in the pack's file name, or 'unknown'."""
    path = Path(path)
    candidates = sorted(path.glob("*.npz")) if path.is_dir() else [path]
    for candidate in candidates:
        match = _DATE.search(candidate.stem)
        if match:
            return match.group(1)
    return "unknown"


def pack_has_q0(path: Path) -> bool | None:
    """Whether a pack stores a Q=0 anchor (0 in perts or a d0 key); None if unreadable."""
    path = Path(path)
    candidates = sorted(path.glob("*.npz")) if path.is_dir() else [path]
    if not candidates:
        return None
    try:
        with np.load(candidates[0], allow_pickle=True) as data:
            if "d0" in data.files:
                return True
            if "perts" in data.files:
                return bool(np.any(np.isclose(data["perts"], 0.0)))
    except Exception:
        return None
    return False


_DEFECT_CELL = re.compile(r"^(.+?)[_-](\d+)(?:_.*)?$")


def _name_parts(path: Path) -> tuple[str | None, str | None]:
    """Defect and cell from the file/folder name when metadata does not name them.

    Defect names may contain underscores (``V_Si_128``), so the cell is the
    first number after the name rather than the last token.
    """
    path = Path(path)
    name = path.name if path.is_dir() else re.sub(r"_\d{8}$", "", path.stem.partition("_conv_")[0])
    match = _DEFECT_CELL.fullmatch(name)
    return (match.group(1), match.group(2)) if match else (None, None)


_MS_SUFFIX = re.compile(r"_ms_[\d.]+(_\d+)?$")


def pack_suffix(path: Path) -> str:
    """The convergence folder suffix in the file/folder name (test_AO, test_all, ...).

    A full-decont sublevel file ends in ``_ms_0``/``_ms_1``/``_ms_-1``; the
    registry strips it, and so do we so all sublevels share one scheme key.
    """
    path = Path(path)
    if path.is_dir():
        for item in sorted(path.glob("*.npz")):
            suffix = _MS_SUFFIX.sub("", item.stem.partition("_conv_")[2])
            if suffix:
                return suffix
        return ""
    return _MS_SUFFIX.sub("", path.stem.partition("_conv_")[2])


def _method_from_suffix(suffix: str) -> str | None:
    """The method a folder suffix names, using the same spellings as packaging."""
    if not suffix:
        return None
    candidate = suffix if suffix.lower().startswith("test") else f"test_{suffix}"
    return normalize_calc_method(method_from_path(candidate))


def _scheme_id(defect: str, cell: str, displacement: str, key: str | None, path: Path) -> str:
    """One scheme across date folders.

    ``package --conv`` writes each date into its own ``<date>/`` leaf, so the
    scheme cannot be keyed by the parent directory.  Decont sets are folders,
    but several dated sets of one method are one scheme with a date dropdown.
    """
    return f"{defect}|{cell}|{displacement}|{key or ''}"


def _scheme_label(method: str | None, key: str | None) -> str:
    if method is None:
        return key or "convergence"
    name = _DISPLAY.get(method, method)
    if key and key not in _QUIET_KEYS:
        return f"{name} [{key}]"
    return name


@dataclass(frozen=True)
class PackFile:
    date: str
    path: Path


@dataclass
class Scheme:
    """One convergence folder and its dated files, newest first."""

    id: str
    label: str
    method: str | None
    method_key: str | None
    folder: Path
    dates: list[PackFile] = field(default_factory=list)
    has_q0: bool | None = None

    @property
    def latest(self) -> PackFile:
        return self.dates[0]

    def file_for(self, date: str | None) -> PackFile:
        if date:
            for item in self.dates:
                if item.date == date:
                    return item
        return self.latest


@dataclass
class Group:
    """One defect + cell + displacement tab."""

    defect: str
    cell: str
    displacement: str
    schemes: list[Scheme] = field(default_factory=list)
    source_path: Path | None = None  # set for a session-only unregistered pack

    @property
    def key(self) -> str:
        if self.source_path is not None:
            return f"unregistered|{self.source_path}"
        return f"{self.defect}|{self.cell}|{self.displacement}"

    @property
    def title(self) -> str:
        parts = [self.defect]
        if self.cell:
            parts.append(str(self.cell))
        if self.displacement:
            parts.append(self.displacement.replace("_", " "))
        return " ".join(parts)

    def scheme(self, scheme_id: str) -> Scheme | None:
        return next((s for s in self.schemes if s.id == scheme_id), None)


class Catalog:
    """All convergence packs in the registry, grouped for the GUI."""

    def __init__(self, conn: sqlite3.Connection | None = None):
        self.conn = conn or ensure_db()
        self.errors: list[str] = []
        self._raw: list[tuple[Path, dict, str, str]] = []
        self._load()

    def _load(self) -> None:
        try:
            paths = find_convergence(self.conn)
        except Exception as exc:  # a missing/corrupt registry must not kill the GUI
            self.errors.append(f"cannot read the registry: {exc}")
            paths = []
        for path in paths:
            path = Path(path)
            meta = read_pack_metadata(path)
            defect = meta.get("defect")
            cell = meta.get("cell")
            if not defect or cell is None:
                name_defect, name_cell = _name_parts(path)
                defect = defect or name_defect
                cell = cell if cell is not None else name_cell
            if not defect or cell is None:
                self.errors.append(f"{path.name}: no defect/cell in metadata or name; not listed")
                continue
            suffix = pack_suffix(path)
            key = meta.get("method_key") or suffix or None
            method = normalize_calc_method(meta.get("method")) or _method_from_suffix(suffix)
            displacement = str(meta.get("displacement") or "mass_weighted")
            scheme_id = _scheme_id(str(defect), str(cell), displacement, key, path)
            self._raw.append((path, meta | {"defect": str(defect), "cell": cell, "method": method,
                                             "method_key": key, "displacement": displacement},
                              pack_date(path), scheme_id))

    def defects(self) -> list[str]:
        return sorted({meta["defect"] for _, meta, _, _ in self._raw})

    def cells(self) -> list[str]:
        return sorted({str(meta["cell"]) for _, meta, _, _ in self._raw}, key=lambda c: (len(c), c))

    def methods(self) -> list[str]:
        return sorted({meta["method"] for _, meta, _, _ in self._raw if meta["method"]},
                      key=lambda m: (_METHOD_ORDER.get(m, 99), m))

    def displacements(self) -> list[str]:
        return sorted({meta["displacement"] for _, meta, _, _ in self._raw})

    def groups(
        self,
        *,
        defect: str | None = None,
        cell: str | None = None,
        method: str | None = None,
        displacement: str | None = None,
        latest_only: bool = False,
    ) -> list[Group]:
        """Groups matching the filters; with ``latest_only`` each scheme keeps its newest file."""
        collected: dict[tuple[str, str, str], dict[str, list[tuple[str, Path]]]] = {}
        for path, meta, date, scheme_id in self._raw:
            if defect and meta["defect"] != defect:
                continue
            if cell and str(meta["cell"]) != str(cell):
                continue
            if method and meta["method"] != method:
                continue
            if displacement and meta["displacement"] != displacement:
                continue
            group_key = (meta["defect"], str(meta["cell"]), meta["displacement"])
            schemes = collected.setdefault(group_key, {})
            schemes.setdefault(scheme_id, []).append((date, path))

        groups: list[Group] = []
        for (defect_name, cell_name, displacement_name), schemes_dict in sorted(
            collected.items(), key=lambda item: (item[0][0], (len(item[0][1]), item[0][1]), item[0][2])
        ):
            schemes: list[Scheme] = []
            for scheme_id, files in schemes_dict.items():
                # Newest date first; "unknown" cannot claim to be newest.
                files.sort(key=lambda item: (item[0] != "unknown", item[0]), reverse=True)
                # One entry per date; a decont folder appears once.
                unique: list[PackFile] = []
                seen: set[str] = set()
                for date, path in files:
                    if date in seen:
                        continue
                    seen.add(date)
                    unique.append(PackFile(date, path))
                if latest_only:
                    unique = unique[:1]
                path0, meta0 = next((p, m) for p, m, d, sid in self._raw if sid == scheme_id)
                method_key = meta0.get("method_key")
                schemes.append(
                    Scheme(
                        id=scheme_id,
                        label=_scheme_label(meta0.get("method"), method_key),
                        method=meta0.get("method"),
                        method_key=method_key,
                        folder=Path(path0).parent if Path(path0).is_file() else Path(path0),
                        dates=unique,
                        has_q0=pack_has_q0(unique[0].path),
                    )
                )
            schemes.sort(key=lambda s: (_METHOD_ORDER.get(s.method or "", 99), s.label.lower()))
            # Disambiguate duplicate labels with the folder/date.
            counts: dict[str, int] = {}
            for scheme in schemes:
                counts[scheme.label] = counts.get(scheme.label, 0) + 1
            for scheme in schemes:
                if counts[scheme.label] > 1:
                    scheme.label = f"{scheme.label} [{scheme.dates[0].date}]"
            groups.append(Group(defect=defect_name, cell=cell_name, displacement=displacement_name, schemes=schemes))
        return groups

    def group(self, key: str, *, latest_only: bool = False) -> Group | None:
        return next((g for g in self.groups(latest_only=latest_only) if g.key == key), None)

    def group_for_path(self, path: Path) -> Group | None:
        """The group whose scheme contains this registered file or decont folder."""
        target = Path(path)
        for group in self.groups():
            for scheme in group.schemes:
                if scheme.folder == target or any(item.path == target for item in scheme.dates):
                    return group
        return None

    def synthetic_group(self, path: Path) -> Group:
        """A group for an unregistered file/folder opened for this session."""
        path = Path(path)
        meta = read_pack_metadata(path)
        defect, cell = meta.get("defect"), meta.get("cell")
        if not defect or cell is None:
            name_defect, name_cell = _name_parts(path)
            defect = defect or name_defect or path.stem
            cell = cell if cell is not None else (name_cell or "?")
        method = normalize_calc_method(meta.get("method"))
        key = meta.get("method_key")
        displacement = str(meta.get("displacement") or "mass_weighted")
        date = pack_date(path)
        scheme = Scheme(
            id=f"unregistered|{path}",
            label=_scheme_label(method, key) or path.stem,
            method=method,
            method_key=key,
            folder=path.parent if path.is_file() else path,
            dates=[PackFile(date, path)],
            has_q0=pack_has_q0(path),
        )
        return Group(
            defect=str(defect), cell=str(cell), displacement=displacement, schemes=[scheme], source_path=path
        )
