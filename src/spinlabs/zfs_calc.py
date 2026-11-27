"""The ZFS of one SCF/ZFS run folder from its OUTCAR(s), by method (``spinlabs calc zfs``).

AO: the folder's OUTCAR tensor as VASP reports it. UO: the same, with the defect-band spin correction
(models.spin_correction_scale: x3/2 for S = 1, x2/3 for S = 3/2; S = NUPDOWN/2 from the INCAR).
full_decont: the ms_*/OUTCAR tensors of its spin sublevels combined into the decontaminated tensor
(Biktagirov; ZFSTensor.decontaminated), S from the high-spin sublevel's name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from spinlabs.methods import METHODS, normalize_calc_method
from spinlabs.models import ZFSTensor, apply_spin_correction, spin_correction_scale
from spinlabs.parsers import parse_outcar_zfs

# folder names the ground-state ZFS runs have (spinlabs init zfs), legacy ones included
_FOLDER_METHODS = {"ZFS_AO": "AO", "ZFS_hyp": "AO", "ZFS_UO": "UO", "ZFS_occup": "UO", "ZFS_full": "full_decont"}


@dataclass
class ZFSResult:
    """The ZFS of a run folder: ``tensor`` (MHz) by ``method``; ``parts`` the OUTCAR tensors it came from."""

    folder: Path
    method: str
    spin: float | None
    tensor: ZFSTensor
    parts: dict[str, ZFSTensor] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        """The numbers, JSON-ready (MHz)."""
        d_xx, d_yy, d_zz, axes = self.tensor.principal_components()
        return {
            "folder": str(self.folder), "method": self.method, "spin": self.spin,
            "D_MHz": self.tensor.D, "E_MHz": self.tensor.E,
            "principal_values_MHz": [d_xx, d_yy, d_zz], "principal_z_axis": axes[:, 2].tolist(),
            "tensor_MHz": self.tensor.matrix.tolist(),
            "parts": {name: {"D_MHz": t.D, "E_MHz": t.E, "tensor_MHz": t.matrix.tolist()}
                      for name, t in self.parts.items()},
            "notes": list(self.notes),
        }


def _sublevel_folders(folder: Path) -> dict[str, Path]:
    """ms_* subfolders with a valid sublevel name (decont.sublevel_info)."""
    from spinlabs.decont import sublevel_info

    found = {}
    for path in sorted(folder.glob("ms_*")):
        if path.is_dir():
            try:
                sublevel_info(path.name)
            except ValueError:
                continue
            found[path.name] = path
    return found


def method_of(folder: Path) -> str | None:
    """The method a run folder is for: full_decont with ms_* sublevel folders, else from its name (ZFS_AO,
    ZFS_UO, ZFS_full and the legacy ZFS_hyp, ZFS_occup); None when it cannot tell."""
    if _sublevel_folders(folder):
        return "full_decont"
    return _FOLDER_METHODS.get(folder.name)


def _spin_from_incar(folder: Path) -> float | None:
    incar = folder / "INCAR"
    if not incar.is_file():
        return None
    values = re.findall(r"^\s*NUPDOWN\s*=\s*([-+0-9.eE]+)", incar.read_text(), re.M | re.I)
    return float(values[-1]) / 2 if values else None


def _outcar_tensor(folder: Path) -> ZFSTensor:
    outcar = folder / "OUTCAR"
    if not outcar.is_file():
        raise ValueError(f"no OUTCAR in {folder}")
    tensor = parse_outcar_zfs(outcar)
    if tensor is None:
        raise ValueError(f"{outcar} has no ZFS block (run not finished, or LDMATRIX off)")
    return tensor.to_unit("MHz")


def compute_zfs(folder: str | Path, method: str | None = None) -> ZFSResult:
    """The ZFS of ``folder`` by ``method`` (AO, UO, full_decont; any spelling; None: method_of the folder).

    ValueError when the method cannot be told, an OUTCAR or its ZFS block is missing, or a full-decont
    folder does not have the m_s = S sublevel and its 2S spin flips.
    """
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"not a folder: {folder}")
    key = normalize_calc_method(method) if method else method_of(folder)
    if key is None:
        raise ValueError(f"cannot tell the method of {folder} from its contents or name: give --method")
    if key not in METHODS:
        raise ValueError(f"unknown method {method!r}")

    if key == "full_decont":
        return _decontaminated(folder)
    raw = _outcar_tensor(folder)
    spin = _spin_from_incar(folder)
    if not METHODS[key].spin_corrected:
        return ZFSResult(folder, key, spin, raw, parts={"OUTCAR": raw})
    notes = []
    if spin is None:
        notes.append("no NUPDOWN in the INCAR: S = 1 assumed for the spin correction")
    scale = spin_correction_scale(spin)
    notes.append(f"defect-band spin correction x{scale:.4g} (S = {spin if spin is not None else 1:g})")
    return ZFSResult(folder, key, spin, apply_spin_correction(raw, scale), parts={"OUTCAR": raw}, notes=notes)


def _decontaminated(folder: Path) -> ZFSResult:
    from spinlabs.decont import high_spin, sublevel_info

    levels = _sublevel_folders(folder)
    if not levels:
        raise ValueError(f"{folder} has no ms_* sublevel folders: not a full-decontamination run")
    tensors = {name: _outcar_tensor(path) for name, path in levels.items()}
    high = high_spin(tensors)
    spin, _ = sublevel_info(high)
    flips = [name for name in tensors if sublevel_info(name)[0] == spin - 1 and sublevel_info(name)[1] is not None]
    if len(flips) != round(2 * spin):
        raise ValueError(f"{folder}: S = {spin:g} needs 2S = {round(2 * spin)} spin-flip sublevels "
                         f"ms_{spin - 1:g}_<band>, found {len(flips)} ({', '.join(flips) or 'none'})")
    tensor = ZFSTensor.decontaminated(tensors[high], [tensors[name] for name in flips], spin)
    parts = {name: tensors[name] for name in [high, *flips]}
    notes = [f"decontaminated from {high} and {', '.join(flips)} (Biktagirov; VASP's per-sublevel scaling undone)"]
    return ZFSResult(folder, "full_decont", spin, tensor, parts=parts, notes=notes)
