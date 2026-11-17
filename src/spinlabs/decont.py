"""Full spin decontamination (Biktagirov): the per-sublevel packages and their loading.

``spinlabs package`` writes one file per spin sublevel of a ``--full-decont`` dataset into one
folder (``data/<defect>_<cell>/pert_<p>/full_decont/<date>/``, convergence runs under
``data/<defect>_<cell>/full_decont/<date>/``). Each file names its sublevel (``spin_sublevel``):
``ms_<S>`` is the high-spin determinant, ``ms_<S-1>_<k>`` the one whose unpaired band k was
flipped to the down channel (spinlabs.vasp.occupations.spin_sublevels). Combining the
sublevels into the decontaminated D is left to the analysis.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from spinlabs.models import RawZFSData

_SUBLEVEL = re.compile(r"^ms_(-?\d+(?:\.\d+)?)(?:_(\d+))?$")


def sublevel_info(name: str) -> tuple[float, int | None]:
    """(ms, flipped band) of a sublevel name: ms_1 -> (1.0, None), ms_0_127 -> (0.0, 127), ms_0.5_6 -> (0.5, 6)."""
    match = _SUBLEVEL.match(name)
    if match is None:
        raise ValueError(f"not a spin sublevel name (ms_<ms> or ms_<ms>_<band>): {name!r}")
    return float(match.group(1)), int(match.group(2)) if match.group(2) else None


def high_spin(names) -> str:
    """The ms = S sublevel (the high-spin determinant) among ``names``."""
    return max(names, key=lambda name: (sublevel_info(name)[0], sublevel_info(name)[1] is None))


def load_sublevel_set(folder: str | Path) -> dict[str, RawZFSData]:
    """{sublevel: RawZFSData} of one packaged full-decontamination set (a folder of sublevel files).

    Only files that carry a ``spin_sublevel`` key count; a folder with two files of one
    sublevel is refused.
    """
    out: dict[str, RawZFSData] = {}
    for path in sorted(Path(folder).glob("*.npz")):
        with np.load(path, allow_pickle=True) as data:
            if "spin_sublevel" not in data.files or ("first_order" not in data.files and "second_order" not in data.files):
                continue
            name = str(data["spin_sublevel"])
        if name in out:
            raise ValueError(f"two files of sublevel {name} in {folder}")
        out[name] = RawZFSData.load(path)
    if not out:
        raise ValueError(f"no spin sublevel files in {folder}")
    return dict(sorted(out.items(), key=lambda item: (-sublevel_info(item[0])[0], item[0])))


def load_conv_sublevel_set(folder: str | Path) -> dict[str, dict]:
    """{sublevel: convergence data} of one packaged full-decontamination convergence set.

    Each value holds the npz arrays (modes, perts, tensors, energies, d0 when stored) and
    ``metadata`` (decoded JSON).
    """
    out: dict[str, dict] = {}
    for path in sorted(Path(folder).glob("*.npz")):
        with np.load(path, allow_pickle=True) as data:
            if "spin_sublevel" not in data.files or "tensors" not in data.files:
                continue
            entry = {key: data[key] for key in data.files if key != "metadata"}
            entry["metadata"] = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
        name = str(entry.pop("spin_sublevel"))
        if name in out:
            raise ValueError(f"two files of sublevel {name} in {folder}")
        entry["path"] = path
        out[name] = entry
    if not out:
        raise ValueError(f"no spin sublevel convergence files in {folder}")
    return dict(sorted(out.items(), key=lambda item: (-sublevel_info(item[0])[0], item[0])))
