"""Collect perturbation convergence runs into one .npz.

Scans ``runs/<mode>_pert_<p>/OUTCAR`` under the given root, parses the
spin-spin ZFS tensor and final energy from each finished run, and saves
everything to a single compressed .npz for offline analysis (scp it off
the cluster).

A ``metadata`` key (a JSON string) records the defect name, cell size,
creation date and, when given, the ground-state OUTCAR used.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np

from spinlabs.parsers import parse_outcar_energy, parse_outcar_zfs
from spinlabs.vasp.trees import tree_displacement

from spinlabs.logging_utils import configure_logging, get_logger

LOGGER = get_logger(__name__)


def discover_runs(run_root: Path) -> list[tuple[int, float, Path]]:
    """Find runs/<mode>_pert_<p>/OUTCAR; returns sorted (mode, pert, dir)."""
    runs = []
    for outcar in sorted(Path(run_root).glob("runs/*/OUTCAR")):
        m = re.fullmatch(r"(\d+)_pert_(.+)", outcar.parent.name)
        if not m:
            LOGGER.warning(f"skipping unrecognised run dir {outcar.parent.name}")
            continue
        try:
            runs.append((int(m.group(1)), float(m.group(2)), outcar))
        except ValueError:
            LOGGER.warning(f"skipping unparsable run dir {outcar.parent.name}")
    return sorted(runs, key=lambda r: (r[0], r[1]))


def infer_defect_cell(folder: Path) -> tuple[str | None, str | None]:
    """Infer (defect, cell) from a folder name like NV_512 or NV_512_extra."""
    m = re.fullmatch(r"([A-Za-z]+[A-Za-z0-9]*)[_-](\d+)(?:_.*)?", folder.name)
    if m:
        return m.group(1), m.group(2)
    return None, None


def pack_convergence(
    run_root: Path,
    output: Path,
    ground_state: Path | None = None,
    defect: str | None = None,
    cell: str | None = None,
    method: str | None = None,
    method_key: str | None = None,
) -> Path:
    """Pack runs/<mode>_pert_<p>/OUTCAR under one root into one .npz."""
    run_root = Path(run_root)
    if not (run_root / "runs").is_dir():
        sys.exit(f"Error: no runs/ under {run_root}")
    runs = [t for t in discover_runs(run_root)]
    if not runs:
        sys.exit(f"Error: no runs/*/OUTCAR found under {run_root}")

    modes = sorted({m for m, _, _ in runs})
    perts = sorted({p for _, p, _ in runs})
    tensors = np.full((len(modes), len(perts), 3, 3), np.nan)
    energies = np.full((len(modes), len(perts)), np.nan)
    mode_idx = {m: i for i, m in enumerate(modes)}
    pert_idx = {p: i for i, p in enumerate(perts)}

    n_missing = n_empty = 0
    for mode, pert, outcar in runs:
        tensor = parse_outcar_zfs(outcar)
        if tensor is None or bool(np.isnan(tensor.matrix).all()):
            LOGGER.warning(f"no ZFS tensor in {outcar} (run unfinished?)")
            n_empty += 1
            continue
        i, j = mode_idx[mode], pert_idx[pert]
        tensors[i, j] = tensor.matrix
        energy = parse_outcar_energy(outcar)
        if energy is not None:
            energies[i, j] = energy
        else:
            n_missing += 1

    extra: dict = {}
    if ground_state is not None:
        d0 = parse_outcar_zfs(ground_state)
        if d0 is None:
            sys.exit(f"Error: no ZFS tensor in ground-state OUTCAR {ground_state}")
        extra["d0"] = d0.matrix  # (3, 3) MHz, the Q=0 point
        e0 = parse_outcar_energy(ground_state)
        if e0 is not None:
            extra["d0_energy"] = np.float64(e0)
        LOGGER.info(f"ground-state tensor from {ground_state}\n{extra['d0']}")

    if defect is None or cell is None:
        inf_d, inf_c = infer_defect_cell(run_root)
        defect = defect or inf_d
        cell = cell or inf_c
    metadata = {
        "defect": defect,
        "cell": cell,
        "created": f"{date.today():%Y-%m-%d}",
        "ground_state": str(ground_state) if ground_state is not None else None,
        "n_runs": len(runs),
        "method": method,
        "method_key": method_key,
        # how the steps displaced the atoms (spinlabs.models.DISPLACEMENTS): normalized steps are
        # Cartesian lengths in Angstrom, the others Q in Angstrom*sqrt(amu)
        "displacement": tree_displacement(run_root) or "mass_weighted",
    }
    np.savez_compressed(
        output,
        modes=np.array(modes, dtype=int),
        perts=np.array(perts, dtype=float),
        tensors=tensors,  # (n_modes, n_perts, 3, 3), MHz
        energies=energies,  # (n_modes, n_perts), eV
        metadata=np.array(json.dumps(metadata)),
        **extra,
    )
    n_done = tensors.size // 9 - n_empty
    LOGGER.info(f"metadata: {json.dumps(metadata)}")
    LOGGER.info(
        f"Packed {n_done}/{tensors.size // 9} runs ({n_empty} unfinished, {n_missing} without energy) -> {output}"
    )
    LOGGER.info(f"modes: {modes}")
    LOGGER.info(f"perts: {perts}")
    return Path(output)


def build_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-r", "--run-root", type=Path, default=Path("."), help="folder containing runs/ (default: .)")
    parser.add_argument("-o", "--output", type=Path, default=Path("pert_runs.npz"), help="output .npz path")
    parser.add_argument(
        "--ground-state",
        type=Path,
        default=None,
        metavar="OUTCAR",
        help="ground-state OUTCAR; its ZFS tensor is stored as 'd0' (the Q=0 point) in the npz",
    )
    parser.add_argument(
        "--defect", type=str, default=None,
        help="defect name for the metadata key (default: inferred from the run-root name)",
    )
    parser.add_argument(
        "--cell", type=str, default=None,
        help="cell size for the metadata key (default: inferred from the run-root name)",
    )


def run(args: argparse.Namespace) -> int:
    pack_convergence(args.run_root, args.output, ground_state=args.ground_state, defect=args.defect, cell=args.cell)
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging(verbosity=1)
    parser = argparse.ArgumentParser(
        description="Parse runs/<mode>_pert_<p>/OUTCAR files and pack ZFS tensors + energies into one .npz"
    )
    build_parser(parser)
    return run(parser.parse_args(argv))
