from __future__ import annotations

import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from pymatgen.core import Structure

from spinlabs.constants import CONSTANTS
from spinlabs.models import PerturbationEntry, PhononSpectrum, RawZFSData, ZFSTensor, normalize_calc_method

from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)

ZFS_REGEX = re.compile(
    r"Spin-spin contribution to zero-field splitting tensor \(MHz\)\s*-+\s*"
    r"D_xx\s+D_yy\s+D_zz\s+D_xy\s+D_xz\s+D_yz\s*-+\s*([\s\d\.\-]+?)(?=\s*-{3,})",
    re.MULTILINE | re.DOTALL,
)

ENERGY_REGEX = re.compile(
    r"free  energy   TOTEN  =\s+([\-\d\.]+)\s+eV",
    re.MULTILINE,
)


def split_defect_folder_name(name: str) -> tuple[str, int | None]:
    """Split a defect folder name like ``NV_512`` or ``V_Si_128``.

    Returns ``(defect, cell_size)`` where ``defect`` is everything up to the
    trailing underscore-separated cell size, so multi-part names like
    ``V_Si_128`` keep their full label (``V_Si``) instead of truncating to
    ``V``.
    """
    match = re.match(r"^(?P<defect>.+?)_(?P<cell>\d+)(?:_.*)?$", name)
    if match:
        return match.group("defect"), int(match.group("cell"))
    return name, None


def parse_outcar_zfs(outcar_path: str | Path) -> ZFSTensor | None:
    """
    Parses the dipole-dipole spin-spin ZFS tensor from a VASP OUTCAR file.
    Returns ZFSTensor in MHz, or None if not found.
    """
    path = Path(outcar_path)
    if not path.is_file():
        return None

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    match = ZFS_REGEX.search(content)
    if not match:
        return None

    try:
        raw_values = match.group(1).split()
        values = [float(v) for v in raw_values]
        if len(values) < 6:
            return None

        # VASP outputs: D_xx, D_yy, D_zz, D_xy, D_xz, D_yz
        D_xx, D_yy, D_zz, D_xy, D_xz, D_yz = values[:6]
        matrix = np.array(
            [
                [D_xx, D_xy, D_xz],
                [D_xy, D_yy, D_yz],
                [D_xz, D_yz, D_zz],
            ],
            dtype=float,
        )

        return ZFSTensor(matrix=matrix, unit="MHz")
    except (ValueError, IndexError):
        return None


def parse_outcar_energy(outcar_path: str | Path) -> float | None:
    """
    Parses the final free energy TOTEN (in eV) from a VASP OUTCAR file.
    """
    path = Path(outcar_path)
    if not path.is_file():
        return None

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    matches = ENERGY_REGEX.findall(content)
    if not matches:
        return None

    try:
        return float(matches[-1])
    except ValueError:
        return None


def _dedupe_phonon_qpoints(raw_data: dict, path: Path) -> None:
    """Deduplicate repeated q-points in raw phonopy yaml data in place.

    Some phonopy runs append the same band twice (e.g. a restart
    concatenates output). Keep only the first occurrence of each
    q-position and warn about duplicates.
    """
    q_entries = raw_data.get("phonon", [])
    if len(q_entries) <= 1:
        return
    seen: dict[tuple, int] = {}
    keep: list[int] = []
    for i, entry in enumerate(q_entries):
        q = tuple(entry.get("q-position", [None, None, None]))
        if q in seen:
            warnings.warn(
                f"Duplicate q-point {q} found in {path} "
                f"(indices {seen[q]} and {i}); keeping the first "
                f"occurrence only.",
                stacklevel=3,
            )
        else:
            seen[q] = i
            keep.append(i)
    if len(keep) != len(q_entries):
        raw_data["phonon"] = [q_entries[i] for i in keep]


def parse_phonopy_yaml(yaml_path: str | Path, poscar_path: str | Path | None = None) -> PhononSpectrum:
    """
    Parses phonon vibrational frequencies, eigenvectors, and structure from a phonopy.yaml file.
    """
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"phonopy file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw_data = yaml.load(f, Loader=Loader)

    # Deduplicate q-points: some phonopy runs append the same band twice
    # (e.g. a restart concatenates output). Keep only the first occurrence
    # of each q-position, and warn about duplicates.
    _dedupe_phonon_qpoints(raw_data, path)
    phonon_data = raw_data["phonon"][0]
    n_phonon = len(phonon_data["band"])
    n_lattice = len(phonon_data["band"][0]["eigenvector"])

    # Frequencies converted to meV
    mode_freqs_mev = np.array([d["frequency"] for d in phonon_data["band"]]) * CONSTANTS["THz2meV"]

    # Eigenvectors: real part at Gamma
    mode_eigenvectors = np.zeros((n_phonon, n_lattice, 3), dtype=float)
    for i in range(n_phonon):
        mode_eigenvectors[i] = np.array([[comp[0] for comp in atom] for atom in phonon_data["band"][i]["eigenvector"]])

    # Structure data
    struct = None
    if poscar_path and Path(poscar_path).is_file():
        struct = Structure.from_file(str(poscar_path))
    else:
        points = raw_data.get("points", [])
        if not points:
            # Fall back to a POSCAR next to the phonopy file, like the old
            # PhononManager.load_all_data did.
            sibling = path.parent / "POSCAR"
            if sibling.is_file():
                sibling_struct = Structure.from_file(str(sibling))
                if len(sibling_struct) != n_lattice:
                    raise ValueError(
                        f"Geometry mismatch: {len(sibling_struct)} atoms in POSCAR "
                        f"({sibling}) vs {n_lattice} atoms in eigenvectors "
                        f"of {path}"
                    )
                LOGGER.info(f"Trying to read data from: {sibling}")
                struct = sibling_struct

    if struct is not None:
        frac_coords = struct.frac_coords
        symbols = [site.specie.symbol for site in struct]
        masses = np.array([site.specie.atomic_mass for site in struct], dtype=float)
        lattice = struct.lattice.matrix
    else:
        points = raw_data.get("points", [])
        if points:
            frac_coords = np.array([p["coordinates"] for p in points], dtype=float)
            symbols = [p.get("symbol", "X") for p in points]
            masses = np.array([p.get("mass", 1.0) for p in points], dtype=float)
        else:
            frac_coords = np.zeros((n_lattice, 3), dtype=float)
            symbols = ["X"] * n_lattice
            masses = np.ones(n_lattice, dtype=float)
            warnings.warn(
                f"No structure data in {path} and no POSCAR found in "
                f"{path.parent}: falling back to degenerate defaults "
                f"(identity lattice, all atoms at origin, symbol X, mass 1.0). "
                f"Symmetry analysis will fail; place a matching POSCAR next "
                f"to the file or run write_full_phonopy_yaml().",
                stacklevel=2,
            )

        lattice = np.array(raw_data.get("lattice", np.eye(3)), dtype=float)

    return PhononSpectrum(
        frequencies_mev=mode_freqs_mev,
        eigenvectors=mode_eigenvectors,
        atom_frac_coords=frac_coords,
        atom_symbols=symbols,
        atomic_masses=masses,
        lattice=lattice,
    )


def _dedupe_text_qpoints(text_lines: list[str]) -> int:
    """Remove duplicate q-point blocks from phonopy yaml text in place.

    A q-point block runs from its ``q-position:`` line to the line before
    the next ``q-position:`` line. The q tuple is read from the line text
    (inline ``[...]`` with optional trailing comment) or, for block
    sequences, from the following ``- value`` lines — so any phonopy
    formatting is handled. First occurrence wins. Returns the number of
    removed blocks.
    """
    qpos_re = re.compile(r"^\s*-?\s*q-position:\s*(.*)$")
    list_re = re.compile(r"^\s*-\s*(-?[\d.eE+-]+)\s*$")

    def parse_payload(payload: str) -> tuple | None:
        payload = payload.split("#", 1)[0].strip().strip("[]() ")
        if not payload:
            return None
        try:
            return tuple(float(x) for x in re.split(r"[,\s]+", payload) if x)
        except ValueError:
            return None

    entries: list[tuple[int, tuple]] = []  # (line index of q-position, q)
    for i, line in enumerate(text_lines):
        m = qpos_re.match(line)
        if not m:
            continue
        q = parse_payload(m.group(1))
        if q is None:
            # block sequence: q-position: followed by "- value" lines
            vals: list[float] = []
            j = i + 1
            while j < len(text_lines):
                lm = list_re.match(text_lines[j])
                if lm:
                    vals.append(float(lm.group(1)))
                    j += 1
                else:
                    break
            if len(vals) == 3:
                q = tuple(vals)
        if q is not None and len(q) == 3:
            entries.append((i, q))
    if len(entries) <= 1:
        return 0
    seen: set[tuple] = set()
    drop: list[tuple[int, int]] = []
    for j, (start, q) in enumerate(entries):
        end = entries[j + 1][0] if j + 1 < len(entries) else len(text_lines)
        if q in seen:
            drop.append((start, end))
        else:
            seen.add(q)
    for start, end in reversed(drop):
        del text_lines[start:end]
    return len(drop)


def _structure_yaml_lines(
    lattice: np.ndarray,
    symbols: list[str],
    frac_coords: np.ndarray,
    masses: np.ndarray,
) -> list[str]:
    """Format a structure block in phonopy's native yaml style."""
    lines = ["lattice:"]
    for row, label in zip(lattice, ("a", "b", "c")):
        lines.append("- [ %21.15f, %21.15f, %21.15f ] # %s" % (row[0], row[1], row[2], label))
    lines.append("points:")
    for i, (symbol, frac, mass) in enumerate(zip(symbols, frac_coords, masses)):
        lines.append(f"- symbol: {symbol} # {i + 1}")
        c = " %18.15f, %18.15f, %18.15f " % tuple(frac)
        lines.append(f"  coordinates: [{c}]")
        lines.append(f"  mass: {mass:.6f}")
    return lines


def prepare_phonopy_yaml(
    yaml_path: str | Path,
    poscar_path: str | Path | None = None,
    out_path: str | Path | None = None,
) -> tuple[PhononSpectrum, Path | None]:
    """Parse a phonopy yaml once and, if needed, write a structure-complete copy.

    Parses ``yaml_path`` exactly once (q-point deduplication warning fires
    here only). If the yaml already contains structure data (``points``),
    returns ``(spectrum, None)``. Otherwise the structure is taken from
    ``poscar_path`` (default: a ``POSCAR`` next to the yaml) — the atom
    count must match the eigenvector dimension or ``ValueError`` is raised
    — the original file text is copied unchanged with duplicate q-point
    blocks removed, the structure block is appended at the end in phonopy's
    native style, and the result is saved to ``out_path`` (default:
    ``<yaml stem>_full.yaml`` next to the input). Returns
    ``(spectrum, out_path)``; if no usable POSCAR is found, the second
    element is ``None`` and nothing is written.
    """
    path = Path(yaml_path)
    spectrum = parse_phonopy_yaml(path, poscar_path=poscar_path)

    has_points = any(re.match(r"^points:", line) for line in path.read_text(encoding="utf-8").splitlines())
    if has_points:
        return spectrum, None
    if poscar_path is None:
        sibling = path.parent / "POSCAR"
        poscar_path = sibling if sibling.is_file() else None
    if poscar_path is None or not Path(poscar_path).is_file():
        return spectrum, None

    text_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    n_dup = _dedupe_text_qpoints(text_lines)
    struct_lines = _structure_yaml_lines(
        spectrum.lattice,
        spectrum.atom_symbols,
        spectrum.atom_frac_coords,
        spectrum.atomic_masses,
    )
    out = Path(out_path) if out_path else path.with_name(f"{path.stem}_full.yaml")
    with out.open("w", encoding="utf-8") as f:
        f.writelines(text_lines)
        f.write("\n")
        f.write("\n".join(struct_lines))
        f.write("\n")
    LOGGER.info(
        f"wrote {out} (structure data from {poscar_path})"
        + (f"; removed {n_dup} duplicate q-point block(s)" if n_dup else "")
    )
    return spectrum, out


def parse_phonon_data(
    path: str | Path,
    poscar_path: str | Path | None = None,
) -> PhononSpectrum:
    """Load a PhononSpectrum from any supported phonon file format.

    Dispatches on the file extension: ``.yaml``/``.yml`` goes to
    :func:`parse_phonopy_yaml`, everything else (``.npz``) to
    :func:`parse_phonon_npz`. Raises ``ValueError`` for unsupported
    extensions.
    """
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        return parse_phonopy_yaml(p, poscar_path=poscar_path)
    if p.suffix == ".npz":
        return parse_phonon_npz(p)
    raise ValueError(f"Unsupported phonon file format: '{p.suffix}' (path: {p}). Expected .yaml, .yml or .npz")


def parse_phonon_npz(npz_path: str | Path) -> PhononSpectrum:
    """Loads a precomputed PhononSpectrum from a .npz file using PhononSpectrum.load."""
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"Phonon npz file not found: {path}")

    # Fall back to legacy keys if needed
    data = np.load(path, allow_pickle=True)
    if "frequencies" in data or "frequencies_mev" in data:
        try:
            return PhononSpectrum.load(path)
        except Exception:
            pass

    freqs = data["freqs"] if "freqs" in data else data["frequencies_mev"]
    eigs = data["eigs"] if "eigs" in data else data["eigenvectors"]
    atoms = data["atoms"] if "atoms" in data else data["atom_frac_coords"]
    symbols = list(data["atom_symbols"])
    if "masses" in data:
        masses = data["masses"]
    elif "atomic_masses" in data:
        masses = data["atomic_masses"]
    else:
        try:
            from pymatgen.core import Element

            masses = np.array([Element(s).atomic_mass for s in symbols], dtype=float)
        except Exception:
            masses = np.ones(len(symbols), dtype=float)
    lattice = data["lattice"]

    symmetries = list(data["symmetries"]) if "symmetries" in data else (list(data["sym"]) if "sym" in data else None)
    iprs = data["iprs"] if "iprs" in data else (data["ipr"] if "ipr" in data else None)
    # np.savez(..., iprs=None) stores a 0-d object array: not None but not a
    # usable per-mode array either. Normalize to None when scalar/empty.
    if iprs is not None:
        iprs = np.asarray(iprs)
        if iprs.ndim == 0 or iprs.size == 0:
            iprs = None
    e_pair_complete = (
        list(bool(x) for x in data["e_pair_complete"])
        if "e_pair_complete" in data and data["e_pair_complete"].ndim > 0
        else None
    )
    # Legacy files store the original DFT-run mode indices under "idx"; the spectrum
    # owns mode identity, so this must not be dropped on load.
    original_indices = None
    if "original_indices" in data and data["original_indices"] is not None:
        original_indices = np.asarray(data["original_indices"], dtype=int)
    elif "idx" in data and data["idx"] is not None:
        # idx is 0-based into the full 3N-mode run (verified: full_freqs[idx] ==
        # sym_freqs and all symmetry labels agree at these indices for NV_512).
        original_indices = np.asarray(data["idx"], dtype=int)

    return PhononSpectrum(
        frequencies_mev=np.asarray(freqs, dtype=float),
        eigenvectors=np.asarray(eigs, dtype=float),
        atom_frac_coords=np.asarray(atoms, dtype=float),
        atom_symbols=symbols,
        atomic_masses=np.asarray(masses, dtype=float),
        lattice=np.asarray(lattice, dtype=float),
        symmetries=symmetries,
        iprs=np.asarray(iprs, dtype=float) if iprs is not None else None,
        e_pair_complete=e_pair_complete,
        original_indices=original_indices,
    )


def save_phonon_npz(spectrum: PhononSpectrum, out_path: str | Path) -> str:
    """Saves a PhononSpectrum object to a .npz archive with explicit unit tracking."""
    return spectrum.save(out_path)


_RUN_FOLDER = {1: re.compile(r"^(\d+)$"), 2: re.compile(r"^(\d+)_(\d+)$")}


def _run_outcars(directory: Path, order: int) -> list[Path]:
    """OUTCARs of the run folders below ``directory``.

    Run folders are ``runs/<i>`` (first order) or ``runs/<i>_<j>`` (second order),
    directly under ``runs/`` and named exactly so. Nothing else counts: a copy such as
    ``old_data/85_85/OUTCAR`` or ``runs/85_85_old`` would otherwise get the same mode
    numbers and silently replace the real run (ClV_128's pre-fix pair runs did).
    """
    pattern = _RUN_FOLDER[order]
    runs = directory / "runs"
    return sorted(p for p in runs.glob("*/OUTCAR") if pattern.match(p.parent.name)) if runs.is_dir() else []


def _worker_parse_outcar_1d(outcar_file: Path) -> tuple[int, ZFSTensor, float | None] | None:
    zfs = parse_outcar_zfs(outcar_file)
    if zfs is None:
        return None
    try:
        match = _RUN_FOLDER[1].match(outcar_file.parent.name)
        if not match:
            return None
        index = int(match.group(1)) - 1
        energy = parse_outcar_energy(outcar_file)
        return index, zfs, energy
    except Exception:
        return None


def _worker_parse_outcar_2d(outcar_file: Path) -> tuple[tuple[int, int], ZFSTensor, float | None] | None:
    zfs = parse_outcar_zfs(outcar_file)
    if zfs is None:
        return None
    try:
        match = _RUN_FOLDER[2].match(outcar_file.parent.name)
        if not match:
            return None
        indices = (int(match.group(1)) - 1, int(match.group(2)) - 1)
        energy = parse_outcar_energy(outcar_file)
        return indices, zfs, energy
    except Exception:
        return None


def parse_perturbation_directory(
    directory: str | Path,
    order: int = 1,
    amplitude: float | tuple[float, float] = 1.0,
    max_workers: int = 4,
    verbose: bool = False,
) -> dict[Any, PerturbationEntry]:
    """
    Parses the run folders' OUTCARs under a directory for 1D or 2D displacements in parallel.

    Only ``runs/<i>/OUTCAR`` (order 1) or ``runs/<i>_<j>/OUTCAR`` (order 2) are read
    (see :func:`_run_outcars`); any other OUTCAR below the directory is ignored.
    """
    dir_path = Path(directory)
    outcar_files = _run_outcars(dir_path, order)
    if not outcar_files:
        if verbose:
            warnings.warn(f"No OUTCARs found in: {dir_path}")
        return {}
    if verbose:
        LOGGER.info(f"Parsing {len(outcar_files)} OUTCARs in {dir_path} ...")

    worker_fn = _worker_parse_outcar_1d if order == 1 else _worker_parse_outcar_2d
    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(worker_fn, outcar_files):
            if res is not None:
                key, zfs, energy = res
                mode_indices = (key,) if isinstance(key, int) else key
                results[key] = PerturbationEntry(
                    order=order,
                    mode_indices=mode_indices,
                    amplitude=amplitude,
                    zfs_tensor=zfs,
                    energy=energy,
                )

    if verbose:
        skipped = len(outcar_files) - len(results)
        suffix = f" ({skipped} skipped)" if skipped else ""
        LOGGER.info(f"  -> {len(results)}/{len(outcar_files)} parsed{suffix}")

    return results


def parse_zfs_simulation_dataset(
    sim_folder: str | Path,
    max_workers: int = 4,
    order: int | None = None,
    pert_scale: float | None = None,
    defect: str | None = None,
    cell_size: int | None = None,
    *,
    calc_method: str | None = None,
    zfs_folder: str | None = None,
    verbose: bool = False,
) -> RawZFSData:
    """
    Parses an entire simulation directory structure:
    - Extracts defect name, cell size, and perturbation scale from directory paths
    - Reads the unperturbed ground-state ZFS tensor
    - Reads all 1D or 2D perturbed calculations

    Explicit overrides (order, pert_scale, defect, cell_size) take precedence over
    values inferred from the directory path.
    """
    sim_path = Path(sim_folder)
    if not sim_path.is_dir():
        raise FileNotFoundError(f"Simulation folder not found: {sim_path}")

    if calc_method in ("all", "all_bands"):
        calc_method, default_zfs = ("all_bands", "ZFS_hyp")
    elif calc_method in ("approx", "defect_band_approx"):
        calc_method, default_zfs = ("defect_band_approx", "ZFS_occup")
    else:
        raise ValueError(f"{calc_method} is not a valid zfs calculation method.")

    final_zfs_folder = zfs_folder or default_zfs

    if "pert" not in sim_path.name:
        raise ValueError("Perturbation scale not found in folder name (expected e.g. 'pert_0.01').")

    defect_name, cell_size_inferred = split_defect_folder_name(sim_path.parent.parent.name)
    defect = defect or defect_name
    cell_size = cell_size or cell_size_inferred
    pert_scale = pert_scale if pert_scale is not None else float(sim_path.name.split("_")[1])

    # Unperturbed relaxed ground state
    relaxed_outcar = sim_path.parent.parent / final_zfs_folder / "OUTCAR"
    ground_state_zfs = parse_outcar_zfs(relaxed_outcar)
    if ground_state_zfs is None:
        raise ValueError(f"Relaxed ZFS tensor not found in: {relaxed_outcar}")

    search_path = sim_path / calc_method
    first_order = {}
    second_order = {}

    if order is None:
        # Infer from parent folder (e.g. 'first_order', 'second_order')
        order = 1 if "first" in sim_path.parent.name else 2 if "second" in sim_path.parent.name else None

    if order == 1:
        first_order = parse_perturbation_directory(
            search_path, order=1, amplitude=pert_scale, max_workers=max_workers, verbose=verbose
        )
    elif order == 2:
        second_order = parse_perturbation_directory(
            search_path, order=2, amplitude=(pert_scale, pert_scale), max_workers=max_workers, verbose=verbose
        )

    data = RawZFSData(
        defect=defect,
        cell_size=cell_size,
        pert_scale=pert_scale,
        calc_method=calc_method,
        ground_state_zfs=ground_state_zfs,
        order=order,
        first_order=first_order,
        second_order=second_order,
        metadata={"calc_method": calc_method, "sim_path": str(sim_path)},
    )
    # Apply the spin correction here so every consumer of parsed sim-folder
    # data sees corrected tensors, matching RawZFSData.load for npz files.
    if normalize_calc_method(data.calc_method) == "defect_band_approx":
        data._apply_spin_correction()
    return data


def parse_zfs_dataset_npz(raw_paths: list[str | Path] | tuple[str | Path, ...]) -> RawZFSData:
    """
    Loads raw 1D and 2D perturbation data from precomputed .npz files using RawZFSData.load.
    """
    return RawZFSData.load(raw_paths)
