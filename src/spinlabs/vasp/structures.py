"""Perturb VASP structures along phonon normal coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import itertools
import re

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar

from spinlabs.models import PhononSpectrum
from spinlabs.parsers import parse_phonon_data


def load_poscar(poscar_file: str | Path) -> Structure:
    """Load a VASP POSCAR file into a pymatgen Structure."""
    return Poscar.from_file(poscar_file).structure


def load_structure(path: str | Path) -> tuple[Structure, Path]:
    """The structure for ``path`` and the file it was read from.

    An OUTCAR (or a run directory containing one) gives the geometry VASP
    actually computed: the lattice and ion positions printed in the OUTCAR,
    with the species from the POSCAR next to it. This is the geometry behind
    that run's ZFS result, even when POSCAR/CONTCAR were rewritten by a later
    attempt. A run directory without an OUTCAR, or any other file, is read as
    POSCAR-format (POSCAR, CONTCAR, ...).
    """
    path = Path(path)
    if path.is_dir():
        path = path / "OUTCAR" if (path / "OUTCAR").is_file() else path / "POSCAR"
    if not path.is_file():
        raise FileNotFoundError(f"Structure file not found: {path}")
    if path.name.startswith("OUTCAR"):
        return _outcar_structure(path), path
    return load_poscar(path), path


def _outcar_structure(outcar: Path) -> Structure:
    """Initial lattice and Cartesian positions from an OUTCAR; species from the run's POSCAR."""
    species_file = next((outcar.with_name(n) for n in ("POSCAR", "CONTCAR") if outcar.with_name(n).is_file()), None)
    if species_file is None:
        raise FileNotFoundError(f"No POSCAR/CONTCAR next to {outcar} to take the species from")
    species = [site.specie for site in load_poscar(species_file)]
    markers = {"direct lattice vectors": 3, "position of ions in cartesian coordinates": len(species)}
    blocks: dict[str, np.ndarray] = {}
    # Both blocks are in the OUTCAR header: stop reading once they are found.
    with open(outcar, errors="replace") as f:
        for line in f:
            marker = next((m for m in markers if m in line and m not in blocks), None)
            if marker is None:
                continue
            rows = [next(f, "").split() for _ in range(markers[marker])]
            try:
                blocks[marker] = np.array([[float(x) for x in row[:3]] for row in rows])
            except (ValueError, IndexError):
                break
            if len(blocks) == len(markers):
                break
    missing = [m for m in markers if m not in blocks]
    if missing:
        raise ValueError(f"{outcar}: no complete '{missing[0]}' block (unfinished run?)")
    lattice, positions = blocks["direct lattice vectors"], blocks["position of ions in cartesian coordinates"]
    return Structure(lattice, species, positions, coords_are_cartesian=True)


@dataclass(frozen=True)
class DefectLocation:
    """A candidate point defect found by find_defect.

    For a vacancy, frac_coords is the Cartesian centroid of the
    under-coordinated neighbours, mapped back into fractional
    coordinates, and site_index is None. For an interstitial,
    frac_coords is the centroid of the close-pair cluster and
    neighbor_indices lists its (ambiguous) members.
    """

    defect_class: str  # "vacancy" | "divacancy" | "interstitial" | "substitutional"
    frac_coords: np.ndarray
    site_index: int | None
    neighbor_indices: tuple[int, ...]
    species: str | None = None  # for substitutionals: the substituent species


def _local_metrics(structure: Structure) -> tuple[np.ndarray, float, np.ndarray]:
    """Per-site coordination numbers, the cutoff used, and nearest-
    neighbour distances.

    The cutoff is 1.15 x the median nearest-neighbour distance, which
    separates first-shell neighbours from second-shell ones in the
    lattices this pipeline targets (diamond-like and wurtzite-like).
    """
    nn_dists = np.array(
        [min(s.distance(other) for j, other in enumerate(structure) if j != i) for i, s in enumerate(structure)]
    )
    cutoff = 1.15 * float(np.median(nn_dists))
    cn = np.array([len(structure.get_neighbors(s, cutoff)) for s in structure])
    return cn, cutoff, nn_dists


def _cluster_indices(indices: list[int], structure: Structure, cutoff: float) -> list[list[int]]:
    """Group indices into clusters whose sites are within cutoff of each
    other (single-linkage, periodic-aware)."""
    remaining = set(indices)
    clusters: list[list[int]] = []
    while remaining:
        seed = remaining.pop()
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            for i in list(remaining):
                if any(structure.get_distance(i, j) <= cutoff for j in cluster):
                    cluster.append(i)
                    remaining.discard(i)
                    changed = True
        clusters.append(sorted(cluster))
    return clusters


def _periodic_mean(fracs: np.ndarray) -> np.ndarray:
    """Mean of fractional coordinates that is safe across cell boundaries.

    Each point is unwrapped relative to the first one before averaging.
    """
    disp = (fracs - fracs[0] + 0.5) % 1.0 - 0.5
    return fracs[0] + disp.mean(axis=0)


def _cluster_centroid(fracs: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Fractional centroid of a bonded cluster, safe across cell edges.

    The cluster is unwrapped breadth-first: start from the first
    member, then repeatedly place the remaining member whose closest
    periodic image to the unwrapped set is nearest (in Cartesian
    distance). Bonded members sit within one bond length of the
    cluster, so that image is unambiguous, and the resulting Cartesian
    mean does not mix up periodic images. Works for vacancy shells
    that straddle a supercell boundary.
    """
    placed = [fracs[0].copy()]
    remaining = list(range(1, len(fracs)))
    while remaining:
        best = None
        for j in remaining:
            for i in placed:
                for t in itertools.product((-1, 0, 1), repeat=3):
                    cand = fracs[j] + np.array(t)
                    dist = np.linalg.norm((cand - i) @ lattice)
                    if best is None or dist < best[0]:
                        best = (dist, j, cand)
        placed.append(best[2])
        remaining.remove(best[1])
    cart = np.mean([q @ lattice for q in placed], axis=0)
    return np.linalg.solve(lattice.T, cart)


def find_defect(
    structure: Structure,
) -> list[DefectLocation]:
    """Locate point defects in a supercell without a pristine reference.

    Self-contained local-geometry heuristic:

    - An atom whose nearest neighbour is much closer than the lattice
      bond length is flagged as an interstitial (it squeezes against
      the lattice; the close-pair cluster marks where).
    - Sites whose coordination number is below the mode coordination
      number of their species are neighbours of a missing atom; their
      clusters each trace one vacancy, whose position is estimated as
      the Cartesian centroid of the cluster.

    Returns one DefectLocation per detected defect, sorted by
    fractional coordinates. An empty list means no defect was found.
    """
    cn, cutoff, nn_dists = _local_metrics(structure)

    # Interstitials: an extra atom squeezes against the lattice, so its
    # nearest neighbour is anomalously short. Sites whose NN distance
    # is below 0.75 x the median form close-pair clusters; the cluster
    # is reported as one interstitial (which member is the extra atom
    # is not resolvable without a pristine reference).
    defects: list[DefectLocation] = []
    close_sites = [i for i in range(len(structure)) if nn_dists[i] < 0.75 * float(np.median(nn_dists))]
    for cluster in _cluster_indices(close_sites, structure, cutoff):
        cart_centroid = np.mean([structure[i].coords for i in cluster], axis=0)
        defects.append(
            DefectLocation(
                defect_class="interstitial",
                frac_coords=structure.lattice.get_fractional_coords(cart_centroid) % 1.0,
                site_index=None,
                neighbor_indices=tuple(cluster),
            )
        )
    interstitial_sites = {i for d in defects if d.defect_class == "interstitial" for i in d.neighbor_indices}
    # Vacancies: clusters of under-coordinated sites relative to the
    # per-species mode coordination number.
    under_coord: dict[str, list[int]] = {}
    mode_cn_by_species: dict[str, int] = {}
    for species in {s.specie.symbol for s in structure}:
        members = [i for i, s in enumerate(structure) if s.specie.symbol == species]
        if len(members) < 2:
            continue
        mode_cn = int(np.bincount(cn[members].astype(int)).argmax())
        mode_cn_by_species[species] = mode_cn
        under_coord[species] = [i for i in members if cn[i] < mode_cn]

    neighbour_pool = sorted(set(sum(under_coord.values(), [])) - interstitial_sites)
    if neighbour_pool:
        # Neighbours of one defect are not all bonded to each other:
        # first-shell neighbours sit ~1.63 bond lengths apart, and a
        # divacancy's shell spans up to ~2 bond lengths. A wide reach
        # (2.1 x) still keeps defects 3+ bond lengths apart separate.
        vac_cutoff = 2.1 * cutoff
        for cluster in _cluster_indices(neighbour_pool, structure, vac_cutoff):
            # Each missing atom removes mode_cn bonds, but bonds
            # between two removed atoms touch no surviving site, so
            # deficit undercounts. A single vacancy leaves deficit
            # mode_cn; an adjacent divacancy leaves 2*mode_cn - 2.
            deficit = sum(mode_cn_by_species[structure[i].specie.symbol] - cn[i] for i in cluster)
            defect_class = "divacancy" if deficit > max(mode_cn_by_species.values()) else "vacancy"
            frac_centroid = _cluster_centroid(
                np.array([structure[i].frac_coords for i in cluster]),
                structure.lattice.matrix,
            )
            defects.append(
                DefectLocation(
                    defect_class=defect_class,
                    frac_coords=frac_centroid % 1.0,
                    site_index=None,
                    neighbor_indices=tuple(cluster),
                )
            )

    # Substitutionals: group sites by their neighbour-species
    # signature; the majority species within a signature defines the
    # host species for that sublattice, so a minority-species site is
    # the substituent.
    signatures: dict[tuple, list[int]] = {}
    for i in range(len(structure)):
        if i in interstitial_sites:
            continue
        counts: dict[str, int] = {}
        for n in structure.get_neighbors(structure[i], cutoff):
            counts[n.specie.symbol] = counts.get(n.specie.symbol, 0) + 1
        sig = tuple(sorted(counts.items()))
        signatures.setdefault(sig, []).append(i)
    for sig, members in signatures.items():
        species_counts = np.bincount([structure[i].specie.Z for i in members])
        majority_z = int(np.argmax(species_counts))
        for i in members:
            if structure[i].specie.Z != majority_z:
                defects.append(
                    DefectLocation(
                        defect_class="substitutional",
                        frac_coords=np.array(structure[i].frac_coords),
                        site_index=i,
                        neighbor_indices=(),
                        species=structure[i].specie.symbol,
                    )
                )

    defects.sort(key=lambda d: tuple(d.frac_coords))
    return defects


def load_phonon_data(phonon_file: str | Path) -> PhononSpectrum:
    """Load phonon data (npz or phonopy yaml) via the general parser.

    The eigenvectors are the raw phonopy eigenvectors of the dynamical
    matrix, normalized as sum_ja |e_ja|^2 = 1.
    """
    if not Path(phonon_file).exists():
        raise FileNotFoundError(f"Phonon data file not found: {phonon_file}")

    spectrum = parse_phonon_data(phonon_file)
    if spectrum.eigenvectors.ndim != 3 or spectrum.eigenvectors.shape[2] != 3:
        raise ValueError(f"Expected eigenvectors shape (n_modes, n_atoms, 3), got {spectrum.eigenvectors.shape}")
    return spectrum


def _displacement(eigenvectors: np.ndarray, masses: np.ndarray, index0: int, amplitude: float) -> np.ndarray:
    """Mass-weighted displacement for one mode, in Angstrom.

    amplitude is the normal coordinate Q in Angstrom*sqrt(amu):
    Delta r_alpha(j) = Q * e_alpha(j) / sqrt(m_j).
    """
    return amplitude * eigenvectors[index0] / np.sqrt(masses[:, None])


def _resolve_mode(mode_index: int, original_indices: np.ndarray | None, n_modes: int) -> int:
    """Resolve a 1-based mode label to an eigenvector array row.

    Symmetry-filtered phonon files keep the *original* mode labels in
    ``original_indices`` while the eigs array is trimmed, so the label is
    not the array row. When original_indices is given (1-based labels,
    same convention as spinlabs.vasp.modes), map label -> row by exact match;
    a missing or non-unique label is a hard error. Otherwise the label
    must simply be within [1, n_modes].
    """
    if original_indices is None or len(original_indices) == 0:
        if not (1 <= mode_index <= n_modes):
            raise ValueError(f"Mode {mode_index} out of range [1, {n_modes}]")
        return mode_index - 1
    hits = np.flatnonzero(original_indices == mode_index - 1)
    if len(hits) == 0:
        raise ValueError(
            f"Mode {mode_index} not present in trimmed phonon data "
            f"(original indices available: "
            f"{int(original_indices.min()) + 1}..{int(original_indices.max()) + 1})"
        )
    if len(hits) > 1:
        raise ValueError(f"Mode {mode_index} is ambiguous: matches rows {hits.tolist()}")
    return int(hits[0])


def _check_mode(eigenvectors: np.ndarray, structure: Structure, masses: np.ndarray, index0: int, n_modes: int) -> None:
    if not (0 <= index0 < n_modes):
        raise ValueError(f"Mode index {index0 + 1} out of range [1, {n_modes}]")
    if len(structure) != eigenvectors.shape[1]:
        raise ValueError(f"Structure atoms ({len(structure)}) != Eigenvector atoms ({eigenvectors.shape[1]})")
    if len(masses) != len(structure):
        raise ValueError(f"Mass array length {len(masses)} != n_atoms {len(structure)}")


def apply_perturbation(
    structure: Structure,
    eigs: np.ndarray,
    masses: np.ndarray,
    mode_index: int,
    amplitude: float,
    original_indices: np.ndarray | None = None,
) -> Structure:
    """Apply a mass-weighted single-mode perturbation to a structure.

    mode_index is 1-based; amplitude is Q in Angstrom*sqrt(amu). For a
    carbon (12 amu), Q=0.1 gives per-atom displacements ~0.1/sqrt(12) ~
    0.029 Angstrom.
    """
    n_modes, _, _ = eigs.shape
    idx = _resolve_mode(mode_index, original_indices, n_modes)
    _check_mode(eigs, structure, masses, idx, n_modes)

    disp = _displacement(eigs, masses, idx, amplitude)  # (n_atoms, 3)

    perturbed = structure.copy()
    for j in range(len(structure)):
        perturbed.sites[j].coords = structure.sites[j].coords + disp[j]

    print(f"Mode {mode_index}: max atomic displacement = {np.max(np.linalg.norm(disp, axis=1)):.6f} Å")
    return perturbed


def apply_combined_perturbation(
    structure: Structure,
    eigenvectors: np.ndarray,
    mode_i: int,
    mode_j: int,
    masses: np.ndarray,
    amplitude: float,
    original_indices: np.ndarray | None = None,
) -> Structure:
    """Apply a two-mode combined perturbation (Mode I + Mode J, 1-based)."""
    n_modes, _, _ = eigenvectors.shape
    idx_i = _resolve_mode(mode_i, original_indices, n_modes)
    idx_j = _resolve_mode(mode_j, original_indices, n_modes)
    _check_mode(eigenvectors, structure, masses, idx_i, n_modes)
    _check_mode(eigenvectors, structure, masses, idx_j, n_modes)

    disp = _displacement(eigenvectors, masses, idx_i, amplitude) + _displacement(eigenvectors, masses, idx_j, amplitude)

    perturbed = structure.copy()
    for j in range(len(structure)):
        perturbed.sites[j].coords = structure.sites[j].coords + disp[j]

    print(
        f"Applied combined perturbation: Mode {mode_i} + Mode {mode_j}, "
        f"amplitude {amplitude} Å; max displacements "
        f"{np.max(np.linalg.norm(_displacement(eigenvectors, masses, idx_i, amplitude), axis=1)):.6f} Å "
        f"and {np.max(np.linalg.norm(_displacement(eigenvectors, masses, idx_j, amplitude), axis=1)):.6f} Å"
    )
    return perturbed


def compare_displacement(reference: str | Path, perturbed: str | Path) -> dict:
    """Displacement of a perturbed structure relative to a reference.

    Returns per-atom displacement vectors (n_atoms, 3) plus summary
    statistics. Atoms are matched by index: both files must come from the
    same phonon/defect tree, so the atom order is identical. Fractional
    differences are wrapped to the nearest periodic image, so a structure
    written into a neighbouring cell by VASP compares as identical.

    Each argument may be a POSCAR/CONTCAR, a run directory, or an OUTCAR;
    see :func:`load_structure`. ``source`` in the result is the file read.
    """
    ref, _ = load_structure(reference)
    pert, source = load_structure(perturbed)
    if len(ref) != len(pert):
        raise ValueError(f"Atom count mismatch: {len(ref)} vs {len(pert)}")
    if ref.composition.reduced_formula != pert.composition.reduced_formula:
        raise ValueError(
            f"Composition mismatch: {ref.composition.reduced_formula} vs {pert.composition.reduced_formula}"
        )
    # Work in fractional coordinates and wrap the difference to the
    # nearest periodic image, so identical structures that VASP re-wrote
    # into a neighbouring cell do not show up as huge fake displacements.
    frac = np.array([p.frac_coords for p in pert]) - np.array([p.frac_coords for p in ref])
    frac -= np.round(frac)
    disp = frac @ np.array(ref.lattice.matrix)
    norms = np.linalg.norm(disp, axis=1)
    return {
        "disp": disp,
        "max": float(norms.max()),
        "rms": float(np.sqrt((norms**2).mean())),
        "max_index": int(norms.argmax()),
        "n_atoms": len(ref),
        "source": source,
    }


def run_modes(path: str | Path) -> tuple[int, ...] | None:
    """1-based modes of a perturbation run folder: ``runs/<i>`` (first order) or ``runs/<i>_<j>`` (pair)."""
    path = Path(path)
    folder = path if path.is_dir() else path.parent
    match = re.fullmatch(r"(\d+)(?:_(\d+))?", folder.name)
    return tuple(int(g) for g in match.groups() if g) if match else None


def run_pert_scale(path: str | Path) -> float | None:
    """Perturbation scale from a ``pert_<x>`` component of the path, or None."""
    for part in reversed(Path(path).resolve().parts):
        match = re.fullmatch(r"pert_(-?[0-9.]+)", part)
        if match:
            return float(match.group(1))
    return None


def expected_displacement(
    spectrum: PhononSpectrum, modes: tuple[int, ...], amplitude: float, *, mass_weighted: bool = True
) -> np.ndarray:
    """Per-atom displacement (n_atoms, 3) in Å for ``amplitude`` along each 1-based mode, summed.

    This is what the generator applies: a pair folder i_j displaces along both
    modes and i_i twice along one. Without mass weighting each mode is displaced
    by Q * e instead of Q * e / sqrt(m), as the earliest perturbation runs were.
    """
    eigs = np.real(np.asarray(spectrum.eigenvectors))
    masses = np.asarray(spectrum.atomic_masses, dtype=float) if mass_weighted else np.ones(eigs.shape[1])
    total = np.zeros(eigs.shape[1:])
    for mode in modes:
        total += _displacement(eigs, masses, _resolve_mode(mode, spectrum.original_indices, eigs.shape[0]), amplitude)
    return total


def compare_displacement_cli(
    reference: str | Path,
    perturbed: list[str | Path],
    *,
    spectrum: PhononSpectrum | None = None,
    amplitude: float | None = None,
    mass_weighted: bool = True,
) -> None:
    """Print a table of displacement statistics for one or more structures.

    With ``spectrum`` each run folder's modes (``runs/<i>`` or ``runs/<i>_<j>``)
    are compared with the expected displacement at ``amplitude`` (default: the
    path's ``pert_<x>``): the fitted scale is actual/expected along the expected
    direction, and the residual is the part of the displacement it leaves out.
    """
    rows = []
    for path in perturbed:
        try:
            stats = compare_displacement(reference, path)
        except (FileNotFoundError, ValueError) as e:
            print(f"{path}: {e}")
            continue
        rows.append((path, stats))
        print(
            f"{path}  (geometry from {stats['source'].name})\n    max |disp| = {stats['max']:.6f} Å "
            f"(atom {stats['max_index'] + 1})"
            f"\n    rms        = {stats['rms']:.6f} Å"
            f"\n    n_atoms    = {stats['n_atoms']}"
        )
        if spectrum is not None:
            print(_expected_report(path, stats["disp"], spectrum, amplitude, mass_weighted))
        print()
    # Relative sizes need a displaced first entry; identical structures (e.g. a ground state
    # checked against the tree's input) have nothing to compare.
    if len(rows) > 1 and rows[0][1]["max"] > 0:
        print("Relative sizes (max |disp|, first entry = 1.000):")
        base = rows[0][1]["max"]
        for path, stats in rows:
            print(f"    {stats['max'] / base:8.3f}  {path}")
    elif rows and all(stats["max"] == 0 for _, stats in rows):
        print("All structures are identical to the reference.")


def _expected_report(
    path: str | Path, disp: np.ndarray, spectrum: PhononSpectrum, amplitude: float | None, mass_weighted: bool
) -> str:
    modes = run_modes(path)
    amplitude = amplitude if amplitude is not None else run_pert_scale(path)
    if modes is None or amplitude is None:
        return "    expected   = ? (modes need a runs/<i> or runs/<i>_<j> folder, amplitude --pert or pert_<x>)"
    try:
        expected = expected_displacement(spectrum, modes, amplitude, mass_weighted=mass_weighted)
    except ValueError as e:
        return f"    expected   = ? ({e})"
    if expected.shape != disp.shape:
        return f"    expected   = ? (phonon file has {len(expected)} atoms, structure {len(disp)})"
    norms = np.linalg.norm(expected, axis=1)
    scale = float(np.sum(disp * expected) / np.sum(expected * expected))
    residual = float(np.linalg.norm(disp - scale * expected) / np.linalg.norm(disp)) if np.any(disp) else 0.0
    weighting = "mass-weighted" if mass_weighted else "not mass-weighted"
    return (
        f"    expected   = max {norms.max():.6f} Å, rms {np.sqrt((norms**2).mean()):.6f} Å "
        f"(modes {'+'.join(map(str, modes))}, Q={amplitude:g}, {weighting})"
        f"\n    scale      = {scale:.3f} x expected (residual {residual:.1%})"
    )


def write_perturbed_poscar(
    structure: Structure, output_file: str, comment: str, template_poscar: str | Path | None = None
) -> None:
    """Write a perturbed POSCAR.

    Without template_poscar, a plain pymatgen Poscar is written.  With a
    template, the template's header (including its species grouping --
    pymatgen's Poscar merges repeated species groups like 'Si Si C C Cl'
    into 'Si C Cl') is kept verbatim and only the coordinate block is
    replaced.  Atom order in `structure` must match the template file,
    which holds for structures loaded from it with load_poscar.
    """
    if template_poscar is None:
        poscar = Poscar(structure)
        poscar.comment = comment
        poscar.write_file(output_file)
        return

    lines = Path(template_poscar).read_text().splitlines()
    out = [comment] + lines[1:7]  # comment, scale, lattice, species, counts
    i = 7
    if lines[i].strip().lower().startswith("s"):  # selective dynamics
        out.append(lines[i])
        i += 1
    coord_mode = lines[i].strip().lower()
    out.append(lines[i])
    direct = coord_mode.startswith("d")
    lat = np.array(structure.lattice.matrix)
    for site in structure:
        frac = site.frac_coords if direct else np.asarray(site.coords) @ np.linalg.inv(lat)
        out.append("  %.16f  %.16f  %.16f" % tuple(frac))
    Path(output_file).write_text("\n".join(out) + "\n")


def spectrum_structure(spectrum: PhononSpectrum) -> Structure:
    """Build a pymatgen Structure from a PhononSpectrum's atom list."""
    frac = np.mod(np.asarray(spectrum.atom_frac_coords, dtype=float), 1.0)
    return Structure(
        spectrum.lattice,
        list(spectrum.atom_symbols),
        frac,
        coords_are_cartesian=False,
    )


def detect_defect_position(spectrum: PhononSpectrum) -> np.ndarray | None:
    """Fractional defect-centre coords from find_defect, or None.

    The PhononSpectrum supercell must contain exactly one point defect;
    multiple detections are a hard error rather than an arbitrary pick.
    """
    defects = find_defect(spectrum_structure(spectrum))
    if not defects:
        return None
    if len(defects) > 1:
        classes = sorted(d.defect_class for d in defects)
        raise ValueError(f"Expected one point defect in supercell, found {len(defects)}: {classes}")
    return defects[0].frac_coords % 1.0
