"""stamp_phonon_frames.py -- one-off: record the defect frame finished phonon files were classified in.

Phonon npz files written before the defect frame was stored (PhononSpectrum.frame) were classified in
a frame fixed by the species: C3 axis [1, 1, 1] with sigma_v1 normal [1, -1, 0] when the cell holds C
and N (NV), else [0, 0, 1] with [1, 0, 0] (ClV, the divacancy, V_Si, ...). spinlabs now reads the
frame from the file, and derives it from the structure only for a file without one
(symmetry.frame_from_structure), which can pick another of the three mirrors (NV in a cubic cell),
swapping Ex and Ey. This stamps that legacy frame into each given file, so its labels, its kept
partners and the runs displaced along them stay as they are.

Each file gets one new member, frame.npy (symmetry.frame_matrix: columns n, z x n, z); every other
member is left byte for byte. Before stamping, the frame's operations are checked to be symmetries of
the file's structure (symmetry.frame_mismatch <= FRAME_TOL about the defect centre). A file that
already has a frame is left alone (reported, with whether it equals the legacy one). A stamped file
that is in the spinlabs registry gets its sha256 updated.

Usage:
    python scripts/stamp_phonon_frames.py <phonon.npz or folder> [...] [--dry-run]

A folder stands for every phonon npz below it (recursively): an npz with eigenvectors (eigs or
eigenvectors), lattice and atom_symbols, whatever its name; ZFS packages, coupling files and
convergence packs below it are passed over.

A phonopy yaml (named explicitly) is the phonon code's full spectrum: it is written as <stem>.npz next
to it, every mode as the yaml has it (not aligned), classified in the legacy frame and with that frame
stored. ``spinlabs symmetrize --align <stem>.npz`` then aligns it exactly as the code before stored
frames did, so the aligned files of runs made then can be recreated (an existing <stem>.npz is not
overwritten).

Run it wherever the finished defects' phonon files live (the cluster's defect folders and data/, and
local copies): an rsync from an unstamped copy overwrites a stamped file.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import sqlite3
import sys
import zipfile

import numpy as np

from spinlabs.models import PhononSpectrum
from spinlabs.registry import REGISTRY_PATH, sha256_of
from spinlabs.symmetry import FRAME_TOL, frame_matrix, frame_mismatch
from spinlabs.utils import resolve_path

NV_FRAME = frame_matrix([1.0, 1.0, 1.0], [1.0, -1.0, 0.0])
DEFAULT_FRAME = frame_matrix([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])


def is_phonon_npz(path) -> bool:
    """Whether ``path`` is a phonon npz: it holds eigenvectors and the structure (lattice, atom_symbols)."""
    try:
        with zipfile.ZipFile(path) as z:
            names = {n.removesuffix(".npy") for n in z.namelist()}
    except (zipfile.BadZipFile, OSError):
        return False
    return bool(names & {"eigs", "eigenvectors"}) and {"lattice", "atom_symbols"} <= names


def phonon_files(arguments) -> tuple[list, list[str]]:
    """The phonon npz files the arguments name (a folder: every one below it), and an error per argument
    that is missing or a file that is not a phonon npz."""
    files, errors = [], []
    for argument in arguments:
        path = resolve_path(argument)
        if path.is_dir():
            found = sorted(f for f in path.rglob("*.npz") if is_phonon_npz(f))
            print(f"{path}: {len(found)} phonon file(s) below it")
            files += found
        elif not path.exists():
            errors.append(f"{path}: no such file or folder")
        elif path.suffix.lower() in (".yaml", ".yml"):
            files.append(path)
        elif not is_phonon_npz(path):
            errors.append(f"{path}: not a phonon npz (no eigenvectors, lattice and atom_symbols)")
        else:
            files.append(path)
    return list(dict.fromkeys(files)), errors


def legacy_frame(symbols) -> np.ndarray:
    """The frame symmetry._defect_frame used before frames were stored."""
    return NV_FRAME if "C" in symbols and "N" in symbols else DEFAULT_FRAME


def structure_only(data) -> PhononSpectrum:
    """A mode-less spectrum of the file's structure (no classification runs), for the centre and the check."""
    frac = np.asarray(data["atom_frac_coords"] if "atom_frac_coords" in data else data["atoms"], dtype=float)
    symbols = [str(s) for s in data["atom_symbols"]]
    return PhononSpectrum(
        frequencies_mev=np.zeros(0),
        eigenvectors=np.zeros((0, len(symbols), 3)),
        atom_frac_coords=frac,
        atom_symbols=symbols,
        atomic_masses=np.ones(len(symbols)),
        lattice=np.asarray(data["lattice"], dtype=float),
    )


def append_member(path, name: str, array: np.ndarray) -> None:
    """Add ``name`` (.npy of ``array``) to the npz at ``path``, compressed like its other members."""
    with zipfile.ZipFile(path) as z:
        compression = z.infolist()[0].compress_type if z.infolist() else zipfile.ZIP_STORED
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(array, dtype=float), allow_pickle=False)
    tmp = path.with_name(path.name + ".stamping")
    shutil.copy2(path, tmp)
    try:
        with zipfile.ZipFile(tmp, "a", compression=compression) as z:
            z.writestr(name, buffer.getvalue())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def update_registry(path) -> bool:
    """Refresh the registered sha256 of ``path``; False when the registry or the row is absent."""
    if not REGISTRY_PATH.is_file():
        return False
    with sqlite3.connect(REGISTRY_PATH) as conn:
        cur = conn.execute("UPDATE files SET sha256 = ? WHERE path = ?", (sha256_of(path), str(path)))
        return cur.rowcount > 0


def npz_from_yaml(path, dry_run: bool) -> bool:
    """Write a phonopy yaml's full spectrum as <stem>.npz, classified in the legacy frame and stamped with it."""
    from dataclasses import replace

    from spinlabs.parsers import prepare_phonopy_yaml, save_phonon_npz

    out = path.with_suffix(".npz")
    if out.exists():
        print(f"{path}: {out.name} exists; not overwritten (stamp it instead: pass {out})")
        return True
    spectrum, _ = prepare_phonopy_yaml(path)
    frame = legacy_frame(list(spectrum.atom_symbols))
    mismatch = frame_mismatch(spectrum, frame)
    axis, normal = np.round(frame[:, 2], 4).tolist(), np.round(frame[:, 0], 4).tolist()
    if mismatch > FRAME_TOL:
        print(f"{path}: legacy frame (axis {axis}, normal {normal}) misplaces an atom by {mismatch:.3f} A "
              f"(> {FRAME_TOL} A); nothing written", file=sys.stderr)
        return False
    if dry_run:
        print(f"{path}: would write {out.name} ({spectrum.n_modes} modes) with axis {axis}, normal {normal}")
        return True
    # Parsing classified the modes in the frame of the structure: classify them again in the legacy one.
    spectrum = replace(spectrum, symmetries=None, deg_groups=None, e_pair_complete=None, pair_ids=None, frame=frame)
    save_phonon_npz(spectrum, out)
    print(f"{path}: wrote {out} ({spectrum.n_modes} modes, not aligned) with axis {axis}, normal {normal} "
          f"(mismatch {mismatch:.4f} A)")
    return True


def stamp(path, dry_run: bool) -> bool:
    """Stamp one file; False when it could not be (frame not a symmetry of the structure)."""
    if path.suffix.lower() in (".yaml", ".yml"):
        return npz_from_yaml(path, dry_run)
    with np.load(path, allow_pickle=True) as npz:
        data = {k: npz[k] for k in npz.files if k in ("atom_frac_coords", "atoms", "atom_symbols", "lattice", "frame")}
    frame = legacy_frame([str(s) for s in data["atom_symbols"]])
    if "frame" in data:
        same = np.allclose(data["frame"], frame, atol=1e-12)
        print(f"{path}: already has a frame ({'the legacy one' if same else 'NOT the legacy one'}); left alone")
        return True
    spectrum = structure_only(data)
    mismatch = frame_mismatch(spectrum, frame)
    axis, normal = np.round(frame[:, 2], 4).tolist(), np.round(frame[:, 0], 4).tolist()
    if mismatch > FRAME_TOL:
        print(f"{path}: legacy frame (axis {axis}, normal {normal}) misplaces an atom by {mismatch:.3f} A "
              f"(> {FRAME_TOL} A); NOT stamped", file=sys.stderr)
        return False
    if dry_run:
        print(f"{path}: would stamp axis {axis}, normal {normal} (mismatch {mismatch:.4f} A)")
        return True
    append_member(path, "frame.npy", frame)
    registered = update_registry(path)
    print(f"{path}: stamped axis {axis}, normal {normal} (mismatch {mismatch:.4f} A)"
          + ("; registry sha256 updated" if registered else ""))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", help="phonon .npz files of finished defects, or folders to search")
    parser.add_argument("--dry-run", action="store_true", help="check and report, write nothing")
    args = parser.parse_args()
    files, errors = phonon_files(args.paths)
    for error in errors:
        print(error, file=sys.stderr)
    ok = [stamp(f, args.dry_run) for f in files]
    return 0 if all(ok) and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
