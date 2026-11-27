#!/usr/bin/env python3
"""align_wavecar.py -- rotate the unpaired down-spin orbitals of a VASP WAVECAR onto the up-spin ones.

Standalone: needs only numpy. The same alignment as spinlabs.vasp.wavecar.align_unpaired (what the
perturbation runner does after every SCF), with --check, --out, --bands and any number of k-points
(the rotation is done per k-point). Needs a spin-polarised (ISPIN = 2) WAVECAR of vasp_std, single
or double precision.

vasp_std only
-------------
A vasp_gam WAVECAR stores only half of the plane waves (G and -G are tied by psi(-G) = psi(G)^*) and
carries the same precision tag, so its header looks like a vasp_std one. The overlaps would then
miss the factor 2 on the G != 0 terms, and a complex rotation of the stored half breaks the
real-wavefunction condition. The script therefore counts the plane waves of the full ENCUT sphere
at each k-point from the header's ENCUT and lattice and refuses (exit 1, nothing written) when the
stored count is not that: about half of it is a vasp_gam file. Rerun the SCF with vasp_std (a
Gamma-centred 1x1x1 KPOINTS is fine) to align it.

Why
---
For a spin-flip (broken-symmetry) determinant built from a high-spin WAVECAR with ALGO = None, the
flipped electron moves from up band k to down band k. That is only the same spatial orbital when the
two spin channels chose the same basis for the unpaired levels (for the NV centre the e pair, bands
127/128 in a 64-atom cell). The SCF picks that basis at random in each channel, so e.g. "down 127"
can be mostly "up 128". This holds also when a displacement splits a degenerate level: an E mode
splits the NV e pair, and the two channels can order the split pair oppositely. The down unpaired
orbitals are empty in the SCF, so the rotation leaves its state, density and energy unchanged; only
which down orbital carries which band index changes.

How
---
For the bands B of a set, M_ij = <up_i|down_j> (plane-wave part, each orbital normalised).
With the SVD M = W S V^H, the unitary U = V W^H maximises Re tr(M U); the new down bands are
down'_j = sum_i U_ij down_i, after which <up_i|down'_j> = W S W^H is Hermitian with the largest
possible diagonal. Eigenvalues and occupations in the WAVECAR are left as they are.

Which bands
-----------
By default, the unpaired bands (occupied up, empty down, occupation threshold 0.5) are read from
the EIGENVAL of the run that wrote the WAVECAR (the one next to it, first k-point), and each run of consecutive unpaired bands
with two or more members is aligned as one set, degenerate or not; a single band needs nothing. Or
give the bands yourself with --bands (1-based, aligned as one set).

The script refuses (exit 1, nothing written) when the up and down bands of a set do not span the
same space (smallest singular value of M below --min-span): then no rotation makes a flip land in
the orbital it left.

Usage
-----
    python align_wavecar.py RUN_DIR                    # RUN_DIR/WAVECAR in place, bands from RUN_DIR/EIGENVAL
    python align_wavecar.py WAVECAR --eigenval EIGENVAL  # explicit files
    python align_wavecar.py RUN_DIR --check            # only report the overlaps, write nothing
    python align_wavecar.py RUN_DIR --out WAVECAR.aligned
    python align_wavecar.py WAVECAR --bands 127 128    # these bands, no EIGENVAL needed

A second run on an aligned WAVECAR changes nothing (rotation 0).
Overlaps use only the plane-wave part (no PAW augmentation), so aligned sets show |<up|down>|
somewhat below 1 (e.g. 0.98); that is not a remaining rotation.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

DTYPES = {45200: np.complex64, 45210: np.complex128}  # rtag -> plane-wave coefficient precision
HSQDTM = 13.605826 * 0.529177249**2  # hbar^2 / 2 m_e in eV A^2 (VASP's RYTOEV * AUTOA^2)
SPHERE_TOL = 0.01  # relative slack of the stored plane-wave count against the ENCUT sphere


def sphere_count(lattice: np.ndarray, encut: float, kpoint: np.ndarray) -> int:
    """Plane waves G with hbar^2 |k + G|^2 / 2m < ``encut`` (eV): the full sphere vasp_std stores.

    ``lattice`` rows are the direct lattice vectors (A), ``kpoint`` is in reciprocal-lattice units.
    """
    recip = 2 * np.pi * np.linalg.inv(lattice).T  # rows b_i, a_i . b_j = 2 pi delta_ij
    gmax = np.sqrt(encut / HSQDTM)
    # |n_i + k_i| = |(k + G) . a_i| / 2 pi <= gmax |a_i| / 2 pi
    nmax = (gmax * np.linalg.norm(lattice, axis=1) / (2 * np.pi) + np.abs(kpoint)).astype(int) + 1
    n1, n2 = np.meshgrid(np.arange(-nmax[0], nmax[0] + 1), np.arange(-nmax[1], nmax[1] + 1), indexing="ij")
    plane = (n1[..., None] + kpoint[0]) * recip[0] + (n2[..., None] + kpoint[1]) * recip[1]
    count = 0
    for n3 in range(-nmax[2], nmax[2] + 1):  # one plane at a time keeps large cells cheap
        q = plane + (n3 + kpoint[2]) * recip[2]
        count += int(np.count_nonzero(HSQDTM * np.einsum("...i,...i", q, q) < encut))
    return count


# --------------------------------------------------------------------------- #
# WAVECAR records
# --------------------------------------------------------------------------- #
class Wavecar:
    """Record layout of a WAVECAR (records of ``recl`` bytes).

    record 0: recl, nspin, rtag; record 1: nk, nb, encut, lattice (9), efermi; then for each spin and
    k-point one header record (nplw, k (3), and eig, 0, occ per band) followed by one record per band
    with its nplw plane-wave coefficients.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            recl, nspin, rtag = np.fromfile(f, dtype=np.float64, count=3).astype(int)
            if rtag not in DTYPES:
                raise ValueError(f"{path}: unknown precision tag {rtag} (known: {sorted(DTYPES)})")
            f.seek(recl)
            rec = np.fromfile(f, dtype=np.float64, count=12)
        if nspin != 2:
            raise ValueError(f"{path}: ISPIN = {nspin}; a spin-polarised (ISPIN = 2) WAVECAR is needed")
        self.recl, self.nspin, self.rtag = int(recl), int(nspin), int(rtag)
        self.nk, self.nb = int(rec[0]), int(rec[1])
        self.encut, self.lattice = float(rec[2]), rec[3:12].reshape(3, 3)
        self.dtype = DTYPES[self.rtag]
        self.check_full_sphere()

    def check_full_sphere(self) -> None:
        """ValueError unless every k-point stores the full ENCUT sphere of plane waves (vasp_std).

        vasp_gam stores about half of it under the same precision tag; the overlaps would then miss
        the factor 2 on G != 0 and a complex rotation would break psi(-G) = psi(G)^*.
        """
        for spin in range(self.nspin):
            for k in range(self.nk):
                with open(self.path, "rb") as f:
                    f.seek(self._k_record(spin, k) * self.recl)
                    rec = np.fromfile(f, dtype=np.float64, count=4)
                nplw, kpoint = int(rec[0]), rec[1:4]
                full = sphere_count(self.lattice, self.encut, kpoint)
                if abs(nplw - full) <= SPHERE_TOL * full:
                    continue
                where = f"{self.path}, spin {spin + 1}, k-point {k + 1}: {nplw} plane waves stored"
                if abs(nplw - (full + 1) / 2) <= SPHERE_TOL * full:
                    raise ValueError(
                        f"{where}, half of the {full} in the ENCUT = {self.encut:g} eV sphere: a Gamma-only "
                        "(vasp_gam) WAVECAR. Only vasp_std WAVECARs can be aligned; rerun the SCF with vasp_std"
                    )
                raise ValueError(
                    f"{where}, but the ENCUT = {self.encut:g} eV sphere holds {full} (vasp_gam would store "
                    f"{(full + 1) // 2}); not a vasp_std WAVECAR this script can read"
                )

    def _k_record(self, spin: int, k: int) -> int:
        return 2 + spin * self.nk * (self.nb + 1) + k * (self.nb + 1)

    def k_header(self, spin: int, k: int) -> tuple[int, np.ndarray]:
        """(nplw, eigenvalues) of 0-based ``spin`` and ``k``."""
        with open(self.path, "rb") as f:
            f.seek(self._k_record(spin, k) * self.recl)
            rec = np.fromfile(f, dtype=np.float64, count=4 + 3 * self.nb)
        return int(rec[0]), rec[4::3]

    def read(self, spin: int, k: int, bands) -> np.ndarray:
        """Coefficients of 1-based ``bands``: (len(bands), nplw), complex128."""
        nplw, _ = self.k_header(spin, k)
        out = []
        with open(self.path, "rb") as f:
            for b in bands:
                f.seek((self._k_record(spin, k) + b) * self.recl)
                out.append(np.fromfile(f, dtype=self.dtype, count=nplw))
        return np.array(out, dtype=np.complex128)

    def write(self, spin: int, k: int, bands, coeffs: np.ndarray) -> None:
        """Overwrite the coefficients of 1-based ``bands`` in place, in the file's precision."""
        with open(self.path, "r+b") as f:
            for b, c in zip(bands, coeffs):
                f.seek((self._k_record(spin, k) + b) * self.recl)
                f.write(np.ascontiguousarray(c, dtype=self.dtype).tobytes())

    def overlap(self, k: int, bands) -> np.ndarray:
        """M_ij = <up_i|down_j> of 1-based ``bands`` at ``k``, each orbital normalised."""
        up, dn = self.read(0, k, bands), self.read(1, k, bands)
        up /= np.linalg.norm(up, axis=1, keepdims=True)
        dn /= np.linalg.norm(dn, axis=1, keepdims=True)
        return up.conj() @ dn.T


# --------------------------------------------------------------------------- #
# bands to align
# --------------------------------------------------------------------------- #
def unpaired_bands(eigenval_path: Path) -> list[int]:
    """1-based bands occupied (> 0.5) in the up channel and empty in the down channel, at the first k-point.

    EIGENVAL of an ISPIN = 2 run: line 6 holds (NELECT, nk, NBANDS); each k-point block is a blank
    line, the k-point line, then one line per band (index, eig up, eig down, occ up, occ down).
    """
    lines = eigenval_path.read_text().splitlines()
    try:
        nk, nbands = (int(x) for x in lines[5].split()[1:3])
    except (IndexError, ValueError):
        raise ValueError(f"{eigenval_path}: not an EIGENVAL (no NELECT/nk/NBANDS on line 6)") from None
    rows = [line.split() for line in lines[7:] if line.strip()][1 : nbands + 1]  # skip the k-point line
    if len(rows) != nbands or any(len(r) != 5 for r in rows):
        raise ValueError(f"{eigenval_path}: expected {nbands} band lines of an ISPIN = 2 run at the first k-point")
    up = [float(r[3]) > 0.5 for r in rows]
    dn = [float(r[4]) > 0.5 for r in rows]
    if sum(up) <= sum(dn):
        raise ValueError(
            f"{eigenval_path}: {sum(up)} occupied up bands, {sum(dn)} down; the up channel must be the majority"
        )
    if any(d and not u for u, d in zip(up, dn)):
        raise ValueError(f"{eigenval_path}: bands occupied down but empty up; cannot tell which bands are unpaired")
    bands = [int(r[0]) for r, u, d in zip(rows, up, dn) if u and not d]
    if nk > 1:
        print(f"{eigenval_path}: {nk} k-points; unpaired bands taken from the first")
    return bands


def consecutive_groups(bands) -> list[tuple[int, ...]]:
    """1-based ``bands`` split into runs of consecutive band indices."""
    groups: list[list[int]] = []
    for b in sorted(bands):
        if groups and b == groups[-1][-1] + 1:
            groups[-1].append(b)
        else:
            groups.append([b])
    return [tuple(g) for g in groups]


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #
def rotation_angle(m: np.ndarray) -> float | None:
    """Rotation of a 2-band down set against the up one (deg; 0 aligned, 90 swapped), else None."""
    a = np.abs(m)
    return float(np.degrees(np.arctan2(a[0, 1], a[0, 0]))) if m.shape == (2, 2) else None


def describe(label: str, m: np.ndarray) -> str:
    a = np.abs(m)
    angle = rotation_angle(m)
    text = f"{label}: |<up_i|down_i>| {np.round(np.diag(a), 3).tolist()}"
    if angle is not None:
        text += f", rotation {angle:.1f} deg"
    return text


def plan(wc: Wavecar, groups, min_span: float) -> None:
    """Check every set at every k-point before anything is written; ValueError on the first problem."""
    for k in range(wc.nk):
        for group in groups:
            span = np.linalg.svd(wc.overlap(k, group), compute_uv=False)
            if span.min() < min_span:
                raise ValueError(
                    f"k-point {k + 1}, bands {list(group)}: the up and down channels do not span the same "
                    f"level (overlap singular values {np.round(span, 3).tolist()} < {min_span})"
                )


def align(wc: Wavecar, k: int, group) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the down bands of ``group`` at ``k`` onto the up ones, in place; (overlap before, after)."""
    m = wc.overlap(k, group)
    w, _, vh = np.linalg.svd(m)
    u = vh.conj().T @ w.conj().T
    down = wc.read(1, k, group)
    wc.write(1, k, group, u.T @ down)  # row j: sum_i U_ij down_i
    return m, wc.overlap(k, group)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rotate the unpaired down-spin orbitals of a vasp_std WAVECAR onto the up-spin ones "
        "(see the docstring)."
    )
    ap.add_argument("path", type=Path, help="run directory (with WAVECAR and EIGENVAL) or a WAVECAR file")
    ap.add_argument("--eigenval", type=Path,
                    help="EIGENVAL with the occupations (default: the one next to the WAVECAR)")
    ap.add_argument("--bands", type=int, nargs="+", help="1-based bands to align as one set (skips the EIGENVAL)")
    ap.add_argument("--out", type=Path, help="write the aligned WAVECAR here instead of in place")
    ap.add_argument("--check", action="store_true", help="only report the overlaps; write nothing")
    ap.add_argument("--min-span", type=float, default=0.9,
                    help="smallest allowed singular value of <up|down> for a set (default: 0.9)")
    args = ap.parse_args(argv)

    wavecar = args.path / "WAVECAR" if args.path.is_dir() else args.path
    try:
        wc = Wavecar(wavecar)
        if args.bands:
            bands = sorted(set(args.bands))
            if bands[0] < 1 or bands[-1] > wc.nb:
                raise ValueError(f"--bands must lie in 1..{wc.nb}")
            groups = [tuple(bands)]
            print(f"bands {bands} (given)")
        else:
            eigenval = args.eigenval or wavecar.parent / "EIGENVAL"
            bands = unpaired_bands(eigenval)
            if not bands:
                raise ValueError(f"{eigenval}: no unpaired band (occupied up, empty down)")
            if bands[-1] > wc.nb:
                raise ValueError(f"{eigenval}: unpaired band {bands[-1]} beyond the WAVECAR's {wc.nb} bands")
            groups = [g for g in consecutive_groups(bands) if len(g) > 1]
            print(f"unpaired bands {bands} (from {eigenval}); sets to align: {groups or 'none'}")
        plan(wc, groups, args.min_span)
    except (ValueError, OSError) as error:
        print(f"WAVECAR alignment failed: {error}")
        return 1

    if args.check:
        for k in range(wc.nk):
            for group in groups:
                print(describe(f"k-point {k + 1}, bands {list(group)}", wc.overlap(k, group)))
        return 0

    if args.out is not None:
        shutil.copyfile(wavecar, args.out)
        wc = Wavecar(args.out)
    for k in range(wc.nk):
        for group in groups:
            before, after = align(wc, k, group)
            print(describe(f"k-point {k + 1}, bands {list(group)}: before", before))
            print(describe(f"k-point {k + 1}, bands {list(group)}: after ", after))
    print(f"written: {wc.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
