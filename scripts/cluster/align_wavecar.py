#!/usr/bin/env python3
"""align_wavecar.py -- rotate degenerate down-spin orbitals of a VASP WAVECAR onto the up-spin ones.

Standalone: needs only numpy. Works on spin-polarised (ISPIN = 2) WAVECARs of vasp_std, single or
double precision, any number of k-points (the rotation is done per k-point).

Why
---
For a spin-flip (broken-symmetry) determinant built from a high-spin WAVECAR with ALGO = None, the
flipped electron moves from up band k to down band k. That is only the same spatial orbital when the
two spin channels chose the same basis for a degenerate level (for the NV centre the e pair, bands
127/128 in a 64-atom cell). The SCF picks that basis at random in each channel, so e.g. "down 127"
can be mostly "up 128". This script rotates the down-channel bands of each degenerate level onto the
up-channel ones. The rotation is unitary within the level, so the SCF state, its density and energy
are unchanged; only which down orbital carries which band index changes.

How
---
For the bands B of a level, M_ij = <up_i|down_j> (plane-wave part, each orbital normalised).
With the SVD M = W S V^H, the unitary U = V W^H maximises Re tr(M U); the new down bands are
down'_j = sum_i U_ij down_i, after which <up_i|down'_j> = W S W^H is Hermitian with the largest
possible diagonal. Eigenvalues and occupations in the WAVECAR are left as they are.

Which bands
-----------
By default, the unpaired bands (occupied up, empty down) are read from the INCAR occupations
(DOCCUP/DOCCDO, else FERWE/FERDO, run-length form "n*x"), and grouped into degenerate levels by the
up-channel eigenvalues in the WAVECAR (tolerance --tol). Each level of two or more bands is
aligned; a single band needs nothing. Or give the bands yourself with --bands (1-based, aligned as
one set whether degenerate or not).

The script refuses (exit 1, nothing written) when a level is degenerate in the up channel but split
in the down channel, or when the up and down bands do not span the same space (smallest singular
value of M below --min-span): then no rotation makes a flip land in the orbital it left.

Usage
-----
    python align_wavecar.py RUN_DIR                    # RUN_DIR/WAVECAR in place, bands from RUN_DIR/INCAR
    python align_wavecar.py WAVECAR --incar INCAR      # explicit files
    python align_wavecar.py RUN_DIR --check            # only report the overlaps, write nothing
    python align_wavecar.py RUN_DIR --out WAVECAR.aligned
    python align_wavecar.py WAVECAR --bands 127 128    # these bands, no INCAR needed

A second run on an aligned WAVECAR changes nothing (rotation 0).
Overlaps use only the plane-wave part (no PAW augmentation), so aligned levels show |<up|down>|
somewhat below 1 (e.g. 0.98); that is not a remaining rotation.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np

DTYPES = {45200: np.complex64, 45210: np.complex128}  # rtag -> plane-wave coefficient precision


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
            nk, nb = np.fromfile(f, dtype=np.float64, count=2).astype(int)
        if nspin != 2:
            raise ValueError(f"{path}: ISPIN = {nspin}; a spin-polarised (ISPIN = 2) WAVECAR is needed")
        self.recl, self.nspin, self.rtag, self.nk, self.nb = int(recl), int(nspin), int(rtag), int(nk), int(nb)
        self.dtype = DTYPES[self.rtag]

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
def _tag(incar: str, tag: str) -> str | None:
    """Last value of ``tag`` in the INCAR text (VASP semantics), comments stripped."""
    values = re.findall(rf"^\s*{tag}\s*=\s*(.+?)\s*(?:[!#].*)?$", incar, re.M | re.I)
    return values[-1] if values else None


def _expand(value: str) -> list[float]:
    """'252*1.0 1*0.0 0.5' -> one occupation per band."""
    occ: list[float] = []
    for item in value.split():
        count, star, x = item.partition("*")
        occ.extend([float(x)] * int(count) if star else [float(item)])
    return occ


def unpaired_bands(incar_path: Path) -> list[int]:
    """1-based bands occupied in the up channel and empty in the down channel."""
    text = incar_path.read_text()
    for up_tag, dn_tag in (("DOCCUP", "DOCCDO"), ("FERWE", "FERDO")):
        up, dn = _tag(text, up_tag), _tag(text, dn_tag)
        if up is not None and dn is not None:
            break
    else:
        raise ValueError(f"{incar_path}: no DOCCUP/DOCCDO or FERWE/FERDO occupations")
    up_occ, dn_occ = _expand(up), _expand(dn)
    bands = [i + 1 for i, (u, d) in enumerate(zip(up_occ, dn_occ)) if u > 0.5 and d < 0.5]
    if not bands:
        raise ValueError(f"{incar_path}: no unpaired band (occupied up, empty down) in {up_tag}/{dn_tag}")
    return bands


def degenerate_levels(bands, eig_up: np.ndarray, tol: float) -> list[tuple[int, ...]]:
    """Consecutive ``bands`` whose up-channel eigenvalues differ by at most ``tol`` eV, as levels."""
    levels: list[list[int]] = []
    for b in sorted(bands):
        if levels and b == levels[-1][-1] + 1 and abs(eig_up[b - 1] - eig_up[b - 2]) <= tol:
            levels[-1].append(b)
        else:
            levels.append([b])
    return [tuple(level) for level in levels]


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #
def rotation_angle(m: np.ndarray) -> float | None:
    """Rotation of a 2-band down level against the up one (deg; 0 aligned, 90 swapped), else None."""
    a = np.abs(m)
    return float(np.degrees(np.arctan2(a[0, 1], a[0, 0]))) if m.shape == (2, 2) else None


def describe(label: str, m: np.ndarray) -> str:
    a = np.abs(m)
    angle = rotation_angle(m)
    text = f"{label}: |<up_i|down_i>| {np.round(np.diag(a), 3).tolist()}"
    if angle is not None:
        text += f", rotation {angle:.1f} deg"
    return text


def plan(wc: Wavecar, levels_by_k, min_span: float, tol: float, explicit: bool) -> None:
    """Check every level before anything is written; ValueError on the first problem."""
    for k in range(wc.nk):
        _, eig_dn = wc.k_header(1, k)
        for level in levels_by_k[k]:
            if not explicit:
                spread = float(np.ptp(eig_dn[[b - 1 for b in level]]))
                if spread > tol:
                    raise ValueError(
                        f"k-point {k + 1}, bands {list(level)}: degenerate in the up channel but spread "
                        f"{spread * 1e3:.2f} meV in the down channel"
                    )
            span = np.linalg.svd(wc.overlap(k, level), compute_uv=False)
            if span.min() < min_span:
                raise ValueError(
                    f"k-point {k + 1}, bands {list(level)}: the up and down channels do not span the same "
                    f"level (overlap singular values {np.round(span, 3).tolist()} < {min_span})"
                )


def align(wc: Wavecar, k: int, level) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the down bands of ``level`` at ``k`` onto the up ones, in place; (overlap before, after)."""
    m = wc.overlap(k, level)
    w, _, vh = np.linalg.svd(m)
    u = vh.conj().T @ w.conj().T
    down = wc.read(1, k, level)
    wc.write(1, k, level, u.T @ down)  # row j: sum_i U_ij down_i
    return m, wc.overlap(k, level)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rotate degenerate down-spin orbitals of a WAVECAR onto the up-spin ones (see the docstring)."
    )
    ap.add_argument("path", type=Path, help="run directory (with WAVECAR and INCAR) or a WAVECAR file")
    ap.add_argument("--incar", type=Path, help="INCAR with the occupations (default: next to the WAVECAR)")
    ap.add_argument("--bands", type=int, nargs="+", help="1-based bands to align as one set (skips the INCAR)")
    ap.add_argument("--out", type=Path, help="write the aligned WAVECAR here instead of in place")
    ap.add_argument("--check", action="store_true", help="only report the overlaps; write nothing")
    ap.add_argument("--tol", type=float, default=1e-3, help="degeneracy tolerance in eV (default: 1e-3)")
    ap.add_argument("--min-span", type=float, default=0.9,
                    help="smallest allowed singular value of <up|down> for a level (default: 0.9)")
    args = ap.parse_args(argv)

    wavecar = args.path / "WAVECAR" if args.path.is_dir() else args.path
    try:
        wc = Wavecar(wavecar)
        if args.bands:
            bands, explicit = sorted(set(args.bands)), True
            if bands[0] < 1 or bands[-1] > wc.nb:
                raise ValueError(f"--bands must lie in 1..{wc.nb}")
            levels_by_k = [[tuple(bands)] for _ in range(wc.nk)]
            print(f"bands {bands} (given)")
        else:
            incar = args.incar or wavecar.parent / "INCAR"
            bands, explicit = unpaired_bands(incar), False
            levels_by_k = [
                [lv for lv in degenerate_levels(bands, wc.k_header(0, k)[1], args.tol) if len(lv) > 1]
                for k in range(wc.nk)
            ]
            print(f"unpaired bands {bands} (from {incar}); degenerate levels: "
                  f"{sorted({lv for levels in levels_by_k for lv in levels}) or 'none'}")
        plan(wc, levels_by_k, args.min_span, args.tol, explicit)
    except (ValueError, OSError) as error:
        print(f"WAVECAR alignment failed: {error}")
        return 1

    if args.check:
        for k in range(wc.nk):
            for level in levels_by_k[k]:
                print(describe(f"k-point {k + 1}, bands {list(level)}", wc.overlap(k, level)))
        return 0

    if args.out is not None:
        shutil.copyfile(wavecar, args.out)
        wc = Wavecar(args.out)
    for k in range(wc.nk):
        for level in levels_by_k[k]:
            before, after = align(wc, k, level)
            print(describe(f"k-point {k + 1}, bands {list(level)}: before", before))
            print(describe(f"k-point {k + 1}, bands {list(level)}: after ", after))
    print(f"written: {wc.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
