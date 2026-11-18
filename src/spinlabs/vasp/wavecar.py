"""Read and rewrite single bands of a VASP WAVECAR, and align the unpaired down-spin orbitals to the up spin.

Used by the full spin decontamination: a spin-flipped determinant moves the electron of up band k
to down band k, which is only the same spatial orbital when the two channels chose the same basis
of a degenerate level (the NV e pair). The SCF picks that basis at random in each channel, so
:func:`align_down_pair` rotates the down-channel bands of the pair onto the up-channel ones before
the ALGO=None runs. The rotation is unitary within the pair, so the states, the density and the
energy of the SCF are unchanged.

WAVECAR layout (records of ``recl`` bytes): record 0 holds (recl, nspin, rtag), record 1
(nk, nb, encut, lattice, efermi); then per spin and k-point a header record (nplw, k, and
(eig, 0, occ) per band) followed by one record per band with its nplw plane-wave coefficients,
complex64 for rtag 45200 and complex128 for 45210.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DTYPES = {45200: np.complex64, 45210: np.complex128}


@dataclass(frozen=True)
class WavecarHeader:
    recl: int
    nspin: int
    rtag: int
    nk: int
    nb: int

    @property
    def dtype(self):
        return _DTYPES[self.rtag]

    def band_record(self, spin: int, k: int, band: int) -> int:
        """Record index of 0-based ``band`` at 0-based ``spin`` and ``k``."""
        return 2 + spin * self.nk * (self.nb + 1) + k * (self.nb + 1) + 1 + band

    def k_header_record(self, spin: int, k: int) -> int:
        return 2 + spin * self.nk * (self.nb + 1) + k * (self.nb + 1)


def read_header(path: str | Path) -> WavecarHeader:
    with open(path, "rb") as f:
        recl, nspin, rtag = np.fromfile(f, dtype=np.float64, count=3).astype(int)
        if rtag not in _DTYPES:
            raise ValueError(f"{path}: unknown WAVECAR precision tag {rtag} (known: {sorted(_DTYPES)})")
        f.seek(recl)
        nk, nb = np.fromfile(f, dtype=np.float64, count=2).astype(int)
    return WavecarHeader(int(recl), int(nspin), int(rtag), int(nk), int(nb))


def _nplw(path: Path, header: WavecarHeader, spin: int, k: int) -> int:
    with open(path, "rb") as f:
        f.seek(header.k_header_record(spin, k) * header.recl)
        return int(np.fromfile(f, dtype=np.float64, count=1)[0])


def read_bands(path: str | Path, spin: int, bands, k: int = 0) -> np.ndarray:
    """Plane-wave coefficients of 1-based ``bands`` in 0-based ``spin`` at ``k``: (len(bands), nplw)."""
    path = Path(path)
    header = read_header(path)
    nplw = _nplw(path, header, spin, k)
    out = []
    with open(path, "rb") as f:
        for band in bands:
            f.seek(header.band_record(spin, k, band - 1) * header.recl)
            out.append(np.fromfile(f, dtype=header.dtype, count=nplw))
    return np.array(out)


def write_bands(path: str | Path, spin: int, bands, coeffs: np.ndarray, k: int = 0) -> None:
    """Overwrite the coefficients of 1-based ``bands`` in place (same nplw, stored in the file's precision)."""
    path = Path(path)
    header = read_header(path)
    nplw = _nplw(path, header, spin, k)
    coeffs = np.asarray(coeffs)
    if coeffs.shape != (len(bands), nplw):
        raise ValueError(f"coefficients have shape {coeffs.shape}, expected {(len(bands), nplw)}")
    with open(path, "r+b") as f:
        for band, c in zip(bands, coeffs):
            f.seek(header.band_record(spin, k, band - 1) * header.recl)
            f.write(np.ascontiguousarray(c, dtype=header.dtype).tobytes())


def pair_overlap(path: str | Path, bands, k: int = 0) -> np.ndarray:
    """<up_i|down_j> of 1-based ``bands`` (plane-wave part, each orbital normalised)."""
    up = read_bands(path, 0, bands, k).astype(np.complex128)
    dn = read_bands(path, 1, bands, k).astype(np.complex128)
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    dn /= np.linalg.norm(dn, axis=1, keepdims=True)
    return up.conj() @ dn.T


@dataclass(frozen=True)
class PairAlignment:
    bands: tuple[int, ...]
    before: np.ndarray  # |<up_i|down_j>| before
    after: np.ndarray  # |<up_i|down_j>| after
    rotation: np.ndarray  # unitary U: new down_j = sum_i old down_i U_ij

    def angle(self, which: str = "before") -> float:
        """Rotation of a 2-band down pair against the up pair, in degrees (0 = aligned, 90 = swapped):
        arctan(|<up_1|down_2>| / |<up_1|down_1>|). The ratio drops the overall size of the overlaps,
        which stays below 1 from the PAW augmentation and the two channels' potentials."""
        m = self.before if which == "before" else self.after
        return float(np.degrees(np.arctan2(m[0, 1], m[0, 0])))


def align_down_pair(path: str | Path, bands, out: str | Path | None = None, k: int = 0) -> PairAlignment:
    """Rotate the down-channel ``bands`` (1-based) onto the up-channel ones.

    The unitary U maximises Re tr <up|down U> (polar decomposition of the overlap block
    M = <up_i|down_j>: M = W S V^H, U = V W^H), so afterwards <up_i|down'_j> = W S W^H is
    Hermitian with the largest possible diagonal. Written in place, or to a copy at ``out``.
    The band eigenvalues and occupations in the k-point header are left as they are. Returns the
    overlaps before and after.
    """
    bands = tuple(int(b) for b in bands)
    path = Path(path)
    target = Path(out) if out is not None else path
    if out is not None:
        shutil.copyfile(path, target)
    m = pair_overlap(target, bands, k)
    w, _, vh = np.linalg.svd(m)
    u = vh.conj().T @ w.conj().T
    down = read_bands(target, 1, bands, k).astype(np.complex128)
    write_bands(target, 1, bands, u.T @ down, k)  # row j of the result: sum_i U_ij down_i
    return PairAlignment(bands, np.abs(m), np.abs(pair_overlap(target, bands, k)), u)


def read_eigenvalues(path: str | Path, spin: int, k: int = 0) -> np.ndarray:
    """Band eigenvalues (eV) of 0-based ``spin`` at ``k``, from the k-point header record."""
    path = Path(path)
    header = read_header(path)
    with open(path, "rb") as f:
        f.seek(header.k_header_record(spin, k) * header.recl)
        rec = np.fromfile(f, dtype=np.float64, count=4 + 3 * header.nb)
    return rec[4::3]


def consecutive_groups(bands) -> list[tuple[int, ...]]:
    """1-based ``bands`` split into runs of consecutive band indices."""
    groups: list[list[int]] = []
    for band in sorted(bands):
        if groups and band == groups[-1][-1] + 1:
            groups[-1].append(band)
        else:
            groups.append([band])
    return [tuple(g) for g in groups]


MIN_SPAN = 0.9  # smallest singular value of the up/down overlap block for the channels to share a level


def align_unpaired(run_dir: str | Path, min_span: float = MIN_SPAN) -> list[PairAlignment]:
    """Align the down channel to the up channel over the unpaired bands, degenerate or not.

    The unpaired bands (occupied up, empty down) come from the run's INCAR occupations
    (DOCCUP/DOCCDO or FERWE/FERDO, spinlabs.vasp.occupations); each run of consecutive unpaired
    bands with two or more members is aligned in place as one set (:func:`align_down_pair`).
    There is no degeneracy condition: a displacement that splits a degenerate level (an E mode
    splitting the NV e pair) can split it in opposite order in the two channels, and the flips
    then need the down bands matched to the up ones all the same. The down unpaired orbitals are
    empty in the SCF, so the rotation leaves its state alone, and ALGO = None keeps the rotated
    orbitals. ValueError when the two channels' bands do not span the same space (smallest
    singular value of the overlap below ``min_span``): the flips would then not move an electron
    into the orbital it left, and no alignment can fix that.
    """
    from spinlabs.vasp.occupations import parse_occupations

    run_dir = Path(run_dir)
    wavecar = run_dir / "WAVECAR"
    up, dn = parse_occupations((run_dir / "INCAR").read_text())
    unpaired = [i + 1 for i, (u, d) in enumerate(zip(up, dn)) if u > 0.5 and d < 0.5]
    if not unpaired:
        raise ValueError(f"no unpaired band in the occupations of {run_dir / 'INCAR'}")
    results = []
    for group in consecutive_groups(unpaired):
        if len(group) < 2:
            continue
        span = np.linalg.svd(pair_overlap(wavecar, group), compute_uv=False)
        if span.min() < min_span:
            raise ValueError(
                f"bands {group}: the up and down channels do not span the same level "
                f"(overlap singular values {np.round(span, 3).tolist()} < {min_span})"
            )
        results.append(align_down_pair(wavecar, group))
    return results


def main(argv: list[str] | None = None) -> int:
    """``python -m spinlabs.vasp.wavecar align DIR``: align DIR/WAVECAR (see :func:`align_unpaired`)."""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m spinlabs.vasp.wavecar")
    ap.add_argument("command", choices=["align"])
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args(argv)
    try:
        results = align_unpaired(args.run_dir)
    except (ValueError, OSError) as error:
        print(f"WAVECAR alignment failed: {error}")
        return 1
    if not results:
        print("WAVECAR: fewer than two unpaired bands, nothing to align")
    for r in results:
        line = f"aligned down bands {list(r.bands)} to up: "
        if len(r.bands) == 2:
            line += f"rotation {r.angle():.1f} -> {r.angle('after'):.2f} deg, "
        line += f"|<up_i|down_i>| {np.round(np.diag(r.before), 3).tolist()} -> {np.round(np.diag(r.after), 3).tolist()}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
