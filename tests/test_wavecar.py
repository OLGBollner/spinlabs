"""Reading and rewriting WAVECAR bands, and aligning a degenerate down-spin pair to the up spin."""

import numpy as np
import pytest

from spinlabs.vasp.wavecar import align_down_pair, pair_overlap, read_bands, read_header

NPLW, NB = 50, 4


def _wavecar(path, up, down):
    """A spin-polarised, one k-point, complex64 WAVECAR holding ``up`` and ``down`` (NB x NPLW each)."""
    recl = NPLW * 8 + 8  # longer than the k header (4 + 3 NB doubles) and a band record
    records = [np.array([recl, 2, 45200], dtype=np.float64), np.array([1, NB, 400.0, *np.eye(3).ravel(), 0.0])]
    for coeffs in (up, down):
        records.append(np.array([NPLW, 0, 0, 0, *[x for b in range(NB) for x in (float(b), 0.0, 1.0)]]))
        records.extend(np.asarray(c, dtype=np.complex64) for c in coeffs)
    with open(path, "wb") as f:
        for rec in records:
            raw = rec.tobytes()
            f.write(raw + b"\0" * (recl - len(raw)))


def _orthonormal(rng, n):
    q, _ = np.linalg.qr(rng.normal(size=(NPLW, n)) + 1j * rng.normal(size=(NPLW, n)))
    return q.T


def test_a_rotated_down_pair_is_aligned_back_to_the_up_pair(tmp_path):
    rng = np.random.default_rng(3)
    up = _orthonormal(rng, NB)
    theta = np.radians(74.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * np.exp(0.4j)
    down = up.copy()
    down[2:4] = rot.T @ up[2:4]  # bands 3 and 4 (1-based): the up pair, rotated and phased
    _wavecar(tmp_path / "WAVECAR", up, down)

    assert read_header(tmp_path / "WAVECAR").nb == NB
    before = np.abs(pair_overlap(tmp_path / "WAVECAR", (3, 4)))
    assert before[0, 0] == pytest.approx(np.cos(theta), abs=1e-5)

    result = align_down_pair(tmp_path / "WAVECAR", (3, 4), out=tmp_path / "aligned")
    assert np.allclose(result.after, np.eye(2), atol=1e-5)
    assert result.angle() == pytest.approx(74.0, abs=1e-3) and result.angle("after") == pytest.approx(0.0, abs=1e-3)
    assert np.allclose(result.rotation.conj().T @ result.rotation, np.eye(2))
    # the up channel and the other down bands are untouched; the input file too
    assert np.array_equal(read_bands(tmp_path / "aligned", 0, range(1, NB + 1)), up.astype(np.complex64))
    assert np.array_equal(read_bands(tmp_path / "aligned", 1, (1, 2)), down[:2].astype(np.complex64))
    assert np.allclose(np.abs(pair_overlap(tmp_path / "WAVECAR", (3, 4))), before)
    new = read_bands(tmp_path / "aligned", 1, (3, 4)).astype(complex)
    assert np.allclose(new.conj() @ new.T, np.eye(2), atol=1e-5)  # still orthonormal
