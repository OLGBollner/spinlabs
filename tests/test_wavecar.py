"""Reading and rewriting WAVECAR bands, and aligning a degenerate down-spin pair to the up spin."""

import numpy as np
import pytest

from spinlabs.vasp.wavecar import (
    align_down_pair,
    align_unpaired,
    degenerate_groups,
    main,
    pair_overlap,
    read_bands,
    read_eigenvalues,
    read_header,
)

NPLW, NB = 50, 4


def _wavecar(path, up, down, eig_up=None, eig_dn=None):
    """A spin-polarised, one k-point, complex64 WAVECAR holding ``up`` and ``down`` (NB x NPLW each)."""
    eigs = [np.arange(NB, dtype=float) if e is None else np.asarray(e, float) for e in (eig_up, eig_dn)]
    recl = NPLW * 8 + 8  # longer than the k header (4 + 3 NB doubles) and a band record
    records = [np.array([recl, 2, 45200], dtype=np.float64), np.array([1, NB, 400.0, *np.eye(3).ravel(), 0.0])]
    for coeffs, eig in zip((up, down), eigs):
        records.append(np.array([NPLW, 0, 0, 0, *[x for b in range(NB) for x in (eig[b], 0.0, 1.0)]]))
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


# 4 bands: 1, 2 paired; 3, 4 the degenerate unpaired pair (occupied up, empty down)
INCAR = "FERWE = 4*1.0\nFERDO = 2*1.0 2*0.0\n"
EIG = [0.0, 1.0, 5.0, 5.0]


def _rotated_run(tmp_path, eig_dn=EIG, theta=74.0, mix_outside=False):
    rng = np.random.default_rng(5)
    up = _orthonormal(rng, NB)
    t = np.radians(theta)
    down = up.copy()
    down[2:4] = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]) @ up[2:4]
    if mix_outside:  # down band 4 is a different orbital: the channels share no level
        down[3] = _orthonormal(np.random.default_rng(9), NB)[0]
    (tmp_path / "INCAR").write_text(INCAR)
    _wavecar(tmp_path / "WAVECAR", up, down, EIG, eig_dn)
    return tmp_path


def test_degenerate_groups():
    assert degenerate_groups([3, 4], np.array(EIG), 1e-3) == [(3, 4)]
    assert degenerate_groups([2, 3, 4], np.array(EIG), 1e-3) == [(2,), (3, 4)]


def test_the_unpaired_degenerate_pair_is_aligned_in_place(tmp_path):
    run = _rotated_run(tmp_path)
    assert list(read_eigenvalues(run / "WAVECAR", 0)) == EIG
    (result,) = align_unpaired(run)
    assert result.bands == (3, 4)
    assert result.angle() == pytest.approx(74.0, abs=1e-3) and result.angle("after") == pytest.approx(0.0, abs=1e-3)
    assert np.allclose(np.abs(pair_overlap(run / "WAVECAR", (3, 4))), np.eye(2), atol=1e-5)
    assert align_unpaired(run)[0].angle() == pytest.approx(0.0, abs=1e-3)  # a second pass changes nothing


def test_a_pair_split_in_the_down_channel_is_refused(tmp_path):
    run = _rotated_run(tmp_path, eig_dn=[0.0, 1.0, 5.0, 5.2])
    with pytest.raises(ValueError, match="down channel"):
        align_unpaired(run)


def test_channels_without_a_common_level_are_refused(tmp_path, capsys):
    run = _rotated_run(tmp_path, mix_outside=True)
    with pytest.raises(ValueError, match="do not span the same level"):
        align_unpaired(run)
    assert main(["align", str(run)]) == 1
    assert "WAVECAR alignment failed" in capsys.readouterr().out


def test_cli_reports_the_rotation(tmp_path, capsys):
    run = _rotated_run(tmp_path)
    assert main(["align", str(run)]) == 0
    assert "aligned down bands [3, 4] to up: rotation 74.0 -> 0.00 deg" in capsys.readouterr().out
