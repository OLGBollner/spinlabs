"""Reading and rewriting WAVECAR bands, and aligning a degenerate down-spin pair to the up spin."""

import numpy as np
import pytest

from spinlabs.vasp.wavecar import (
    HSQDTM,
    align_down_pair,
    align_unpaired,
    check_full_sphere,
    consecutive_groups,
    main,
    pair_overlap,
    read_bands,
    read_eigenvalues,
    read_header,
    sphere_count,
)

NB, ENCUT, LATTICE = 4, 400.0, 1.5 * np.eye(3)
NPLW = sphere_count(LATTICE, ENCUT, np.zeros(3))  # a vasp_std WAVECAR stores the full sphere at Gamma


def _wavecar(path, up, down, eig_up=None, eig_dn=None, nplw=NPLW):
    """A spin-polarised, one k-point (Gamma), complex64 WAVECAR holding ``up`` and ``down`` (NB x NPLW
    each); ``nplw`` is the plane-wave count its k headers claim."""
    eigs = [np.arange(NB, dtype=float) if e is None else np.asarray(e, float) for e in (eig_up, eig_dn)]
    recl = NPLW * 8 + 8  # longer than the k header (4 + 3 NB doubles) and a band record
    records = [np.array([recl, 2, 45200], dtype=np.float64), np.array([1, NB, ENCUT, *LATTICE.ravel(), 0.0])]
    for coeffs, eig in zip((up, down), eigs):
        records.append(np.array([nplw, 0, 0, 0, *[x for b in range(NB) for x in (eig[b], 0.0, 1.0)]]))
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
OCC_UP, OCC_DN = [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]
EIG = [0.0, 1.0, 5.0, 5.0]


def _eigenval(path, occ_up=OCC_UP, occ_dn=OCC_DN, eig_up=EIG, eig_dn=EIG):
    """An ISPIN = 2, one k-point EIGENVAL with these occupations."""
    lines = ["    1    1    1    2", "  0.1E+02  0.1E-09  0.1E-09  0.1E-09  0.5E-15", "  1.0E-004", "  CAR ",
             " unknown system", f"  {int(sum(occ_up) + sum(occ_dn))}  1  {NB}", "",
             "  0.0  0.0  0.0  0.1000000E+01"]
    lines += [f"  {b + 1}  {eig_up[b]:.6f}  {eig_dn[b]:.6f}  {occ_up[b]:.6f}  {occ_dn[b]:.6f}" for b in range(NB)]
    path.write_text("\n".join(lines) + "\n")


def _rotated_run(tmp_path, eig_dn=EIG, theta=74.0, mix_outside=False):
    rng = np.random.default_rng(5)
    up = _orthonormal(rng, NB)
    t = np.radians(theta)
    down = up.copy()
    down[2:4] = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]) @ up[2:4]
    if mix_outside:  # down band 4 is a different orbital: the channels share no level
        down[3] = _orthonormal(np.random.default_rng(9), NB)[0]
    _eigenval(tmp_path / "EIGENVAL", eig_dn=eig_dn)
    _wavecar(tmp_path / "WAVECAR", up, down, EIG, eig_dn)
    return tmp_path


def test_consecutive_groups():
    assert consecutive_groups([4, 3]) == [(3, 4)]
    assert consecutive_groups([2, 4, 5]) == [(2,), (4, 5)]


def test_the_unpaired_degenerate_pair_is_aligned_in_place(tmp_path):
    run = _rotated_run(tmp_path)
    assert list(read_eigenvalues(run / "WAVECAR", 0)) == EIG
    (result,) = align_unpaired(run)
    assert result.bands == (3, 4)
    assert result.angle() == pytest.approx(74.0, abs=1e-3) and result.angle("after") == pytest.approx(0.0, abs=1e-3)
    assert np.allclose(np.abs(pair_overlap(run / "WAVECAR", (3, 4))), np.eye(2), atol=1e-5)
    assert align_unpaired(run)[0].angle() == pytest.approx(0.0, abs=1e-3)  # a second pass changes nothing


def test_a_split_pair_is_aligned_too(tmp_path):
    """An E displacement splits the pair, possibly in opposite order in the two channels: still aligned."""
    run = _rotated_run(tmp_path, eig_dn=[0.0, 1.0, 5.2, 5.0], theta=90.0)  # down pair swapped
    (result,) = align_unpaired(run)
    assert result.angle() == pytest.approx(90.0, abs=1e-3) and result.angle("after") == pytest.approx(0.0, abs=1e-3)
    assert np.allclose(np.abs(pair_overlap(run / "WAVECAR", (3, 4))), np.eye(2), atol=1e-5)


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


def test_sphere_count_of_a_cubic_cell():
    # 1 A cubic cell: |G| = 2 pi |n|; this ENCUT keeps |n|^2 <= 2, i.e. 1 + 6 + 12 plane waves
    encut = HSQDTM * (2 * np.pi) ** 2 * 2.5
    assert sphere_count(np.eye(3), encut, np.zeros(3)) == 19
    assert sphere_count(np.eye(3), encut, np.array([0.5, 0.0, 0.0])) == sphere_count(
        np.eye(3), encut, np.array([-0.5, 0.0, 0.0])
    )


def test_a_gamma_only_wavecar_is_refused(tmp_path, capsys):
    """vasp_gam stores half the sphere under the same precision tag: refused, nothing written."""
    run = _rotated_run(tmp_path)
    up = read_bands(run / "WAVECAR", 0, range(1, NB + 1))
    down = read_bands(run / "WAVECAR", 1, range(1, NB + 1))
    _wavecar(run / "WAVECAR", up, down, EIG, EIG, nplw=(NPLW + 1) // 2)
    before = (run / "WAVECAR").read_bytes()
    with pytest.raises(ValueError, match="vasp_gam"):
        align_unpaired(run)
    with pytest.raises(ValueError, match="vasp_gam"):
        align_down_pair(run / "WAVECAR", (3, 4))
    assert main(["align", str(run)]) == 1
    assert "Gamma-only (vasp_gam) WAVECAR" in capsys.readouterr().out
    assert (run / "WAVECAR").read_bytes() == before


def test_a_wavecar_with_another_plane_wave_count_is_refused(tmp_path):
    run = _rotated_run(tmp_path)
    up = read_bands(run / "WAVECAR", 0, range(1, NB + 1))
    _wavecar(run / "WAVECAR", up, up, nplw=NPLW - 10)
    with pytest.raises(ValueError, match="not a vasp_std WAVECAR"):
        check_full_sphere(run / "WAVECAR")


def test_the_bands_come_from_the_runs_eigenval(tmp_path):
    run = _rotated_run(tmp_path)
    (run / "INCAR").write_text("FERWE = 4*1.0\nFERDO = 4*0.0\n")  # not read
    assert [r.bands for r in align_unpaired(run)] == [(3, 4)]


def test_an_eigenval_with_swapped_spin_channels_is_refused(tmp_path):
    run = _rotated_run(tmp_path)
    _eigenval(run / "EIGENVAL", occ_up=OCC_DN, occ_dn=OCC_UP)
    with pytest.raises(ValueError, match="up channel must be the majority"):
        align_unpaired(run)


def test_a_missing_eigenval_fails_the_cli(tmp_path, capsys):
    run = _rotated_run(tmp_path)
    (run / "EIGENVAL").unlink()
    assert main(["align", str(run)]) == 1
    assert "WAVECAR alignment failed" in capsys.readouterr().out
