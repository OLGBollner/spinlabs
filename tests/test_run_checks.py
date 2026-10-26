"""Checks of perturbation runs (displacement, VASP symmetry handling, ZFS change shape) and the noise/SNR estimates."""

from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Poscar

from spinlabs.models import PhononSpectrum, RawZFSData, ZFSTensor
from spinlabs.vasp.run_checks import (
    check_perturbation,
    check_run,
    check_symmetry_ops,
    check_zfs_shape,
    dataset_shape_flags,
    outcar_symmetry,
    reference_outcar,
)

GS = np.array([[0.0, 1000.0, 1000.0], [1000.0, 0.0, 1000.0], [1000.0, 1000.0, 0.0]])  # NV along [111]
SYMS = ["A1", "Ex", "Ey", "A2"]


def _principal_change(a1: float = 0.0, exx: float = 0.0, xz: float = 0.0) -> np.ndarray:
    """GS plus a change given in GS's principal frame: A1 (D_zz) and E (D_xx - D_yy, D_xz) parts."""
    _, rot = np.linalg.eigh(GS)
    d = np.diag([-a1 / 2 + exx / 2, -a1 / 2 - exx / 2, a1])
    d[0, 2] = d[2, 0] = xz
    return GS + rot @ d @ rot.T


def _outcar(path: Path, tensor: np.ndarray, *, ops: int = 2, isym: int = 3, finished: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = tensor
    path.write_text(
        f"   ISYM   =      {isym}\n The static configuration has the point symmetry C_s .\n"
        f" GETGRP returns: Found  {ops} space group operations\n"
        " Spin-spin contribution to zero-field splitting tensor (MHz)\n -------\n"
        "      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz\n -------\n"
        f"  {t[0, 0]:.4f}  {t[1, 1]:.4f}  {t[2, 2]:.4f}  {t[0, 1]:.4f}  {t[0, 2]:.4f}  {t[1, 2]:.4f}\n -------\n"
        + (" General timing and accounting informations for this job:\n" if finished else "")
    )


def _spectrum(n_atoms: int = 4) -> PhononSpectrum:
    rng = np.random.default_rng(5)
    eigs = rng.normal(size=(4, n_atoms, 3))
    eigs /= np.linalg.norm(eigs.reshape(4, -1), axis=1)[:, None, None]
    return PhononSpectrum(
        frequencies_mev=np.array([20.0, 40.0, 40.0, 60.0]),
        symmetries=SYMS,
        eigenvectors=eigs,
        atom_frac_coords=np.zeros((n_atoms, 3)),
        atom_symbols=["C"] * (n_atoms - 1) + ["N"],
        atomic_masses=np.array([12.011] * (n_atoms - 1) + [14.007]),
        lattice=np.eye(3) * 3.57,
    )


# ---- ZFS shape ----

@pytest.mark.parametrize(
    "labels, change, flagged",
    [
        (["A1"], dict(a1=20.0), False),
        (["A1"], dict(a1=5.0, exx=10.0), True),  # an A1 mode cannot make D_xx and D_yy differ
        (["A2", "A2"], dict(a1=1.0), False),
        (["Ex"], dict(exx=8.0, xz=6.0), False),
        (["Ex"], dict(a1=30.0, exx=0.5), True),  # an E mode that only changes D_zz
        (["Ex", "A1"], dict(a1=30.0, exx=0.5), False),  # mixed pairs have no fixed shape
        (["A1"], dict(exx=0.1), False),  # below the noise floor
    ],
)
def test_zfs_shape(labels, change, flagged):
    _, problem = check_zfs_shape(_principal_change(**change), GS, labels)
    assert (problem is not None) is flagged


# ---- VASP symmetry handling ----

def test_symmetry_ops_within_and_above_the_bound():
    sym = outcar_symmetry(" ISYM = 3\n has the point symmetry C_s .\n GETGRP returns: Found  2 space\n General timing and accounting")
    assert sym["n_ops"] == 2 and sym["isym"] == 3 and sym["finished"]
    assert check_symmetry_ops(sym, ["Ex"])[1] == []
    sym["n_ops"] = 6
    assert "at most 2 ops expected but VASP found 6" in check_symmetry_ops(sym, ["Ex"])[1][0]
    assert check_symmetry_ops(sym, ["A1"])[1] == []


def test_symmetry_off_and_invgrp():
    assert check_symmetry_ops(outcar_symmetry(" ISYM = -1\n"), ["Ex"])[1] == []
    sym = outcar_symmetry(" internal error in subroutine INVGRP\n ISYM = 2\n GETGRP returns: Found 1 space")
    assert any("INVGRP" in p for p in check_symmetry_ops(sym, ["A1"])[1])


# ---- displacement ----

def _structures(tmp_path: Path, factor: float):
    from spinlabs.vasp.structures import expected_displacement

    spectrum = _spectrum()
    ref = Structure(Lattice.cubic(3.57), ["C", "C", "C", "N"], [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0], [0.75, 0.75, 0.75]])
    spectrum.atom_frac_coords = np.array([s.frac_coords for s in ref])
    pert = ref.copy()
    disp = factor * expected_displacement(spectrum, (2, 3), 0.1)
    for site, d in zip(pert.sites, disp):
        site.coords = site.coords + d
    Poscar(ref).write_file(tmp_path / "ref_POSCAR")
    Poscar(pert).write_file(tmp_path / "POSCAR")
    return spectrum


def test_perturbation_check_accepts_the_intended_displacement(tmp_path):
    spectrum = _structures(tmp_path, 1.0)
    ok, line = check_perturbation(tmp_path / "POSCAR", tmp_path / "ref_POSCAR", spectrum, (2, 3), 0.1)
    assert ok, line
    assert "perturbation OK: modes 2+3, step 0.1 (mass_weighted)" in line and "scale 1.000 x expected" in line


def test_perturbation_check_rejects_a_doubled_displacement(tmp_path):
    spectrum = _structures(tmp_path, 2.0)
    ok, line = check_perturbation(tmp_path / "POSCAR", tmp_path / "ref_POSCAR", spectrum, (2, 3), 0.1)
    assert not ok and "perturbation WRONG" in line and "scale 2.000" in line


# ---- whole run ----

def _tree_run(tmp_path: Path, folder: str, tensor: np.ndarray, **outcar) -> Path:
    """<defect>/second_order/pert_0.1/all_bands/runs/<folder> with the ground state in <defect>/ZFS_hyp."""
    defect = tmp_path / "NV_64"
    _outcar(defect / "ZFS_hyp" / "OUTCAR", GS, ops=6)
    run = defect / "second_order" / "pert_0.1" / "all_bands" / "runs" / folder
    _outcar(run / "OUTCAR", tensor, **outcar)
    return run


def test_reference_is_found_from_the_tree_layout(tmp_path):
    run = _tree_run(tmp_path, "1_1", _principal_change(a1=10.0))
    assert reference_outcar(run) == tmp_path / "NV_64" / "ZFS_hyp" / "OUTCAR"


def test_healthy_run_is_ok(tmp_path):
    run = _tree_run(tmp_path, "1_1", _principal_change(a1=10.0), ops=6)
    check = check_run(run, _spectrum(), (1, 1))
    assert check.status == "ok", check.report()
    assert "modes 1+1: A1/A1" in check.report()


def test_unfinished_wrongly_shaped_run_is_suspicious(tmp_path):
    run = _tree_run(tmp_path, "1_4", _principal_change(a1=2.0, exx=10.0), ops=6, finished=False)
    check = check_run(run, _spectrum(), (1, 4))
    assert check.status == "suspicious"
    text = check.report()
    assert "did not finish" in text and "E fraction" in text


def test_dataset_flags_name_run_folders(tmp_path):
    raw = RawZFSData(
        defect="NV", cell_size=64, pert_scale=0.1, calc_method="all_bands",
        ground_state_zfs=ZFSTensor(matrix=GS, unit="MHz"),
        first_order={0: {"tensor": _principal_change(a1=10.0), "unit": "MHz"},
                     3: {"tensor": _principal_change(exx=10.0), "unit": "MHz"}},  # A2 mode with an E change
        second_order={(0, 0): {"tensor": _principal_change(a1=20.0), "unit": "MHz"}},
    )
    flags = dataset_shape_flags(raw, _spectrum())
    assert [folder for folder, _ in flags] == ["4"]


# ---- signal-to-noise of the finite differences (convergence module) ----

def test_snr_report_recovers_coefficients_and_window():
    from spinlabs.convergence import snr_report

    rng = np.random.default_rng(1)
    q = np.array([-0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1])
    b, c, d = 100.0, 10.0, 50.0  # MHz/Q^k, all in the A1 part
    parts = np.zeros((len(q), 5))
    parts[:, 0] = b * q + c * q**2 + d * q**3
    parts += rng.normal(scale=1e-4, size=parts.shape)
    r = snr_report(q, parts, sigma=1e-4, offset=np.zeros(5), target=10.0)
    assert r.c_resolved and r.d_resolved
    assert abs(r.b - b) / b < 0.01 and abs(r.c - c) / c < 0.05 and abs(r.d - d) / d < 0.1
    # SNR 10 for 2c Q^2 against sqrt 6 sigma, truncation 3 d Q / c = 10%
    assert np.isclose(r.q_lo2, np.sqrt(10 * np.sqrt(6) * 1e-4 / (2 * r.c)))
    assert np.isclose(r.q_hi2, 0.1 * r.c / (3 * r.d))
    noise, trunc = r.second_order(r.q_lo2)
    assert np.isclose(noise, 0.1)


def test_snr_report_bounds_an_unresolved_cubic_term():
    from spinlabs.convergence import format_snr, snr_report

    q = np.array([-0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1])
    parts = np.zeros((7, 5))
    parts[:, 1] = 20.0 * q + 5.0 * q**2  # E part, no cubic term
    r = snr_report(q, parts, sigma=1e-3, offset=np.zeros(5))
    assert not r.d_resolved and r.q_hi2 > 0
    assert "second order: " in format_snr(r, [0.05, 0.1]) and ">=" in format_snr(r, [0.05, 0.1])
    assert snr_report(q[1:-1], parts[1:-1]) is None  # 4 amplitudes besides D0: too few for a cubic


def test_pooled_noise_from_cubic_scatter():
    from spinlabs.convergence import pooled_noise

    rng = np.random.default_rng(2)
    q = np.array([-0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1])
    modes = [np.outer(q, rng.normal(size=5)) * 50 + np.outer(q**2, rng.normal(size=5)) * 10
             + rng.normal(scale=0.002, size=(7, 5)) for _ in range(20)]
    sigma, dof = pooled_noise([q] * 20, modes)
    assert dof == 20 * 2 * 5  # 6 amplitudes besides Q=0, 4 cubic coefficients, 5 parts
    assert 0.0017 < sigma < 0.0023


def test_d0_offset_shows_in_the_error_budget():
    from spinlabs.convergence import d0_offset, snr_report

    q = np.array([-0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1])
    parts = np.zeros((7, 5))
    parts[:, 0] = 100.0 * q + 10.0 * q**2
    parts[3, 2] = -0.01  # D0 off by 0.01 MHz in one E part
    offset = d0_offset([q, q], [parts, parts], parts[3])
    assert np.allclose(offset, [0, 0, 0.01, 0, 0])
    r = snr_report(q, parts, sigma=1e-5, offset=offset, target=10.0)
    assert np.isclose(r.offset, 0.01)
    # second order needs 2 c Q^2 >= 10 x offset: Q >= sqrt(0.1 / 20)
    assert np.isclose(r.q_lo2, np.sqrt(10 * r.err2 / (2 * r.c))) and r.q_lo2 > 0.07


def test_load_convergence_prepends_d0_when_no_q0_run(tmp_path):
    from spinlabs.convergence import load_convergence

    perts = np.array([-0.1, 0.1])
    tensors = np.stack([np.stack([GS + 1.0, GS + 2.0]), np.stack([GS + 3.0, GS + 4.0])])  # (2 modes, 2 Q)
    path = tmp_path / "conv.npz"
    np.savez(path, modes=np.array([5, 7]), perts=perts, tensors=tensors, d0=GS, metadata='{"defect": "NV"}')
    conv = load_convergence(path)
    assert np.allclose(conv.perts, [0.0, -0.1, 0.1])
    assert conv.tensors.shape == (2, 3, 3, 3)
    assert np.allclose(conv.tensors[:, 0], GS) and np.allclose(conv.q0, GS)
    assert np.allclose(conv.tensors[1, 2], GS + 4.0)
    assert conv.metadata == {"defect": "NV"}


def test_load_convergence_borrows_q0_for_a_mode_without_it(tmp_path):
    from spinlabs.convergence import load_convergence

    perts = np.array([-0.1, 0.0, 0.1])
    tensors = np.stack([np.stack([GS + 1.0, GS, GS + 2.0]), np.stack([GS + 3.0, np.full((3, 3), np.nan), GS + 4.0])])
    path = tmp_path / "conv.npz"
    np.savez(path, modes=np.array([5, 7]), perts=perts, tensors=tensors)
    conv = load_convergence(path)
    assert np.allclose(conv.tensors[1, 1], GS) and np.allclose(conv.q0, GS)
    assert conv.metadata == {}
    # a pack with neither a Q=0 run nor d0 has no anchor
    np.savez(path, modes=np.array([5]), perts=np.array([-0.1, 0.1]), tensors=tensors[:1, [0, 2]])
    assert load_convergence(path).q0 is None


def test_unconverged_run_is_suspicious(tmp_path):
    run = _tree_run(tmp_path, "1_1", _principal_change(a1=10.0), ops=6)
    (run / "INCAR").write_text("NELM = 30\nEDIFF = 1E-8\n")
    (run / "OSZICAR").write_text(
        "".join(f"DAV: {n:3d}    -0.1E+04   {1e-5:.5E}   -0.1E-03  1   0.1E-02\n" for n in range(1, 31))
        + "   1 F= -.1E+04 E0= -.1E+04  d E =-.1E-07\n"
    )
    check = check_run(run, _spectrum(), (1, 1))
    assert check.status == "suspicious"
    assert "SCF did not converge: stopped at NELM = 30 steps" in check.report()
