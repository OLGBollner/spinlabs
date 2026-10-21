"""check_run.py, run as the job does after each VASP run."""

import subprocess
import sys
from pathlib import Path

import numpy as np

from spinlabs.models import PhononSpectrum

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "cluster"
GS = np.array([[0.0, 1000.0, 1000.0], [1000.0, 0.0, 1000.0], [1000.0, 1000.0, 0.0]])  # NV along [111]


def _principal_change(a1: float = 0.0) -> np.ndarray:
    """GS plus an A1 change (D_zz) in GS's principal frame."""
    _, rot = np.linalg.eigh(GS)
    return GS + rot @ np.diag([-a1 / 2, -a1 / 2, a1]) @ rot.T


def _outcar(path: Path, t: np.ndarray, *, ops: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "   ISYM   =      3\n The static configuration has the point symmetry C_s .\n"
        f" GETGRP returns: Found  {ops} space group operations\n"
        " Spin-spin contribution to zero-field splitting tensor (MHz)\n -------\n"
        "      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz\n -------\n"
        f"  {t[0, 0]:.4f}  {t[1, 1]:.4f}  {t[2, 2]:.4f}  {t[0, 1]:.4f}  {t[0, 2]:.4f}  {t[1, 2]:.4f}\n -------\n"
        " General timing and accounting informations for this job:\n"
    )


def _spectrum() -> PhononSpectrum:
    eigs = np.random.default_rng(5).normal(size=(4, 4, 3))
    eigs /= np.linalg.norm(eigs.reshape(4, -1), axis=1)[:, None, None]
    return PhononSpectrum(
        frequencies_mev=np.array([20.0, 40.0, 40.0, 60.0]), symmetries=["A1", "Ex", "Ey", "A2"],
        eigenvectors=eigs, atom_frac_coords=np.zeros((4, 3)), atom_symbols=["C", "C", "C", "N"],
        atomic_masses=np.array([12.011] * 3 + [14.007]), lattice=np.eye(3) * 3.57,
    )


def test_check_run_script_writes_check_file(tmp_path):
    defect = tmp_path / "NV_64"
    _outcar(defect / "ZFS_hyp" / "OUTCAR", GS, ops=6)
    runs = defect / "second_order" / "pert_0.1" / "all_bands" / "runs"
    bad, good = runs / "2", runs / "1"
    _outcar(bad / "OUTCAR", _principal_change(a1=30.0), ops=6)  # E mode: too many ops, A1-only change
    _outcar(good / "OUTCAR", _principal_change(a1=10.0), ops=6)
    phonons = tmp_path / "phonons.npz"
    _spectrum().save(phonons)
    r = subprocess.run([sys.executable, str(SCRIPTS / "check_run.py"), str(good), str(bad), "--phonons", str(phonons),
                        "--write-check"], capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "checked 2 run(s): 1 suspicious" in r.stdout
    assert (bad / "CHECK").is_file() and not (good / "CHECK").exists()
    assert "at most 2 ops expected but VASP found 6" in (bad / "CHECK").read_text()
