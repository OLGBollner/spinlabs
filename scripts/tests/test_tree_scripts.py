"""create_perturbation_tree.py and the structure scripts, run as on the cluster."""

import re
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]  # this checkout, which holds the runner
SCRIPTS = REPO / "scripts" / "cluster"


def _save(spec, path):
    from spinlabs.parsers import save_phonon_npz

    save_phonon_npz(spec, path)


def _tiny(freqs, syms):
    """Minimal PhononSpectrum with preset labels (bypasses classification)."""
    from spinlabs.models import PhononSpectrum

    n = len(freqs)
    return PhononSpectrum(
        frequencies_mev=np.asarray(freqs, dtype=float),
        eigenvectors=np.zeros((n, 1, 3)),
        atom_frac_coords=np.zeros((1, 3)),
        atom_symbols=["C"],
        atomic_masses=np.array([12.0]),
        lattice=np.eye(3),
        symmetries=list(syms),
        original_indices=np.arange(n),
    )


def test_cli_expands_tilde_binary(tmp_path, monkeypatch):
    """--vasp-binary ../... must resolve to an absolute path baked into the
    sbatch scripts, like --phonon."""
    defect = tmp_path / "NV_512"
    (defect / "data").mkdir(parents=True)
    _save(_tiny([10.0, 20.0, 30.0], ["A1", "Ex", "A2"]), defect / "data" / "phonon_data.npz")
    for stage in ("ZFS_hyp", "ZFS_occup"):
        d = defect / stage
        d.mkdir()
        (d / "INCAR").write_text("ENCUT = 520\n")
        for name in ("POSCAR", "KPOINTS", "POTCAR"):
            (d / name).write_text(name + "\n")
        (d / "OUTCAR").write_text("Elapsed time (sec):   100.000\n")
    args = [str(defect), "--pert", "0.0001", "--array-jobs", "1", "--vasp-binary", "../me/vasp_std"]
    r = subprocess.run(
        ["python", str(SCRIPTS / "create_perturbation_tree.py"), *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": __import__("os").environ["PATH"]},
    )
    assert r.returncode == 0, r.stdout + r.stderr
    sbatch = next(defect.rglob("run_perturbation_first_order.sh"))
    assert f"binary={tmp_path.parent}/me/vasp_std" in sbatch.read_text()


def test_generated_scripts_keep_the_time_estimate_and_name_the_repo(tmp_path):
    """The tree generator fills the wrapper: computed SBATCH time/array/name, config and SPINLABS_REPO."""
    defect = tmp_path / "NV_512"
    (defect / "data").mkdir(parents=True)
    _save(_tiny([10.0, 20.0, 30.0], ["A1", "Ex", "A2"]), defect / "data" / "phonon_data.npz")
    for stage in ("ZFS_hyp", "ZFS_occup"):
        d = defect / stage
        d.mkdir()
        (d / "INCAR").write_text("ENCUT = 520\nLDMATRIX = .TRUE.\n")
        for name in ("POSCAR", "KPOINTS", "POTCAR"):
            (d / name).write_text(name + "\n")
        (d / "OUTCAR").write_text("Elapsed time (sec):   3600.000\n")
    r = subprocess.run(
        ["python", str(SCRIPTS / "create_perturbation_tree.py"), str(defect),
         "--pert", "0.1", "--array-jobs", "2", "--vasp-binary", "/bin/true"],
        capture_output=True, text=True, cwd=tmp_path,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": __import__("os").environ["PATH"]},
    )
    assert r.returncode == 0, r.stdout + r.stderr
    for script in defect.rglob("run_perturbation_*_order.sh"):
        text = script.read_text()
        assert f"SPINLABS_REPO={REPO}" in text
        assert re.search(r"^#SBATCH -a 0-1$", text, re.M)
        assert re.search(r"^#SBATCH -t \d+:\d\d:\d\d$", text, re.M)
        assert re.search(r"^#SBATCH -J NV_512_(first|second)_order_", text, re.M)
        assert "PERT=0.1" in text and re.search(r"^binary=\S*/true$", text, re.M)
        assert "<" not in "".join(ln for ln in text.splitlines() if re.match(r"^[A-Za-z_]+=", ln))


def test_unweighted_tree_keeps_q_times_e_through_generation_structures_and_packaging(tmp_path):
    """--no-mass-weighted writes MASS_WEIGHTED=0; the structure script then displaces by Q*e, which the
    perturbation check accepts only as unweighted, and packaging reads the setting back from the tree."""
    import os

    import numpy as np
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.vasp.inputs import Poscar

    from spinlabs.cli.package import tree_mass_weighted

    defect = tmp_path / "NV_64"
    (defect / "data").mkdir(parents=True)
    ref = Structure(Lattice.cubic(3.57), ["C", "C", "N"], [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0]])
    from spinlabs.models import PhononSpectrum

    eig = np.random.default_rng(4).normal(size=(3, 3, 3))
    eig /= np.linalg.norm(eig.reshape(3, -1), axis=1)[:, None, None]
    spec = PhononSpectrum(
        frequencies_mev=np.array([10.0, 20.0, 30.0]), eigenvectors=eig,
        atom_frac_coords=np.array([s.frac_coords for s in ref]), atom_symbols=["C", "C", "N"],
        atomic_masses=np.array([12.011, 12.011, 14.007]), lattice=np.eye(3) * 3.57,
        symmetries=["A1", "Ex", "A2"], original_indices=np.arange(3),
    )
    _save(spec, defect / "data" / "phonon_data.npz")
    for stage in ("ZFS_hyp", "ZFS_occup"):
        d = defect / stage
        d.mkdir()
        (d / "INCAR").write_text("ENCUT = 520\n")
        Poscar(ref).write_file(d / "POSCAR")
        for name in ("KPOINTS", "POTCAR"):
            (d / name).write_text(name + "\n")
        (d / "OUTCAR").write_text("Elapsed time (sec):   3600.000\n")
    env = {"PYTHONPATH": str(REPO / "src"), "PATH": os.environ["PATH"]}
    scripts = SCRIPTS
    r = subprocess.run(
        ["python", str(scripts / "create_perturbation_tree.py"), str(defect), "--pert", "0.1", "--no-mass-weighted"],
        capture_output=True, text=True, cwd=tmp_path, env=env,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    basis = defect / "first_order" / "pert_0.1" / "all_bands"
    assert re.search(r"^MASS_WEIGHTED=0\b", (basis / "run_perturbation_first_order.sh").read_text(), re.M)
    assert tree_mass_weighted(basis) is False

    phonons = defect / "data" / "phonon_data.npz"
    run = lambda *a: subprocess.run(["python", *map(str, a)], capture_output=True, text=True, cwd=tmp_path, env=env)  # noqa: E731
    made = run(scripts / "create_phonon_struct.py", "-o", tmp_path / "POSCAR_q", basis / "input" / "POSCAR",
               phonons, 1, 0.1, "--no-mass-weighted")
    assert made.returncode == 0, made.stdout + made.stderr
    check = [scripts / "check_perturbation.py", tmp_path / "POSCAR_q", basis / "input" / "POSCAR", phonons, 1, "--pert", 0.1]
    assert run(*check, "--no-mass-weighted").returncode == 0  # Q*e, as intended
    assert run(*check).returncode == 1  # not Q*e/sqrt(m)
