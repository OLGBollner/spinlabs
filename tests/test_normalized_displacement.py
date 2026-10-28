"""Normalized displacements (the same Cartesian step along every mode) and the convergence folders."""

import json
import re

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from scipy import constants as Cn

from spinlabs.constants import CONSTANTS
from spinlabs.models import PhononSpectrum, RawZFSData, ZFSTensor
from spinlabs.vasp.structures import apply_combined_perturbation, apply_perturbation, run_modes, run_pert_scale

STEP = 0.025
MASSES = np.array([12.011, 12.011, 28.0855])  # two carbons and a heavy (Si) atom: n_k differs between modes


def _spectrum() -> PhononSpectrum:
    """9 orthonormal mass-weighted modes of a 3-atom cell, so normal coordinates are projections."""
    q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(9, 9)))
    return PhononSpectrum(
        frequencies_mev=np.linspace(10.0, 90.0, 9),
        eigenvectors=q.T.reshape(9, 3, 3),
        atom_frac_coords=np.array([[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0]]),
        atom_symbols=["C", "C", "Si"],
        atomic_masses=MASSES,
        lattice=np.eye(3) * 3.57,
        symmetries=["A1"] * 9,
        original_indices=np.arange(9),
    )


def _structure() -> Structure:
    return Structure(Lattice.cubic(3.57), ["C", "C", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0]])


def _displacement(perturbed: Structure, reference: Structure) -> np.ndarray:
    return np.array([p.coords - r.coords for p, r in zip(perturbed, reference)])


def _normal_coordinates(spec: PhononSpectrum, disp: np.ndarray) -> np.ndarray:
    """Q_k = e_k . (sqrt(m) dr) for every mode."""
    weighted = (np.sqrt(MASSES)[:, None] * disp).ravel()
    return np.asarray(spec.eigenvectors).reshape(9, -1) @ weighted


def test_norm_factors_are_one_over_the_mass_weighted_eigenvector_length():
    spec = _spectrum()
    u = np.asarray(spec.eigenvectors) / np.sqrt(MASSES)[None, :, None]
    np.testing.assert_allclose(spec.mode_norm_factors(), 1 / np.linalg.norm(u.reshape(9, -1), axis=1))
    assert np.ptp(spec.mode_norm_factors()) > 0.3  # the heavy atom makes the modes differ


def test_normalized_step_moves_every_mode_by_the_same_cartesian_length():
    spec, ref = _spectrum(), _structure()
    for mode in range(1, 10):
        normalized = _displacement(apply_perturbation(ref, spec.eigenvectors, MASSES, mode, STEP,
                                                      displacement="normalized"), ref)
        weighted = _displacement(apply_perturbation(ref, spec.eigenvectors, MASSES, mode, STEP), ref)
        assert np.linalg.norm(normalized) == pytest.approx(STEP)
        assert np.linalg.norm(weighted) == pytest.approx(STEP / spec.mode_norm_factors()[mode - 1])
        np.testing.assert_allclose(normalized / np.linalg.norm(normalized), weighted / np.linalg.norm(weighted))


def _unit_mode(rng, atoms) -> np.ndarray:
    """A unit eigenvector (3 atoms, 3) moving only ``atoms``."""
    e = np.zeros((3, 3))
    e[atoms] = rng.normal(size=(len(atoms), 3))
    return e / np.linalg.norm(e)


@pytest.mark.parametrize("atoms, mass", [([0, 1], 12.011), ([2], 28.0855)], ids=["carbon-only", "silicon-only"])
def test_a_single_species_mode_is_displaced_alike_normalized_and_unweighted(atoms, mass):
    """A mode moving one species only has u = e/sqrt(m) with one m, so u/|u| = e: the normalized step and
    the unweighted step move every atom by the same vector, and the mass-weighted one by that over sqrt(m)."""
    ref = _structure()
    eigs = _unit_mode(np.random.default_rng(3), atoms)[None]
    disp = {conv: _displacement(apply_perturbation(ref, eigs, MASSES, 1, STEP, displacement=conv), ref)
            for conv in ("normalized", "unweighted", "mass_weighted")}
    np.testing.assert_allclose(disp["normalized"], disp["unweighted"], atol=1e-12)
    assert np.linalg.norm(disp["normalized"]) == pytest.approx(STEP)
    np.testing.assert_allclose(disp["mass_weighted"], disp["unweighted"] / np.sqrt(mass), atol=1e-12)


def test_a_mixed_species_mode_is_displaced_differently_normalized_and_unweighted():
    """With carbon and silicon in one mode, normalizing u = e/sqrt(m) leans the step towards the light atoms."""
    ref = _structure()
    eigs = _unit_mode(np.random.default_rng(3), [0, 1, 2])[None]
    normalized = _displacement(apply_perturbation(ref, eigs, MASSES, 1, STEP, displacement="normalized"), ref)
    unweighted = _displacement(apply_perturbation(ref, eigs, MASSES, 1, STEP, displacement="unweighted"), ref)
    assert np.linalg.norm(normalized) == pytest.approx(STEP) and np.linalg.norm(unweighted) == pytest.approx(STEP)
    assert not np.allclose(normalized, unweighted, atol=1e-6)
    # the silicon atom moves relatively less, the carbons more
    assert np.linalg.norm(normalized[2]) < np.linalg.norm(unweighted[2])


def test_normal_coordinates_of_a_normalized_pair_are_step_times_n():
    spec, ref = _spectrum(), _structure()
    n = spec.mode_norm_factors()
    pair = _displacement(apply_combined_perturbation(ref, spec.eigenvectors, 2, 7, MASSES, STEP,
                                                     displacement="normalized"), ref)
    q = _normal_coordinates(spec, pair)
    expected = np.zeros(9)
    expected[[1, 6]] = STEP * n[[1, 6]]  # each mode of the pair moves by step * n_k, not by one shared Q
    np.testing.assert_allclose(q, expected, atol=1e-12)


def test_analysis_amplitude_follows_the_convention():
    spec = _spectrum()
    scale = STEP * CONSTANTS["ang_amu2SI"]
    omega = spec.frequencies_mev * CONSTANTS["meV2rads"]
    zero_point = np.sqrt(2 * omega / Cn.hbar)
    normalized = np.array(spec.get_phonon_pert(scale, displacement="normalized")["disp"])
    weighted = np.array(spec.get_phonon_pert(scale)["disp"])
    np.testing.assert_allclose(weighted, scale * zero_point)
    np.testing.assert_allclose(normalized, scale * spec.mode_norm_factors() * zero_point)


def _raw(displacement=None, mass_weighted=True):
    gs = np.diag([-1000.0, -1000.0, 2000.0])
    first = {i: {"tensor": gs + (i + 1) * np.diag([-1.0, -1.0, 2.0]), "unit": "MHz"} for i in range(2)}
    return RawZFSData(defect="SiV", cell_size=3, pert_scale=STEP, calc_method="all_bands",
                      mass_weighted=mass_weighted, displacement=displacement,
                      ground_state_zfs=ZFSTensor(matrix=gs, unit="MHz"), first_order=first)


def test_convention_round_trips_and_mixing_is_refused(tmp_path):
    path = _raw("normalized").save(tmp_path / "normalized.npz")
    assert RawZFSData.load(path).displacement_convention == "normalized"
    assert _raw().displacement_convention == "mass_weighted"  # data from before the key
    assert _raw(mass_weighted=False).displacement_convention == "unweighted"
    with pytest.raises(ValueError, match="normalized and mass_weighted"):
        _raw("normalized").combine(_raw())


def test_manager_uses_the_normalized_amplitudes():
    from spinlabs.zfs_manager import ZFSManager

    spec = _spectrum()
    manager = ZFSManager(spectrum=spec, raw_data=_raw("normalized"))
    assert manager.displacement == "normalized"
    q = manager.get_phonon_pert(STEP * CONSTANTS["ang_amu2SI"])["disp"]
    reference = spec.get_phonon_pert(STEP * CONSTANTS["ang_amu2SI"])["disp"]
    np.testing.assert_allclose(np.array(q) / np.array(reference), spec.mode_norm_factors())


# ---- trees and convergence folders ----

def _defect(tmp_path, name="SiV_3"):
    from spinlabs.parsers import save_phonon_npz

    defect = tmp_path / name
    (defect / "data").mkdir(parents=True)
    save_phonon_npz(_spectrum(), defect / "data" / "phonon_data.npz")
    for stage in ("ZFS_hyp", "ZFS_occup"):
        d = defect / stage
        d.mkdir()
        (d / "INCAR").write_text("ENCUT = 520\n")
        for name in ("POSCAR", "KPOINTS", "POTCAR"):
            (d / name).write_text(name + "\n")
        (d / "OUTCAR").write_text("Elapsed time (sec):   100.000\n")
    return defect


def _config(script, key):
    return re.search(rf"^{key}=(.*?)(\s+#.*)?$", script.read_text(), re.M).group(1)


def test_new_tree_is_normalized_and_gets_convergence_folders(tmp_path):
    from spinlabs.vasp.trees import build_perturbation_tree, tree_displacement

    defect = _defect(tmp_path)
    failures = build_perturbation_tree(defect, [STEP], defect / "data" / "phonon_data.npz", displacement="normalized",
                                       conv_modes=[2, 7], vasp_binary=tmp_path / "vasp_std")
    assert failures == []
    tree = defect / "first_order" / f"pert_{STEP}" / "all_bands"
    assert _config(tree / "run_perturbation_first_order.sh", "DISPLACEMENT") == "normalized"
    assert tree_displacement(tree) == "normalized"
    for folder in ("convergence_test_all", "convergence_test_occup"):
        script = defect / folder / "run_convergence.sh"
        assert (defect / folder / "input" / "INCAR").is_file()
        assert _config(script, "MODES") == '"2 7"'
        assert _config(script, "PERTS") == '"0.0125 0.025 0.05 0.1"'  # p/2, p, 2p, 4p, each run at both signs
        assert _config(script, "DISPLACEMENT") == "normalized"
        assert "<" not in "".join(ln for ln in script.read_text().splitlines() if re.match(r"^[A-Za-z_]+=", ln))


@pytest.mark.parametrize("line, expected", [
    ("MASS_WEIGHTED=0", "unweighted"),
    ("MASS_WEIGHTED=${MASS_WEIGHTED:-0}  # 1: displace by Q*e/sqrt(m); 0: by Q*e", "unweighted"),
    ("MASS_WEIGHTED=${MASS_WEIGHTED:-1}", "mass_weighted"),
    ("DISPLACEMENT=${DISPLACEMENT:-normalized}", "normalized"),
    ("PERT=0.1", None),
])
def test_tree_displacement_reads_set_and_default_forms(tmp_path, line, expected):
    from spinlabs.vasp.trees import tree_displacement

    (tmp_path / "run_perturbation_first_order.sh").write_text(f"DEFECT=NV_64\n{line}\n")
    assert tree_displacement(tmp_path) == expected


def test_tree_needs_a_step(tmp_path):
    from spinlabs.vasp.trees import build_perturbation_tree

    defect = _defect(tmp_path)
    with pytest.raises(ValueError, match="step"):
        build_perturbation_tree(defect, [], defect / "data" / "phonon_data.npz", displacement="normalized",
                                conv_modes=None)


def test_convergence_added_to_an_existing_tree_matches_its_displacement(tmp_path):
    from spinlabs.vasp.trees import build_convergence_tree, defect_displacement, defect_vasp_binary

    defect = _defect(tmp_path)
    old = defect / "first_order" / "pert_0.1" / "all_bands"
    old.mkdir(parents=True)
    (old / "run_perturbation_first_order.sh").write_text("binary=/opt/vasp_std\nMASS_WEIGHTED=0\n")
    assert defect_displacement(defect) == "unweighted"
    assert defect_vasp_binary(defect).as_posix() == "/opt/vasp_std"
    phonon = defect / "data" / "phonon_data.npz"
    assert build_convergence_tree(defect, phonon, [1], [-STEP, STEP], displacement="unweighted",
                                  vasp_binary=defect_vasp_binary(defect)) == []
    assert _config(defect / "convergence_test_all" / "run_convergence.sh", "DISPLACEMENT") == "unweighted"
    with pytest.raises(ValueError, match=r"modes \[42\]"):
        build_convergence_tree(defect, phonon, [42], [STEP], displacement="normalized", force=True)


def test_convergence_run_folders_resolve_modes_step_and_ground_state(tmp_path):
    from spinlabs.vasp.run_checks import reference_outcar

    run = tmp_path / "SiV_3" / "convergence_test_occup" / "runs" / "7_pert_-0.025"
    run.mkdir(parents=True)
    (tmp_path / "SiV_3" / "ZFS_occup").mkdir()
    (tmp_path / "SiV_3" / "ZFS_occup" / "OUTCAR").write_text("")
    assert run_modes(run) == (7,)
    assert run_pert_scale(run) == -0.025
    assert reference_outcar(run) == tmp_path / "SiV_3" / "ZFS_occup" / "OUTCAR"


def test_convergence_pack_records_the_displacement(tmp_path):
    from spinlabs.pack_convergence import pack_convergence

    root = tmp_path / "SiV_3" / "convergence_test_all"
    run = root / "runs" / "7_pert_0.025"
    run.mkdir(parents=True)
    (root / "run_convergence.sh").write_text("DISPLACEMENT=normalized\n")
    (run / "OUTCAR").write_text(
        " Spin-spin contribution to zero-field splitting tensor (MHz)\n ----\n"
        "      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz\n ----\n"
        "  -1.0  -1.0  2.0  0.0  0.0  0.0\n ----\n"
    )
    out = pack_convergence(root, tmp_path / "pack.npz")
    with np.load(out, allow_pickle=True) as data:
        assert json.loads(str(data["metadata"]))["displacement"] == "normalized"
