"""Mirror projection of forward diagonal second differences (ZFSManager mirror_projection).

The forward difference (D(2Q) - 2 D(Q) + D0) / Q^2 = D'' + Q D''' + ... carries the odd powers of Q. For a
mode that a sigma_v maps onto -Q the even terms are mirror-symmetric and the odd ones mirror-antisymmetric,
so the symmetric part of d2D is the curvature without that error.
"""

from pathlib import Path

import numpy as np
import pytest

from spinlabs.constants import CONSTANTS
from spinlabs.models import PhononSpectrum, RawZFSData, ZFSTensor
from spinlabs.parsers import parse_phonon_npz
from spinlabs.symmetry import mirror_parities
from spinlabs.zfs_manager import ZFSManager

REPO_ROOT = Path(__file__).resolve().parents[1]
NV_64_PHONONS = REPO_ROOT / "NV_64" / "phonon_data.npz"  # plain git blob
NV_512_SUBSET = REPO_ROOT / "NV_512" / "phonon_data_subset.npz"  # plain git blob

Q = 0.01 * CONSTANTS["ang_amu2SI"]  # perturbation amplitude of the synthetic runs
MHZ = CONSTANTS["MHz2J"]
D0 = np.diag([-2870.0 / 3, -2870.0 / 3, 2 * 2870.0 / 3]) * MHZ
MIRROR = np.diag([1.0, -1.0, 1.0])  # sigma_v with normal y: xz, xx, yy, zz even; xy, yz odd


def _sym(t):
    t = np.asarray(t, float)
    return 0.5 * (t + t.T)


# D(Q) = D0 + B q + C q^2 + T q^3 (q = Q / Q0): slope and cubic M-odd, curvature M-even (MHz).
B = _sym([[0.0, 30.0, 0.0], [0.0, 0.0, 20.0], [0.0, 0.0, 0.0]])
C = _sym([[3.0, 0.0, 2.0], [0.0, -1.0, 0.0], [0.0, 0.0, -2.0]])
T = _sym([[0.0, 4.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 0.0]])
assert np.allclose(MIRROR @ B @ MIRROR, -B) and np.allclose(MIRROR @ T @ MIRROR, -T)
assert np.allclose(MIRROR @ C @ MIRROR, C)


def _D(q, rot=np.eye(3)):
    """D at q steps (J), stored as the manager stores tensors: rotated into the D0 frame (R.T D R)."""
    return rot.T @ (D0 + (B * q + C * q**2 + T * q**3) * MHZ) @ rot


def _couplings(d2):
    """(V_00, V_+-, V_0+-) of a diagonal second derivative, the production formulas."""
    tr, diff = d2[0, 0] + d2[1, 1], d2[0, 0] - d2[1, 1]
    return (
        abs(d2[2, 2] - 0.5 * tr) / 6.0,
        0.25 * np.sqrt(diff**2 + 4.0 * d2[0, 1] ** 2),
        np.hypot(d2[0, 2], d2[1, 2]) / (2.0 * np.sqrt(2)),
    )


@pytest.fixture
def manager():
    n_atoms = 2
    spectrum = PhononSpectrum(
        frequencies_mev=np.array([10.0, 20.0, 20.0, 30.0]),
        eigenvectors=np.random.default_rng(0).standard_normal((4, n_atoms, 3)),
        atom_frac_coords=np.zeros((n_atoms, 3)),
        atom_symbols=["C", "C"],
        atomic_masses=np.array([12.011, 12.011]),
        lattice=np.eye(3) * 3.56,
        symmetries=["A1", "Ex", "Ey", "A1"],
        iprs=np.ones(4) * 0.5,
    )
    raw = RawZFSData(defect="NV", cell_size=64, pert_scale=0.01, ground_state_zfs=ZFSTensor(matrix=D0 / MHZ, unit="MHz"))
    m = ZFSManager(spectrum=spectrum, raw_data=raw)
    m.zfs_relaxed = D0.copy()
    # Mode 1 (Ex) is mirror-odd, its twin 2 (Ey) mirror-even; the synthetic geometry gives no real mirrors.
    m._mirror_parity = {0: (np.eye(3), 1.0), 1: (MIRROR, -1.0), 2: (MIRROR, 1.0), 3: (np.eye(3), 1.0)}
    return m


def _forward_entry(mode, sym, rot=np.eye(3), scale=1.0):
    """The (i, i) pair run D(2Q) of a mode whose response is ``scale`` times D(q)."""
    tensor = rot.T @ D0 @ rot + scale * (_D(2.0, rot) - rot.T @ D0 @ rot)
    return {"tensor": tensor, "pert": (Q, Q), "symmetry": (sym, sym), "ipr": (0.5, 0.5)}


def _first_order(rot=np.eye(3), modes=(1, 2), scales=None):
    """Forward first-order derivatives (D(Q) - D0) / Q of the synthetic modes."""
    d1 = np.zeros((4, 3, 3))
    for k in modes:
        s = 1.0 if scales is None else scales[k]
        d1[k] = s * (_D(1.0, rot) - rot.T @ D0 @ rot) / Q
    return d1


def _run(m, entries, d1):
    m.zfs_tensors_2d = entries
    m.treated_modes = {0, 1, 2, 3}
    return m.calculate_second_order_derivatives(d1)


def test_projection_recovers_the_curvature_of_a_mirror_odd_mode(manager):
    exact = _couplings(2.0 * C * MHZ / Q**2)  # d2D/dQ^2 = 2 C per step^2
    entries = {(1, 1): _forward_entry(1, "Ex")}

    _, V00, Vpm, V0pm = _run(manager, entries, _first_order())  # default: off
    assert manager.mirror_projected == []
    assert not np.isclose(V0pm[1, 1], exact[2], rtol=1e-6)  # the forward value carries 6 T q (yz, xy)

    manager.mirror_projection = True
    _, V00, Vpm, V0pm = _run(manager, entries, _first_order())
    assert manager.mirror_projected == [1]
    assert np.allclose([V00[1, 1], Vpm[1, 1], V0pm[1, 1]], exact, rtol=1e-9)


def test_projection_is_independent_of_the_principal_frame_rotation(manager):
    t = np.radians(35.0)  # a rotation about the axis: what an in-plane principal-frame choice can give
    rot = np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])
    manager.mirror_projection = True
    _, V00, Vpm, V0pm = _run(manager, {(1, 1): _forward_entry(1, "Ex")}, _first_order())
    manager.eigen_rotation = rot
    manager.zfs_relaxed = rot.T @ D0 @ rot
    _, V00r, Vpmr, V0pmr = _run(manager, {(1, 1): _forward_entry(1, "Ex", rot)}, _first_order(rot))
    assert np.allclose([V00r[1, 1], Vpmr[1, 1], V0pmr[1, 1]], [V00[1, 1], Vpm[1, 1], V0pm[1, 1]], rtol=1e-9)


def test_mirror_even_partner_takes_its_odd_twins_couplings(manager):
    manager.mirror_projection = True
    # The even partner's own forward entry is off (its odd terms are mirror-even and cannot be removed).
    entries = {(1, 1): _forward_entry(1, "Ex"), (2, 2): _forward_entry(2, "Ey", scale=1.7)}
    _, V00, Vpm, V0pm = _run(manager, entries, _first_order(scales={1: 1.0, 2: 1.7}))
    assert manager.mirror_projected == [1]
    assert manager.mirror_inherited == [(2, 1)]
    assert manager.mirror_uncorrected == []
    for V in (V00, Vpm, V0pm):
        assert V[2, 2] == V[1, 1]


def test_even_partner_without_a_projected_twin_is_reported_and_kept(manager, caplog):
    manager.mirror_projection = True
    entries = {(2, 2): _forward_entry(2, "Ey")}
    _, V00_on, Vpm_on, V0pm_on = _run(manager, entries, _first_order(modes=(2,)))
    assert manager.mirror_projected == [] and manager.mirror_inherited == []
    assert manager.mirror_uncorrected == [2]
    assert "keep their odd-in-Q error" in caplog.text
    manager.mirror_projection = False
    _, V00, Vpm, V0pm = _run(manager, entries, _first_order(modes=(2,)))
    assert np.array_equal(V0pm_on, V0pm) and np.array_equal(Vpm_on, Vpm) and np.array_equal(V00_on, V00)


def test_weakly_aligned_and_central_entries_are_left_alone(manager):
    manager.mirror_projection = True
    entry = _forward_entry(1, "Ex")
    manager._mirror_parity[1] = (MIRROR, -0.5)  # below MIRROR_MIN_OVERLAP: not taken as mirror-odd
    _, _, _, V0pm_weak = _run(manager, {(1, 1): dict(entry)}, _first_order())
    assert manager.mirror_projected == []

    manager._mirror_parity[1] = (MIRROR, -1.0)
    central = {"tensor": _D(1.0), "tensor_minus": _D(-1.0), "central": True,
               "pert": (Q, Q), "symmetry": ("Ex", "Ex"), "ipr": (0.5, 0.5)}
    _, _, _, V0pm_c = _run(manager, {(1, 1): central}, _first_order())
    assert manager.mirror_projected == []  # central differences have no odd terms
    assert np.isclose(V0pm_c[1, 1], _couplings(2.0 * C * MHZ / Q**2)[2], rtol=1e-9)


def test_non_c3v_structure_skips_the_projection_with_a_warning(manager, caplog):
    # The fixture's labels imply C3v (manager.point_group == "3m"), but its structure (atoms at the
    # origin) has no detectable C3v: the mirrors must come from the structure, so nothing is projected.
    manager.mirror_projection = True
    manager._mirror_parity = None
    assert manager.point_group == "3m"
    _run(manager, {(1, 1): _forward_entry(1, "Ex")}, _first_order())
    assert manager.mirror_projected == []
    assert "needs a C3v structure" in caplog.text


# -- mirror parities from real geometry ---------------------------------------------------------------------


def _parities(path):
    spec = parse_phonon_npz(path)
    from spinlabs.symmetry import classify_and_pair

    labels, groups = classify_and_pair(spec, force=True)
    return np.array(labels), groups, mirror_parities(spec)


def test_nv64_parities_follow_c3v():
    labels, groups, par = _parities(NV_64_PHONONS)
    chi = np.array([par[k][1] for k in range(len(labels))])
    assert np.allclose(chi[labels == "A1"], 1.0, atol=1e-3)
    assert np.allclose(chi[labels == "A2"], -1.0, atol=1e-3)
    pairs = [g for g in groups if len(g) == 2]
    assert len(pairs) == 63
    for a, b in pairs:
        # Partners of an E pair overlap with the same mirror with opposite sign, |chi| >= cos 30 deg.
        assert np.isclose(chi[a], -chi[b], atol=1e-3)
        assert abs(chi[a]) >= np.cos(np.radians(30.0)) - 1e-3
    # NV_64's pairs come from the phonon code in an arbitrary orientation inside each pair.
    assert np.min(np.abs(chi[np.isin(labels, ["Ex", "Ey"])])) < 0.95


def test_nv512_subset_e_modes_are_mirror_adapted():
    labels, _, par = _parities(NV_512_SUBSET)
    chi = np.array([par[k][1] for k in range(len(labels))])
    e = np.isin(labels, ["Ex", "Ey"])
    assert e.any()
    assert np.allclose(np.abs(chi[e]), 1.0, atol=1e-3)
