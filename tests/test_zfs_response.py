"""The shared ZFS symmetry-response check (cluster runs, packaging and ZFSManager) and its projectors."""

import numpy as np
import pytest

from spinlabs.models import PhononSpectrum
from spinlabs.symmetry import direct_product, tensor_irrep_fractions
from spinlabs.zfs_response import allowed_irreps, response_point_group, verify_zfs_response

NV_AXIS = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)


def _axis_frame(axis: np.ndarray) -> np.ndarray:
    """Rotation taking z to ``axis`` (columns: x', y', axis)."""
    x = np.cross(axis, [0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(axis, x), axis])


def _lab(tensor_in_defect_frame: np.ndarray, axis=NV_AXIS) -> np.ndarray:
    rot = _axis_frame(axis)
    return rot @ tensor_in_defect_frame @ rot.T


GS = _lab(np.diag([-957.0, -957.0, 1913.0]))  # NV ground state along [111], MHz
A1 = np.diag([-5.0, -5.0, 10.0])
E = np.array([[4.0, 3.0, 2.0], [3.0, -4.0, 1.0], [2.0, 1.0, 0.0]])


def test_projectors_are_complete_and_symmetric_tensors_have_no_a2():
    assert tensor_irrep_fractions(A1)["A1"] == pytest.approx(1.0)
    assert tensor_irrep_fractions(E)["E"] == pytest.approx(1.0)  # the dimension factor d_E = 2 included
    t = np.random.default_rng(3).normal(size=(3, 3))
    fractions = tensor_irrep_fractions(t + t.T)
    assert sum(f**2 for f in fractions.values()) == pytest.approx(1.0)
    assert fractions["A2"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("a, b, product", [
    ("Ex", "Ey", ["A1", "A2", "E"]), ("A1", "Ex", ["E"]), ("A2", "A2", ["A1"]), ("A1", "A2", ["A2"]),
])
def test_direct_products(a, b, product):
    assert direct_product("3m", a, b) == product


@pytest.mark.parametrize("labels, kind, allowed", [
    (["A1"], "displacement", {"A1"}),
    (["A2"], "displacement", {"A1", "A2"}),  # linear A2 (absent from D) plus the A1 curvature
    (["Ex"], "displacement", {"A1", "A2", "E"}),
    (["A1", "A1"], "derivative", {"A1"}),
    (["A2", "A2"], "derivative", {"A1", "A2"}),  # + A2 x A2 x A2 = A2 at O(q)
    (["A1", "Ex"], "derivative", {"A1", "A2", "E"}),  # + A1 x E x E at O(q)
    (["Ex", "Ey"], "derivative", {"A1", "A2", "E"}),
])
def test_allowed_irreps(labels, kind, allowed):
    assert allowed_irreps("3m", labels, kind) == allowed


@pytest.mark.parametrize("labels, strict", [
    (["A1", "Ex"], {"E"}), (["A1", "A2"], {"A2"}), (["A2", "A2"], {"A1"}), (["Ex"], {"E"}),
])
def test_strict_rule_is_the_exact_product(labels, strict):
    assert allowed_irreps("3m", labels, "derivative", strict=True) == strict


def test_finite_difference_terms_are_reported_but_not_warned_about():
    curvature = _lab(A1)  # an A1 x E curvature that is pure A1: only the O(q D''') terms can give that
    check = verify_zfs_response(curvature, ["A1", "Ex"], kind="derivative", point_group="3m", reference=GS)
    assert check.problem is None and check.strict_forbidden == pytest.approx(1.0)
    broken = verify_zfs_response(_lab(E), ["A1", "A1"], kind="derivative", point_group="3m", reference=GS)
    assert broken.problem is not None  # A1 x A1 allows A1 only, at every order


def test_the_defect_axis_comes_from_the_ground_state():
    # The same physical change gives the same verdict for an NV along [111] or along z.
    for axis in (NV_AXIS, np.array([0.0, 0.0, 1.0])):
        gs = _lab(np.diag([-957.0, -957.0, 1913.0]), axis)
        ok = verify_zfs_response(_lab(A1, axis), ["A1"], kind="displacement", point_group="3m", reference=gs)
        bad = verify_zfs_response(_lab(A1 + E, axis), ["A1"], kind="displacement", point_group="3m", reference=gs)
        assert ok.problem is None and ok.fractions["A1"] == pytest.approx(1.0)
        assert bad.problem is not None and "allow only A1" in bad.problem


def test_e_dominance_heuristic_applies_to_displacements_along_e_modes_only():
    only_a1 = _lab(A1 * 3)
    run = verify_zfs_response(only_a1, ["Ex"], kind="displacement", point_group="3m", reference=GS)
    assert "should mainly change the E part" in run.problem
    curvature = verify_zfs_response(only_a1, ["Ex", "Ey"], kind="derivative", point_group="3m", reference=GS)
    assert curvature.problem is None  # E x E allows A1: no heuristic on derivatives


def test_small_changes_and_unknown_point_groups_are_not_judged():
    tiny = verify_zfs_response(_lab(E * 0.01), ["A1"], kind="displacement", point_group="3m", reference=GS)
    assert tiny.problem is None and "too small to judge" in tiny.line
    unknown = verify_zfs_response(_lab(E), ["A1"], kind="displacement", point_group=None, reference=GS)
    assert unknown.problem is None and "not checked" in unknown.line


def _spectrum(labels):
    n = len(labels)
    return PhononSpectrum(
        frequencies_mev=np.arange(1.0, n + 1), eigenvectors=np.zeros((n, 2, 3)), atom_frac_coords=np.zeros((2, 3)),
        atom_symbols=["C", "N"], atomic_masses=np.array([12.011, 14.007]), lattice=np.eye(3) * 3.57,
        symmetries=labels,
    )


def test_point_group_falls_back_to_the_mode_labels():
    # All atoms at the origin: no detectable symmetry, so the labels decide.
    assert response_point_group(_spectrum(["A1", "Ex", "Ey", "A2"])) == "3m"
    assert response_point_group(_spectrum(["B1", "B2"])) is None
