"""V_{Delta n} channel analysis on synthetic convergence packs."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from spinlabs.convergence import traceless_parts
from spinlabs.convergence_channels import (
    CHANNELS,
    allowed_channels,
    analyse_channels,
    channel_value,
    channel_values,
)
from spinlabs.pack_convergence import ConvergenceScan, save_convergence

D0 = np.diag([-950.0, -1000.0, 1950.0])
STEPS = [-0.2, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.2]
E_PART = np.array([[1.0, 0.0, 0.5], [0.0, -1.0, 0.0], [0.5, 0.0, 0.0]])
A1_PART = np.diag([-1.0, -1.0, 2.0])


def _ground_state(path):
    path.write_text(
        " Spin-spin contribution to zero-field splitting tensor (MHz)\n ---\n"
        "      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz\n ---\n"
        "   -950.0  -1000.0  1950.0  0.0  0.0  0.0\n ---\n"
    )
    return path


def _pack(path, tensors, modes=(5, 7), displacement="normalized"):
    scan = ConvergenceScan(
        path.parent, list(modes), STEPS, np.asarray(tensors, dtype=float), np.zeros((len(modes), len(STEPS))),
        len(modes) * len(STEPS), displacement=displacement,
    )
    gs = _ground_state(path.parent / "gs_OUTCAR")
    path.parent.mkdir(parents=True, exist_ok=True)
    return save_convergence(
        scan, path, ground_state=gs, defect="NV", cell="512", method="AO", method_key="test_AO"
    )


def _polynomial(mode, *, extra_e=0.0):
    """Exact D0 + q B + q^2 C; mode 5 has E and A1, mode 7 A1 only."""
    b = 40.0 * (mode - 4) * E_PART + 10.0 * A1_PART
    c = 300.0 * A1_PART + 20.0 * E_PART
    if extra_e:
        b = b + extra_e * E_PART
    return np.array([D0 + q * b + q**2 * c for q in STEPS])


def test_channel_functionals_match_production_formulas():
    rng = np.random.default_rng(3)
    for _ in range(5):
        tensor = rng.normal(size=(3, 3))
        tensor = (tensor + tensor.T) / 2
        tensor -= np.trace(tensor) / 3 * np.eye(3)
        parts = traceless_parts(tensor, np.eye(3))
        got = channel_values(parts)
        expected = {
            "d0": abs(tensor[2, 2] - 0.5 * (tensor[0, 0] + tensor[1, 1])) / 3.0,
            "d2": 0.5 * np.sqrt((tensor[0, 0] - tensor[1, 1]) ** 2 + 4 * tensor[0, 1] ** 2),
            "d1": np.sqrt(tensor[0, 2] ** 2 + tensor[1, 2] ** 2) / np.sqrt(2.0),
        }
        for key in CHANNELS:
            assert got[key] == pytest.approx(expected[key], abs=1e-12)


@pytest.mark.parametrize(
    ("symmetry", "first", "second"),
    [
        ("A1", {"d0"}, {"d0"}),
        ("A2", set(), {"d0"}),
        ("Ex", {"d1", "d2"}, {"d0", "d1", "d2"}),
        ("Ey", {"d1", "d2"}, {"d0", "d1", "d2"}),
        (None, {"d0", "d1", "d2"}, {"d0", "d1", "d2"}),
    ],
)
def test_allowed_channels_follow_production_symmetry(symmetry, first, second):
    assert allowed_channels(symmetry, 1) == frozenset(first)
    assert allowed_channels(symmetry, 2) == frozenset(second)


def test_all_three_channels_are_returned_for_every_mode(tmp_path):
    mode5 = _polynomial(5)
    mode7 = _polynomial(7)
    pack = _pack(tmp_path / "NV_512.npz", [mode5, mode7])
    analysis = analyse_channels(pack, mode_info={5: ("Ex", 10.0), 7: ("A1", 20.0)})
    for mode in analysis.modes:
        for order in (1, 2):
            for scheme in ("forward", "central"):
                for channel in CHANNELS:
                    assert mode.get(order, channel, scheme) is not None
    assert analysis.mode(5).allowed[1] == frozenset({"d1", "d2"})
    assert analysis.mode(5).allowed[2] == frozenset({"d0", "d1", "d2"})
    assert analysis.mode(7).allowed[1] == frozenset({"d0"})
    assert analysis.mode(7).allowed[2] == frozenset({"d0"})


def test_q_to_zero_channel_values_come_from_the_polynomial(tmp_path):
    mode5 = _polynomial(5)
    pack = _pack(tmp_path / "NV_512.npz", [mode5], modes=(5,))
    analysis = analyse_channels(pack, mode_info={5: ("Ex", 10.0)}, schemes=("forward", "central"))
    b = traceless_parts(40.0 * E_PART + 10.0 * A1_PART, np.eye(3))
    c = traceless_parts(300.0 * A1_PART + 20.0 * E_PART, np.eye(3))
    mode = analysis.mode(5)
    for scheme in ("forward", "central"):
        for channel in CHANNELS:
            first = mode.get(1, channel, scheme)
            second = mode.get(2, channel, scheme)
            assert first.q0 == pytest.approx(channel_value(b, channel), rel=1e-9, abs=1e-9)
            assert second.q0 == pytest.approx(channel_value(c, channel), rel=1e-9, abs=1e-9)


def test_forbidden_channels_do_not_narrow_the_common_window(tmp_path):
    clean = _pack(tmp_path / "clean.npz", [_polynomial(5), _polynomial(7)])
    polluted = _pack(
        tmp_path / "polluted.npz", [_polynomial(5), _polynomial(7, extra_e=500.0)]
    )
    info = {5: ("Ex", 10.0), 7: ("A1", 20.0)}
    clean_analysis = analyse_channels(clean, mode_info=info, noise=0.01, schemes=("forward",))
    polluted_analysis = analyse_channels(polluted, mode_info=info, noise=0.01, schemes=("forward",))
    assert clean_analysis.common[(1, "forward")] == pytest.approx(
        polluted_analysis.common[(1, "forward")], rel=1e-9, abs=1e-9
    )
    # The A1 mode's forbidden E channels are present for the visual zero check.
    forbidden = polluted_analysis.mode(7).get(1, "d2", "forward")
    assert forbidden is not None
    assert forbidden.window.allowed is False


def test_unknown_symmetry_warns_and_shows_all_channels(tmp_path):
    pack = _pack(tmp_path / "NV_512.npz", [_polynomial(5)], modes=(5,))
    analysis = analyse_channels(pack)
    assert any("symmetry unknown" in warning for warning in analysis.warnings)
    assert analysis.mode(5).allowed[1] == frozenset(CHANNELS)
