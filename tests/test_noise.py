"""spinlabs.noise: the noise of the diagonal second differences and couplings with it removed."""

from types import SimpleNamespace

import numpy as np
import pytest

from spinlabs.constants import CONSTANTS
from spinlabs.noise import SecondOrderNoise, denoised_couplings, mode_amplitudes, second_difference_noise

MHZ = CONSTANTS["MHz2J"]
D0 = np.diag([-1.0, -1.0, 2.0]) * 1000.0 * MHZ


def _central(forbidden_out, forbidden_in, allowed=0.0):
    """A central A1 entry whose second difference has the given E-type parts (MHz): (xz, yz) and ((xx-yy)/sqrt2, xy)."""
    xz, yz = forbidden_out
    w, xy = forbidden_in
    d2 = np.array([[w / np.sqrt(2), xy, xz], [xy, -w / np.sqrt(2), yz], [xz, yz, allowed]]) * MHZ
    return {"tensor": D0 + d2, "tensor_minus": D0.copy(), "central": True, "pert": (1e-24, 1e-24)}


def _manager(entries, labels, groups=None, settings=None):
    spectrum = SimpleNamespace(symmetries=labels, deg_groups=groups or [], n_modes=len(labels))
    return SimpleNamespace(
        spectrum=spectrum, zfs_tensors_2d=entries, zfs_relaxed=D0, eigen_rotation=None, first_order={},
        raw_data=SimpleNamespace(vasp_settings=settings), _mirror_parities=lambda: {},
        mirror_projected=[], mirror_projected_central=[], mirror_inherited=[],
        computed_pairs={(min(i, j), max(i, j)) for i, j in entries}, offdiag_from_diagonal=True,
    )


def test_noise_is_measured_from_the_forbidden_parts_per_orientation():
    entries = {(0, 0): _central((0.001, -0.003), (0.010, 0.020), allowed=5.0),
               (1, 1): _central((0.002, 0.004), (-0.030, 0.040), allowed=-3.0)}
    noise = second_difference_noise(_manager(entries, ["A1", "A1"]))
    # median |x| / 0.6745 over the four samples of each orientation; the allowed (A1) part does not enter
    assert np.isclose(noise.out_of_plane, np.median([0.001, 0.003, 0.002, 0.004]) / 0.6745)
    assert np.isclose(noise.in_plane, np.median([0.010, 0.020, 0.030, 0.040]) / 0.6745)
    assert (noise.n_out, noise.n_in, noise.from_e_modes) == (4, 4, False)


def test_a_mode_only_estimate_on_symmetrized_runs_warns(caplog):
    settings = {"trees": {"first_order": {"runs": {"ISYM": {"3": 9, "0": 1}}}}}
    noise = second_difference_noise(_manager({(0, 0): _central((0.001, 0.001), (0.01, 0.01))}, ["A1"], settings=settings))
    assert noise.symmetrized_runs == pytest.approx(0.9)
    assert "may be understated" in caplog.text


def test_channel_factors():
    n = SecondOrderNoise(out_of_plane=2.0, in_plane=4.0, n_out=1, n_in=1)
    assert n.per_component("V2_0_0") == (4.0 / 6, 0.0)  # |zz - (xx+yy)/2| / 6, in-plane stand-in
    assert n.per_component("V2_0_pm") == (2.0 / (2 * np.sqrt(2)), 2.0 / (2 * np.sqrt(2)))  # |p| / (2 sqrt 2)
    assert n.per_component("V2_p_m") == (np.sqrt(2) * 4.0 / 4, 2 * 4.0 / 4)  # |w| / 4, w = (xx - yy, 2 xy)


def _coupling(diag00, diag0pm, diagpm, labels, unit="J"):
    """Geometric-mean filled matrices as production writes them (A x E pairs from V00 of the A mode)."""
    def fill(d, a_from=None):
        v = np.array(d, float)
        if a_from is not None:  # A modes enter Delta1/Delta2 through their Delta0
            v = np.where([lab.startswith("A") for lab in labels], np.array(a_from, float), v)
        m = np.sqrt(np.outer(v, v))
        a = np.array([lab.startswith("A") for lab in labels])
        if a_from is not None:
            m[np.ix_(a, a)] = 0.0  # A x A has no Delta1/Delta2
        np.fill_diagonal(m, d)
        return m
    return {"V2_0_0": fill(diag00), "V2_0_pm": fill(diag0pm, diag00), "V2_p_m": fill(diagpm, diag00),
            "coupling_unit": np.array(unit)}


def test_denoised_couplings_remove_the_noise_in_quadrature():
    labels = ["A1", "Ex", "Ey"]
    q = 2e-12
    entries = {(i, i): {"pert": (q, q)} for i in range(3)}
    manager = _manager(entries, labels, groups=[[0], [1, 2]])
    noise = SecondOrderNoise(out_of_plane=0.003, in_plane=0.006, n_out=10, n_in=10)
    sv = {ch: np.hypot(*noise.per_component(ch)) * MHZ / q**2 for ch in ("V2_0_0", "V2_0_pm", "V2_p_m")}
    d00, d0pm, dpm = 4 * sv["V2_0_0"], 2 * sv["V2_0_pm"], 0.5 * sv["V2_p_m"]  # the last is below its noise
    coupling = _coupling([d00, 0, 0], [0, d0pm, d0pm], [0, dpm, dpm], labels)
    out = denoised_couplings(manager, coupling, noise)
    assert np.isclose(out["V2_0_0"][0, 0], np.sqrt(d00**2 - sv["V2_0_0"] ** 2))
    assert np.isclose(out["V2_0_pm"][1, 1], np.sqrt(d0pm**2 - sv["V2_0_pm"] ** 2))
    assert out["V2_p_m"][1, 1] == 0.0  # pure noise
    # the A x E pair follows the A mode's Delta0 scale and the E mode's Delta1 scale
    s00, s1 = out["V2_0_0"][0, 0] / d00, out["V2_0_pm"][1, 1] / d0pm
    assert np.isclose(out["V2_0_pm"][0, 1], coupling["V2_0_pm"][0, 1] * np.sqrt(s00 * s1))


def test_projected_modes_lose_only_the_allowed_noise():
    labels = ["A1", "Ex", "Ey"]
    q = 2e-12
    manager = _manager({(i, i): {"pert": (q, q)} for i in range(3)}, labels, groups=[[0], [1, 2]])
    manager.mirror_projected_central = [1]  # its twin 2 inherits the projected value
    noise = SecondOrderNoise(out_of_plane=0.003, in_plane=0.006, n_out=10, n_in=10)
    allowed = noise.per_component("V2_p_m")[0] * MHZ / q**2
    d = 3 * allowed
    out = denoised_couplings(manager, _coupling([1e-30, 0, 0], [0, d, d], [0, d, d], labels), noise)
    assert np.isclose(out["V2_p_m"][1, 1], np.sqrt(d**2 - allowed**2))
    assert np.isclose(out["V2_p_m"][2, 2], np.sqrt(d**2 - allowed**2))


def test_couplings_stored_in_mhz_are_handled():
    labels = ["Ex", "Ey"]
    q = 2e-12
    manager = _manager({(i, i): {"pert": (q, q)} for i in range(2)}, labels, groups=[[0, 1]])
    noise = SecondOrderNoise(out_of_plane=0.003, in_plane=0.006, n_out=10, n_in=10)
    sv_mhz = np.hypot(*noise.per_component("V2_0_pm")) / q**2  # MHz units: no MHz -> J
    d = 2 * sv_mhz
    out = denoised_couplings(manager, _coupling([0, 0], [d, d], [d, d], labels, unit="MHz"), noise)
    assert np.isclose(out["V2_0_pm"][0, 0], np.sqrt(d**2 - sv_mhz**2))


def test_mode_amplitudes_fill_twins():
    manager = _manager({(1, 1): {"pert": (3.0, 3.0)}}, ["A1", "Ex", "Ey"], groups=[[0], [1, 2]])
    assert mode_amplitudes(manager).tolist() == [0.0, 3.0, 3.0]


def test_from_e_modes_is_set_by_the_e_samples_themselves():
    # Two forward A1 diagonals, one without its first-order run (no sample), and one aligned central E mode:
    # 3 out-of-plane samples against 2 A diagonals, which a count-based flag reads as A-only.
    mirror = np.diag([-1.0, 1.0, 1.0])
    d2 = np.array([[0.01, 0.02, 0.003], [0.02, -0.01, 0.004], [0.003, 0.004, 1.0]]) * MHZ
    entries = {(0, 0): {"tensor": D0 + d2, "central": False, "pert": (1e-24, 1e-24)},
               (1, 1): {"tensor": D0 + d2, "central": False, "pert": (1e-24, 1e-24)},
               (2, 2): _central((0.002, 0.001), (0.01, 0.02))}
    manager = _manager(entries, ["A1", "A1", "Ex"])
    manager.first_order = {0: {"tensor": D0}}  # mode 1 has no first-order run
    manager._mirror_parities = lambda: {2: (mirror, -1.0)}
    noise = second_difference_noise(manager)
    assert noise.n_out == 3 and noise.from_e_modes is True


def test_computed_off_diagonal_pairs_keep_their_values():
    labels = ["A1", "Ex", "Ey"]
    q = 2e-12
    entries = {(i, i): {"pert": (q, q)} for i in range(3)}
    entries[(0, 1)] = {"pert": (q, q)}  # a computed pair
    manager = _manager(entries, labels, groups=[[0], [1, 2]])
    noise = SecondOrderNoise(out_of_plane=0.003, in_plane=0.006, n_out=10, n_in=10)
    sv = {ch: np.hypot(*noise.per_component(ch)) * MHZ / q**2 for ch in ("V2_0_0", "V2_0_pm", "V2_p_m")}
    coupling = _coupling([4 * sv["V2_0_0"], 0, 0], [0, 2 * sv["V2_0_pm"], 2 * sv["V2_0_pm"]],
                         [0, 2 * sv["V2_p_m"], 2 * sv["V2_p_m"]], labels)
    out = denoised_couplings(manager, coupling, noise)
    for channel in ("V2_0_0", "V2_0_pm", "V2_p_m"):
        assert out[channel][0, 1] == coupling[channel][0, 1] and out[channel][1, 0] == coupling[channel][1, 0]
    assert out["V2_0_pm"][0, 2] != coupling["V2_0_pm"][0, 2]  # the filled pair still follows its diagonals


def test_a_dropped_pair_is_refilled_from_the_denoised_diagonals():
    # (1, 2) has a run but the manager dropped it (e.g. inconsistent), so its coupling is the fill
    labels = ["A1", "Ex", "Ex"]
    q = 2e-12
    entries = {(i, i): {"pert": (q, q)} for i in range(3)}
    entries[(1, 2)] = {"pert": (q, q)}
    manager = _manager(entries, labels, groups=[[0], [1], [2]])
    manager.computed_pairs = {(0, 0), (1, 1), (2, 2)}
    noise = SecondOrderNoise(out_of_plane=0.003, in_plane=0.006, n_out=10, n_in=10)
    sv = {ch: np.hypot(*noise.per_component(ch)) * MHZ / q**2 for ch in ("V2_0_0", "V2_0_pm", "V2_p_m")}
    coupling = _coupling([4 * sv["V2_0_0"], 3 * sv["V2_0_0"], 2 * sv["V2_0_0"]],
                         [0, 2 * sv["V2_0_pm"], 3 * sv["V2_0_pm"]], [0, 2 * sv["V2_p_m"], 3 * sv["V2_p_m"]], labels)
    out = denoised_couplings(manager, coupling, noise)
    for channel in ("V2_0_0", "V2_0_pm", "V2_p_m"):
        d = np.diag(out[channel])
        assert np.isclose(out[channel][1, 2], np.sqrt(d[1] * d[2]))
        assert out[channel][1, 2] < coupling[channel][1, 2]


def test_without_the_fill_off_diagonal_entries_are_untouched():
    labels = ["Ex", "Ex"]
    q = 2e-12
    manager = _manager({(i, i): {"pert": (q, q)} for i in range(2)}, labels, groups=[[0], [1]])
    manager.offdiag_from_diagonal = False
    noise = SecondOrderNoise(out_of_plane=0.003, in_plane=0.006, n_out=10, n_in=10)
    d = 2 * np.hypot(*noise.per_component("V2_0_pm")) * MHZ / q**2
    coupling = _coupling([0, 0], [d, d], [d, d], labels)
    out = denoised_couplings(manager, coupling, noise)
    assert out["V2_0_pm"][0, 1] == coupling["V2_0_pm"][0, 1]
    assert out["V2_0_pm"][0, 0] < d


def test_denoising_needs_the_managers_computed_pairs():
    manager = _manager({(0, 0): {"pert": (1e-12, 1e-12)}}, ["A1"])
    manager.computed_pairs = None
    with pytest.raises(ValueError, match="computed pairs"):
        denoised_couplings(manager, _coupling([1.0], [0], [0], ["A1"]), SecondOrderNoise(1, 1, 1, 1))
