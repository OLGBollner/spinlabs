"""Central differences from first-order runs at +Q and -Q (RawZFSData.first_order_minus).

First order (D(Q) - D(-Q))/2Q, diagonal second order (D(Q) + D(-Q) - 2 D0)/Q^2; they cannot be
mixed with forward second-order pair runs.
"""

import argparse
import re
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from spinlabs.constants import CONSTANTS
from spinlabs.models import PerturbationEntry, PhononSpectrum, RawZFSData, ZFSTensor
from spinlabs.zfs_manager import ZFSManager

D0 = np.diag([-950.0, -1000.0, 1950.0])  # MHz, distinct principal values: no frame ambiguity
B = np.array([[0.0, 0.0, 30.0], [0.0, 0.0, 0.0], [30.0, 0.0, 50.0]])  # MHz per unit q
C = np.diag([-20.0, -20.0, 40.0])  # MHz per unit q^2
E = np.diag([5.0, -5.0, 0.0])  # MHz per unit q^3


def _spectrum(n=3):
    rng = np.random.default_rng(3)
    return PhononSpectrum(
        frequencies_mev=np.array([10.0, 20.0, 30.0][:n]),
        eigenvectors=rng.normal(size=(n, 2, 3)),
        atom_frac_coords=np.zeros((2, 3)),
        atom_symbols=["C", "C"],
        atomic_masses=np.array([12.011, 12.011]),
        lattice=np.eye(3) * 3.56,
        symmetries=["A1", "A1", "A1"][:n],
        iprs=np.ones(n) * 0.5,
    )


def _entry(tensor, pert):
    return {"tensor": tensor, "unit": "MHz", "pert": pert}


def _response(q):
    """D(q) in MHz for a step q in units of the run's step (so q = +-1)."""
    return D0 + B * q + C * q**2 + E * q**3


def _central_raw(modes=(0, 1), minus_modes=(0, 1), **kw):
    return RawZFSData(
        defect="NV",
        cell_size=64,
        pert_scale=0.05,
        ground_state_zfs=ZFSTensor(matrix=D0, unit="MHz"),
        first_order={m: _entry(_response(1.0), 0.05) for m in modes},
        first_order_minus={m: _entry(_response(-1.0), -0.05) for m in minus_modes},
        **kw,
    )


# ---- the data model ----

def test_central_and_forward_differences_cannot_be_mixed():
    pair = {(0, 0): _entry(_response(2.0), (0.05, 0.05))}
    with pytest.raises(ValueError, match="Cannot mix central and forward differences"):
        _central_raw(second_order=pair)


def test_central_data_counts_as_second_order_and_gets_its_own_file_name():
    raw = _central_raw()
    assert raw.is_central and raw.has_second_order and raw.combined_order == 2
    assert raw._default_name().endswith("_central.npz")  # never overwrites a forward _12d package
    forward = RawZFSData(defect="NV", cell_size=64, pert_scale=0.05, first_order={0: _entry(D0, 0.05)})
    assert not forward.is_central and not forward.has_second_order


def test_minus_runs_without_plus_runs_are_refused_on_save(tmp_path):
    raw = RawZFSData(defect="NV", cell_size=64, pert_scale=0.05, first_order_minus={0: _entry(D0, -0.05)})
    with pytest.raises(ValueError, match="without first_order"):
        raw.save(tmp_path / "x.npz")


def test_with_minus_attaches_the_minus_tree_and_checks_the_step():
    plus = RawZFSData(defect="NV", cell_size=64, pert_scale=0.05, first_order={0: _entry(_response(1.0), 0.05)})
    minus = RawZFSData(defect="NV", cell_size=64, pert_scale=-0.05, first_order={0: _entry(_response(-1.0), -0.05)})
    raw = plus.with_minus(minus)
    assert raw.is_central and raw.pert_scale == 0.05
    assert np.allclose(raw.first_order_minus[0]["tensor"], _response(-1.0))
    assert np.allclose(raw.first_order[0]["tensor"], _response(1.0))
    wrong = RawZFSData(defect="NV", cell_size=64, pert_scale=-0.1, first_order={0: _entry(D0, -0.1)})
    with pytest.raises(ValueError, match="steps \\+p and -p"):
        plus.with_minus(wrong)


def test_save_and_load_keep_the_minus_runs_and_their_negative_si_step(tmp_path):
    raw = _central_raw()
    raw.first_order_minus[0] = PerturbationEntry(
        order=1, mode_indices=(0,), amplitude=-0.05, zfs_tensor=ZFSTensor(matrix=_response(-1.0), unit="MHz")
    )
    loaded = RawZFSData.load(raw.save(tmp_path / raw._default_name()))
    assert loaded.is_central and sorted(loaded.first_order_minus) == [0, 1]
    # -0.05 Angstrom*sqrt(amu) is converted to SI like a positive step
    assert loaded.first_order_minus[0]["pert"] == pytest.approx(-0.05 * CONSTANTS["ang_amu2SI"])
    assert np.allclose(loaded.to_unit("MHz").first_order_minus[1]["tensor"], _response(-1.0))


def test_the_spin_correction_scales_the_minus_runs_too(tmp_path):
    raw = _central_raw(calc_method="defect_band_approx")  # saved uncorrected (flag False)
    loaded = RawZFSData.load(raw.save(tmp_path / "approx.npz")).to_unit("MHz")
    assert loaded.spin_correction_applied
    assert np.allclose(loaded.first_order[0]["tensor"], 1.5 * _response(1.0))
    assert np.allclose(loaded.first_order_minus[0]["tensor"], 1.5 * _response(-1.0))


# ---- the analysis ----

def _manager(raw):
    manager = ZFSManager(spectrum=_spectrum(), raw_data=raw)
    R = manager.eigen_rotation
    return manager, (lambda m: R.T @ m @ R * CONSTANTS["MHz2J"])  # MHz in the raw frame -> J in the manager's


def test_central_differences_cancel_the_terms_a_forward_difference_keeps():
    manager, frame = _manager(_central_raw())
    derivs, *_ = manager.calculate_first_order_derivatives()
    second, *_ = manager.calculate_second_order_derivatives(derivs)
    for i in (0, 1):
        q = manager.first_order[i]["pert"]
        assert manager.first_order_minus[i]["pert"] == pytest.approx(-q)
        # (D(Q) - D(-Q))/2Q = B + E, with no C (forward would give B + C + E)
        assert np.allclose(derivs[i] * q, frame(B + E))
        # (D(Q) + D(-Q) - 2 D0)/Q^2 = 2C, with no odd term (forward would add 6E)
        assert np.allclose(second[i, i] * q**2, frame(2 * C))
    assert manager.inconsistent_pairs == []  # no pair run, nothing to check


def test_a_mode_without_its_minus_run_gets_no_diagonal_and_a_forward_first_order(caplog):
    with caplog.at_level("WARNING", logger="spinlabs"):
        manager, frame = _manager(_central_raw(modes=(0, 1), minus_modes=(0,)))
    assert "1 modes have no -Q run" in caplog.text
    assert sorted(manager.zfs_tensors_2d) == [(0, 0)]
    assert manager.treated_modes == {0, 1}  # the -Q runs do not narrow first order
    derivs, *_ = manager.calculate_first_order_derivatives()
    second, *_ = manager.calculate_second_order_derivatives(derivs)
    q = manager.first_order[1]["pert"]
    assert np.allclose(derivs[1] * q, frame(B + C + E))  # forward: (D(Q) - D0)/Q
    assert np.allclose(second[1, 1], 0.0)


# ---- trees and packaging ----

def test_central_tree_builds_first_order_at_plus_and_minus_p_only(tmp_path):
    from spinlabs.parsers import save_phonon_npz
    from spinlabs.vasp import build_perturbation_tree

    defect = tmp_path / "NV_512"
    (defect / "data").mkdir(parents=True)
    spec = PhononSpectrum(
        frequencies_mev=np.array([10.0, 20.0]), eigenvectors=np.zeros((2, 1, 3)), atom_frac_coords=np.zeros((1, 3)),
        atom_symbols=["C"], atomic_masses=np.array([12.0]), lattice=np.eye(3), symmetries=["A1", "A1"],
    )
    save_phonon_npz(spec, defect / "data" / "phonon_data.npz")
    for stage in ("ZFS_hyp", "ZFS_occup"):
        (defect / stage).mkdir()
        (defect / stage / "INCAR").write_text("ENCUT = 520\n")
        for name in ("POSCAR", "KPOINTS", "POTCAR"):
            (defect / stage / name).write_text(name + "\n")
        (defect / stage / "OUTCAR").write_text("Elapsed time (sec):   100.000\n")
    failures = build_perturbation_tree(
        out=defect, pert=[0.05], displacement="normalized", conv_modes=None,
        phonon=defect / "data" / "phonon_data.npz", array_jobs=1, vasp_binary=Path("/bin/true"), central=True,
    )
    assert failures == []
    assert sorted(p.name for p in (defect / "first_order").iterdir()) == ["pert_-0.05", "pert_0.05"]
    assert not (defect / "second_order").exists()
    script = (defect / "first_order" / "pert_-0.05" / "all_bands" / "run_perturbation_first_order.sh").read_text()
    assert re.search(r"^PERT=-0\.05$", script, re.M)
    with pytest.raises(ValueError, match="positive steps"):
        build_perturbation_tree(out=defect, pert=[-0.05], displacement="normalized", conv_modes=None,
                                phonon=defect / "data" / "phonon_data.npz", central=True)


def _package_args(tmp_path, central=True, orders="first,second"):
    return argparse.Namespace(
        defect_folder=[tmp_path / "NV_512"], orders=orders, pert=None, method=["all"],
        output_root=tmp_path / "data", scan=False, yes=True, central=central,
    )


def _parsed(sim_folder, **_kw):
    """parse_zfs_simulation_dataset stand-in: the step from the folder name, a response at that sign."""
    pert = float(Path(sim_folder).name.split("_")[1])
    tensor = _response(np.sign(pert))
    entry = PerturbationEntry(order=1, mode_indices=(0,), amplitude=pert, zfs_tensor=ZFSTensor(matrix=tensor))
    return RawZFSData(defect="NV", cell_size=512, pert_scale=pert, calc_method="all_bands", order=1,
                      ground_state_zfs=ZFSTensor(matrix=D0), first_order={0: entry})


def _folders(tmp_path, *names):
    for order, pert in names:
        run = tmp_path / "NV_512" / order / f"pert_{pert}" / "all_bands" / "OUTCAR"
        run.parent.mkdir(parents=True, exist_ok=True)
        run.touch()


def test_package_central_pairs_each_step_with_its_minus_tree(tmp_path, monkeypatch, caplog):
    from spinlabs.cli.package import _run

    monkeypatch.setattr("spinlabs.registry.REGISTRY_PATH", tmp_path / "registry.db")
    _folders(tmp_path, ("first_order", "0.05"), ("first_order", "-0.05"), ("second_order", "0.05"))
    with patch("spinlabs.cli.package.parse_zfs_simulation_dataset", side_effect=_parsed) as parse:
        with caplog.at_level("INFO", logger="spinlabs"):
            _run(_package_args(tmp_path))
    parsed = sorted(Path(c.kwargs["sim_folder"]).parent.name + "/" + Path(c.kwargs["sim_folder"]).name
                    for c in parse.call_args_list)
    assert parsed == ["first_order/pert_-0.05", "first_order/pert_0.05"]  # no second order, -0.05 not on its own
    assert "second-order trees are not packaged" in caplog.text
    day = date.today().strftime("%Y%m%d")
    saved = tmp_path / "data" / "NV_512" / "pert_0.05" / "all_bands" / day / "NV_512_raw_zfs_data_all_bands_central.npz"
    assert [p for p in (tmp_path / "data").rglob("*.npz")] == [saved]
    raw = RawZFSData.load(saved).to_unit("MHz")
    assert raw.is_central and np.allclose(raw.first_order_minus[0]["tensor"], _response(-1.0))


def test_package_central_skips_a_step_without_its_minus_tree(tmp_path, monkeypatch, caplog):
    from spinlabs.cli.package import _run

    monkeypatch.setattr("spinlabs.registry.REGISTRY_PATH", tmp_path / "registry.db")
    _folders(tmp_path, ("first_order", "0.05"))
    with patch("spinlabs.cli.package.parse_zfs_simulation_dataset", side_effect=_parsed):
        _run(_package_args(tmp_path))
    assert "no first_order/pert_-0.05 folder" in caplog.text
    assert not list((tmp_path / "data").rglob("*.npz"))
