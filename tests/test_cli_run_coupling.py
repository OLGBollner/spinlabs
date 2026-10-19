"""``spinlabs run coupling`` runs only the ZFSManager stage and saves nothing."""

import numpy as np
import pytest

import spinlabs.cli as cli
import spinlabs.registry as registry
from spinlabs.models import PhononSpectrum, RawZFSData, ZFSTensor

N_MODES = 4


def _inputs(tmp_path, *, second_order=True):
    """A raw ZFS file with a linear response per mode, plus a matching phonon file."""
    gs = np.diag([-1000.0, -1000.0, 2000.0])
    step = np.diag([-1.0, -1.0, 2.0])
    syms = ["A1", "Ex", "Ey", "A1"]
    first = {
        i: {"tensor": gs + (i + 1) * step, "unit": "MHz", "pert": 1e-25, "symmetry": syms[i], "ipr": 0.5}
        for i in range(N_MODES)
    }
    second = {
        (i, i): {"tensor": gs + 2.1 * (i + 1) * step, "unit": "MHz", "pert": (1e-25, 1e-25),
                 "symmetry": (syms[i], syms[i]), "ipr": (0.5, 0.5)}
        for i in range(N_MODES)
    } if second_order else {}
    raw = RawZFSData(
        defect="NV", cell_size=64, pert_scale=0.1, calc_method="all_bands",
        ground_state_zfs=ZFSTensor(matrix=gs, unit="MHz"), first_order=first, second_order=second,
    )
    raw_file = tmp_path / "in" / "NV_64_raw_zfs_data_all_bands_12d.npz"
    raw_file.parent.mkdir()
    raw.save(raw_file)
    freqs = np.array([10.0, 30.0, 30.0, 40.0])  # degenerate Ex/Ey pair
    PhononSpectrum(
        frequencies_mev=freqs, symmetries=syms, iprs=np.full(N_MODES, 0.5),
        eigenvectors=np.ones((N_MODES, 1, 3)), atom_frac_coords=np.zeros((1, 3)), atom_symbols=["C"],
        atomic_masses=np.array([12.0]), lattice=np.eye(3) * 5.0,
    ).save(raw_file.parent / "phonon_data.npz")
    return raw_file


def _files(root):
    return sorted(p for p in root.rglob("*") if p.is_file() and ".spinlabs" not in p.parts)


def test_coupling_stage_prints_both_orders_and_saves_nothing(tmp_path, capsys, monkeypatch):
    raw_file = _inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    before = _files(tmp_path)

    cli.main(["run", "coupling", "--raw-zfs-file", str(raw_file), "--top", "2", "--mode", "1", "-q"])

    out = capsys.readouterr().out
    assert "NV_64  all_bands  pert 0.1" in out
    assert "4 first order, 4 second order (4 diagonal, 0 off-diagonal)" in out
    assert "First order (MHz/q):" in out and "Second order (MHz/q^2):" in out
    assert "      1  A1    10.00" in out  # --mode 1 is listed (1-based)
    assert _files(tmp_path) == before
    assert registry.ensure_db().execute("SELECT COUNT(*) FROM runs").fetchone() == (0,)


def test_coupling_stage_order_1_skips_second_order(tmp_path, capsys):
    raw_file = _inputs(tmp_path)

    cli.main(["run", "coupling", "--raw-zfs-file", str(raw_file), "--order", "1", "-q"])

    out = capsys.readouterr().out
    assert "First order" in out
    assert "Second order" not in out


def test_coupling_stage_rejects_order_2_without_second_order_data(tmp_path):
    raw_file = _inputs(tmp_path, second_order=False)

    with pytest.raises(SystemExit, match="no second-order tensors"):
        cli.main(["run", "coupling", "--raw-zfs-file", str(raw_file), "--order", "2", "-q"])


def test_coupling_stage_needs_exactly_one_input(tmp_path):
    with pytest.raises(SystemExit, match="exactly one of"):
        cli.main(["run", "coupling", "-q"])
