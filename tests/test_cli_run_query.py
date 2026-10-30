"""Regression tests for SPI-17: `spinlabs run --query` must forward CLI tuning flags.

Covers:
- t_start / t_end / t_step actually reach the pipeline (temperature grid reflects them).
- init_state reaches the pipeline (recorded in run_info.json).
- run_name is used; with several matched datasets each gets its own
  <run_name>_<stem> directory instead of colliding suffixes.
"""
import argparse
import json

import pytest

import numpy as np

from spinlabs.cli import run as run_mod
from spinlabs.models import PhononSpectrum, RawZFSData, ZFSTensor
from spinlabs.registry import ensure_db, register_file


def _make_raw_zfs(tmp_path, name):
    n_modes = 4
    pert_si = 0.025 * 1.60218e-22
    raw = RawZFSData(
        defect="NV",
        cell_size=512,
        pert_scale=0.025,
        calc_method="all_bands",
        order=1,
        ground_state_zfs=ZFSTensor(matrix=np.diag([-1000.0, -1000.0, 2000.0]), unit="MHz"),
        first_order={
            i: {
                "tensor": np.diag([-1000.0, -1000.0, 2000.0]) * 6.626e-28,
                "pert": pert_si,
                "symmetry": "A1" if i % 2 == 0 else "Ex",
                "ipr": 0.5,
            }
            for i in range(n_modes)
        },
        second_order={},
    )
    f = tmp_path / name
    raw.save(f)
    return f


def _make_phonon_alongside(npz_path):
    n_modes = 4
    freqs = np.linspace(10.0, 40.0, n_modes)
    freqs[1] = freqs[2] = 30.0
    spec = PhononSpectrum(
        frequencies_mev=freqs,
        symmetries=["A1", "Ex", "Ey", "A1"],
        iprs=np.ones(n_modes) * 0.5,
        eigenvectors=np.ones((n_modes, 1, 3)),
        atom_frac_coords=np.zeros((1, 3)),
        atom_symbols=["C"],
        atomic_masses=np.array([12.0]),
        lattice=np.eye(3) * 5.0,
    )
    spec.save(npz_path.parent / "phonon_data.npz")


def _query_args(tmp_path, out_root, run_name="spi17"):
    return argparse.Namespace(
        defect=["NV"],
        cell_size=[512],
        method=["all_bands"],
        pert_scale=None,
        order=None,
        t_start=0.0,
        t_end=500.0,
        t_step=100.0,
        temperatures=None,
        init_state="ms_1",
        run_name=run_name,
        plot=False,
        two_phonon=None,
        phonon_file=None,
        output_root=str(out_root),
        exclude_inconsistent_pairs=False,
        degenerate_pairs_from_symmetry=False,
        offdiag_from_diagonal=False,
        sigma=7.5,
    )


def test_run_query_forwards_temperature_and_init_state(tmp_path):
    """--t-start/--t-end/--t-step/--init-state must reach the actual run."""
    db_dir = tmp_path / "db"
    import spinlabs.registry as reg

    reg.REGISTRY_DIR = db_dir
    reg.REGISTRY_PATH = db_dir / "registry.db"
    conn = ensure_db()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data = _make_raw_zfs(data_dir, "NV_512_raw_zfs_data_all_bands_1d.npz")
    _make_phonon_alongside(data)
    register_file(conn, data, "packaged_zfs", defect="NV", cell_size=512,
                  calc_method="all_bands", pert_order=[1], pert_scale=0.025)

    out = tmp_path / "out"
    run_mod._run_query(_query_args(tmp_path, out))

    run_dirs = list(out.glob("spi17*"))
    assert len(run_dirs) == 1, f"expected one run dir, got {run_dirs}"
    info = json.loads((run_dirs[0] / "run_info.json").read_text())
    assert info["init_state"] == ["ms_1"]
    assert info["temperatures"]["min"] == 0.0
    # pipeline appends a +1 K guard point, so max is t_end + 1
    assert info["temperatures"]["max"] == pytest.approx(501.0)


def test_run_query_run_name_per_dataset(tmp_path):
    """With >1 match, each dataset gets <run_name>_<stem> (unique), not duplicate _N."""
    import spinlabs.registry as reg

    db_dir = tmp_path / "db"
    reg.REGISTRY_DIR = db_dir
    reg.REGISTRY_PATH = db_dir / "registry.db"
    conn = ensure_db()
    names = ["NV_512_raw_zfs_data_all_bands_1d.npz", "NV_512_raw_zfs_data_rpa_1d.npz"]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Two configurations (different pert-scales), so --query runs both.
    for name, pert in zip(names, (0.025, 0.1)):
        data = _make_raw_zfs(data_dir, name)
        register_file(conn, data, "packaged_zfs", defect="NV", cell_size=512,
                      calc_method="all_bands", pert_order=[1], pert_scale=pert)
    _make_phonon_alongside(data_dir / names[0])

    out = tmp_path / "out"
    run_mod._run_query(_query_args(tmp_path, out))

    dirs = sorted(d.name for d in out.iterdir() if d.is_dir())
    assert len(dirs) == 2, f"expected 2 distinct run dirs, got {dirs}"
    assert all(d.startswith("spi17_") for d in dirs), dirs
    assert len(set(dirs)) == len(dirs), "run dirs must not collide"
