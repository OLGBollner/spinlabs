import logging

import pytest


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path, monkeypatch):
    """Point the spinlabs registry at a per-test directory."""
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / ".spinlabs"))
    import spinlabs.registry as reg

    monkeypatch.setattr(reg, "REGISTRY_DIR", tmp_path / ".spinlabs")
    monkeypatch.setattr(reg, "REGISTRY_PATH", tmp_path / ".spinlabs" / "registry.db")
    yield


@pytest.fixture(autouse=True)
def _reset_spinlabs_logger():
    """Give each test a clean ``spinlabs`` logger that propagates to caplog.

    ``configure_logging`` disables propagation and attaches a stdout handler;
    without this reset that state would leak between tests.
    """
    logger = logging.getLogger("spinlabs")
    saved = logger.level, list(logger.handlers), logger.propagate
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    yield
    level, handlers, propagate = saved
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.fixture
def dummy_coupling_file(tmp_path):
    import numpy as np
    from spinlabs.models import SpinPhononCouplingData, ZFSTensor

    n_modes = 5
    coupling = SpinPhononCouplingData(
        order=2,
        defect="NV",
        cell_size=64,
        pert_scale=0.025,
        calc_method="all_bands",
        frequencies=np.linspace(10.0, 50.0, n_modes) * 1.60218e-22,  # in J
        frequency_unit="J",
        ground_state_zfs=ZFSTensor(matrix=np.diag([-1e-25, -1e-25, 2e-25]), unit="J"),
        V_0_0=np.array([1e-25, 2e-25, 1.5e-25, 3e-25, 2.2e-25]),
        V_p_m=np.array([2e-25, 1e-25, 2.5e-25, 1.8e-25, 3.1e-25]),
        V_0_pm=np.array([1.2e-25, 2.1e-25, 1.9e-25, 2.7e-25, 1.4e-25]),
        V2_0_0=np.ones((n_modes, n_modes)) * 1e-25,
        V2_p_m=np.ones((n_modes, n_modes)) * 2e-25,
        V2_0_pm=np.ones((n_modes, n_modes)) * 1.5e-25,
        coupling_unit="J",
        symmetries=["A1", "E", "E", "A1", "E"],
        iprs=np.full(n_modes, 0.2),
    )
    save_file = tmp_path / "test_coupling.npz"
    coupling.save(save_file)
    return save_file
