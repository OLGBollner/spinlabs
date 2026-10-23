"""Fixtures for the script suite (pytest scripts/tests), as in the package suite."""

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
