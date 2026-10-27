"""Tests for spinlabs.logging_utils and pipeline logging."""

import logging
import sqlite3

import numpy as np
import pytest

from spinlabs.logging_utils import configure_logging, get_logger
from spinlabs.pipeline import run_full_pipeline
from spinlabs.models import SpinPhononCouplingData, ZFSTensor

logger = get_logger()


@pytest.fixture
def dummy_coupling_file(tmp_path):
    n_modes = 5
    coupling = SpinPhononCouplingData(
        order=2,
        defect="NV",
        cell_size=64,
        pert_scale=0.025,
        calc_method="all_bands",
        frequencies=np.linspace(10.0, 50.0, n_modes) * 1.60218e-22,
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


def _capture_records():
    class Capture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record):
            self.records.append(record)

    return Capture


def test_configure_logging_default_console_level_is_info():
    get_logger()
    configure_logging()
    # The logger itself stays at DEBUG so DB logging sees the full stream;
    # the console handler carries the requested level.
    from spinlabs.logging_utils import _stream_handler as sh

    assert sh is not None and sh in logger.handlers
    assert logger.level == logging.DEBUG
    assert sh.level == logging.INFO


def test_configure_logging_verbosity_count_and_quiet():
    import spinlabs.logging_utils as lu

    get_logger()

    configure_logging(verbosity=0)
    assert lu._stream_handler.level == logging.INFO
    configure_logging(verbosity=1)
    assert lu._stream_handler.level == logging.DEBUG
    configure_logging(verbosity=5)  # extra -v caps at DEBUG
    assert lu._stream_handler.level == logging.DEBUG
    configure_logging(quiet=True)
    assert lu._stream_handler.level == logging.WARNING


def test_console_layout_sections_indent_and_prefixes(capsys):
    from spinlabs.logging_utils import section

    configure_logging()
    log = get_logger("spinlabs.test")
    log.info("before any chapter")
    section(log, "Plot convergence: NV_512 test_all")
    log.info("saved figure: a.png\nsecond line")
    log.warning("mode 7: fewer than 2 finished runs, skipping")
    log.error("plot failed")
    log.debug("hidden at the default level")
    assert capsys.readouterr().out == (
        "before any chapter\n"
        "\n== Plot convergence: NV_512 test_all ==\n"
        "  saved figure: a.png\n  second line\n"
        "  warning: mode 7: fewer than 2 finished runs, skipping\n"
        "  error: plot failed\n"
    )


def test_configure_logging_does_not_stack_handlers():
    get_logger()
    configure_logging()
    configure_logging(verbosity=2)
    streams = [h for h in logger.handlers if type(h) is logging.StreamHandler]
    assert len(streams) == 1
    assert logger.level == logging.DEBUG


def test_get_logger_returns_child_of_root():
    assert get_logger("spinlabs.pipeline").name == "spinlabs.pipeline"
    assert get_logger().name == "spinlabs"


def test_pipeline_uses_logging_not_print(tmp_path, dummy_coupling_file, capsys):
    """Pipeline messages go through the spinlabs logger, not bare print()."""
    from spinlabs.pipeline import run_full_pipeline

    Capture = _capture_records()()
    logger = logging.getLogger("spinlabs")
    logger.addHandler(Capture)
    try:
        run_full_pipeline(
            coupling_file=str(dummy_coupling_file),
            output_root=str(tmp_path / "runs"),
            save_plots=False,
        )
    finally:
        logger.removeHandler(Capture)

    messages = [r.getMessage() for r in Capture.records if r.name.startswith("spinlabs")]
    assert messages, "expected pipeline log records"
    assert any(m.startswith("run folder: ") for m in messages)
    assert any(m.startswith("run finished: ") for m in messages)
    # Nothing may be printed to stdout anymore.
    assert capsys.readouterr().out == ""


def test_pipeline_logs_persist_to_registry(tmp_path, dummy_coupling_file, _isolated_registry):
    """run_full_pipeline attaches the DB handler and its INFO logs land in run_logs."""

    import spinlabs.registry as reg
    from spinlabs.logging_utils import detach_db_logging

    try:
        run_full_pipeline(
            coupling_file=str(dummy_coupling_file),
            output_root=str(tmp_path / "runs"),
            save_plots=False,
        )
    finally:
        detach_db_logging()

    conn = sqlite3.connect(tmp_path / ".spinlabs" / "registry.db")
    rows = reg.list_run_logs(conn)
    assert any(r["message"].startswith("run folder: ") for r in rows)
    assert any(r["level"] == "INFO" for r in rows)


# ---------------------------------------------------------------------------
# New SPI-14 regressions: handler ownership, backfilling, level separation
# ---------------------------------------------------------------------------


def test_attach_db_logging_reuses_single_handler(_isolated_registry):
    """Repeated attaches re-point the same handler instead of stacking copies."""
    import spinlabs.registry as reg
    from spinlabs.logging_utils import attach_db_logging, db_logging_attached, get_logger

    conn = reg.ensure_db()
    h1 = attach_db_logging(conn)
    get_logger()
    n_handlers = len(logger.handlers)
    h2 = attach_db_logging(reg.ensure_db())
    assert h1 is h2
    assert len(logger.handlers) == n_handlers  # no duplicate handler
    assert db_logging_attached()
    from spinlabs.logging_utils import detach_db_logging

    detach_db_logging()


def test_db_logging_attached_reflects_state(_isolated_registry):
    import spinlabs.registry as reg
    from spinlabs.logging_utils import attach_db_logging, db_logging_attached, detach_db_logging

    assert not db_logging_attached()
    attach_db_logging(reg.ensure_db())
    assert db_logging_attached()
    detach_db_logging()
    assert not db_logging_attached()


def test_set_db_run_id_backfills_pending_records(tmp_path):
    """Records logged before the run is registered get backfilled to run_id."""
    import spinlabs.registry as reg
    from spinlabs.logging_utils import attach_db_logging, detach_db_logging, get_logger, set_db_run_id

    conn = reg.ensure_db()
    attach_db_logging(conn)
    get_logger()
    logger.info("before registration")

    run_id = reg.register_run(conn, run_dir=tmp_path / "demo", method="all_bands")
    set_db_run_id(run_id)
    logger.info("after registration")
    detach_db_logging()

    rows = reg.list_run_logs(conn)
    by_msg = {r["message"]: r["run_id"] for r in rows}
    assert by_msg["before registration"] == run_id  # backfilled
    assert by_msg["after registration"] == run_id


def test_set_db_run_id_does_not_touch_older_unattached_records(tmp_path):
    """Unattached leftovers from a previous run in the same process stay NULL."""
    import spinlabs.registry as reg
    from spinlabs.logging_utils import attach_db_logging, detach_db_logging, set_db_run_id

    conn = reg.ensure_db()
    reg.insert_run_log(conn, "INFO", "other", "old run leftover", run_id=None)
    baseline_before = reg.list_run_logs(conn)[-1]["id"]
    # a run row must exist so run_id=42 is a plausible id, but the point is
    # that the pre-existing NULL log row stays unattributed
    reg.register_run(conn, run_dir=tmp_path, method="all_bands")

    h = attach_db_logging(conn)
    assert h._baseline >= baseline_before
    logger = logging.getLogger("spinlabs")
    logger.info("new run record")
    set_db_run_id(42)
    detach_db_logging()

    rows = reg.list_run_logs(conn)
    assert rows[0]["run_id"] is None  # old record untouched
    assert any(r["message"] == "new run record" and r["run_id"] == 42 for r in rows)


def test_set_db_run_id_safe_without_handler():
    from spinlabs.logging_utils import set_db_run_id

    set_db_run_id(None)  # must not raise when no DB handler is attached
    set_db_run_id(1)


def test_console_level_independent_of_db_capture(tmp_path):
    """DEBUG records are captured to the DB even when the console shows only INFO."""
    import spinlabs.logging_utils as lu
    import spinlabs.registry as reg
    from spinlabs.logging_utils import (
        attach_db_logging,
        configure_logging,
        detach_db_logging,
        get_logger,
    )

    configure_logging(verbosity=0)  # console: INFO
    assert lu._stream_handler is not None
    assert lu._stream_handler.level == logging.INFO
    attach_db_logging(reg.ensure_db())
    get_logger()
    logger.debug("dbg-only message")
    detach_db_logging()

    rows = reg.list_run_logs(reg.ensure_db())
    assert any(r["message"] == "dbg-only message" and r["level"] == "DEBUG" for r in rows)


@pytest.mark.parametrize(
    "argv",
    [
        ["package", "--defect-folder", "x", "-v"],
        ["plot", "t1", "-q"],
        ["plot", "convergence", "x.npz", "-vv"],
        ["run", "--coupling-file", "x.npz", "-v"],
    ],
)
def test_every_subcommand_accepts_verbosity_flags(argv):
    from spinlabs.cli import build_parser

    args = build_parser().parse_args(argv)
    assert hasattr(args, "verbose") and hasattr(args, "quiet")
