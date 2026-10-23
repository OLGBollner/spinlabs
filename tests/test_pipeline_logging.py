"""E2E tests for DB log capture during pipeline runs (SPI-15)."""

import sqlite3

from spinlabs.logging_utils import attach_db_logging, detach_db_logging
from spinlabs.pipeline import run_full_pipeline


def _run_logs(reg):
    conn = sqlite3.connect(reg.REGISTRY_PATH)
    return conn.execute("SELECT level, logger, message, run_id FROM run_logs").fetchall()


def test_pipeline_logs_captured_with_run_id(tmp_path, dummy_coupling_file):
    """A full pipeline run produces DB rows tagged with the registered run id."""
    import spinlabs.registry as reg

    handler = attach_db_logging(reg.ensure_db())
    try:
        run_full_pipeline(
            coupling_file=dummy_coupling_file,
            output_root=tmp_path / "runs",
            save_plots=False,
        )
    finally:
        detach_db_logging()
        handler.close()

    rows = _run_logs(reg)
    assert rows, "pipeline produced no DB log rows"
    run_ids = {r[3] for r in rows}
    assert len(run_ids) == 1 and run_ids != {None}
    # pipeline.banner uses INFO; none of the pipeline messages are ERROR
    levels = {r[0] for r in rows}
    assert levels <= {"DEBUG", "INFO", "WARNING", "ERROR"}


def test_pipeline_logs_captured_without_cli(tmp_path, dummy_coupling_file):
    """DB capture works in the library path, with no configure_logging call."""
    import spinlabs.registry as reg

    handler = attach_db_logging(reg.ensure_db())
    try:
        run_full_pipeline(
            coupling_file=dummy_coupling_file,
            output_root=tmp_path / "runs",
            save_plots=False,
        )
    finally:
        detach_db_logging()
        handler.close()

    messages = {r[2] for r in _run_logs(reg)}
    assert any("Starting spinlabs pipeline run" in m for m in messages)


def test_pipeline_reattaches_own_handler_after_detach(tmp_path, dummy_coupling_file):
    """Each pipeline run owns its handler: after a manual detach, the next run
    re-attaches and captures rows tagged with its own (new) run id."""
    import spinlabs.registry as reg

    handler = attach_db_logging(reg.ensure_db())
    run_full_pipeline(
        coupling_file=dummy_coupling_file,
        output_root=tmp_path / "runs_a",
        save_plots=False,
    )
    detach_db_logging()
    handler.close()

    run_full_pipeline(
        coupling_file=dummy_coupling_file,
        output_root=tmp_path / "runs_b",
        save_plots=False,
    )
    run_ids = {r[3] for r in _run_logs(reg)}
    assert run_ids == {1, 2}
