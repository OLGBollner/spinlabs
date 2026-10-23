"""Central logging configuration for spinlabs.

Modules log through ``logging.getLogger(__name__)`` (children of the
``spinlabs`` logger). The CLI configures the root ``spinlabs`` logger once
via :func:`configure_logging`; library users may attach their own handlers.

Two handlers live on the ``spinlabs`` logger:

* a single console stream handler, reconfigured (never duplicated) by
  :func:`configure_logging`;
* at most one :class:`SQLiteLogHandler`, owned by whoever called
  :func:`attach_db_logging`. The pipeline attaches it at entry when no
  handler is present and detaches it on exit only if it attached it itself,
  so a library caller's own DB logging is never torn down.
"""

from __future__ import annotations

import logging
import sqlite3
import sys

from spinlabs.registry import insert_run_log

ROOT_LOGGER_NAME = "spinlabs"

_db_handler: SQLiteLogHandler | None = None
_stream_handler: logging.StreamHandler | None = None


class SQLiteLogHandler(logging.Handler):
    """Persist log records into the ``run_logs`` table of the registry DB.

    Uses its own connection so the per-record commit never interleaves with
    the caller's transactions (WAL mode keeps writers from starving readers).
    ``run_id`` starts as ``None`` (logs emitted before the run is registered)
    and is backfilled for this handler's own records once the run row exists
    (see :func:`set_db_run_id`).
    """

    def __init__(self, conn: sqlite3.Connection, run_id: int | None = None) -> None:
        super().__init__()
        self.conn = conn
        self.run_id = run_id
        # Log records emitted before the run is registered carry run_id NULL;
        # remember the highest existing id so backfilling later only touches
        # records emitted through this handler, never older unattached ones.
        self._baseline = self._max_run_log_id(conn)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            pass  # non-fatal: fall back to the journal mode already in use

    @staticmethod
    def _max_run_log_id(conn: sqlite3.Connection) -> int:
        try:
            row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM run_logs").fetchone()
        except sqlite3.Error:
            return 0
        return int(row[0]) if row else 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            insert_run_log(self.conn, record.levelname, record.name, self.format(record), self.run_id)
        except Exception:
            self.handleError(record)


def attach_db_logging(conn: sqlite3.Connection, run_id: int | None = None) -> SQLiteLogHandler:
    """Attach (or re-point) the shared DB log handler for the ``spinlabs`` logger.

    Returns the handler so callers can keep a reference or detach it.
    """
    global _db_handler
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    # The attached handler must see the full stream: raise the logger level to
    # DEBUG so per-level filtering is done by handlers (console verbosity, DB
    # capture) rather than by the logger, which would silently drop INFO/DEBUG
    # records before they reach the DB handler.
    logger.setLevel(logging.DEBUG)
    if _db_handler is None or _db_handler not in logger.handlers:
        _db_handler = SQLiteLogHandler(conn, run_id)
        logger.addHandler(_db_handler)
    else:
        _db_handler.conn = conn
        _db_handler.run_id = run_id
    return _db_handler


def db_logging_attached() -> bool:
    """Return True when the shared DB log handler is currently attached."""
    global _db_handler
    return _db_handler is not None and _db_handler in logging.getLogger(ROOT_LOGGER_NAME).handlers


def detach_db_logging() -> None:
    """Remove the shared DB log handler, if attached."""
    global _db_handler
    if _db_handler is not None:
        logging.getLogger(ROOT_LOGGER_NAME).removeHandler(_db_handler)
        _db_handler = None


def set_db_run_id(run_id: int | None) -> None:
    """Link subsequent log records to a registered run id.

    Also backfills the ``run_id`` of records emitted through this handler
    before the run was registered (records older than the handler's baseline
    id, e.g. left over from an earlier run in the same process, are left at
    NULL so they are never misattributed).
    """
    global _db_handler
    if _db_handler is None:
        return
    try:
        _db_handler.conn.execute(
            "UPDATE run_logs SET run_id = ? WHERE run_id IS NULL AND id > ?",
            (run_id, _db_handler._baseline),
        )
        _db_handler.conn.commit()
    except sqlite3.Error:
        pass
    _db_handler.run_id = run_id


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the logger for ``name`` (or the shared ``spinlabs`` logger)."""
    return logging.getLogger(name or ROOT_LOGGER_NAME)


def configure_logging(verbosity: int = 0, quiet: bool = False) -> None:
    """Set up the ``spinlabs`` logger with a single console stream handler.

    ``verbosity`` is the counted ``-v`` flag: 0 shows WARNING and above,
    1 (-v) adds INFO, 2 (-vv) adds DEBUG. ``quiet=True`` (-q) forces ERROR.
    The logger itself always runs at DEBUG so that DB logging — when
    attached — receives the full stream regardless of console verbosity.
    Repeated calls reconfigure the single handler instead of stacking
    duplicates.
    """
    if quiet:
        console_level = logging.ERROR
    elif verbosity >= 2:
        console_level = logging.DEBUG
    elif verbosity == 1:
        console_level = logging.INFO
    else:
        console_level = logging.WARNING

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    global _stream_handler
    if _stream_handler is None or _stream_handler not in logger.handlers:
        _stream_handler = logging.StreamHandler(sys.stdout)
        _stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(_stream_handler)
    _stream_handler.setLevel(console_level)
    logger.propagate = False
