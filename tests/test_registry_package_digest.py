"""registry.package_digest: the registered sha256 of a package as a cache key, re-hashed after a rewrite."""
import logging
import os

import pytest

from spinlabs.registry import (
    UnregisteredPackageError,
    ensure_db,
    package_digest,
    register_file,
    sha256_of,
)


def _package(tmp_path, content=b"first"):
    path = tmp_path / "data" / "NV_512" / "pert_0.05" / "all_bands" / "20261030" / "NV_512_raw_zfs_data_all_bands_12d.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return path


def _register(conn, path):
    register_file(conn, path, "packaged_zfs", defect="NV", cell_size=512, calc_method="all_bands",
                  pert_order=[1, 2], pert_scale=0.05)


def _row(conn, path):
    return conn.execute(
        "SELECT kind, defect, cell_size, calc_method, pert_order, pert_scale, sha256, registered_at FROM files WHERE path = ?",
        (str(path.resolve()),),
    ).fetchone()


def test_registered_file_returns_stored_hash(tmp_path):
    conn = ensure_db()
    path = _package(tmp_path)
    _register(conn, path)
    before = _row(conn, path)
    assert package_digest(conn, path) == sha256_of(path) == before[6]
    assert _row(conn, path) == before  # nothing re-registered


def test_symlinked_path_resolves_to_the_registered_file(tmp_path):
    conn = ensure_db()
    path = _package(tmp_path)
    _register(conn, path)
    link = tmp_path / "link.npz"
    link.symlink_to(path)
    assert package_digest(conn, link) == sha256_of(path)


def test_rewritten_file_is_rehashed_with_its_metadata(tmp_path, caplog):
    conn = ensure_db()
    path = _package(tmp_path)
    _register(conn, path)
    old = _row(conn, path)
    path.write_bytes(b"second")  # rewritten in place, as rsync does
    later = path.stat().st_mtime + 10
    os.utime(path, (later, later))  # registered_at has one-second resolution
    with caplog.at_level(logging.WARNING, logger="spinlabs.registry"):
        digest = package_digest(conn, path)
    new = _row(conn, path)
    assert digest == sha256_of(path) == new[6] != old[6]
    assert new[:6] == old[:6]  # kind, defect, cell_size, calc_method, pert_order, pert_scale
    assert "re-registered" in caplog.text
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1


def test_unregistered_file_raises_with_register_command(tmp_path):
    conn = ensure_db()
    path = _package(tmp_path)
    with pytest.raises(UnregisteredPackageError, match="spinlabs package --register-only") as exc:
        package_digest(conn, path)
    assert str(path.resolve().parents[3]) in str(exc.value)
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0  # not registered silently
