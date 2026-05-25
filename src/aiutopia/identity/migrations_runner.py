"""Minimal forward-only SQLite migration runner.

Migrations are filename-ordered (e.g. `001_initial.sql`, `002_add_index.sql`).
A `_schema_migrations` table tracks which have been applied; running twice is
a no-op.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


_BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS _schema_migrations (
    name        TEXT PRIMARY KEY,
    applied_at  INTEGER NOT NULL
)
"""


def applied_migrations(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_BOOTSTRAP_SQL)
        rows = conn.execute(
            "SELECT name FROM _schema_migrations ORDER BY applied_at, name"
        ).fetchall()
    return [r[0] for r in rows]


def apply_migrations(db_path: Path, migrations_dir: Path) -> list[str]:
    """Apply every `*.sql` file in `migrations_dir` (sorted) not yet applied.

    Returns the list of migration names applied during this call (empty if
    already up-to-date). Each migration runs in its own transaction; on
    failure the transaction rolls back and the exception propagates.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in migrations_dir.iterdir()
                   if p.is_file() and p.suffix == ".sql")
    newly_applied: list[str] = []

    with sqlite3.connect(db_path) as conn:
        conn.executescript(_BOOTSTRAP_SQL)
        already = {r[0] for r in conn.execute(
            "SELECT name FROM _schema_migrations").fetchall()}

        for f in files:
            if f.name in already:
                continue
            sql = f.read_text()
            try:
                conn.execute("BEGIN")
                conn.executescript(sql)
                conn.execute(
                    "INSERT INTO _schema_migrations (name, applied_at) "
                    "VALUES (?, ?)",
                    (f.name, int(time.time())),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            newly_applied.append(f.name)

    return newly_applied
