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


def _split_statements(sql: str) -> list[str]:
    """Split a SQL script into complete statements using sqlite3's parser.

    Walks the buffer with `sqlite3.complete_statement()` so triggers,
    multi-line expressions, and `;` inside string literals are handled
    correctly (string-split on `;` would mis-handle these)."""
    statements: list[str] = []
    buf = ""
    for line in sql.splitlines(keepends=True):
        buf += line
        # Strip trailing whitespace-only/comment-only trail for the
        # complete_statement check; sqlite3 wants a terminated statement
        # ending in ';' on a non-comment line.
        if sqlite3.complete_statement(buf):
            # Strip leading comment-only / blank lines from the buffer so
            # that a header comment immediately preceding a statement (very
            # common in migration files) doesn't cause the whole statement
            # to be discarded by the `startswith("--")` check below.
            lines = buf.splitlines(keepends=True)
            while lines and (not lines[0].strip()
                             or lines[0].lstrip().startswith("--")):
                lines.pop(0)
            stmt = "".join(lines).strip()
            if stmt:
                statements.append(stmt)
            buf = ""
    tail = buf.strip()
    if tail and not tail.startswith("--"):
        # Trailing fragment without a final ';' — let sqlite raise a
        # clear error rather than swallowing it.
        statements.append(tail)
    return statements


def apply_migrations(db_path: Path, migrations_dir: Path) -> list[str]:
    """Apply every `*.sql` file in `migrations_dir` (sorted) not yet applied.

    Returns the list of migration names applied during this call (empty if
    already up-to-date). Each migration runs as a SINGLE real transaction
    (parsed into individual statements via `sqlite3.complete_statement`
    rather than `executescript`, because `executescript` issues an implicit
    COMMIT that would defeat rollback-on-failure). On failure the
    transaction rolls back, the `_schema_migrations` row is NOT written,
    and the exception propagates.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in migrations_dir.iterdir()
                   if p.is_file() and p.suffix == ".sql")
    newly_applied: list[str] = []

    with sqlite3.connect(db_path) as conn:
        # Bootstrap the tracking table via executescript (acceptable: this
        # is a single idempotent CREATE TABLE IF NOT EXISTS, no rollback
        # semantics needed).
        conn.executescript(_BOOTSTRAP_SQL)
        already = {r[0] for r in conn.execute(
            "SELECT name FROM _schema_migrations").fetchall()}

        for f in files:
            if f.name in already:
                continue
            sql = f.read_text(encoding="utf-8")
            statements = _split_statements(sql)
            try:
                conn.execute("BEGIN")
                for stmt in statements:
                    conn.execute(stmt)
                conn.execute(
                    "INSERT INTO _schema_migrations (name, applied_at) "
                    "VALUES (?, ?)",
                    (f.name, int(time.time())),
                )
                conn.execute("COMMIT")
            except Exception:
                # Real rollback now — no implicit COMMIT was issued
                # before the BEGIN, so the transaction is genuinely
                # active and can be rolled back.
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass  # already auto-rolled-back; preserve original exc
                raise
            newly_applied.append(f.name)

    return newly_applied
