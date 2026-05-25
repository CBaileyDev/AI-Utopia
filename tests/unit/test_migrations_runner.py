import sqlite3
from pathlib import Path

import pytest

from aiutopia.identity.migrations_runner import (
    apply_migrations,
    applied_migrations,
)


def test_apply_migrations_creates_versions_table_and_runs_files(tmp_path: Path) -> None:
    mig_dir = tmp_path / "migrations"
    mig_dir.mkdir()
    (mig_dir / "001_create_widgets.sql").write_text(
        "CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT);"
    )
    (mig_dir / "002_seed_widget.sql").write_text(
        "INSERT INTO widgets (name) VALUES ('hello');"
    )

    db = tmp_path / "test.db"
    apply_migrations(db, mig_dir)

    with sqlite3.connect(db) as conn:
        widgets = conn.execute("SELECT name FROM widgets").fetchall()
        applied = conn.execute(
            "SELECT name FROM _schema_migrations ORDER BY applied_at"
        ).fetchall()

    assert widgets == [("hello",)]
    assert [r[0] for r in applied] == [
        "001_create_widgets.sql",
        "002_seed_widget.sql",
    ]


def test_apply_migrations_is_idempotent(tmp_path: Path) -> None:
    mig_dir = tmp_path / "migrations"
    mig_dir.mkdir()
    (mig_dir / "001_create.sql").write_text(
        "CREATE TABLE t (id INTEGER PRIMARY KEY);"
    )
    db = tmp_path / "test.db"
    apply_migrations(db, mig_dir)
    apply_migrations(db, mig_dir)  # second run is a no-op
    assert applied_migrations(db) == ["001_create.sql"]


def test_apply_migrations_runs_in_filename_order(tmp_path: Path) -> None:
    mig_dir = tmp_path / "migrations"
    mig_dir.mkdir()
    (mig_dir / "002_b.sql").write_text("CREATE TABLE b (id INTEGER PRIMARY KEY);")
    (mig_dir / "001_a.sql").write_text("CREATE TABLE a (id INTEGER PRIMARY KEY);")
    db = tmp_path / "test.db"
    apply_migrations(db, mig_dir)
    assert applied_migrations(db) == ["001_a.sql", "002_b.sql"]
