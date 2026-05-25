"""IdentityService — CRUD over identity.db with §3.6 succession semantics.

In M0 the service is in-process synchronous SQLite. In M5+ it gains a
Py4J callback hook so on-death events from Fabric drive the same code
path. Death/spawn methods here are unit-tested with dry-runs only —
the Carpet `/player kill` and `/player spawn` calls are wired in CLI
T30 once the bridge is up.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from aiutopia.common.ids import (
    is_ulid, memory_id_for, new_agent_uuid, skill_library_id_for,
)
from aiutopia.identity.migrations_runner import apply_migrations
from aiutopia.identity.models import Agent, Role, RoleId


# Subset of migrations belonging to identity.db (vs planner_state.db).
_IDENTITY_MIGRATIONS = ("identity_",)
_PLANNER_MIGRATIONS  = ("planner_state_",)


def _migrations_for(prefixes: tuple[str, ...], dir_path: Path) -> Path:
    """Materialize a temp dir containing only the matching migrations,
    so the shared runner only applies the right subset."""
    out = Path(tempfile.mkdtemp(prefix="aiutopia_mig_"))
    for f in dir_path.iterdir():
        if any(f.name.startswith(p) for p in prefixes):
            shutil.copy(f, out / f.name)
    return out


def init_identity_db(db_path: Path, migrations_dir: Path) -> list[str]:
    """Apply only the identity_*.sql migrations to db_path."""
    subset = _migrations_for(_IDENTITY_MIGRATIONS, migrations_dir)
    try:
        return apply_migrations(db_path, subset)
    finally:
        shutil.rmtree(subset)


def init_planner_state_db(db_path: Path, migrations_dir: Path) -> list[str]:
    """Apply only the planner_state_*.sql migrations to db_path."""
    subset = _migrations_for(_PLANNER_MIGRATIONS, migrations_dir)
    try:
        return apply_migrations(db_path, subset)
    finally:
        shutil.rmtree(subset)


class IdentityService:
    """All reads + writes for identity.db. Thread-safe per connection."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ───── reads ─────
    def get_role(self, role_id: RoleId) -> Role:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM roles WHERE role_id = ?", (role_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"no such role: {role_id!r}")
        return Role(
            role_id=row["role_id"],
            display_name=row["display_name"],
            policy_weights_path=row["policy_weights_path"],
            policy_version=row["policy_version"],
            observation_schema_version=row["observation_schema_version"],
            action_schema_version=row["action_schema_version"],
            max_lives=row["max_lives"],
            default_skin_pool=json.loads(row["default_skin_pool"] or "[]"),
        )

    def get_agent(self, agent_uuid: str) -> Agent:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM agents WHERE agent_uuid = ?", (agent_uuid,)
            ).fetchone()
        if row is None:
            raise KeyError(f"no such agent: {agent_uuid!r}")
        return Agent(
            agent_uuid=row["agent_uuid"],
            role_id=row["role_id"],
            agent_name=row["agent_name"],
            skill_library_id=row["skill_library_id"],
            memory_id=row["memory_id"],
            status=row["status"],
            born_at=row["born_at"],
            died_at=row["died_at"],
            spawn_position_json=row["spawn_position_json"],
            current_skin=row["current_skin"],
        )

    def list_living_agents(self) -> list[Agent]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT agent_uuid FROM agents WHERE status = 'alive' ORDER BY born_at"
            ).fetchall()
        return [self.get_agent(r["agent_uuid"]) for r in rows]

    # ───── writes ─────
    def spawn_agent(self, role_id: RoleId, agent_name: str, born_at: int,
                    spawn_position_json: str | None = None,
                    skin: str | None = None) -> Agent:
        agent_uuid = new_agent_uuid()
        skill_lib  = skill_library_id_for(agent_uuid)
        mem_id     = memory_id_for(agent_uuid)
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO agents
                       (agent_uuid, role_id, agent_name, skill_library_id,
                        memory_id, status, born_at, died_at,
                        spawn_position_json, current_skin)
                   VALUES (?, ?, ?, ?, ?, 'alive', ?, NULL, ?, ?)""",
                (agent_uuid, role_id, agent_name, skill_lib, mem_id,
                 born_at, spawn_position_json, skin),
            )
            conn.execute(
                """INSERT INTO agent_lives
                       (agent_uuid, role_id, born_at)
                   VALUES (?, ?, ?)""",
                (agent_uuid, role_id, born_at),
            )
        return self.get_agent(agent_uuid)

    def record_death(self, agent_uuid: str, died_at: int,
                     cause_of_death: str) -> None:
        if not is_ulid(agent_uuid):
            raise ValueError(f"not a ULID: {agent_uuid!r}")
        with self._conn() as conn:
            conn.execute(
                "UPDATE agents SET status='dead', died_at=? "
                "WHERE agent_uuid=? AND status='alive'",
                (died_at, agent_uuid),
            )
            conn.execute(
                "UPDATE agent_lives SET died_at=?, cause_of_death=? "
                "WHERE agent_uuid=? AND died_at IS NULL",
                (died_at, cause_of_death, agent_uuid),
            )

    def record_funeral(self, deceased_agent_uuid: str,
                       witness_uuids: list[str],
                       event_summary: str, written_at: int) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO funerals
                       (deceased_agent_uuid, witness_agent_uuids_json,
                        event_summary, written_to_memory_at)
                   VALUES (?, ?, ?, ?)""",
                (deceased_agent_uuid, json.dumps(witness_uuids),
                 event_summary, written_at),
            )
            return cur.lastrowid or -1
