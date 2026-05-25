"""ULID-only identifier helpers (§3.5 conventions)."""
from __future__ import annotations

import re

import ulid

# Crockford base32: 0-9 + A-HJKMNP-TV-Z (no I, L, O, U)
ULID_REGEX = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def new_ulid() -> str:
    """Return a new ULID as a 26-character Crockford-base32 string."""
    return str(ulid.new())


def is_ulid(value: str) -> bool:
    """True iff `value` matches the ULID regex."""
    return bool(ULID_REGEX.fullmatch(value))


# Domain factories (each returns a fresh ULID; named for grep-ability)
def new_agent_uuid() -> str:
    return new_ulid()


def new_plan_id() -> str:
    return new_ulid()


def new_subgoal_id() -> str:
    return new_ulid()


def new_report_id() -> str:
    return new_ulid()


def new_event_id() -> str:
    return new_ulid()


def skill_library_id_for(agent_uuid: str) -> str:
    """Chroma collection name for an agent's skill library (§3.5 convention)."""
    if not is_ulid(agent_uuid):
        raise ValueError(f"not a ULID: {agent_uuid!r}")
    return f"skill_lib_{agent_uuid}"


def memory_id_for(agent_uuid: str) -> str:
    """Chroma collection name for an agent's episodic memory (§3.5 convention)."""
    if not is_ulid(agent_uuid):
        raise ValueError(f"not a ULID: {agent_uuid!r}")
    return f"mem_{agent_uuid}"
