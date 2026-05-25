"""ULID-only identifier helpers (§3.5 conventions)."""
from __future__ import annotations

import re

import ulid

# Crockford base32: 0-9 + A-HJKMNP-TV-Z (no I, L, O, U)
ULID_REGEX = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def new_ulid() -> str:
    """Return a new ULID as a 26-character Crockford-base32 string.

    NOTE: not monotonic across calls — each ULID's random component is
    independent. Fine at our event rate (orders of magnitude below the
    1M/sec where collisions become a concern). If strict ordering within
    the same millisecond is ever needed, switch to a MonotonicULIDFactory.
    """
    return str(ulid.ULID())


def is_ulid(value: object) -> bool:
    """True iff `value` is a string matching the ULID regex.

    Safe to call with any input — returns False for non-strings instead
    of raising. This makes it usable as a Pydantic validator + as the
    gate inside `skill_library_id_for` / `memory_id_for` without those
    raising TypeError instead of the documented ValueError."""
    return isinstance(value, str) and ULID_REGEX.fullmatch(value) is not None


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
