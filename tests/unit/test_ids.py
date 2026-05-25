import re

import pytest

from aiutopia.common.ids import (
    ULID_REGEX,
    is_ulid,
    new_agent_uuid,
    new_plan_id,
    new_subgoal_id,
    new_report_id,
    new_event_id,
    new_ulid,
    skill_library_id_for,
    memory_id_for,
)


def test_new_ulid_returns_valid_crockford_base32() -> None:
    value = new_ulid()
    assert re.fullmatch(ULID_REGEX, value), value


def test_factories_each_return_ulid_strings() -> None:
    for fn in (new_agent_uuid, new_plan_id, new_subgoal_id,
               new_report_id, new_event_id):
        assert is_ulid(fn())


def test_chroma_id_helpers_use_agent_uuid_verbatim() -> None:
    uuid = "01J0CABCDEFGHJKMNPQRSTVWXY"
    assert skill_library_id_for(uuid) == f"skill_lib_{uuid}"
    assert memory_id_for(uuid) == f"mem_{uuid}"


def test_is_ulid_rejects_uuidv4_and_garbage() -> None:
    assert not is_ulid("550e8400-e29b-41d4-a716-446655440000")
    assert not is_ulid("not-a-ulid")
    assert not is_ulid("")
    # I, L, O, U are excluded from Crockford base32
    assert not is_ulid("01J0CABCDEFGHIJKLMNPQRSTVW")
