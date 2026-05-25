import re

import pytest

from aiutopia.common.ids import (
    ULID_REGEX,
    is_ulid,
    memory_id_for,
    new_agent_uuid,
    new_event_id,
    new_plan_id,
    new_report_id,
    new_subgoal_id,
    new_ulid,
    skill_library_id_for,
)


def test_new_ulid_returns_valid_crockford_base32() -> None:
    value = new_ulid()
    assert re.fullmatch(ULID_REGEX, value), value


def test_new_ulid_is_unique_across_calls() -> None:
    assert new_ulid() != new_ulid()


def test_factories_each_return_ulid_strings() -> None:
    for fn in (new_agent_uuid, new_plan_id, new_subgoal_id,
               new_report_id, new_event_id):
        assert is_ulid(fn())


def test_chroma_id_helpers_use_agent_uuid_verbatim() -> None:
    uuid = "01J0CABCDEFGHJKMNPQRSTVWXY"
    assert skill_library_id_for(uuid) == f"skill_lib_{uuid}"
    assert memory_id_for(uuid) == f"mem_{uuid}"


def test_chroma_id_helpers_reject_non_ulid() -> None:
    for bad in ("not-a-ulid",
                "550e8400-e29b-41d4-a716-446655440000",
                ""):
        with pytest.raises(ValueError):
            skill_library_id_for(bad)
        with pytest.raises(ValueError):
            memory_id_for(bad)


def test_is_ulid_rejects_uuidv4_and_garbage() -> None:
    assert not is_ulid("550e8400-e29b-41d4-a716-446655440000")
    assert not is_ulid("not-a-ulid")
    assert not is_ulid("")


@pytest.mark.parametrize("forbidden_char", ["I", "L", "O", "U"])
def test_is_ulid_rejects_each_forbidden_crockford_char(forbidden_char: str) -> None:
    # 25 valid '0' chars + 1 forbidden char = 26 total but the forbidden
    # char is at a known position, isolating each character independently.
    candidate = "0" * 25 + forbidden_char
    assert not is_ulid(candidate)


def test_is_ulid_returns_false_on_non_string_input() -> None:
    """Defensive: callers may pass JSON-derived values that could be None,
    int, bytes, etc. Must return False, NOT raise TypeError."""
    for value in (None, 123, 12.5, b"01J0CABCDEFGHJKMNPQRSTVWXY", ["x"], {"x": 1}):
        assert is_ulid(value) is False
