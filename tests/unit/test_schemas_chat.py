import pytest

from aiutopia.schemas.chat import ChatEvent


def test_minimal_chat_event_validates() -> None:
    e = ChatEvent(
        sender_player_uuid="player-mojang-uuid",
        sender_player_name="Carte",
        addressed_agent_uuid="01J0CABCDEFGHJKMNPQRSTVWXY",
        addressed_agent_name="Bjorn",
        text="where is the iron?",
        timestamp=1_700_000_000,
    )
    assert e.expected_reply_type == "text"
    assert e.suppressed_in_chat is True
    assert e.schema_version == "1.0.0"


def test_chat_event_rejects_too_long_text() -> None:
    with pytest.raises(Exception):
        ChatEvent(
            sender_player_uuid="x", sender_player_name="x",
            addressed_agent_uuid="x", addressed_agent_name="x",
            text="x" * 1_001, timestamp=1,
        )


def test_chat_event_rejects_empty_text() -> None:
    with pytest.raises(Exception):
        ChatEvent(
            sender_player_uuid="x", sender_player_name="x",
            addressed_agent_uuid="x", addressed_agent_name="x",
            text="", timestamp=1,
        )
