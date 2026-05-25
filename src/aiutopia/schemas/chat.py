"""§6.5 — ChatEvent."""
from __future__ import annotations

from pydantic import BaseModel, Field

from aiutopia.common.ids import new_event_id
from aiutopia.schemas.enums import ExpectedReplyType, SCHEMA_VERSION_LLM_PLAN


_ULID_PATTERN = r"^[0-9A-HJKMNP-TV-Z]{26}$"


class ChatEvent(BaseModel):
    event_id:             str  = Field(default_factory=new_event_id,
                                        pattern=_ULID_PATTERN)
    schema_version:       str  = SCHEMA_VERSION_LLM_PLAN
    sender_player_uuid:   str  = Field(..., description="Mojang UUID of player")
    sender_player_name:   str  = Field(..., min_length=1, max_length=16)
    addressed_agent_uuid: str  = Field(..., pattern=_ULID_PATTERN)
    addressed_agent_name: str  = Field(..., min_length=1, max_length=16)
    text:                 str  = Field(..., min_length=1, max_length=1000,
        description="raw chat text WITHOUT the leading @<agent_name> prefix")
    timestamp:            int
    expected_reply_type:  ExpectedReplyType = "text"
    suppressed_in_chat:   bool = True
